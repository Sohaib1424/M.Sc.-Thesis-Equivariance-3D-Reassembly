"""
Vector Neuron primitives -- SO(3)-equivariant layers on "vector-list" features
of shape (..., C, 3): C channels, each a 3-vector.

The unifying rule (Deng et al., 2021): never let a learned weight touch the
trailing xyz axis. Every op here either mixes across the CHANNEL axis only
(which commutes with rotation, since rotation acts on the xyz axis), or uses
inner products between co-rotating vectors (which are invariant) to decide how
to combine them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6
_HALF = (torch.float16, torch.bfloat16)


def _at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """
    Promote half precision to float32, but leave float64 alone.

    An unconditional `.float()` would silently DOWNCAST a float64 model -- which
    the equivariance tests run in, precisely because they need the extra digits
    -- and would then raise on any float64 tensor it was combined with.
    """
    return x.float() if x.dtype in _HALF else x


class VNLinear(nn.Module):
    """(..., C_in, 3) -> (..., C_out, 3). Channel mixing only.

    VNLinear(R x) == R VNLinear(x) for every R in SO(3), because R acts on the
    xyz axis and W acts on the channel axis and the two commute trivially.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.map = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE for callers: the trailing transpose makes the result
        # NON-CONTIGUOUS, so downstream head-splitting must use `.reshape()`,
        # never `.view()` (which raises on an incompatible stride).
        return self.map(x.transpose(-1, -2)).transpose(-1, -2)


class VNLeakyReLU(nn.Module):
    """
    Equivariant nonlinearity.

    For each channel a learned direction q = VNLinear(x) (itself equivariant);
    the component of x along q is attenuated only when <x, q> < 0:

        out = x                                     if <x, q> >= 0
        out = x - (1 - slope) * proj_q(x)           otherwise

    Equivariant because the branch decision <x, q> is a rotation-invariant
    scalar -- it never consults an externally fixed frame.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2, share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_dir = VNLinear(channels, 1 if share_nonlinearity else channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.map_to_dir(x)
        if q.shape[-2] != x.shape[-2]:
            q = q.expand_as(x)
        dot = (x * q).sum(dim=-1, keepdim=True)
        q_sq = (q * q).sum(dim=-1, keepdim=True).clamp_min(EPS)
        proj = (dot / q_sq) * q
        return torch.where(dot >= 0, x, x - (1 - self.negative_slope) * proj)


class VNLayerNorm(nn.Module):
    """
    Equivariant normalisation with NO batch statistics.

    Rotation changes a vector's direction, never its length, so the per-channel
    norm ||x_c|| is an invariant scalar. This normalises those norms ACROSS
    CHANNELS within each node independently, then rescales the (unit)
    directions -- directions are never touched, so equivariance holds.

    Why this is the default here rather than VNBatchNorm: the "batch" in this
    pipeline is a variable-size set of mesh vertices whose count swings by more
    than an order of magnitude between scenes (a 2-piece break vs. an 86-piece
    one), the two DDP ranks see completely different scenes, and micro-batching
    splits a batch into one scene at a time. Batch statistics under those
    conditions are noisy, rank-dependent, and mismatched between train and
    eval. Per-node normalisation removes the problem rather than synchronising
    it -- no running buffers, no SyncBatchNorm collective, identical behaviour
    in both modes.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1)                                    # (..., C) invariant
        rms = norm.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        scale = (self.weight / rms).unsqueeze(-1)                # (..., C, 1)
        return x * scale


class VNBatchNorm(nn.Module):
    """
    Equivariant batch norm on the invariant per-channel norms.

    Kept for comparison against `VNLayerNorm` (select with `norm='batch'`),
    since it is what the original design used. Note it introduces running
    buffers that DDP will broadcast every step and that are estimated from
    whatever scenes a rank happened to draw.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-2], 3)
        norm = flat.norm(dim=-1) + EPS
        norm_bn = self.bn(norm)
        out = flat / norm.unsqueeze(-1) * norm_bn.unsqueeze(-1)
        return out.reshape(shape)


def make_norm(kind: str, channels: int) -> nn.Module:
    kind = (kind or "layer").lower()
    if kind == "layer":
        return VNLayerNorm(channels)
    if kind == "batch":
        return VNBatchNorm(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm must be 'layer', 'batch' or 'none', got {kind!r}")


class VNInvariant(nn.Module):
    """
    Equivariant vectors -> rotation-invariant scalars, via the pairwise Gram
    matrix G_ij = <x_i, x_j> (invariant: both vectors co-rotate and rotations
    preserve inner products), flattened.

    `bottleneck` is a mandatory memory control, not a modelling nicety. The
    Gram matrix is O(C^2) PER ENTITY: at hidden=64 the edge head's input has
    2*64+3 = 131 channels, so a batch with ~95k undirected edges would allocate
    95k x 131 x 131 floats -- 6.5 GB for a single tensor that is then retained
    for backward. Projecting to `bottleneck` channels first (equivariantly,
    with a VNLinear, so invariance is preserved exactly) makes it
    95k x 16 x 16, i.e. ~100 MB.
    """

    def __init__(self, in_channels: int, bottleneck: int = 16):
        super().__init__()
        self.bottleneck = min(bottleneck, in_channels)
        self.project = (
            VNLinear(in_channels, self.bottleneck) if self.bottleneck != in_channels else nn.Identity()
        )

    @property
    def out_features(self) -> int:
        return self.bottleneck * self.bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.project(x)
        gram = torch.matmul(z, z.transpose(-1, -2))
        return gram.flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Rotation head
# ---------------------------------------------------------------------------
def gram_schmidt_frame(a1: torch.Tensor, a2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    6D rotation representation (Zhou et al., CVPR 2019): orthonormalise two
    equivariant 3-vector channels into a proper rotation.

        b1 = normalize(a1)
        b2 = normalize(a2 - <b1, a2> b1)
        b3 = b1 x b2
        F  = [b1 b2 b3]   (as COLUMNS)

    Always a valid rotation (det = +1, guaranteed by the cross product rather
    than an arbitrary third vector). Replaces the design document's quaternion
    regression, which is not a linear representation of SO(3) and cannot be
    produced equivariantly by a VN backbone.

    Returns (..., 3, 3). This is the FRAME, and it is LEFT-equivariant:
        F(A a1, A a2) = A F(a1, a2).
    Read `predict_rotation` before using it as a predicted rotation.

    Computed in at least float32, even under autocast: the two normalisations
    and the cross product are exactly where float16 hurts, and a rotation matrix
    is only (F, 3, 3) -- widening it costs nothing. float64 is preserved rather
    than downcast, so a double-precision equivariance check stays meaningful.
    """
    a1 = _at_least_float32(a1)
    a2 = _at_least_float32(a2)
    # The epsilon CLAMPS the norm rather than being added to it. Adding it
    # gives ||b1|| = ||a1|| / (||a1|| + eps), which is short of 1 by
    # eps/||a1|| -- a relative error that grows without bound as the predicted
    # vectors shrink. Measured: at ||a|| = 1e-6 the "rotation" is 3% away from
    # orthogonal, so R R^T != I, det != 1, and the geodesic loss's
    # trace formula (which assumes a rotation) silently stops measuring an
    # angle. Small activations are exactly what an untrained network produces.
    # Clamping leaves the normalisation exact for every ||a|| >= eps and only
    # guards the genuinely degenerate zero-vector case.
    # `sqrt(sum + eps^2)` rather than `norm().clamp_min(eps)`: clamping leaves
    # the gradient of `.norm()` undefined at the zero vector, and `a2_orth` is
    # exactly zero whenever the two predicted channels are parallel -- which an
    # untrained head does produce. Folding eps inside the sqrt is smooth
    # everywhere and still exact to machine precision for any ||a|| >> eps.
    b1 = a1 / torch.sqrt(a1.pow(2).sum(-1, keepdim=True) + eps * eps)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2_orth / torch.sqrt(a2_orth.pow(2).sum(-1, keepdim=True) + eps * eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def predict_rotation(a1: torch.Tensor, a2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    The predicted rotation R_pred = F^T, where F is the Gram-Schmidt frame.

    ### Why the transpose is REQUIRED, not cosmetic

    Every VN primitive in this file is LEFT-equivariant: rotate the input
    geometry by A (v -> A v) and every feature, and hence the frame, satisfies

        F(A x) = A F(x).                                                  (1)

    The supervision target points the other way. The network is fed the
    diffused fragment (geometry A x, where A is the unknown scattering
    rotation) and must output the rotation that undoes it:

        R_gt = A^T.                                                       (2)

    Suppose the head returned F directly. Combining (1) and (2) it would have
    to satisfy A F(x_clean) = A^T for the clean geometry x_clean, i.e.
    F(x_clean) = (A^2)^T. But x_clean does not depend on A -- the same clean
    fragment is scattered by a different random A on every draw -- so a single
    F(x_clean) would have to equal (A^2)^T for every A simultaneously. It
    cannot. The target is unlearnable by ANY left-equivariant head, and the
    failure mode is silent: training simply plateaus near chance with no error.

    Transposing fixes it. With G(x) := F(x)^T, equation (1) becomes

        G(A x) = (A F(x))^T = F(x)^T A^T = G(x) A^T,                      (3)

    i.e. G is RIGHT-equivariant. Now a single, A-independent thing needs to be
    learned -- G(x_clean) = I on canonically-oriented geometry -- and (3) then
    delivers G(A x_clean) = A^T = R_gt automatically for EVERY A. That is
    exactly the object the equivariance is supposed to buy.

    Verified numerically in `tests/test_rotation_convention.py`, which checks
    both that (3) holds and that the un-transposed version is inconsistent
    across two different diffusion rotations.
    """
    return gram_schmidt_frame(a1, a2, eps=eps).transpose(-1, -2)


def geodesic_rotation_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic angle on SO(3) in radians, shape (...,).

    Computed as `atan2(sin(theta), cos(theta))` rather than
    `arccos((tr - 1) / 2)`, taking

        cos(theta) = (tr(R) - 1) / 2
        sin(theta) = ||[R32 - R23, R13 - R31, R21 - R12]|| / 2

    (For R = exp(theta K), R - R^T = 2 sin(theta) K with K the unit-axis skew
    matrix, and theta lies in [0, pi] so sin(theta) >= 0 and the branch is
    unambiguous.)

    Two reasons this matters, both practical rather than cosmetic:

    * ARCCOS NEEDS A CLAMP AND THE CLAMP IS A FLOOR. Guarding the domain with
      `clamp(-1 + eps, 1 - eps)` means a numerically perfect prediction reports
      `sqrt(2 * eps)` instead of zero -- about 0.026 degrees at eps = 1e-7,
      before float32 noise in the trace roughly doubles it. `atan2` needs no
      clamp and returns exactly 0 for a perfect prediction.

    * ARCCOS'S GRADIENT DIVERGES EXACTLY WHERE TRAINING ENDS UP. d/dx arccos(x)
      is -1/sqrt(1 - x^2), which blows up as the prediction converges and
      cos(theta) approaches 1 -- so this term's gradient grows without bound
      the better the model gets, and after clipping it crowds out every other
      loss term. The atan2 form has a bounded, well-behaved gradient
      everywhere.

    Always evaluated in at least float32: near theta = 0 the trace is the
    difference of numbers close to 3, and float16 has nothing left to resolve it
    with.
    """
    R_pred = _at_least_float32(R_pred)
    R_gt = _at_least_float32(R_gt).to(R_pred.dtype)
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)

    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    axis = torch.stack([
        R_diff[..., 2, 1] - R_diff[..., 1, 2],
        R_diff[..., 0, 2] - R_diff[..., 2, 0],
        R_diff[..., 1, 0] - R_diff[..., 0, 1],
    ], dim=-1)
    # NOT `axis.norm(dim=-1)`. `axis` is exactly zero whenever the residual
    # rotation is symmetric -- theta = 0 (a perfect prediction) or theta = pi
    # (the worst possible one) -- and torch's `.norm()` has an UNDEFINED
    # gradient there, which propagates NaN into every weight on the next step
    # and never recovers. Folding eps inside the sqrt makes the gradient 0 at
    # the origin instead of NaN, while leaving the value untouched to ~1e-13
    # relative for any axis of realistic magnitude.
    sin_theta = torch.sqrt(axis.pow(2).sum(-1) + 1e-12) / 2
    return torch.atan2(sin_theta, cos_theta)
