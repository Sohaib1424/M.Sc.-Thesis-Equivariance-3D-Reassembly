"""
Vector Neuron primitives: SO(3)-equivariant layers on "vector-list" features
of shape ``(..., C, 3)`` -- C channels, each a 3-vector.

The unifying rule (Deng et al., 2021): a learned weight never touches the
trailing xyz axis. Every operation here either mixes across the *channel* axis
(which commutes with rotation, since rotation acts on the xyz axis) or uses
inner products between co-rotating vectors (invariant) to decide how to combine
them -- never introducing a direction of its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6


class VNLinear(nn.Module):
    """``(..., C_in, 3) -> (..., C_out, 3)``, mixing channels only.

    Equivariant because ``R`` acts on the xyz axis and ``W`` on the channel
    axis, and they commute: ``VNLinear(R x) == R VNLinear(x)``.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x.transpose(-1, -2)).transpose(-1, -2)


class VNLeakyReLU(nn.Module):
    """Equivariant nonlinearity.

    Learns a direction ``q = VNLinear(x)`` (itself equivariant) per channel and
    attenuates only the component of ``x`` along ``q`` when ``<x, q> < 0``.
    The branch decision uses ``<x, q>``, which is rotation-invariant, so it
    never depends on an externally-fixed frame.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2,
                 share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_dir = VNLinear(channels, 1 if share_nonlinearity else channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.map_to_dir(x)
        if q.shape[-2] != x.shape[-2]:
            q = q.expand_as(x)
        dot = (x * q).sum(dim=-1, keepdim=True)                  # invariant
        q_sq = (q * q).sum(dim=-1, keepdim=True).clamp_min(EPS)
        proj = (dot / q_sq) * q
        return torch.where(dot >= 0, x, x - (1 - self.negative_slope) * proj)


class VNLayerNorm(nn.Module):
    """Equivariant layer norm: normalizes the per-channel magnitudes of each
    node across channels, leaving every direction untouched.

    PREFERRED OVER BATCH NORM HERE, for three reasons:

    1. Batch norm over the vertex axis makes one scene's output depend on which
       *other* scenes landed in the same batch. That is the same class of
       defect as cross-scene attention leakage -- less severe, but it still
       means "the model's prediction for this object" is not a function of that
       object alone.
    2. Under DDP it adds a running-stats buffer broadcast every iteration and
       makes the two ranks' statistics differ, for no benefit.
    3. It behaves badly when a fragment is tiny (the pruned fracture-surface
       variant can produce very small graphs).

    ``VNBatchNorm`` is kept below for parity with the original implementation
    and for ablations.
    """

    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1)                                    # (..., C) invariant
        mean = norm.mean(dim=-1, keepdim=True)
        var = norm.var(dim=-1, keepdim=True, unbiased=False)
        scaled = (norm - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            scaled = scaled * self.weight
        # Rescale each direction by its normalized magnitude.
        direction = x / norm.clamp_min(EPS).unsqueeze(-1)
        return direction * scaled.unsqueeze(-1)


class VNBatchNorm(nn.Module):
    """Equivariant batch norm on the (invariant) per-channel magnitudes."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1) + EPS
        norm_bn = self.bn(norm)
        return (x / norm.unsqueeze(-1)) * norm_bn.unsqueeze(-1)


def make_norm(kind: str, channels: int) -> nn.Module:
    kind = (kind or "layer").lower()
    if kind in ("layer", "layernorm", "ln"):
        return VNLayerNorm(channels)
    if kind in ("batch", "batchnorm", "bn"):
        return VNBatchNorm(channels)
    if kind in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"unknown norm kind {kind!r}")


class VNInvariant(nn.Module):
    """``(..., C, 3) -> (..., C*C)``: the Gram matrix of channel inner products.

    A complete, standard rotation invariant -- richer than per-channel norms
    alone, since it also captures relative angles between channels.

    Note the output size is quadratic in ``C``. With the head-dim change in
    ``vn_gat.py``, ``C`` is small, so this stays cheap; if you raise
    ``hidden_channels`` a lot, this MLP input grows as ``C^2``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = torch.matmul(x, x.transpose(-1, -2))
        return gram.flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Rotation representation
# ---------------------------------------------------------------------------
def gram_schmidt_frame(a1: torch.Tensor, a2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Two equivariant 3-vectors -> an orthonormal frame with columns b1,b2,b3.

    (The "6D rotation representation", Zhou et al., CVPR 2019.) Always a proper
    rotation: ``b3 = b1 x b2`` forces ``det = +1``.

    This frame is LEFT-equivariant: ``frame(R a1, R a2) == R frame(a1, a2)``.
    Which side that lands on matters -- see :func:`rotation_6d_to_matrix`.
    """
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2_orth / (a2_orth.norm(dim=-1, keepdim=True) + eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_6d_to_matrix(a1: torch.Tensor, a2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Predicted rotation from two equivariant vector channels.

    THE TRANSPOSE IS THE POINT, and it was missing before.

    The network's Gram-Schmidt frame ``F`` is left-equivariant: rotate the
    input scene by ``Q`` and ``F -> Q F``. But the quantity being supervised
    transforms the *other* way. ``R_gt = R_diffuse^T``; further rotating the
    scattered input by ``Q`` makes the effective diffusion ``Q R_diffuse``, so

        R_gt  ->  (Q R_diffuse)^T  =  R_gt Q^T

    Returning ``F`` directly gives ``Q R_pred``, which is not ``R_pred Q^T``
    for any non-trivial ``Q``. The model was therefore structurally unable to
    be equivariantly consistent with its own target: it could still fit the
    data, but only by *fighting* its inductive bias instead of using it --
    precisely the sample efficiency this thesis is trying to buy.

    Returning ``F^T`` gives ``(Q F)^T = F^T Q^T = R_pred Q^T``, matching the
    target's law exactly. Interpretation: ``F`` is the object's own frame in
    the current (scattered) pose, and the rotation that undoes the scatter is
    the map from that frame back to the canonical one, i.e. its inverse.

    Verified numerically (``validation/v01_rotation_convention.py``), and
    guarded by ``tests/test_model.py::test_rotation_head_matches_target_law``.
    """
    return gram_schmidt_frame(a1, a2, eps=eps).transpose(-1, -2)


# ---------------------------------------------------------------------------
# Rotation losses / metrics
# ---------------------------------------------------------------------------
def geodesic_rotation_angle(
    R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Geodesic angle on SO(3), in radians: ``arccos((tr(R_pred^T R_gt) - 1)/2)``.

    This is the right *metric* (it is what GARF's RMSE(R) reports, in degrees)
    but a poor *loss*: ``d/du arccos(u) = -1/sqrt(1-u^2)`` diverges as ``u -> 1``,
    i.e. the gradient blows up exactly as the prediction becomes correct. With
    ``eps = 1e-7`` the gradient magnitude is capped around 2000, which is more
    than enough to destabilize late training. Use
    :func:`chordal_rotation_loss` to optimize and this to report.
    """
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_theta)


def chordal_rotation_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance ``||R_pred - R_gt||_F^2``, per rotation.

    Equals ``2 * (3 - tr(R_pred^T R_gt)) = 4 * (1 - cos(theta/2)^2) * 2``, i.e.
    a strictly increasing function of the geodesic angle on [0, pi]. Same
    minimizer, same ordering, but smooth everywhere with bounded gradients.
    This is what most modern SE(3) regression work optimizes.
    """
    return ((R_pred - R_gt) ** 2).sum(dim=(-1, -2))


def geodesic_rotation_loss(
    R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Kept for parity with the design document. Prefer the chordal loss."""
    return geodesic_rotation_angle(R_pred, R_gt, eps=eps)


def rotation_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    kind: str = "chordal",
    eps: float = 1e-7,
) -> torch.Tensor:
    kind = kind.lower()
    if kind == "chordal":
        return chordal_rotation_loss(R_pred, R_gt)
    if kind == "geodesic":
        return geodesic_rotation_angle(R_pred, R_gt, eps=eps)
    if kind == "hybrid":
        return chordal_rotation_loss(R_pred, R_gt) + 0.1 * geodesic_rotation_angle(
            R_pred, R_gt, eps=eps
        )
    raise ValueError(f"unknown rotation loss kind {kind!r}")
