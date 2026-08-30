"""
Vector Neuron primitives.

Every module here maps ``(N, C, 3)`` to ``(N, C', 3)`` and satisfies

    f(x R^T) = f(x) R^T          for every R in SO(3)

*exactly* -- as an algebraic identity, not approximately and not as something
learned from augmented data. ``tests/test_vn.py`` asserts it to 1e-6 on random
rotations for every module in this file.

What that identity forbids
--------------------------
Three ordinary neural-network habits break it, and each has bitten this
project at least once:

* **A bias term.** ``W x + b`` adds a fixed vector, and a fixed vector does not
  rotate. Every linear map here is bias-free; this is not an oversight.
* **A pointwise nonlinearity.** ``relu(x)`` clamps each *coordinate*, which is
  a statement about the axes of the coordinate system. :class:`VNLeakyReLU`
  instead reflects about a learned, equivariant hyperplane, so the decision it
  makes travels with the data.
* **Any per-coordinate normalisation.** Dividing the x, y and z components by
  different numbers is a shear. :class:`VNLayerNorm` divides all three by one
  invariant scalar.

The general rule: a scalar computed from vector features may be used freely as
long as it is *invariant*, because an invariant scalar times an equivariant
vector is still equivariant. Inner products, norms and Gram matrices are
invariant; individual components are not.

Layout
------
``(N, C, 3)`` with the spatial axis last, so a rotation is ``x @ R.T`` and
broadcasts over both leading axes. The channel axis is second because every
linear map acts on it and ``W @ x`` then broadcasts without a transpose.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

# Guards ``sqrt`` and division near zero. Zero vectors are not hypothetical:
# a vertex normal of a degenerate triangle is zero, and so is any channel a
# ``VNLinear`` happens to annihilate. ``torch.linalg.norm`` has a NaN gradient
# at exactly zero, which propagates through the whole graph.
EPS = 1e-8


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False,
              eps: float = EPS) -> Tensor:
    """
    ``||x||`` with a finite gradient at zero, and *exact* away from it.

    Clamping the squared sum rather than adding an epsilon to it matters more
    than it looks. ``sqrt(s + eps)`` perturbs every norm, including large ones,
    which leaves the Gram-Schmidt head's output only orthonormal to ~1e-8 and
    its determinant at 0.99999995 instead of 1. Clamping is exact wherever
    ``||x|| > eps`` and merely caps the gradient below it -- and zero is not a
    differentiable point of the norm anyway, so a zero subgradient there is the
    honest answer rather than a fudge.

    ``eps`` is exposed because the right floor depends on what the norm feeds.
    Dividing by it (normalisation) needs a floor large enough that a
    near-zero vector cannot produce an enormous quotient, and 1e-8 is that.
    Feeding it into ``atan2`` does not: the composite derivative there is
    bounded by 1/2 whatever the floor, so a floor of 1e-8 buys no stability and
    costs a 5e-9 rad error at exactly 0 and pi -- which is why
    :func:`~reassembly.nn.losses.geodesic_angle` passes a far smaller one.
    """
    squared = torch.sum(x * x, dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(squared, min=eps * eps))


def normalize(x: Tensor, dim: int = -1) -> Tensor:
    """``x / ||x||``, unit-length away from the origin and finite at it."""
    return x / safe_norm(x, dim=dim, keepdim=True)


class VNLinear(nn.Module):
    """
    Channel mixing, ``y_c = sum_k W_ck x_k``.

    Equivariant because ``W`` acts on the channel axis and ``R`` on the spatial
    axis, and the two commute. No bias -- see the module docstring.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming-uniform on fan_in = in_channels. The spatial axis is not a
        # fan-in: it is carried along, not summed over.
        bound = 1.0 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """``(..., C_in, 3) -> (..., C_out, 3)``."""
        return torch.matmul(self.weight, x)

    def extra_repr(self) -> str:
        return f"{self.in_channels} -> {self.out_channels}"


class VNLeakyReLU(nn.Module):
    """
    Deng et al.'s vector nonlinearity.

    A learned direction ``d = W_d x`` splits space with a hyperplane. Channels
    on the positive side pass through; channels on the negative side have their
    component along ``d`` attenuated by ``negative_slope``::

        q'  =  q                                  if <q, d> >= 0
        q'  =  q - (1 - a) <q, d_hat> d_hat       otherwise

    Both ``q`` and ``d`` rotate together, so ``<q, d>`` is invariant and the
    branch is taken identically in every frame -- which is the whole point.
    ``a = 1`` is the identity; ``a = 0`` is the hard version, projecting
    negative channels onto the hyperplane.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2,
                 share_direction: bool = False) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)
        # One direction per channel by default. ``share_direction`` collapses
        # to a single global hyperplane, which is cheaper and occasionally
        # enough, but loses per-channel selectivity.
        self.direction = VNLinear(channels, 1 if share_direction else channels)

    def forward(self, x: Tensor) -> Tensor:
        d = self.direction(x)                                 # (..., C or 1, 3)
        dot = torch.sum(x * d, dim=-1, keepdim=True)          # invariant
        d_sq = torch.sum(d * d, dim=-1, keepdim=True) + EPS
        # Component of x along d, kept as a vector so the result stays equivariant.
        projection = (dot / d_sq) * d
        negative = (1.0 - self.negative_slope) * projection
        return torch.where(dot >= 0.0, x, x - negative)


class VNLayerNorm(nn.Module):
    """
    RMS normalisation over channel *magnitudes*.

    The norms ``||x_c||`` are invariant, so normalising by a statistic of them
    and rescaling is a single invariant multiplier per node -- equivariant, and
    unlike a signed affine normalisation it can never flip a vector's
    direction. A learned per-channel ``gain`` restores scale freedom.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        squared = torch.sum(x * x, dim=-1)                    # (..., C)
        rms = torch.sqrt(torch.mean(squared, dim=-1, keepdim=True) + EPS)
        return x / rms.unsqueeze(-1) * self.gain.view(-1, 1)


class VNInvariant(nn.Module):
    """
    Equivariant vectors to invariant scalars -- the bridge to ordinary MLPs.

    Learns ``k`` equivariant directions ``d = W_d x`` and returns every inner
    product ``<x_c, d_j>``, flattened to ``(..., C * k)``. Each is invariant
    because both arguments rotate. Setting ``k = 3`` recovers Deng's
    ``VNStdFeature``: the three directions are a learned local frame and the
    inner products are the coordinates of ``x`` in it.

    Optionally appends the channel norms, which are invariant too and which the
    inner products only recover if ``d`` happens to span ``x``.
    """

    def __init__(self, in_channels: int, directions: int = 3,
                 include_norms: bool = True) -> None:
        super().__init__()
        self.directions = VNLinear(in_channels, directions)
        self.include_norms = include_norms
        self.out_features = in_channels * directions + (in_channels if include_norms else 0)

    def forward(self, x: Tensor) -> Tensor:
        d = self.directions(x)                                # (..., k, 3)
        # <x_c, d_j> for every (c, j).
        gram = torch.einsum("...ci,...ki->...ck", x, d)
        flat = gram.flatten(start_dim=-2)
        if self.include_norms:
            flat = torch.cat([flat, safe_norm(x, dim=-1)], dim=-1)
        return flat


class VNScaleGate(nn.Module):
    """
    Conditions the vector stream on invariant scalars without leaving it.

    Fragment scale has no direction, so it cannot be concatenated into a
    ``(C, 3)`` tensor -- there is no legal third axis to put it on. Instead it
    joins the *invariant* description of the node and the pair produces a
    positive per-channel gain::

        gain = softplus(MLP([invariants(x), scalars]))    # invariant
        x    = gain * x                                   # still equivariant

    ``softplus`` keeps the gain positive, so the gate can attenuate or amplify
    a channel but never reflect it. Feed ``log(scale)``: fragments span 4 to
    83,039 vertices, so raw scale is heavy-tailed enough to dominate whatever
    it is concatenated with. GARF applies a positional encoding for the same
    reason.
    """

    def __init__(self, channels: int, scalar_features: int = 1,
                 hidden: int = 64, directions: int = 3) -> None:
        super().__init__()
        self.invariant = VNInvariant(channels, directions=directions)
        self.mlp = nn.Sequential(
            nn.Linear(self.invariant.out_features + scalar_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        # Start at gain ~= 1 (softplus(0.5413) = 1.0) so an untrained gate is
        # close to the identity and does not scramble the signal at step 0.
        # The weight is *small*, not zero: a zero final weight makes the gate's
        # output constant, which zeroes the gradient of everything feeding it --
        # the whole MLP and the invariant projection go dead on the first step
        # and only wake up once the bias has moved. Near-identity is the goal;
        # a dead subnetwork is not.
        nn.init.normal_(self.mlp[-1].weight, std=1e-2 / math.sqrt(hidden))
        nn.init.constant_(self.mlp[-1].bias, 0.5413248546129181)

    def forward(self, x: Tensor, scalars: Optional[Tensor] = None) -> Tensor:
        features = self.invariant(x)
        if scalars is not None:
            features = torch.cat([features, scalars], dim=-1)
        gain = nn.functional.softplus(self.mlp(features))
        return gain.unsqueeze(-1) * x


class VNMLP(nn.Module):
    """``VNLinear -> VNLeakyReLU -> VNLinear``, the standard two-layer block."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            VNLinear(in_channels, hidden_channels),
            VNLeakyReLU(hidden_channels, negative_slope),
            VNLinear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def gram_schmidt(v: Tensor) -> Tensor:
    """
    Two equivariant 3-vectors to a proper rotation matrix.

    ``v`` is ``(..., 2, 3)``. The first vector fixes the first axis; the second
    is orthogonalised against it; the third is their cross product, which forces
    ``det = +1`` without a sign test.

    This is Zhou et al.'s 6D parameterisation, and it is the *only* rotation
    output compatible with vector features. Quaternion regression is not: a
    quaternion's four components are coordinates in a fixed basis, so a linear
    map from ``(C, 3)`` features cannot produce them equivariantly -- rotating
    the input would have to permute and mix the output components in a way no
    fixed linear layer does. Here, rotating the input rotates ``v``, which
    rotates every column of the result: ``f(x R^T) = R f(x)``.
    """
    a, b = v[..., 0, :], v[..., 1, :]
    e1 = normalize(a)
    b_perp = b - torch.sum(b * e1, dim=-1, keepdim=True) * e1
    e2 = normalize(b_perp)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)                  # columns
