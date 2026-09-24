"""
Diagnostics: the measurements that say *why* a rotation error is what it is.

A mean geodesic angle is one number and it is compatible with several quite
different situations. 90 degrees can mean the model has found the object's
symmetry axis and not the rotation about it, or that it has found nothing;
those call for opposite responses and the mean cannot tell them apart. Every
function here exists to split one number into two that can.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# ------------------------------------------------------------ swing/twist --

def matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    ``(..., 3, 3)`` rotations to ``(..., 4)`` quaternions ``(w, x, y, z)``.

    Shepperd's method: four algebraically equivalent expressions, of which the
    one with the largest denominator is used. The textbook formula
    ``w = sqrt(1 + tr) / 2`` is the first of them and it loses all its
    precision near 180 degrees, where ``1 + tr -> 0`` -- which is exactly the
    region a model at chance occupies, so the naive version degrades worst
    precisely where the diagnostic is being read.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    candidates = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    branch = candidates.argmax(dim=-1)
    root = torch.sqrt(candidates.clamp_min(0.0).gather(-1, branch.unsqueeze(-1)).squeeze(-1))
    # Every branch divides by 2*root; the clamp only guards the exactly-zero
    # case, which the argmax makes unreachable for a proper rotation (the four
    # candidates sum to 4).
    scale = 0.5 / root.clamp_min(1e-12)

    quaternions = torch.stack([
        torch.stack([0.5 * root, (m21 - m12) * scale, (m02 - m20) * scale, (m10 - m01) * scale], -1),
        torch.stack([(m21 - m12) * scale, 0.5 * root, (m01 + m10) * scale, (m02 + m20) * scale], -1),
        torch.stack([(m02 - m20) * scale, (m01 + m10) * scale, 0.5 * root, (m12 + m21) * scale], -1),
        torch.stack([(m10 - m01) * scale, (m02 + m20) * scale, (m12 + m21) * scale, 0.5 * root], -1),
    ], dim=-2)
    quaternion = quaternions.gather(
        -2, branch[..., None, None].expand(*branch.shape, 1, 4)).squeeze(-2)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def swing_twist_error(predicted: Tensor, target: Tensor,
                      axis: str = "z") -> Tuple[Tensor, Tensor]:
    """
    Split the residual rotation into **tilt** off an axis and **twist** about it.

    Returns two ``(...,)`` tensors in degrees. Reported, never optimised.

    WHAT IT IS FOR
    --------------
    Breaking Bad's Everyday categories are mostly surfaces of revolution --
    bottles, bowls, mugs, plates -- so a model can plausibly recover an
    object's up-axis while learning nothing about the rotation around it. That
    state and "learned nothing at all" both read as roughly 90 degrees of mean
    geodesic error, and the response to them is opposite: the first says the
    remaining information is in the fracture boundary and the model is not
    using it, the second says the problem is upstream of symmetry entirely.

        tilt ~ 90                 the axis is not being recovered either
        tilt ~ 0, twist ~ 90      the axis is recovered, the azimuth is not

    The second is a real finding about the geometry. It is **not** a hard
    floor, and it was described as one in this project's earlier design: a
    fragment of a symmetric object is not itself symmetric, because its
    fracture boundary is jagged and unique. The earlier model reached 30.9
    degrees on eight Everyday objects, well inside the supposed floor.

    WHY IT IS BUILT THIS WAY
    ------------------------
    The swing of a swing-twist decomposition is the minimal rotation carrying
    ``a`` to ``E a``, so its angle *is* the angle between those two vectors and
    no quaternion algebra is needed for it. Both angles come from ``atan2``
    rather than ``arccos``: the regime this metric exists to detect is
    ``tilt ~ 0``, and an ``arccos`` with the usual clamp bottoms out around
    0.026 degrees there, which is indistinguishable from a genuinely small
    tilt.

    ``axis`` names the dataset's canonical up-axis. Verify it rather than
    assume it -- run with x, y and z and see which shows the signature. A wrong
    choice makes the diagnostic meaningless, not the training wrong.
    """
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {axis!r}")
    residual = torch.matmul(predicted.transpose(-1, -2), target)
    direction = residual.new_tensor(_AXES[axis])

    # -- tilt: how far the axis itself moves -------------------------------
    moved = torch.matmul(residual, direction)
    cross = torch.linalg.cross(direction.expand_as(moved), moved, dim=-1)
    dot = (direction * moved).sum(-1)
    tilt = torch.atan2(cross.norm(dim=-1), dot)

    # -- twist: the rotation about the axis, folded into [0, 180] ----------
    quaternion = matrix_to_quaternion(residual)
    w = quaternion[..., 0]
    along = (quaternion[..., 1:] * direction).sum(-1)
    twist = 2.0 * torch.atan2(along.abs(), w.abs())

    degrees = 180.0 / torch.pi
    return tilt * degrees, twist * degrees


# ----------------------------------------------------------------- head ----

def head_collinearity(axes: Tensor) -> Tensor:
    """
    ``|cos|`` between the two channels the rotation head feeds to Gram-Schmidt.

    ``axes`` is ``(F, 2, 3)``. Returns one scalar, the mean over fragments.

    Near 1 means the two vectors have become parallel, so the frame's second
    column is determined almost entirely by numerical noise in the component of
    the second vector orthogonal to the first. The predicted rotation is then
    effectively a one-vector prediction with a random roll, and the run is
    stuck for a structural reason rather than a data one.

    This is the cheapest measurement that separates "in a bad basin" from
    "still descending". In the earlier design two seeds of an otherwise
    identical eight-object run finished 74 degrees apart (30.9 against 104.7),
    and the healthy runs sat at 0.78-0.90. It costs one dot product per
    fragment and belongs in the epoch table for that reason alone.
    """
    if axes.ndim != 3 or axes.shape[-2:] != (2, 3):
        raise ValueError(f"expected (F, 2, 3) head axes, got {tuple(axes.shape)}")
    first = torch.nn.functional.normalize(axes[:, 0].double(), dim=-1, eps=1e-12)
    second = torch.nn.functional.normalize(axes[:, 1].double(), dim=-1, eps=1e-12)
    return (first * second).sum(-1).abs().mean()


# --------------------------------------------------------------- assembly --

def chamfer_distance(predicted: Tensor, target: Tensor,
                     chunk: Optional[int] = None) -> Tensor:
    """
    Symmetric Chamfer distance between two point clouds, as a mean of **squared**
    distances, in float64.

    Two deliberate choices, both about the same failure.

    ``torch.cdist`` is the obvious tool. It expands
    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b``, and that expansion cancels
    catastrophically when the two points nearly coincide -- which is the case
    every correct prediction is made of. Measured, in float32, scoring a 2,000
    point cloud against *itself*:

        coordinate scale    cdist        direct difference
        unit                8.8e-08      0.0
        x10                 1.1e-05      0.0

    So the floor is not fixed; it grows with the square of the coordinate
    magnitude, and Breaking Bad's part-accuracy threshold is an absolute 0.01.
    At unit scale the error is 1e-5 of the threshold and harmless; it is the
    *coupling to scale* that makes it unsafe to leave in a number that gets
    compared across datasets and normalisations.

    Casting to float64 removes it for cdist as well (both columns read 0.0
    there), so either fix alone would do. Both are applied because they are
    free: the direct difference costs the same memory once chunked, and float64
    on a point cloud of this size is not the bottleneck.

    Chunked over each cloud in turn. A fixed row count bounds only one side of
    the ``(rows, other, 3)`` difference tensor -- 4,096 rows against a
    40,000-point assembly is 3.9 GB in float64 -- so by default the row count
    is derived from the other cloud's size, keeping each block near 16M
    elements (128 MB). Pass ``chunk`` to fix it.
    """
    a = predicted.double()
    b = target.double()
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64, device=a.device)

    def rows(other: int) -> int:
        return int(chunk) if chunk else max(1, (1 << 24) // (3 * max(other, 1)))

    forward = a.new_empty(a.shape[0])
    step = rows(b.shape[0])
    for start in range(0, a.shape[0], step):
        block = a[start:start + step]
        distances = (block.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
        forward[start:start + step] = distances.amin(dim=1)
    backward = b.new_empty(b.shape[0])
    step = rows(a.shape[0])
    for start in range(0, b.shape[0], step):
        block = b[start:start + step]
        distances = (block.unsqueeze(1) - a.unsqueeze(0)).pow(2).sum(-1)
        backward[start:start + step] = distances.amin(dim=1)
    return forward.mean() + backward.mean()


def part_accuracy(chamfer: Tensor, threshold: float = 0.01) -> Tensor:
    """
    Fraction of parts whose Chamfer distance is below ``threshold`` -- Breaking
    Bad's headline metric alongside RMSE.

    Non-finite entries are **excluded** rather than counted as failures. A part
    that could not be scored at all (an empty cloud, a fragment dropped
    upstream) is missing data, and scoring it as a miss quietly converts a
    pipeline defect into a worse-looking model, which is the direction of error
    that never gets investigated.
    """
    values = torch.as_tensor(chamfer)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    return (finite < threshold).double().mean()


# -------------------------------------------------------------- grouping --

def group_means(values: Sequence[float], groups: Sequence[str]
                ) -> Dict[str, Tuple[float, int]]:
    """
    Mean of ``values`` per group, with the count, sorted worst-first.

    The per-category breakdown is the honest counterpart to the balanced
    sampler: balancing changes what the model is *shown*, and this shows what
    it then *does*, per category, on a validation set that is never reweighted.
    Without it, "the validation number improved" cannot be distinguished from
    "the validation number is now dominated by different categories".
    """
    if len(values) != len(groups):
        raise ValueError("values and groups must be the same length")
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for value, group in zip(values, groups):
        if value != value:                      # NaN
            continue
        totals[group] = totals.get(group, 0.0) + float(value)
        counts[group] = counts.get(group, 0) + 1
    means = {name: (totals[name] / counts[name], counts[name]) for name in totals}
    return dict(sorted(means.items(), key=lambda kv: -kv[1][0]))


def format_group_table(means: Dict[str, Tuple[float, int]], title: str,
                       unit: str = "deg", top: Optional[int] = None) -> str:
    """The breakdown as an aligned block, worst category first."""
    if not means:
        return f"  {title}: nothing to report"
    rows = list(means.items())
    shown = rows if top is None else rows[:top]
    width = max(len(name) for name, _ in shown)
    lines = [f"  {title} (worst first)"]
    for name, (value, count) in shown:
        lines.append(f"    {name:<{width}}  {value:8.2f} {unit}  n={count}")
    if top is not None and len(rows) > top:
        lines.append(f"    ... {len(rows) - top} more")
    return "\n".join(lines)
