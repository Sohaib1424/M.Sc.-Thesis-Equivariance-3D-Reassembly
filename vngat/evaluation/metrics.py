"""
Evaluation metrics, matched to what GARF (and the Breaking Bad benchmark
generally) reports: RMSE(R), RMSE(T), Part Accuracy and Chamfer Distance.

TWO ROTATION ERRORS, REPORTED SEPARATELY -- THIS MATTERS FOR THE THESIS
-----------------------------------------------------------------------
There are two different numbers in the literature both called "rotation
error", and they are NOT interchangeable:

  * GEODESIC angle: the single rotation angle between R_pred and R_gt,
    arccos((tr(R_pred^T R_gt) - 1) / 2). This is the natural metric on SO(3)
    and is what the training loss uses.

  * EULER RMSE: the RMSE across the three Euler angles of the residual
    rotation. This is what the Breaking Bad benchmark and GARF's tables report.

For the same prediction the Euler RMSE is typically the SMALLER number,
because it averages three angles and each one only sees part of the error.
Quoting a geodesic angle against a table of Euler RMSEs would understate this
model's accuracy; quoting the reverse would overstate it. Both are returned
here, named unambiguously, and the comparison tables should say which one they
use.

Part Accuracy follows the standard definition: the fraction of fragments whose
Chamfer distance to their ground-truth placement is below `pa_threshold`
(0.01 in the benchmark, on shapes normalised into a unit cube).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

_HALF = (torch.float16, torch.bfloat16)


def _at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """Promote half precision, preserve float64 (see vngat/models/vn_layers.py)."""
    return x.float() if x.dtype in _HALF else x


def geodesic_angle(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(F,) rotation angle in DEGREES between predicted and ground truth."""
    R_pred = _at_least_float32(R_pred)
    R_gt = _at_least_float32(R_gt).to(R_pred.dtype)
    diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos) * (180.0 / torch.pi)


def matrix_to_euler_xyz(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    (F, 3) intrinsic XYZ Euler angles in degrees.

    Gimbal-lock branch handled explicitly: near |sin(pitch)| = 1 the yaw/roll
    split is undefined, and the usual atan2 formula produces large spurious
    values that would corrupt an RMSE average.
    """
    R = _at_least_float32(R)
    sy = -R[..., 2, 0]
    locked = sy.abs() > 1 - eps
    pitch = torch.asin(sy.clamp(-1 + eps, 1 - eps))
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll_locked = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    roll = torch.where(locked, roll_locked, roll)
    yaw = torch.where(locked, torch.zeros_like(yaw), yaw)
    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / torch.pi)


def euler_rmse(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """(F,) per-fragment RMSE over the three Euler angles of the residual."""
    R_pred = _at_least_float32(R_pred)
    residual = torch.matmul(R_pred.transpose(-1, -2), _at_least_float32(R_gt).to(R_pred.dtype))
    angles = matrix_to_euler_xyz(residual)
    # Wrap into (-180, 180] so a residual of 359.9 deg counts as 0.1 deg.
    angles = (angles + 180.0) % 360.0 - 180.0
    return angles.pow(2).mean(dim=-1).sqrt()


def translation_rmse(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """(F,) per-fragment RMSE over the three translation components."""
    return (t_pred - t_gt).pow(2).mean(dim=-1).sqrt()


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """
    Symmetric Chamfer distance between two (N, 3) / (M, 3) point sets.

    Chunked so a dense fragment does not allocate an N x M distance matrix.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("nan"), device=a.device)

    def one_way(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = x.new_zeros(())
        for start in range(0, x.shape[0], chunk):
            d = torch.cdist(x[start:start + chunk], y)
            total = total + d.min(dim=1).values.pow(2).sum()
        return total / x.shape[0]

    return one_way(a, b) + one_way(b, a)


def per_fragment_chamfer(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    point_frag: torch.Tensor,
    num_fragments: int,
) -> torch.Tensor:
    """(F,) Chamfer distance per fragment, comparing like with like (the two
    point sets are the same vertices, so this is a placement error)."""
    out = torch.full((num_fragments,), float("nan"), device=pred_points.device)
    for f in range(num_fragments):
        mask = point_frag == f
        if not bool(mask.any()):
            continue
        out[f] = chamfer_distance(pred_points[mask], gt_points[mask])
    return out


def part_accuracy(chamfer: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Fraction of fragments placed within `threshold` Chamfer distance."""
    valid = ~torch.isnan(chamfer)
    if not bool(valid.any()):
        return torch.tensor(float("nan"), device=chamfer.device)
    return (chamfer[valid] < threshold).float().mean()


def evaluate_scene(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    pred_points: Optional[torch.Tensor] = None,
    gt_points: Optional[torch.Tensor] = None,
    point_frag: Optional[torch.Tensor] = None,
    t_pred: Optional[torch.Tensor] = None,
    t_gt: Optional[torch.Tensor] = None,
    pa_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    All metrics for one scene. Translation-dependent entries are omitted
    (rather than faked) when translations are not supplied, so a
    rotation-only evaluation cannot silently report a meaningless RMSE(T).
    """
    num_fragments = R_pred.shape[0]
    metrics: Dict[str, float] = {
        "rmse_R_euler_deg": float(euler_rmse(R_pred, R_gt).mean()),
        "geodesic_deg": float(geodesic_angle(R_pred, R_gt).mean()),
        "geodesic_median_deg": float(geodesic_angle(R_pred, R_gt).median()),
        "num_fragments": float(num_fragments),
    }
    if t_pred is not None and t_gt is not None:
        metrics["rmse_T"] = float(translation_rmse(t_pred, t_gt).mean())
    if pred_points is not None and gt_points is not None and point_frag is not None:
        chamfer = per_fragment_chamfer(pred_points, gt_points, point_frag, num_fragments)
        metrics["chamfer"] = float(torch.nanmean(chamfer))
        metrics["part_accuracy"] = float(part_accuracy(chamfer, pa_threshold))
    return metrics


def aggregate(scene_metrics: list) -> Dict[str, float]:
    """Mean over scenes, ignoring keys a given scene did not produce."""
    if not scene_metrics:
        return {}
    keys = sorted({k for m in scene_metrics for k in m})
    out = {}
    for key in keys:
        values = [m[key] for m in scene_metrics if key in m and m[key] == m[key]]
        if values:
            out[key] = float(sum(values) / len(values))
    return out
