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


def geodesic_angle(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    (F,) rotation angle in DEGREES between predicted and ground truth.

    Shares the `atan2(sin, cos)` formulation used by the training loss (see
    `vngat.models.vn_layers.geodesic_rotation_loss`), so a perfect prediction
    reports 0 rather than the ~0.03 degree floor an `arccos` clamp imposes.
    """
    from ..models.vn_layers import geodesic_rotation_loss

    return geodesic_rotation_loss(R_pred, R_gt) * (180.0 / torch.pi)


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


# Reference levels, for reading any rotation number against something.
CHANCE_GEODESIC_DEG = 126.47
"""Mean geodesic angle between a rotation and a Haar-uniform one: pi/2 + 2/pi."""
CHANCE_EULER_RMSE_DEG = 83.20
"""Mean Euler RMSE of a Haar-uniform residual (Monte Carlo, 5e5 samples)."""
SYMMETRY_FLOOR_GEODESIC_DEG = 89.9
"""Error a per-fragment canonicaliser would show if the azimuth about an
object's symmetry axis were unrecoverable: the residual would be a uniform
rotation about that axis.

MEASURED AND REFUTED as a floor for this task. A scaling run on 8 Everyday
objects (bottles, cups, mirrors -- surfaces of revolution) reached 30.90 deg
training error, far below this value. The argument fails because a FRAGMENT of
a symmetric object is not itself symmetric: its fracture boundary is jagged and
unique, so the azimuth is recoverable from the fragment even though it would
not be from the intact surface. The intact object's symmetry does not transfer
to its pieces.

Kept as a reference line for reading results -- crossing it is evidence that
per-fragment azimuth is being resolved -- NOT as a predicted ceiling."""
SYMMETRY_FLOOR_EULER_RMSE_DEG = 51.9


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) as (w, x, y, z), via Shepperd's branch selection.

    Branching on the largest of the four candidate terms avoids the
    catastrophic cancellation the naive `w = sqrt(1 + trace)/2` form suffers
    near 180 degrees.
    """
    R = _at_least_float32(R)
    m = [[R[..., i, j] for j in range(3)] for i in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    cand = torch.stack([
        1 + trace,
        1 + m[0][0] - m[1][1] - m[2][2],
        1 - m[0][0] + m[1][1] - m[2][2],
        1 - m[0][0] - m[1][1] + m[2][2],
    ], dim=-1)
    quats = torch.stack([
        torch.stack([1 + trace, m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]], -1),
        torch.stack([m[2][1] - m[1][2], 1 + m[0][0] - m[1][1] - m[2][2], m[0][1] + m[1][0], m[0][2] + m[2][0]], -1),
        torch.stack([m[0][2] - m[2][0], m[0][1] + m[1][0], 1 - m[0][0] + m[1][1] - m[2][2], m[1][2] + m[2][1]], -1),
        torch.stack([m[1][0] - m[0][1], m[0][2] + m[2][0], m[1][2] + m[2][1], 1 - m[0][0] - m[1][1] + m[2][2]], -1),
    ], dim=-2)
    pick = cand.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(*cand.shape[:-1], 1, 4)
    q = torch.gather(quats, -2, pick).squeeze(-2)
    return q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-12)


def swing_twist_error(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    axis: str = "z",
) -> tuple:
    """
    Split the residual rotation into TILT (swing) off a symmetry axis and TWIST
    about it. Returns (tilt_deg, twist_deg), both (F,).

    This is the measurement that decides whether the model is failing for a
    fixable reason or a structural one. If the objects are surfaces of
    revolution then the azimuth about their axis is not identifiable from a
    single fragment, so a perfectly-trained per-fragment model would show
    TILT -> 0 with TWIST staying uniform (mean 90 deg). Tilt still near 90 means
    the model has not learned even the axis, and there is real headroom left.

    `axis` names the canonical up-axis of the dataset's meshes. Verify it for
    your copy rather than trusting the default -- run the evaluation with each
    of 'x', 'y', 'z' and see which one shows the signature.
    """
    residual = torch.matmul(_at_least_float32(R_pred).transpose(-1, -2),
                            _at_least_float32(R_gt).to(_at_least_float32(R_pred).dtype))
    index = {"x": 0, "y": 1, "z": 2}[axis]
    a = torch.zeros(3, device=residual.device, dtype=residual.dtype)
    a[index] = 1.0

    # tilt: how far the axis itself is rotated.
    rotated_axis = torch.matmul(residual, a)
    # atan2(|a x Ra|, a.Ra), not arccos(a.Ra): the clamp arccos needs to stay in
    # domain is a floor -- it reports 0.026 deg for an exactly-zero tilt, which
    # is precisely the regime this metric exists to detect.
    cos_tilt = (rotated_axis * a).sum(-1)
    sin_tilt = torch.sqrt(torch.cross(a.expand_as(rotated_axis), rotated_axis, dim=-1)
                          .pow(2).sum(-1) + 1e-12)
    tilt = torch.atan2(sin_tilt, cos_tilt)

    # twist: the component of the residual about `a`, by swing-twist
    # decomposition of the quaternion (robust where a matrix construction is
    # degenerate at 180 degrees).
    q = matrix_to_quaternion(residual)
    w, v = q[..., 0], q[..., 1:]
    proj = (v * a).sum(-1)
    twist_norm = torch.sqrt(w * w + proj * proj + 1e-12)
    twist_angle = 2.0 * torch.atan2(proj.abs(), w.abs().clamp_min(1e-12))
    twist_angle = torch.where(twist_norm > 1e-6, twist_angle, torch.zeros_like(twist_angle))
    return tilt * (180.0 / torch.pi), twist_angle * (180.0 / torch.pi)


def translation_rmse(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """(F,) per-fragment RMSE over the three translation components."""
    return (t_pred - t_gt).pow(2).mean(dim=-1).sqrt()


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """
    Symmetric Chamfer distance between two (N, 3) / (M, 3) point sets.

    Works in SQUARED distances throughout, in float64, rather than taking
    `torch.cdist(...)` and squaring the result. Two reasons:

    * `cdist` reaches for a matmul expansion of ||x - y||^2 for speed. On
      identical points the expansion cancels to a residual of order 1e-7 in
      float32, and the sqrt turns that into ~4e-4 -- so two IDENTICAL clouds
      score ~2e-7 instead of 0. Squaring the distance immediately afterwards
      just undoes the sqrt while keeping its error.
    * Clamping the squared distance at zero removes the negative residuals the
      expansion can produce, which would otherwise become NaN under a sqrt.

    Chamfer here is compared against a Part-Accuracy threshold of 0.01, so
    float64 leaves roughly thirteen orders of magnitude of headroom instead of
    five. Chunked so a dense fragment never allocates an N x M matrix.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("nan"), device=a.device)
    a64, b64 = a.double(), b.double()

    def one_way(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_sq = (y * y).sum(-1)
        total = x.new_zeros(())
        for start in range(0, x.shape[0], chunk):
            block = x[start:start + chunk]
            d2 = (block * block).sum(-1, keepdim=True) + y_sq.unsqueeze(0) - 2.0 * (block @ y.T)
            total = total + d2.clamp_min(0).amin(dim=1).sum()
        return total / x.shape[0]

    return (one_way(a64, b64) + one_way(b64, a64)).to(a.dtype)


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
    symmetry_axis: str = "z",
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
    tilt, twist = swing_twist_error(R_pred, R_gt, axis=symmetry_axis)
    metrics["tilt_deg"] = float(tilt.mean())
    metrics["twist_deg"] = float(twist.mean())
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
