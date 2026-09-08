"""
Evaluation metrics, TensorFlow.

TWO ROTATION ERRORS, REPORTED SEPARATELY -- THIS MATTERS FOR THE THESIS

  * GEODESIC angle: the single rotation angle between R_pred and R_gt. Natural
    metric on SO(3); what the training loss uses.
  * EULER RMSE: RMSE across the three Euler angles of the residual. What the
    Breaking Bad benchmark and GARF's tables report.

For the same prediction the Euler RMSE is typically SMALLER, because it averages
three angles each seeing part of the error. Quoting one against a table of the
other misstates results in either direction. Both are returned here, named
unambiguously.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import tensorflow as tf

CHANCE_GEODESIC_DEG = 126.47
"""Mean geodesic angle to a Haar-uniform rotation: pi/2 + 2/pi, in degrees."""
CHANCE_EULER_RMSE_DEG = 83.20
"""Mean Euler RMSE of a Haar-uniform residual (Monte Carlo, 5e5 samples)."""
SYMMETRY_FLOOR_GEODESIC_DEG = 89.9
"""Error a per-fragment canonicaliser would show if the azimuth about an
object's symmetry axis were unrecoverable. Kept as a REFERENCE LINE, not a
predicted ceiling: a matched-budget run reached 30.9 degrees training error on
8 objects, well below it, because a FRAGMENT of a symmetric object is not itself
symmetric -- its fracture boundary is jagged and unique."""
SYMMETRY_FLOOR_EULER_RMSE_DEG = 51.9

_HALF = (tf.float16, tf.bfloat16)
_RAD2DEG = 180.0 / math.pi


def _at_least_float32(x):
    return tf.cast(x, tf.float32) if x.dtype in _HALF else x


def geodesic_angle(R_pred, R_gt):
    """(F,) rotation angle in DEGREES. Shares the atan2 formulation of the
    training loss, so a perfect prediction reports ~0 rather than the ~0.03
    degree floor an arccos clamp imposes."""
    from ..models.vn_layers import geodesic_rotation_loss

    return geodesic_rotation_loss(R_pred, R_gt) * _RAD2DEG


def matrix_to_euler_xyz(R, eps: float = 1e-7):
    """
    (F, 3) intrinsic XYZ Euler angles in degrees.

    Gimbal lock handled explicitly: near |sin(pitch)| = 1 the yaw/roll split is
    undefined, and the usual atan2 formula produces large spurious values that
    would corrupt an RMSE average.
    """
    R = _at_least_float32(R)
    sy = -R[..., 2, 0]
    locked = tf.abs(sy) > 1 - eps
    pitch = tf.asin(tf.clip_by_value(sy, -1 + eps, 1 - eps))
    roll = tf.where(locked, tf.atan2(-R[..., 1, 2], R[..., 1, 1]),
                    tf.atan2(R[..., 2, 1], R[..., 2, 2]))
    yaw = tf.where(locked, tf.zeros_like(sy), tf.atan2(R[..., 1, 0], R[..., 0, 0]))
    return tf.stack([roll, pitch, yaw], -1) * _RAD2DEG


def euler_rmse(R_pred, R_gt):
    """(F,) per-fragment RMSE over the three Euler angles of the residual."""
    R_pred = _at_least_float32(R_pred)
    residual = tf.matmul(R_pred, tf.cast(_at_least_float32(R_gt), R_pred.dtype),
                         transpose_a=True)
    angles = matrix_to_euler_xyz(residual)
    # Wrap into (-180, 180] so a residual of 359.9 counts as 0.1 degrees.
    angles = tf.math.floormod(angles + 180.0, 360.0) - 180.0
    return tf.sqrt(tf.reduce_mean(tf.square(angles), -1))


def matrix_to_quaternion(R):
    """(..., 4) as (w, x, y, z), via Shepperd's branch selection. Branching on
    the largest candidate avoids the catastrophic cancellation the naive
    `w = sqrt(1 + trace)/2` form suffers near 180 degrees."""
    R = _at_least_float32(R)
    m = [[R[..., i, j] for j in range(3)] for i in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    cand = tf.stack([1 + trace,
                     1 + m[0][0] - m[1][1] - m[2][2],
                     1 - m[0][0] + m[1][1] - m[2][2],
                     1 - m[0][0] - m[1][1] + m[2][2]], -1)
    quats = tf.stack([
        tf.stack([1 + trace, m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]], -1),
        tf.stack([m[2][1] - m[1][2], cand[..., 1], m[0][1] + m[1][0], m[0][2] + m[2][0]], -1),
        tf.stack([m[0][2] - m[2][0], m[0][1] + m[1][0], cand[..., 2], m[1][2] + m[2][1]], -1),
        tf.stack([m[1][0] - m[0][1], m[0][2] + m[2][0], m[1][2] + m[2][1], cand[..., 3]], -1),
    ], axis=-2)
    pick = tf.argmax(cand, axis=-1, output_type=tf.int32)
    q = tf.gather(quats, pick, batch_dims=len(R.shape) - 2)
    return q / tf.sqrt(tf.reduce_sum(q * q, -1, keepdims=True) + 1e-12)


def swing_twist_error(R_pred, R_gt, axis: str = "z"):
    """
    Split the residual into TILT (swing) off a symmetry axis and TWIST about it.
    Returns (tilt_deg, twist_deg), both (F,).

    This is the measurement that decides whether a failure is fixable or
    structural. For a surface of revolution the azimuth about the axis may not
    be identifiable from a single fragment, in which case a perfectly-trained
    per-fragment model shows TILT -> 0 with TWIST uniform (mean 90). Tilt near
    90 means the axis itself has not been learned, and there is real headroom.

    `axis` names the dataset's canonical up-axis. Verify it rather than trusting
    the default -- run with x, y and z and see which shows the signature.
    """
    R_pred = _at_least_float32(R_pred)
    residual = tf.matmul(R_pred, tf.cast(_at_least_float32(R_gt), R_pred.dtype),
                         transpose_a=True)
    a = tf.constant({"x": [1., 0., 0.], "y": [0., 1., 0.], "z": [0., 0., 1.]}[axis],
                    residual.dtype)

    rotated = tf.linalg.matvec(residual, a)
    # atan2(|a x Ra|, a.Ra), not arccos(a.Ra): the clamp arccos needs to stay in
    # domain is a floor -- it reports 0.026 degrees for an exactly-zero tilt,
    # which is precisely the regime this metric exists to detect.
    cos_tilt = tf.reduce_sum(rotated * a, -1)
    cross = tf.linalg.cross(tf.broadcast_to(a, tf.shape(rotated)), rotated)
    sin_tilt = tf.sqrt(tf.reduce_sum(cross * cross, -1) + 1e-12)
    tilt = tf.atan2(sin_tilt, cos_tilt)

    q = matrix_to_quaternion(residual)
    w, v = q[..., 0], q[..., 1:]
    proj = tf.reduce_sum(v * a, -1)
    twist_norm = tf.sqrt(w * w + proj * proj + 1e-12)
    twist = 2.0 * tf.atan2(tf.abs(proj), tf.maximum(tf.abs(w), 1e-12))
    twist = tf.where(twist_norm > 1e-6, twist, tf.zeros_like(twist))
    return tilt * _RAD2DEG, twist * _RAD2DEG


def translation_rmse(t_pred, t_gt):
    """(F,) per-fragment RMSE over the three translation components."""
    return tf.sqrt(tf.reduce_mean(tf.square(t_pred - t_gt), -1))


def chamfer_distance(a, b, chunk: int = 1024):
    """
    Symmetric Chamfer distance between two (N, 3) / (M, 3) point sets.

    Works in SQUARED distances in float64 rather than taking a distance and
    squaring it: the matmul expansion of ||x-y||^2 cancels to a residual ~1e-7
    in float32 on identical points, and a sqrt turns that into ~4e-4 -- so two
    IDENTICAL clouds would score ~2e-7 instead of 0. Clamping at zero also
    removes the negative residuals the expansion can produce.
    """
    if tf.size(a) == 0 or tf.size(b) == 0:
        return tf.constant(float("nan"), tf.float64)
    a64, b64 = tf.cast(a, tf.float64), tf.cast(b, tf.float64)

    def one_way(x, y):
        y_sq = tf.reduce_sum(y * y, -1)
        total = tf.zeros((), tf.float64)
        n = tf.shape(x)[0]
        for start in tf.range(0, n, chunk):
            block = x[start:start + chunk]
            d2 = (tf.reduce_sum(block * block, -1, keepdims=True)
                  + tf.expand_dims(y_sq, 0) - 2.0 * tf.matmul(block, y, transpose_b=True))
            total += tf.reduce_sum(tf.reduce_min(tf.maximum(d2, 0.0), axis=1))
        return total / tf.cast(n, tf.float64)

    return one_way(a64, b64) + one_way(b64, a64)


def part_accuracy(chamfer, threshold: float = 0.01):
    """Fraction of fragments placed within `threshold` Chamfer distance."""
    valid = tf.logical_not(tf.math.is_nan(chamfer))
    return tf.cond(tf.reduce_any(valid),
                   lambda: tf.reduce_mean(tf.cast(
                       tf.boolean_mask(chamfer, valid) < threshold, tf.float32)),
                   lambda: tf.constant(float("nan")))


def evaluate_scene(R_pred, R_gt, pred_points=None, gt_points=None, point_frag=None,
                   t_pred=None, t_gt=None, pa_threshold: float = 0.01,
                   symmetry_axis: str = "z") -> Dict[str, float]:
    """All metrics for one scene. Translation-dependent entries are OMITTED
    rather than faked when translations are not supplied, so a rotation-only
    evaluation cannot silently report a meaningless RMSE(T)."""
    geo = geodesic_angle(R_pred, R_gt)
    tilt, twist = swing_twist_error(R_pred, R_gt, axis=symmetry_axis)
    metrics = {
        "rmse_R_euler_deg": float(tf.reduce_mean(euler_rmse(R_pred, R_gt))),
        "geodesic_deg": float(tf.reduce_mean(geo)),
        "geodesic_median_deg": float(tf.sort(geo)[tf.size(geo) // 2]),
        "tilt_deg": float(tf.reduce_mean(tilt)),
        "twist_deg": float(tf.reduce_mean(twist)),
        "num_fragments": float(R_pred.shape[0]),
    }
    if t_pred is not None and t_gt is not None:
        metrics["rmse_T"] = float(tf.reduce_mean(translation_rmse(t_pred, t_gt)))
    if pred_points is not None and gt_points is not None and point_frag is not None:
        num = int(R_pred.shape[0])
        vals = []
        for f in range(num):
            m = point_frag == f
            if bool(tf.reduce_any(m)):
                vals.append(float(chamfer_distance(tf.boolean_mask(pred_points, m),
                                                   tf.boolean_mask(gt_points, m))))
        if vals:
            import numpy as np
            arr = np.array(vals)
            metrics["chamfer"] = float(np.nanmean(arr))
            metrics["part_accuracy"] = float(part_accuracy(tf.constant(arr), pa_threshold))
    return metrics


def aggregate(scene_metrics: list) -> Dict[str, float]:
    """Mean over scenes, ignoring keys a given scene did not produce."""
    if not scene_metrics:
        return {}
    out = {}
    for key in sorted({k for m in scene_metrics for k in m}):
        vals = [m[key] for m in scene_metrics if key in m and m[key] == m[key]]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out
