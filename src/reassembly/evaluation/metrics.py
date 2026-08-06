"""
Evaluation metrics for fracture reassembly, matched to the Breaking Bad
benchmark protocol that GARF and its peers report against.

WHY THE EXACT CONVENTION MATTERS HERE
-------------------------------------
"Rotation error in degrees" is not one number. The Breaking Bad benchmark and
the assembly papers built on it (Neural Shape Mating, Jigsaw, PuzzleFusion++,
GARF) report ``RMSE(R)`` as the root-mean-square error over the three EULER
ANGLES of the predicted versus ground-truth rotation, in degrees. The geodesic
angle on SO(3) is a different, generally SMALLER quantity for the same
prediction.

Quoting a geodesic number against a table of Euler-RMSE numbers would flatter
this thesis by a factor that varies with the error distribution -- which is
exactly the kind of comparison a thesis defence should not have to survive.
Both are computed here, reported side by side, and named unambiguously.
``rmse_rotation_euler`` is the one that goes in the comparison table.

METRICS
-------
RMSE(R)  root-mean-square Euler-angle error, degrees (lower better)
RMSE(T)  root-mean-square translation error, in dataset units (lower better);
         the literature usually tabulates this x100
PA       Part Accuracy: fraction of parts whose Chamfer distance to their
         ground-truth placement is below ``pa_threshold`` (0.01 by convention)
CD       Chamfer distance between the assembled shape and the ground truth

A CAVEAT WORTH STATING IN THE THESIS
------------------------------------
Every one of these except RMSE(R) needs a fully placed assembly, i.e. rotation
AND translation. A rotation-only model cannot be scored on them. That is what
``reassembly.assembly.translation`` exists to close, and it is also why any
comparison table must state whether translations came from the solver or from
ground truth -- the two answer different questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class AssemblyMetrics:
    rmse_rotation_euler_deg: float
    rmse_rotation_geodesic_deg: float
    rmse_translation: float
    part_accuracy: float
    chamfer_distance: float
    num_parts: int
    per_part: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "RMSE_R_euler_deg": self.rmse_rotation_euler_deg,
            "RMSE_R_geodesic_deg": self.rmse_rotation_geodesic_deg,
            "RMSE_T": self.rmse_translation,
            "PA": self.part_accuracy,
            "CD": self.chamfer_distance,
            "num_parts": self.num_parts,
        }

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"RMSE(R)={self.rmse_rotation_euler_deg:7.3f} deg [euler] "
            f"({self.rmse_rotation_geodesic_deg:7.3f} deg [geodesic])  "
            f"RMSE(T)={self.rmse_translation:.5f}  "
            f"PA={self.part_accuracy:6.3f}  CD={self.chamfer_distance:.6f}  "
            f"({self.num_parts} parts)"
        )


def rotation_error_euler_deg(
    R_pred: np.ndarray, R_gt: np.ndarray, seq: str = "xyz"
) -> np.ndarray:
    """Per-part Euler-angle error in degrees, the Breaking Bad convention.

    Angles are compared with wraparound: a difference of 359 degrees is an
    error of 1 degree, not 359. Skipping that check is a classic way to get
    inflated, bimodal-looking error histograms.
    """
    e_pred = Rotation.from_matrix(np.asarray(R_pred)).as_euler(seq, degrees=True)
    e_gt = Rotation.from_matrix(np.asarray(R_gt)).as_euler(seq, degrees=True)
    diff = (e_pred - e_gt + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def rotation_error_geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Per-part geodesic angle on SO(3), in degrees."""
    R_pred, R_gt = np.asarray(R_pred), np.asarray(R_gt)
    rel = np.einsum("nji,njk->nik", R_pred, R_gt)
    trace = np.einsum("nii->n", rel)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def rmse_rotation_euler(R_pred: np.ndarray, R_gt: np.ndarray, seq: str = "xyz") -> float:
    return float(np.sqrt(np.mean(rotation_error_euler_deg(R_pred, R_gt, seq) ** 2)))


def rmse_rotation_geodesic(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(rotation_error_geodesic_deg(R_pred, R_gt) ** 2)))


def rmse_translation(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Root mean square of the per-part translation error magnitude."""
    err = np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt), axis=-1)
    return float(np.sqrt(np.mean(err**2)))


def chamfer_distance(
    a: np.ndarray,
    b: np.ndarray,
    squared: bool = True,
    bidirectional: bool = True,
) -> float:
    """Chamfer distance between two point sets.

    ``squared=True`` (the convention in this literature) averages SQUARED
    nearest-neighbour distances, which is why reported CD values look small --
    a 0.01 threshold on squared distance is a 0.1 gap in the units of the
    shape. Getting this wrong flips Part Accuracy dramatically, so it is
    explicit rather than assumed.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    d_ab, _ = cKDTree(b).query(a, k=1)
    if squared:
        d_ab = d_ab**2
    if not bidirectional:
        return float(np.mean(d_ab))

    d_ba, _ = cKDTree(a).query(b, k=1)
    if squared:
        d_ba = d_ba**2
    return float(np.mean(d_ab) + np.mean(d_ba))


def sample_surface(vertices: np.ndarray, faces: Optional[np.ndarray],
                   num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted surface sampling, falling back to vertex sampling.

    Sampling the SURFACE rather than the vertices matters for Chamfer distance:
    vertex density varies wildly across a decimated mesh, and vertex-based CD
    then measures meshing artefacts as much as shape error.
    """
    vertices = np.asarray(vertices)
    if faces is None or len(faces) == 0:
        if len(vertices) == 0:
            return vertices
        idx = rng.integers(0, len(vertices), num_points)
        return vertices[idx]

    faces = np.asarray(faces)
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    total = areas.sum()
    if total <= 0:
        idx = rng.integers(0, len(vertices), num_points)
        return vertices[idx]

    face_idx = rng.choice(len(faces), size=num_points, p=areas / total)
    u = rng.random((num_points, 1))
    v = rng.random((num_points, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[face_idx]
    return t[:, 0] + u * (t[:, 1] - t[:, 0]) + v * (t[:, 2] - t[:, 0])


def evaluate_assembly(
    R_pred: np.ndarray,
    R_gt: np.ndarray,
    t_pred: np.ndarray,
    t_gt: np.ndarray,
    fragment_vertices: Sequence[np.ndarray],
    fragment_faces: Optional[Sequence[np.ndarray]] = None,
    pa_threshold: float = 0.01,
    points_per_part: int = 1000,
    seed: int = 0,
) -> AssemblyMetrics:
    """Full metric suite for one assembled scene.

    ``fragment_vertices[i]`` must be the CENTRALIZED vertices of fragment i, so
    that the placed geometry is ``v @ R.T + t`` under both the prediction and
    the ground truth.
    """
    rng = np.random.default_rng(seed)
    R_pred, R_gt = np.asarray(R_pred), np.asarray(R_gt)
    t_pred, t_gt = np.asarray(t_pred), np.asarray(t_gt)
    n_parts = len(fragment_vertices)

    euler_err = rotation_error_euler_deg(R_pred, R_gt)
    geo_err = rotation_error_geodesic_deg(R_pred, R_gt)

    per_part_cd = np.zeros(n_parts)
    pred_cloud, gt_cloud = [], []

    for i in range(n_parts):
        verts = np.asarray(fragment_vertices[i])
        faces = fragment_faces[i] if fragment_faces is not None else None

        base = sample_surface(verts, faces, points_per_part, rng)
        placed_pred = base @ R_pred[i].T + t_pred[i]
        placed_gt = base @ R_gt[i].T + t_gt[i]

        per_part_cd[i] = chamfer_distance(placed_pred, placed_gt)
        pred_cloud.append(placed_pred)
        gt_cloud.append(placed_gt)

    part_accuracy = float(np.mean(per_part_cd < pa_threshold))
    cd_scene = chamfer_distance(
        np.concatenate(pred_cloud, axis=0), np.concatenate(gt_cloud, axis=0)
    ) if pred_cloud else float("nan")

    return AssemblyMetrics(
        rmse_rotation_euler_deg=float(np.sqrt(np.mean(euler_err**2))),
        rmse_rotation_geodesic_deg=float(np.sqrt(np.mean(geo_err**2))),
        rmse_translation=rmse_translation(t_pred, t_gt),
        part_accuracy=part_accuracy,
        chamfer_distance=cd_scene,
        num_parts=n_parts,
        per_part={"cd": per_part_cd, "euler_deg": euler_err, "geodesic_deg": geo_err},
    )


def aggregate(metrics: Sequence[AssemblyMetrics]) -> Dict[str, float]:
    """Mean over scenes, weighted equally per scene (the reporting convention)."""
    if not metrics:
        return {}
    keys = ("RMSE_R_euler_deg", "RMSE_R_geodesic_deg", "RMSE_T", "PA", "CD")
    dicts = [m.to_dict() for m in metrics]
    out = {k: float(np.nanmean([d[k] for d in dicts])) for k in keys}
    out["num_scenes"] = len(metrics)
    out["total_parts"] = int(sum(d["num_parts"] for d in dicts))
    return out
