"""
Animating a predicted reassembly, from a dump written by
``scripts/dump_prediction.py``.

Needs numpy, and trimesh to build scenes -- no torch -- so a dump copied off
Kaggle can be viewed on any machine with a clone of this repository.

What moves
----------
Fragment ``f``'s vertices, centred on its own centroid, are ``x = V_f - c_f``.
The dump holds the scatter rotation ``A_f`` the model was shown, a display-only
scatter offset, the predicted rotation ``R_f`` and the solver's predicted
placement ``p_f`` (world units, gauge-fixed on the mean of the true centroids):

    s = 0   scattered     A_f x + c_f + shift_f
    s = 1   reassembled   R_f A_f x + p_f           (``reassemble``)
            rotation only R_f A_f x + c_f           (``rotation``: true centroids,
                                                     so only rotation error shows)
            ground truth  x + c_f

Rotation is interpolated by SLERP from the identity to ``R_f`` on top of the
fixed scatter ``A_f``, and position linearly. The animation therefore shows the
prediction being applied, not a blend between two unrelated poses. A perfect
model has ``R_f = A_f^T`` and ``p_f = c_f``, and its last frame is the object.

Colours are :func:`reassembly.viz.scene.fragment_colour`, the same palette as
every other view in this package, so fragment ``k`` is the same colour in every
frame, every panel and every figure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scene import fragment_colour

TARGETS = ("reassemble", "rotation", "truth")


def load_dump(path) -> Dict:
    """Read a dump. Plain arrays with offsets, so no ``allow_pickle``."""
    data = np.load(path, allow_pickle=False)
    v_off, f_off = data["vertex_offsets"], data["face_offsets"]
    count = len(v_off) - 1
    fragments = [{"vertices": data["vertices"][v_off[i]:v_off[i + 1]].astype(np.float64),
                  "faces": data["faces"][f_off[i]:f_off[i + 1]],
                  "centroid": data["centroids"][i].astype(np.float64)}
                 for i in range(count)]
    dump = {"fragments": fragments}
    for name in ("A", "shift", "R_pred", "R_gt", "placement", "geodesic_deg",
                 "tilt_deg", "twist_deg", "part_chamfer"):
        if name in data.files:
            dump[name] = data[name].astype(np.float64)
    for name in ("scene", "checkpoint"):
        dump[name] = str(data[name]) if name in data.files else ""
    for name in ("epoch", "seed"):
        dump[name] = int(data[name]) if name in data.files else -1
    for name in ("rmse_t", "part_accuracy", "chamfer", "matches"):
        dump[name] = float(data[name]) if name in data.files else float("nan")
    if "placement" not in dump:
        # A rotation-only dump: place every fragment at its true centroid.
        dump["placement"] = np.stack([f["centroid"] for f in fragments])
    return dump


# -- rotations --------------------------------------------------------------

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)``, choosing the branch with the largest denominator so it
    stays accurate near 180 degrees -- where a model at chance lives."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    candidates = np.array([1 + trace,
                           1 + m[0, 0] - m[1, 1] - m[2, 2],
                           1 - m[0, 0] + m[1, 1] - m[2, 2],
                           1 - m[0, 0] - m[1, 1] + m[2, 2]])
    k = int(np.argmax(candidates))
    if k == 0:
        q = [candidates[0], m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]
    elif k == 1:
        q = [m[2, 1] - m[1, 2], candidates[1], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]]
    elif k == 2:
        q = [m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], candidates[2], m[1, 2] + m[2, 1]]
    else:
        q = [m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], candidates[3]]
    q = np.asarray(q)
    return q / max(np.linalg.norm(q), 1e-12)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp_from_identity(R: np.ndarray, s: float) -> np.ndarray:
    """The rotation ``s`` of the way from the identity to ``R``, shortest arc."""
    q = matrix_to_quaternion(R)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    if angle < 1e-9:
        return np.eye(3)
    axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
    half = s * angle / 2.0
    return quaternion_to_matrix(np.r_[np.cos(half), np.sin(half) * axis])


# -- poses ------------------------------------------------------------------

def pose_at(dump: Dict, index: int, s: float, target: str = "reassemble") -> np.ndarray:
    """``4x4`` transform of fragment ``index`` at animation parameter ``s``."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
    centroid = dump["fragments"][index]["centroid"]
    A = dump["A"][index]
    shift = dump["shift"][index] if "shift" in dump else np.zeros(3)
    if target == "truth":
        R, end = dump["R_gt"][index], centroid
    elif target == "rotation":
        R, end = dump["R_pred"][index], centroid
    else:
        R, end = dump["R_pred"][index], dump["placement"][index]
    rotation = slerp_from_identity(R, s) @ A
    start = centroid + shift
    position = (1.0 - s) * start + s * end
    M = np.eye(4)
    M[:3, :3] = rotation
    M[:3, 3] = position - rotation @ centroid     # rotate about the centroid
    return M


def posed_vertices(dump: Dict, s: float, target: str = "reassemble") -> List[np.ndarray]:
    out = []
    for i, fragment in enumerate(dump["fragments"]):
        M = pose_at(dump, i, s, target)
        out.append(fragment["vertices"] @ M[:3, :3].T + M[:3, 3])
    return out


def layouts(mode: str, s: float, span: float) -> List[Tuple[float, str, float]]:
    """``(x offset, target, parameter)`` for each copy a mode draws."""
    if mode == "compare":
        return [(-span, "reassemble", 0.0), (0.0, "reassemble", s), (span, "truth", 1.0)]
    return [(0.0, mode, s)]


def frame_parameter(step: int, steps: int, hold: int) -> float:
    """
    ``s`` for a ping-pong loop with both ends held.

    A plain 0 -> 1 loop snaps back the instant it finishes, which reads as a
    glitch and gives nobody time to judge the result.
    """
    period = 2 * (hold + steps)
    t = step % period
    if t < hold:
        return 0.0
    t -= hold
    if t < steps:
        return t / max(steps - 1, 1)
    t -= steps
    if t < hold:
        return 1.0
    return 1.0 - (t - hold) / max(steps - 1, 1)


def world_bounds(dumps: Sequence[Dict], mode: str, samples: int = 9) -> Tuple[np.ndarray, float]:
    """
    One bounding sphere over every frame of every panel, for a camera that
    never moves. An autofit camera shrinks as the fragments converge, which
    looks like the object being pulled toward the viewer -- motion made by the
    renderer, not by the model.
    """
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    targets = ("reassemble", "truth") if mode == "compare" else (mode,)
    for dump in dumps:
        for target in targets:
            for s in np.linspace(0.0, 1.0, samples):
                for v in posed_vertices(dump, s, target):
                    lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    centre = (lo + hi) / 2.0
    return centre, max(float(np.linalg.norm(hi - lo)) / 2.0, 1e-6)


def build_trimesh_scene(dump: Dict, s: float, mode: str = "reassemble",
                        span: Optional[float] = None):
    """A ``trimesh.Scene`` of one frame. ``compare`` draws scattered | predicted | truth."""
    import trimesh

    scene = trimesh.Scene()
    if span is None:
        span = 2.2 * world_bounds([dump], mode, samples=3)[1]
    for i, fragment in enumerate(dump["fragments"]):
        for dx, target, parameter in layouts(mode, s, span):
            mesh = trimesh.Trimesh(fragment["vertices"], fragment["faces"], process=False)
            mesh.apply_transform(pose_at(dump, i, parameter, target))
            if dx:
                mesh.apply_translation([dx, 0.0, 0.0])
            mesh.visual.face_colors = fragment_colour(i)
            scene.add_geometry(mesh, geom_name=f"fragment{i}_{target}_{dx:+.0f}")
    return scene


def describe(dump: Dict) -> str:
    """Two lines on what the dump shows, printed by the viewers."""
    geo = dump.get("geodesic_deg", np.array([np.nan]))
    lines = [f"scene    : {dump['scene']}   ({len(dump['fragments'])} fragments, "
             f"checkpoint epoch {dump['epoch']}, seed {dump['seed']})",
             f"rotation : mean {np.nanmean(geo):.2f} deg, worst {np.nanmax(geo):.2f}, "
             f"best {np.nanmin(geo):.2f}   (chance 126.47)"]
    if "tilt_deg" in dump:
        lines.append(f"           tilt {np.nanmean(dump['tilt_deg']):.2f}, "
                     f"twist {np.nanmean(dump['twist_deg']):.2f}")
    if np.isfinite(dump.get("rmse_t", np.nan)):
        lines.append(f"assembly : RMSE(T) {dump['rmse_t']:.4f}   Chamfer {dump['chamfer']:.5f}   "
                     f"part accuracy {dump['part_accuracy']:.3f}   "
                     f"({dump['matches']:.0f} matches)")
    return "\n".join(lines)
