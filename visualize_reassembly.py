#!/usr/bin/env python
"""
Animate a VN-GAT reassembly locally, from a dump produced on Kaggle by
`scripts/dump_prediction.py`.

    pip install trimesh numpy
    python visualize_reassembly.py --dump prediction.npz            # interactive
    python visualize_reassembly.py --dump prediction.npz --mode compare
    python visualize_reassembly.py --dump prediction.npz --export frames/

Standalone: needs only numpy and trimesh, no part of the project.

WHAT IS BEING ANIMATED

Fragment f's vertices centred on its own centroid are `x = V_f - c_f`. The dump
holds the scatter rotation `A_f` and the model's prediction `R_pred_f`:

    t = 0   scattered    :  A_f x + c_f + shift_f
    t = 1   reassembled  :  R_pred_f A_f x + c_f

Rotation is interpolated by SLERP from the identity to `R_pred_f`, applied on
top of the fixed scatter `A_f`; translation is interpolated linearly from the
scattered offset to zero. So the animation shows the model's answer being
applied gradually, not a blend between two unrelated poses.

A perfect prediction satisfies `R_pred = A^T`, so `R_pred A = I` and the final
frame coincides with the original object. Anything left misaligned at t = 1 is
exactly the model's error -- which is the point of watching it.

MODES
  reassemble  (default) scattered -> the model's prediction
  compare               three copies side by side: scattered | predicted | truth
  truth                 scattered -> ground truth, as an upper bound on what a
                        perfect model would show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
def load_dump(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    v_off, f_off = d["vertex_offsets"], d["face_offsets"]
    frags = [
        {"vertices": d["vertices"][v_off[i]:v_off[i + 1]],
         "faces": d["faces"][f_off[i]:f_off[i + 1]],
         "centroid": d["centroids"][i]}
        for i in range(len(v_off) - 1)
    ]
    return {
        "fragments": frags, "A": d["A"], "t": d["t"],
        "R_pred": d["R_pred"], "R_gt": d["R_gt"],
        "geodesic_deg": d["geodesic_deg"],
        "tilt_deg": d["tilt_deg"], "twist_deg": d["twist_deg"],
        "scene": str(d["scene"]), "fracture": str(d["fracture"]),
        "epoch": int(d["epoch"]),
    }


# ---------------------------------------------------------------------------
def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z), branch-selected to stay stable near 180 degrees."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    cand = np.array([1 + tr,
                     1 + m[0, 0] - m[1, 1] - m[2, 2],
                     1 - m[0, 0] + m[1, 1] - m[2, 2],
                     1 - m[0, 0] - m[1, 1] + m[2, 2]])
    k = int(np.argmax(cand))
    if k == 0:
        q = np.array([1 + tr, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    elif k == 1:
        q = np.array([m[2, 1] - m[1, 2], cand[1], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]])
    elif k == 2:
        q = np.array([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], cand[2], m[1, 2] + m[2, 1]])
    else:
        q = np.array([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], cand[3]])
    return q / max(np.linalg.norm(q), 1e-12)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp_from_identity(R: np.ndarray, s: float) -> np.ndarray:
    """Rotation `s` of the way from the identity to R, along the shortest arc."""
    q = matrix_to_quat(R)
    if q[0] < 0:                      # shortest path
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    if angle < 1e-8:
        return np.eye(3)
    axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
    half = s * angle / 2.0
    return quat_to_matrix(np.r_[np.cos(half), np.sin(half) * axis])


# ---------------------------------------------------------------------------
def pose_at(dump: dict, index: int, s: float, target: str = "pred") -> np.ndarray:
    """4x4 transform for fragment `index` at animation parameter s in [0, 1]."""
    A = dump["A"][index]
    R = dump["R_pred"][index] if target == "pred" else dump["R_gt"][index]
    c = dump["fragments"][index]["centroid"]
    shift = dump["t"][index]

    rot = slerp_from_identity(R, s) @ A          # scatter, then part of the fix
    offset = (1.0 - s) * shift

    M = np.eye(4)
    M[:3, :3] = rot
    # rotate about the fragment's own centroid, then translate
    M[:3, 3] = c + offset - rot @ c
    return M


def colours(n: int) -> np.ndarray:
    """Distinct, stable per-fragment colours."""
    hues = (np.arange(n) * 0.618033988749895) % 1.0
    out = []
    for h in hues:
        i = int(h * 6) % 6
        f = h * 6 - int(h * 6)
        v, p, q, tt = 235, 90, int(235 * (1 - f * 0.6)), int(235 * (1 - (1 - f) * 0.6))
        rgb = [(v, tt, p), (q, v, p), (p, v, tt), (p, q, v), (tt, p, v), (v, p, q)][i]
        out.append([*rgb, 255])
    return np.array(out, np.uint8)


def build_scene(dump: dict, s: float, mode: str) -> trimesh.Scene:
    scene = trimesh.Scene()
    cols = colours(len(dump["fragments"]))
    span = float(max(f["vertices"].max(0).max() - f["vertices"].min(0).min()
                     for f in dump["fragments"])) * 3.0

    def add(index, transform, offset_x, colour):
        f = dump["fragments"][index]
        mesh = trimesh.Trimesh(f["vertices"], f["faces"], process=False)
        mesh.apply_transform(transform)
        if offset_x:
            mesh.apply_translation([offset_x, 0, 0])
        mesh.visual.face_colors = colour
        scene.add_geometry(mesh)

    for i in range(len(dump["fragments"])):
        if mode == "compare":
            add(i, pose_at(dump, i, 0.0, "pred"), -span, cols[i])   # scattered
            add(i, pose_at(dump, i, s, "pred"), 0.0, cols[i])       # prediction
            add(i, pose_at(dump, i, 1.0, "gt"), span, cols[i])      # truth
        elif mode == "truth":
            add(i, pose_at(dump, i, s, "gt"), 0.0, cols[i])
        else:
            add(i, pose_at(dump, i, s, "pred"), 0.0, cols[i])
    return scene


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--mode", default="reassemble", choices=["reassemble", "compare", "truth"])
    p.add_argument("--steps", type=int, default=60, help="Frames from scattered to assembled.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--export", default="", help="Write per-frame .glb files here instead of showing.")
    p.add_argument("--at", type=float, default=None,
                   help="Show a single frame at this s in [0,1] rather than animating.")
    args = p.parse_args(argv)

    dump = load_dump(args.dump)
    geo = dump["geodesic_deg"]
    print(f"scene    : {dump['scene']}")
    print(f"fracture : {dump['fracture']}   ({len(dump['fragments'])} fragments, "
          f"checkpoint epoch {dump['epoch']})")
    print(f"error    : mean {geo.mean():.2f} deg, worst {geo.max():.2f}, best {geo.min():.2f}")
    print(f"           tilt {dump['tilt_deg'].mean():.2f}, twist {dump['twist_deg'].mean():.2f}")
    print(f"           (chance = 126.47 deg)")
    worst = int(np.argmax(geo))
    print(f"worst fragment: {worst} at {geo[worst]:.2f} deg\n")

    if args.at is not None:
        build_scene(dump, float(np.clip(args.at, 0, 1)), args.mode).show()
        return 0

    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.steps + 1):
            s = i / args.steps
            build_scene(dump, s, args.mode).export(out / f"frame_{i:04d}.glb")
        print(f"wrote {args.steps + 1} frames to {out}/")
        print("To make a video (needs ffmpeg + a glb renderer), or just open the")
        print("first and last frames to compare start and end.")
        return 0

    # Interactive: step with the arrow keys via repeated windows is clumsy, so
    # drive it as a smooth loop and let the user close the window to advance.
    try:
        import pyglet  # noqa: F401
    except ImportError:
        print("Interactive playback needs pyglet:  pip install 'pyglet<2'")
        print("Falling back to three static views: scattered, halfway, assembled.")
        for s in (0.0, 0.5, 1.0):
            build_scene(dump, s, args.mode).show()
        return 0

    print("Playing. Close the window to exit.")
    scene = build_scene(dump, 0.0, args.mode)
    state = {"i": 0}

    def callback(scene_obj):
        state["i"] = (state["i"] + 1) % (args.steps + 1)
        s = state["i"] / args.steps
        fresh = build_scene(dump, s, args.mode)
        scene_obj.geometry.clear()
        for name, geom in fresh.geometry.items():
            scene_obj.add_geometry(geom, geom_name=name)

    scene.show(callback=callback, callback_period=1.0 / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
