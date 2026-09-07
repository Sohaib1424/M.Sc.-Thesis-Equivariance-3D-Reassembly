#!/usr/bin/env python
"""
Visualisation, consolidating what used to be six near-identical vis_*.py
scripts into one `--mode` switch.

    python -m scripts.visualize --mode default
    python -m scripts.visualize --mode diffused
    python -m scripts.visualize --mode fractures
    python -m scripts.visualize --mode diffused_fractures
    python -m scripts.visualize --mode side_by_side
    python -m scripts.visualize --mode prediction --checkpoint checkpoints/best.pt

`--mode prediction` is the one that matters for the thesis: it renders the
scattered input, the model's reassembly, and the ground truth side by side, so
a failure is visible as geometry rather than only as a number.

Needs a display for `--show`; use `--out scene.glb` on a headless machine and
open the file locally.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from vngat.data.io import diffuse_fragments, load_random_scene  # noqa: E402
from vngat.data.mesh_ops import extract_fractures  # noqa: E402
from vngat.data.splits import get_random_directory  # noqa: E402


def _colour(mesh: trimesh.Trimesh, rgba=None) -> trimesh.Trimesh:
    mesh.visual.face_colors = rgba if rgba is not None else trimesh.visual.random_color()
    return mesh


def _shift(mesh: trimesh.Trimesh, offset) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(np.asarray(offset, dtype=float))
    return out


def build_scene(args) -> trimesh.Scene:
    scene_dir = Path(args.scene) if args.scene else get_random_directory(
        args.root_dir, split=args.split, max_scenes=args.max_scenes
    )
    print(f"scene: {scene_dir}")
    meshes = load_random_scene(str(scene_dir), fracture_pattern=args.fracture_pattern or None)
    print(f"{len(meshes)} fragments, {sum(len(m.vertices) for m in meshes):,} vertices")

    scene = trimesh.Scene()
    span = float(max(m.extents.max() for m in meshes)) * 3.0

    if args.mode == "default":
        for m in meshes:
            scene.add_geometry(_colour(m.copy()))

    elif args.mode == "diffused":
        diffused, _ = diffuse_fragments(meshes)
        for m in diffused:
            scene.add_geometry(_colour(m))

    elif args.mode == "fractures":
        for m in meshes:
            scene.add_geometry(_colour(extract_fractures(m)))

    elif args.mode == "diffused_fractures":
        diffused, _ = diffuse_fragments(meshes)
        for m in diffused:
            scene.add_geometry(_colour(extract_fractures(m)))

    elif args.mode == "side_by_side":
        diffused, _ = diffuse_fragments(meshes)
        for m in meshes:
            scene.add_geometry(_colour(m.copy()))
        for m in diffused:
            scene.add_geometry(_shift(_colour(m), (span, 0, 0)))
        for m in meshes:
            scene.add_geometry(_shift(_colour(extract_fractures(m)), (2 * span, 0, 0)))

    elif args.mode == "prediction":
        scene = build_prediction_scene(meshes, args, span)

    else:  # pragma: no cover
        raise ValueError(f"unknown mode {args.mode!r}")
    return scene


def build_prediction_scene(meshes, args, span: float) -> trimesh.Scene:
    """Scattered input | model reassembly | ground truth, left to right."""
    import torch

    from vngat.data.features import get_features
    from vngat.data.graph import merge_fragments
    from vngat.evaluation.metrics import euler_rmse, geodesic_angle
    from vngat.training.bridge import build_model_inputs, ground_truth_rotation
    from scripts.evaluate import load_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _cfg, _state = load_model(args.checkpoint, device)

    input_meshes = [extract_fractures(m) for m in meshes] if args.input_source == "frac" else meshes
    input_meshes = [
        fm if (len(fm.faces) and len(fm.vertices) >= 3) else m
        for fm, m in zip(input_meshes, meshes)
    ]

    target = merge_fragments([get_features(m) for m in meshes]).to(device)
    model_graph = (target if args.input_source == "full"
                   else merge_fragments([get_features(m) for m in input_meshes]).to(device))

    rot = torch.from_numpy(
        np.stack([trimesh.transformations.random_rotation_matrix()[:3, :3] for _ in meshes])
        .astype(np.float32)
    ).to(device)

    with torch.no_grad():
        R_pred = model(**build_model_inputs(model_graph.rotate_per_fragment(rot)))["R_pred"]
    R_gt = ground_truth_rotation(rot)

    geo = geodesic_angle(R_pred, R_gt)
    print(f"  geodesic error: mean {geo.mean():.2f} deg, median {geo.median():.2f} deg")
    print(f"  Euler RMSE    : {euler_rmse(R_pred, R_gt).mean():.2f} deg")

    rot_np = rot.cpu().numpy()
    pred_np = R_pred.cpu().numpy()
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        centroid = mesh.vertices.mean(axis=0)
        local = mesh.vertices - centroid
        scattered = local @ rot_np[i].T
        reassembled = scattered @ pred_np[i].T

        scene.add_geometry(_colour(trimesh.Trimesh(scattered + centroid, mesh.faces, process=False)))
        scene.add_geometry(_shift(
            _colour(trimesh.Trimesh(reassembled + centroid, mesh.faces, process=False)), (span, 0, 0)))
        scene.add_geometry(_shift(_colour(mesh.copy()), (2 * span, 0, 0)))
    print("  left: scattered input | middle: model reassembly | right: ground truth")
    return scene


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="default",
                   choices=["default", "diffused", "fractures", "diffused_fractures",
                            "side_by_side", "prediction"])
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--scene", type=str, default="", help="Explicit scene directory.")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--fracture_pattern", type=str, default="fractured_")
    p.add_argument("--input_source", type=str, default="full", choices=["full", "frac"])
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="", help="Export to .glb/.ply instead of showing.")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    scene = build_scene(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.out)
        print(f"wrote {args.out}")
    if args.show or not args.out:
        scene.show()


if __name__ == "__main__":
    main()
