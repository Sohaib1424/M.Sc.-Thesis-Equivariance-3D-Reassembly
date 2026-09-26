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

from vngat.config import Config  # noqa: E402
from vngat.data.dataset import draw_scene  # noqa: E402
from vngat.data.io import diffuse_fragments, list_fracture_dirs, load_scene  # noqa: E402
from vngat.data.mesh_ops import extract_fractures  # noqa: E402


def _colour(mesh: trimesh.Trimesh, rgba=None) -> trimesh.Trimesh:
    mesh.visual.face_colors = rgba if rgba is not None else trimesh.visual.random_color()
    return mesh


def _shift(mesh: trimesh.Trimesh, offset) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(np.asarray(offset, dtype=float))
    return out


def _data_config(args) -> Config:
    """The checkpoint's data definition in prediction mode, else the flags."""
    if args.mode == "prediction" and Path(args.checkpoint).is_file():
        import torch

        from scripts.evaluate import load_config

        cfg = load_config(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    else:
        cfg = Config(split_source="hash")
    cfg.root_dir = args.root_dir or cfg.root_dir
    if args.data_subsets is not None:
        cfg.data_subsets = args.data_subsets
    return cfg


def build_scene(args) -> trimesh.Scene:
    rng = np.random.default_rng(args.seed)
    if args.scene:
        scene_dir = args.scene
        patterns = list_fracture_dirs(scene_dir, args.fracture_pattern or None)
        mode = args.fracture or patterns[int(rng.integers(len(patterns)))]
    else:
        scene_dir, mode, _ = draw_scene(_data_config(args), args.split, rng)
    print(f"scene: {scene_dir} / {mode}")
    meshes = load_scene(str(scene_dir), mode)
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

    from vngat.data.dataset import build_graphs
    from vngat.evaluation.metrics import euler_rmse, geodesic_angle
    from vngat.training.bridge import build_model_inputs, ground_truth_rotation
    from scripts.evaluate import load_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, _state = load_model(args.checkpoint, device)
    # The checkpoint decides how the scene is built: same normalisation, same
    # input source (the model was trained on one of them only).
    target, model_input, _ = build_graphs(meshes, normalize_mode=cfg.normalize_mode,
                                          input_source=cfg.input_source,
                                          with_correspondence=False)
    target = target.to(device)
    model_graph = target if model_input is None else model_input.to(device)

    from vngat.data.io import random_rotation_matrices

    rot = torch.from_numpy(random_rotation_matrices(
        len(meshes), np.random.default_rng(args.seed)).astype(np.float32)).to(device)

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
    p.add_argument("--root_dir", type=str, default=None,
                   help="Default: the checkpoint's (prediction mode) or 'data'.")
    p.add_argument("--data_subsets", type=str, default=None)
    p.add_argument("--scene", type=str, default="", help="Explicit scene directory.")
    p.add_argument("--fracture", type=str, default="", help="Pattern inside --scene.")
    p.add_argument("--split", type=str, default="val",
                   help="Split to draw a random scene from (prediction mode uses the "
                        "checkpoint's own split definition, so 'val' is held out).")
    p.add_argument("--fracture_pattern", type=str, default="fractured_")
    p.add_argument("--seed", type=int, default=0)
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
