#!/usr/bin/env python
"""
Show what a trained model actually reconstructs, next to the ground truth.

    python scripts/vis_prediction.py --checkpoint checkpoints/best.pt --root-dir data
    python scripts/vis_prediction.py --checkpoint checkpoints/best.pt --root-dir data \
        --save prediction.png

Four copies are laid out left to right:

    scattered  ->  predicted  ->  ground truth  ->  overlay

`scattered` is the network's input. `predicted` applies the predicted rotation
and the solved translation. `ground truth` is the assembled object. `overlay`
draws both in the same frame -- prediction in colour, truth in translucent
grey -- which is the only one of the four that shows you *where* the error is
rather than that there is some.

Per-fragment error is printed alongside, so a figure that looks fine can still
be checked against numbers.

`--oracle-rotation` substitutes the true rotations and runs only the
translation solver. Comparing that with the normal output separates "the
network got the orientation wrong" from "the solver placed it wrong", which is
the first thing to want to know when a result looks bad.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from reassembly.utils.console import quiet_third_party_warnings

quiet_third_party_warnings()


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--root-dir", type=str, default=None,
                   help="Defaults to the path stored in the checkpoint's config")
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--fracture", type=str, default=None)
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--oracle-rotation", action="store_true",
                   help="Use TRUE rotations; isolates the translation solver")
    p.add_argument("--collision-weight", type=float, default=0.0)
    p.add_argument("--offset", type=float, default=None,
                   help="Spacing between the four copies; auto-scales by default")
    p.add_argument("--save", type=str, default=None,
                   help="Write a PNG instead of opening a window (headless-safe)")
    p.add_argument("--no-overlay", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = build_args(argv)

    import trimesh

    from reassembly.assembly.matching import match_scene
    from reassembly.assembly.translation import solve_translations
    from reassembly.config import Config
    from reassembly.data.augment import diffuse_fragments
    from reassembly.data.collate import merge_fragments
    from reassembly.data.features import get_features
    from reassembly.data.scene_io import load_scene
    from reassembly.data.splits import SceneIndex
    from reassembly.evaluation.metrics import (
        evaluate_assembly, rotation_error_geodesic_deg,
    )
    from reassembly.models.vn_gat_model import VNGATModel
    from reassembly.training.bridge import build_model_inputs

    # ------------------------------------------------------------- model
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(state["config"]) if "config" in state else Config()
    if args.root_dir:
        cfg.data.root_dir = args.root_dir
    device = torch.device(args.device)

    model = VNGATModel(
        hidden_channels=cfg.model.hidden_channels, num_layers=cfg.model.num_layers,
        num_vn_slots=cfg.model.num_vn_slots, heads=cfg.model.heads,
        head_dim=cfg.model.head_dim, embed_dim=cfg.model.embed_dim,
        norm=cfg.model.norm, angular=cfg.model.angular,
    ).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # ------------------------------------------------------------- scene
    rng = random.Random(args.seed)
    scene_dir = args.scene or str(
        SceneIndex(cfg.data.root_dir, split=args.split,
                   val_frac=cfg.data.val_frac, test_frac=cfg.data.test_frac,
                   seed=cfg.data.split_seed).sample(rng)
    )
    meshes = load_scene(scene_dir, fracture_id=args.fracture, rng=rng,
                        fracture_pattern=cfg.data.fracture_pattern)
    if len(meshes) < 2:
        raise SystemExit(f"{scene_dir} yielded {len(meshes)} fragment(s); nothing to assemble.")

    diffused, transforms = diffuse_fragments(meshes)
    n_frag = len(meshes)
    print(f"scene    : {scene_dir}")
    print(f"fragments: {n_frag}")

    clean_graph = merge_fragments([get_features(m) for m in meshes])
    diff_graph = merge_fragments([get_features(m) for m in diffused])
    diff_graph.fragment_scene_id = torch.zeros(n_frag, dtype=torch.long)

    # ------------------------------------------------------------- predict
    R_gt = np.stack([np.asarray(t)[:3, :3].T for t in transforms])   # R_gt = R_diffuse^T
    with torch.no_grad():
        out = model(**build_model_inputs(diff_graph.to(device)))
    R_pred = R_gt if args.oracle_rotation else out["R_pred"].cpu().numpy().astype(np.float64)

    fid = diff_graph.fragment_id.cpu().numpy()
    emb = out["vertex_embedding"].cpu().numpy()
    pts = diff_graph.x[:, 0:3].cpu().numpy()
    nrm = diff_graph.x[:, 3:6].cpu().numpy()

    per_frag = lambda a: [a[fid == f] for f in range(n_frag)]
    oriented_pts = [p @ R_pred[f].T for f, p in enumerate(per_frag(pts))]
    oriented_nrm = [n @ R_pred[f].T for f, n in enumerate(per_frag(nrm))]

    matches = match_scene(per_frag(emb), oriented_pts, oriented_nrm)
    result = solve_translations(matches, n_frag)
    print(f"matches  : {len(matches)}  components={result.num_components} "
          f"{'(fully constrained)' if result.fully_constrained else '(UNDERCONSTRAINED)'}")

    # ------------------------------------------------------------- metrics
    t_gt = clean_graph.fragment_centroid.cpu().numpy().astype(np.float64)
    t_gt -= t_gt.mean(0, keepdims=True)
    t_pred = result.translations - result.translations.mean(0, keepdims=True)
    verts_local = per_frag(clean_graph.x[:, 0:3].cpu().numpy().astype(np.float64))

    metrics = evaluate_assembly(R_pred, R_gt, t_pred, t_gt, verts_local)
    print(f"\n{metrics}")
    ang = rotation_error_geodesic_deg(R_pred, R_gt)
    tra = np.linalg.norm(t_pred - t_gt, axis=1)
    print(f"\n{'frag':>5} {'verts':>7} {'rot err deg':>12} {'trans err':>11}")
    for f in range(n_frag):
        print(f"{f:>5} {len(verts_local[f]):>7,} {ang[f]:>12.3f} {tra[f]:>11.5f}")

    # ------------------------------------------------------------- render
    allv = np.concatenate(verts_local)
    span = float((allv.max(0) - allv.min(0)).max())
    step = args.offset if args.offset is not None else span * 1.6

    import colorsys
    colors = []
    for i in range(n_frag):
        r, g, b = colorsys.hsv_to_rgb((i / max(n_frag, 1)) % 1.0, 0.65, 0.95)
        colors.append(np.array([int(r * 255), int(g * 255), int(b * 255), 255], np.uint8))

    scene = trimesh.Scene()

    def add(vertices_list, faces_list, shift, alpha=255, grey=False):
        for i, (v, f) in enumerate(zip(vertices_list, faces_list)):
            m = trimesh.Trimesh(np.asarray(v) + shift, np.asarray(f), process=False)
            c = np.array([150, 150, 150, alpha], np.uint8) if grey else \
                np.concatenate([colors[i][:3], [alpha]]).astype(np.uint8)
            m.visual.face_colors = c
            scene.add_geometry(m)

    faces = [np.asarray(m.faces) for m in meshes]
    diff_local = per_frag(diff_graph.x[:, 0:3].cpu().numpy().astype(np.float64))
    placed_pred = [oriented_pts[f] + t_pred[f] for f in range(n_frag)]
    placed_gt = [verts_local[f] + t_gt[f] for f in range(n_frag)]

    add(diff_local, faces, np.array([-1.5 * step, 0, 0]))
    add(placed_pred, faces, np.array([-0.5 * step, 0, 0]))
    add(placed_gt, faces, np.array([0.5 * step, 0, 0]))
    if not args.no_overlay:
        add(placed_gt, faces, np.array([1.5 * step, 0, 0]), alpha=70, grey=True)
        add(placed_pred, faces, np.array([1.5 * step, 0, 0]), alpha=255)

    print(f"\nlayout (left to right): scattered | predicted | ground truth"
          f"{'' if args.no_overlay else ' | overlay'}")

    if args.save:
        png = scene.save_image(resolution=(2200, 900), visible=True)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "wb") as fh:
            fh.write(png)
        print(f"wrote {args.save}")
    else:
        try:
            scene.show(caption="scattered | predicted | truth | overlay")
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"Could not open a viewer ({type(exc).__name__}: {exc}).\n"
                f"On Kaggle or any headless machine, pass --save prediction.png."
            )


if __name__ == "__main__":
    main()
