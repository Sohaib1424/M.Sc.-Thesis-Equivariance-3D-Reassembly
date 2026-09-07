#!/usr/bin/env python
"""
Evaluate a trained checkpoint with GARF-comparable metrics.

    python -m scripts.evaluate --checkpoint checkpoints/best.pt --split test \
        --num_scenes 200 --solve_translation true

Rotation metrics need only the network. RMSE(T), Chamfer and Part Accuracy
need a full assembly, so they are produced by running the classical
translation solver on top of the predicted rotations; pass
`--solve_translation false` to report rotation only rather than have the
translation-dependent numbers quietly omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vngat.assembly.translation import assemble  # noqa: E402
from vngat.config import Config  # noqa: E402
from vngat.data.dataset import BreakingBadDataset, collate_fn  # noqa: E402
from vngat.evaluation.metrics import (  # noqa: E402
    CHANCE_EULER_RMSE_DEG, CHANCE_GEODESIC_DEG, SYMMETRY_FLOOR_EULER_RMSE_DEG,
    SYMMETRY_FLOOR_GEODESIC_DEG, aggregate, evaluate_scene,
)
from vngat.models.vn_gat import VNGATModel  # noqa: E402
from vngat.training.bridge import build_model_inputs, ground_truth_rotation, prepare_scene  # noqa: E402
from vngat.utils.env import dataloader_worker_init, seed_everything  # noqa: E402
from vngat.utils.progress import make_bar, write  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(state["config"]) if state.get("config") else Config()
    model = VNGATModel(
        hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
        num_vn_slots=cfg.num_vn_slots, heads=cfg.heads, embed_dim=cfg.embed_dim,
        gram_bottleneck=cfg.gram_bottleneck, norm=cfg.norm,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, state


@torch.no_grad()
def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, state = load_model(args.checkpoint, device)
    seed_everything(args.seed)

    write(f"checkpoint: epoch {state.get('epoch')} | val_total {state.get('val_loss'):.4f}"
          if state.get("val_loss") is not None else f"checkpoint: epoch {state.get('epoch')}")

    dataset = BreakingBadDataset(
        root_dir=args.root_dir or cfg.root_dir,
        split=args.split,
        val_frac=cfg.val_frac, test_frac=cfg.test_frac, split_seed=cfg.split_seed,
        max_scenes=cfg.max_scenes,
        fracture_pattern=cfg.fracture_pattern or None,
        input_source=cfg.input_source,
        with_correspondence=cfg.correspondence,
        correspondence_tol=cfg.correspondence_tol,
        nominal_length=args.num_scenes,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, worker_init_fn=dataloader_worker_init,
    )

    per_scene = []
    bar = make_bar(args.num_scenes, "evaluating")
    for batch in loader:
        scene = prepare_scene(batch, device, non_blocking=False)
        outputs = model(**build_model_inputs(scene["diffused_input"]))
        R_pred = outputs["R_pred"]
        R_gt = ground_truth_rotation(scene["rot"])

        target = scene["clean_target"]
        diffused = scene["diffused_target"]
        model_graph = scene["diffused_input"]
        num_frags = target.num_fragments

        # Rotate the scattered geometry by the prediction: this is the
        # reassembled fragment up to an unknown translation.
        rot_per_node = R_pred.index_select(0, diffused.node_frag)
        rotated = torch.einsum('nij,nj->ni', rot_per_node, diffused.node_vec[:, 0])

        kwargs = {}
        kwargs_extra = {}
        if args.solve_translation:
            # Matching runs on the graph the network embedded, which in
            # input_source='frac' mode has a DIFFERENT node count from the full
            # target mesh -- so its own positions, normals and fragment ids must
            # be used here, not the target's.
            match_rot = R_pred.index_select(0, model_graph.node_frag)
            match_pts = torch.einsum('nij,nj->ni', match_rot, model_graph.node_vec[:, 0])
            match_normals = torch.einsum('nij,nj->ni', match_rot, model_graph.node_vec[:, 1])

            t_pred, matches = assemble(
                match_pts, match_normals, model_graph.node_frag,
                outputs["vertex_embedding"], num_frags,
                max_points_per_fragment=args.max_match_points,
                collision=args.collision,
            )
            # Ground truth translation: each fragment's centroid in the
            # assembled object frame, gauge-fixed the same way the solver is.
            t_gt = target.frag_centroid - target.frag_centroid.mean(0, keepdim=True)
            placed = rotated + t_pred.index_select(0, target.node_frag)
            truth = target.node_vec[:, 0] + t_gt.index_select(0, target.node_frag)
            kwargs = dict(
                pred_points=placed, gt_points=truth, point_frag=target.node_frag,
                t_pred=t_pred, t_gt=t_gt,
            )
            kwargs_extra = {"num_matches": float(matches.src_idx.numel())}

        metrics = evaluate_scene(R_pred, R_gt, pa_threshold=args.pa_threshold,
                                 symmetry_axis=args.symmetry_axis, **kwargs)
        metrics.update(kwargs_extra)
        # Second-to-last path component is the category for the everyday
        # layout (.../everyday_compressed/<Category>/<hash>).
        parts = Path(batch["scene_dirs"][0]).parts
        metrics["_category"] = parts[-2] if len(parts) >= 2 else "unknown"
        per_scene.append(metrics)
        bar.update(1)
    bar.close()

    summary = aggregate([{k: v for k, v in m.items() if not k.startswith("_")}
                         for m in per_scene])
    write("\n=== results ===")
    for key in sorted(summary):
        write(f"  {key:<24} {summary[key]:.5f}")

    # Reference levels, so a number can be read against something.
    write("\n=== reference levels ===")
    write(f"  chance, geodesic         {CHANCE_GEODESIC_DEG:.2f} deg")
    write(f"  chance, Euler RMSE       {CHANCE_EULER_RMSE_DEG:.2f} deg")
    write(f"  symmetry floor, geodesic {SYMMETRY_FLOOR_GEODESIC_DEG:.2f} deg   "
          f"(axis learnable from one fragment, azimuth about it is not)")
    write(f"  symmetry floor, Euler    {SYMMETRY_FLOOR_EULER_RMSE_DEG:.2f} deg")
    if "geodesic_deg" in summary:
        captured = (CHANCE_GEODESIC_DEG - summary["geodesic_deg"])
        available = CHANCE_GEODESIC_DEG - SYMMETRY_FLOOR_GEODESIC_DEG
        write(f"\n  captured {captured:.2f} of the {available:.2f} deg available "
              f"above the symmetry floor ({100 * captured / available:.1f}%)")
    if "tilt_deg" in summary:
        write(f"\n  tilt (off the {args.symmetry_axis}-axis)  {summary['tilt_deg']:.2f} deg   "
              f"-> ~90 means the axis itself is not learned; ~0 means it is")
        write(f"  twist (about the axis)   {summary['twist_deg']:.2f} deg   "
              f"-> stays ~90 if the azimuth is genuinely unidentifiable")

    if args.by_category:
        from collections import defaultdict

        groups = defaultdict(list)
        for m in per_scene:
            groups[m["_category"]].append(m)
        write("\n=== by category ===")
        write(f"  {'category':<22}{'n':>5}{'geodesic':>11}{'tilt':>9}{'twist':>9}")
        for cat in sorted(groups, key=lambda c: aggregate(
                [{k: v for k, v in m.items() if not k.startswith('_')}
                 for m in groups[c]]).get("geodesic_deg", 1e9)):
            agg = aggregate([{k: v for k, v in m.items() if not k.startswith("_")}
                             for m in groups[cat]])
            write(f"  {cat:<22}{len(groups[cat]):>5}{agg.get('geodesic_deg', float('nan')):>11.2f}"
                  f"{agg.get('tilt_deg', float('nan')):>9.2f}{agg.get('twist_deg', float('nan')):>9.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "per_scene": per_scene}, fh, indent=2)
        write(f"\nwrote {args.out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a VN-GAT checkpoint.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num_scenes", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pa_threshold", type=float, default=0.01)
    p.add_argument("--max_match_points", type=int, default=4096)
    p.add_argument("--solve_translation", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--collision", action="store_true")
    p.add_argument("--symmetry_axis", type=str, default="z", choices=["x", "y", "z"],
                   help="Canonical up-axis of the meshes, for the tilt/twist split. "
                        "Try all three if unsure -- only the right one shows the signature.")
    p.add_argument("--by_category", action="store_true",
                   help="Break results down by object category (Bottle, Bowl, ...). "
                        "Symmetric categories scoring worse than asymmetric ones is "
                        "direct evidence for the azimuth-ambiguity explanation.")
    p.add_argument("--out", type=str, default="")
    return p


def main(argv=None):
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
