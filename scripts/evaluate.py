#!/usr/bin/env python
"""
Evaluate a trained checkpoint with GARF-comparable metrics.

    python -m scripts.evaluate --checkpoint checkpoints/best.pt --split val \
        --num_scenes 0 --solve_translation true

The DATA is rebuilt from the checkpoint's own config -- split source, split
mode, subsets, normalisation, input source -- so a model is always scored on
the split it was trained against. (This script used to rebuild the dataset
from a subset of those settings and silently fell back to the HASH split over
EVERY subset on disk; a model trained on the official split was then scored
partly on official training objects.)

The scenes are a FIXED, deterministic set -- the same objects, break patterns
and rotations on every run -- so two checkpoints evaluated with the same
arguments are compared on identical inputs. `--num_scenes 0` means one break
pattern per object in the split.

THE FRAGMENT RANGE. The checkpoint's own `max_fragments` applies unless
`--max_fragments` is given: `--max_fragments 20` scores any checkpoint on the
benchmark's 2-20 pieces, `--max_fragments 0` on every pattern.

Rotation metrics need only the network. RMSE(T), Chamfer and Part Accuracy
need a full assembly, so they are produced by running the classical
translation solver on top of the predicted rotations, in WORLD units (the
network works in per-scene normalised units; `frag_unit` converts back, so the
0.01 part-accuracy threshold means what the benchmark means by it). Pass
`--solve_translation false` to report rotation only rather than have the
translation-dependent numbers quietly omitted.

Averaging follows the benchmark: per scene, then over scenes. Training logs
are fragment-weighted instead; the two differ when scenes differ in fragment
count, and the thesis tables should use this script's numbers.

THE ANCHOR. Every number is read the benchmark's way: each scene's largest
fragment is set to its true pose -- one rotation of the whole predicted
assembly, and translations measured from the anchor's -- and the other
fragments are scored (`vngat.evaluation.anchor`). `absolute_geodesic_deg` is
the model's own prediction against each object's stored frame, over every
fragment. Both apply to any checkpoint, whatever rotation target it was
trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vngat.assembly.translation import assemble  # noqa: E402
from vngat.config import Config  # noqa: E402
from vngat.data.dataset import BreakingBadDataset, collate_fn, dataset_kwargs  # noqa: E402
from vngat.evaluation.anchor import anchor_alignment  # noqa: E402
from vngat.evaluation.metrics import (  # noqa: E402
    AXIS_ONLY_EULER_RMSE_DEG, AXIS_ONLY_GEODESIC_DEG, CHANCE_EULER_RMSE_DEG,
    CHANCE_GEODESIC_DEG, aggregate, evaluate_scene,
)
from vngat.training.bridge import build_model_inputs, ground_truth_rotation, prepare_scene  # noqa: E402
from vngat.utils.env import dataloader_worker_init  # noqa: E402
from vngat.utils.progress import make_bar, write  # noqa: E402


def load_config(state: dict) -> Config:
    """The checkpoint's config, with a readable error for the previous version's."""
    stored = dict(state.get("config") or {})
    if stored and "head_dim" not in stored:
        raise SystemExit(
            "This checkpoint was written by the previous VN-GAT (Thesis 1): its mesh "
            "layer and edge features differ from this version's, so it cannot be "
            "loaded here. Evaluate it with the Thesis 1 code."
        )
    return Config.from_dict(stored) if stored else Config()


def load_model(checkpoint_path: str, device: torch.device):
    from vngat.training.trainer import build_model

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_config(state)
    model = build_model(cfg, device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, state


@torch.no_grad()
def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, state = load_model(args.checkpoint, device)
    if args.root_dir:
        cfg.root_dir = args.root_dir
    if args.max_fragments is not None:
        # Scoring on a different range than the model trained on is a fair
        # question -- how does a model trained on everything do on the
        # benchmark's 2-20? -- so the flag overrides the checkpoint's value.
        if args.max_fragments < 0 or 0 < args.max_fragments < cfg.min_fragments:
            raise SystemExit(f"--max_fragments must be 0 (no limit) or at least "
                             f"min_fragments ({cfg.min_fragments})")
        cfg.max_fragments = args.max_fragments

    monitor = cfg.checkpoint_monitor
    trained_on = (state.get("config") or {}).get("rotation_target", "absolute")
    write(f"checkpoint: epoch {state.get('epoch')}"
          + (f" | val {monitor} {state['val_loss']:.4f}" if state.get("val_loss") is not None else "")
          + f" | trained on the {trained_on!r} rotation target")

    dataset = BreakingBadDataset(
        split=args.split, fixed=True, fixed_count=args.num_scenes, **dataset_kwargs(cfg),
    )
    write(f"data: {args.split} split ({cfg.split_source}, split_by={cfg.split_by}) -- "
          f"{len(dataset)} scenes from {dataset.num_objects} objects, "
          f"normalize={cfg.normalize_mode}, input={cfg.input_source}")
    limit = dataset.fragment_limit
    write(f"fragments: {cfg.min_fragments}-{cfg.max_fragments} pieces per scene, keeping "
          f"{limit.describe()}" if limit else
          f"fragments: {cfg.min_fragments}+ pieces per scene, no upper limit "
          f"(--max_fragments 20 for the benchmark's 2-20)")
    if cfg.split_by == "fracture":
        write("!! split_by=fracture: these are held-out BREAK PATTERNS of objects seen in "
              "training -- not comparable to object-split (benchmark) numbers.")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, worker_init_fn=dataloader_worker_init,
    )

    per_scene = []
    bar = make_bar(len(dataset), "evaluating")
    for batch in loader:
        scene = prepare_scene(batch, device, non_blocking=False)
        outputs = model(**build_model_inputs(scene["diffused_input"]))
        R_gt = ground_truth_rotation(scene["rot"])
        # The whole predicted assembly turned so this scene's largest fragment
        # sits at its true pose; everything below is placed and scored in that
        # frame, and the anchor itself is left out of every per-fragment mean.
        R_own = outputs["R_pred"]
        R_pred, keep = anchor_alignment(R_own, R_gt, scene["clean_target"])
        R_gt = R_gt.to(R_pred.dtype)
        anchor = int((~keep).nonzero()[0]) if bool((~keep).any()) else None

        target = scene["clean_target"]
        diffused = scene["diffused_target"]
        model_graph = scene["diffused_input"]
        num_frags = target.num_fragments

        kwargs, extra = {}, {}
        if args.solve_translation:
            # Everything geometric in WORLD units: the solver's Huber scale and
            # the part-accuracy threshold are both defined there.
            rotated = torch.einsum("nij,nj->ni", R_pred.index_select(0, diffused.node_frag),
                                   diffused.world_pos())
            # Matching runs on the graph the network embedded, which in
            # input_source='frac' mode has a DIFFERENT node count from the full
            # target mesh -- so its own positions, normals and fragment ids.
            match_rot = R_pred.index_select(0, model_graph.node_frag)
            match_pts = torch.einsum("nij,nj->ni", match_rot, model_graph.world_pos())
            match_normals = torch.einsum("nij,nj->ni", match_rot, model_graph.node_vec[:, 1])
            t_pred, matches = assemble(
                match_pts, match_normals, model_graph.node_frag,
                outputs["vertex_embedding"], num_frags,
                max_points_per_fragment=args.max_match_points,
                collision=args.collision,
                # Centred fragments sit at the origin before translation; the
                # radius is the world radius the scale feature was built from.
                centroids=torch.zeros(num_frags, 3, device=device),
                radii=target.frag_log_scale.squeeze(-1).exp(),
            )
            # Ground truth translation: each fragment's centroid in the
            # assembled object frame. Both measured from the anchor's, which
            # therefore sits exactly at its true position (the benchmark's
            # gauge); a one-fragment scene falls back to the mean.
            t_gt = target.frag_centroid - target.frag_centroid.mean(0, keepdim=True)
            if anchor is not None:
                t_gt = target.frag_centroid - target.frag_centroid[anchor]
                t_pred = t_pred - t_pred[anchor]
            placed = rotated + t_pred.index_select(0, target.node_frag)
            truth = target.world_pos() + t_gt.index_select(0, target.node_frag)
            kwargs = dict(pred_points=placed, gt_points=truth, point_frag=target.node_frag,
                          t_pred=t_pred, t_gt=t_gt)
            extra = {"num_matches": float(matches.src_idx.numel())}

        metrics = evaluate_scene(R_pred, R_gt, pa_threshold=args.pa_threshold,
                                 symmetry_axis=args.symmetry_axis, keep=keep,
                                 R_absolute=R_own.to(R_pred.dtype), **kwargs)
        metrics.update(extra)
        metrics["_category"] = (batch.get("categories") or ["unknown"])[0] or "unknown"
        metrics["_scene"] = f"{batch['scene_dirs'][0]}/{(batch.get('modes') or [''])[0]}"
        per_scene.append(metrics)
        bar.update(1)
    bar.close()

    def public(rows):
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in rows]

    summary = aggregate(public(per_scene))
    write("\n=== results ===")
    for key in sorted(summary):
        write(f"  {key:<24} {summary[key]:.5f}")
    write("  (every figure with each scene's largest fragment set to its true pose and the "
          "other\n   fragments scored, except absolute_geodesic_deg: every fragment in its "
          "object's stored frame)")

    write("\n=== reference levels ===")
    write(f"  chance, geodesic            {CHANCE_GEODESIC_DEG:.2f} deg")
    write(f"  chance, Euler RMSE          {CHANCE_EULER_RMSE_DEG:.2f} deg")
    write(f"  axis-only landmark, geo.    {AXIS_ONLY_GEODESIC_DEG:.2f} deg   "
          f"(symmetry axis right, azimuth about it random)")
    write(f"  axis-only landmark, Euler   {AXIS_ONLY_EULER_RMSE_DEG:.2f} deg")
    write("  The landmark is NOT a floor: a fragment's fracture boundary is unique even on a")
    write("  surface of revolution, and a measured run went well below it.")
    if "geodesic_deg" in summary:
        gained = CHANCE_GEODESIC_DEG - summary["geodesic_deg"]
        side = "below" if summary["geodesic_deg"] < AXIS_ONLY_GEODESIC_DEG else "above"
        write(f"\n  {gained:.2f} deg better than chance; {side} the axis-only landmark "
              f"by {abs(summary['geodesic_deg'] - AXIS_ONLY_GEODESIC_DEG):.2f} deg")
    if "tilt_deg" in summary:
        write(f"\n  tilt (off the {args.symmetry_axis}-axis)  {summary['tilt_deg']:.2f} deg   "
              f"-> ~90 means the axis itself is not learned; ~0 means it is")
        write(f"  twist (about the axis)   {summary['twist_deg']:.2f} deg   "
              f"-> ~90 with a small tilt is the axis-only pattern")

    groups = defaultdict(list)
    for m in per_scene:
        groups[m["_category"]].append(m)
    by_category = {cat: aggregate(public(rows)) for cat, rows in groups.items()}
    if args.by_category or len(groups) > 1:
        write("\n=== by category ===")
        write(f"  {'category':<22}{'n':>5}{'geodesic':>11}{'tilt':>9}{'twist':>9}")
        for cat in sorted(by_category, key=lambda c: by_category[c].get("geodesic_deg", 1e9)):
            agg = by_category[cat]
            write(f"  {cat:<22}{len(groups[cat]):>5}{agg.get('geodesic_deg', float('nan')):>11.2f}"
                  f"{agg.get('tilt_deg', float('nan')):>9.2f}{agg.get('twist_deg', float('nan')):>9.2f}")
        values = [a["geodesic_deg"] for a in by_category.values() if "geodesic_deg" in a]
        if values:
            write(f"  {'category mean':<22}{len(values):>5}{sum(values) / len(values):>11.2f}"
                  f"   (every category counts once)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "by_category": by_category,
                       "per_scene": per_scene, "checkpoint": args.checkpoint,
                       "split": args.split, "min_fragments": cfg.min_fragments,
                       "max_fragments": cfg.max_fragments}, fh, indent=2)
        write(f"\nwrote {args.out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a VN-GAT checkpoint.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--root_dir", type=str, default=None,
                   help="Override the data location stored in the checkpoint.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="'val' is what published Breaking Bad tables use: the official "
                        "release has no test split.")
    p.add_argument("--num_scenes", type=int, default=0,
                   help="0 = one break pattern per object in the split.")
    p.add_argument("--max_fragments", type=int, default=None,
                   help="Score only break patterns of up to this many pieces: 20 = the "
                        "benchmark's 2-20, 0 = every pattern. Default: the checkpoint's own "
                        "setting.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pa_threshold", type=float, default=0.01)
    p.add_argument("--max_match_points", type=int, default=4096)
    p.add_argument("--solve_translation", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--collision", action="store_true")
    p.add_argument("--symmetry_axis", type=str, default="z", choices=["x", "y", "z"],
                   help="Canonical up-axis of the meshes, for the tilt/twist split. "
                        "Try all three if unsure -- only the right one shows the signature.")
    p.add_argument("--by_category", action="store_true",
                   help="Always print the per-category table (it prints anyway when the "
                        "split spans more than one category).")
    p.add_argument("--out", type=str, default="")
    return p


def main(argv=None):
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
