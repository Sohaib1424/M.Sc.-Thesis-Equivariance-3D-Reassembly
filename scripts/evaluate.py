#!/usr/bin/env python
"""
Evaluate a trained checkpoint with the GARF-comparable metric suite.

    # rotation only (no translation solver) -- fastest, answers "how good is stage 1"
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --rotation-only

    # full assembly: network rotations + classical translation solver
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --num-scenes 200

    # oracle ablation: TRUE rotations + the solver, to isolate solver quality
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --oracle-rotation

The three modes together are what make a thesis table interpretable. Reporting
only the middle one leaves "is the remaining error coming from the network or
from the solver?" unanswered, and that is the first question a reader has.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from tqdm.auto import tqdm

from reassembly.assembly.matching import match_scene
from reassembly.assembly.translation import solve_translations
from reassembly.config import Config
from reassembly.data.dataset import BreakingBadDataset
from reassembly.evaluation.metrics import aggregate, evaluate_assembly
from reassembly.models.vn_gat_model import VNGATModel
from reassembly.training.bridge import build_model_inputs


def split_per_fragment(tensor: torch.Tensor, fragment_id: torch.Tensor, n: int):
    """Split a per-node tensor into a list of per-fragment numpy arrays."""
    out = []
    fid = fragment_id.cpu().numpy()
    arr = tensor.detach().cpu().numpy()
    for f in range(n):
        out.append(arr[fid == f])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Defaults to the config saved inside the checkpoint.")
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rotation-only", action="store_true",
                        help="Skip the translation solver; use ground-truth translations.")
    parser.add_argument("--oracle-rotation", action="store_true",
                        help="Use ground-truth rotations; isolates translation-solver quality.")
    parser.add_argument("--collision-weight", type=float, default=0.0)
    parser.add_argument("--pa-threshold", type=float, default=0.01)
    parser.add_argument("--match-report", action="store_true",
                        help="Print embedding-distance statistics for calibrating the matcher.")
    parser.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.config:
        cfg = Config.from_yaml(args.config)
    elif "config" in state:
        cfg = Config.from_dict(state["config"])
    else:
        cfg = Config()
    if args.root_dir:
        cfg.data.root_dir = args.root_dir

    model = VNGATModel(
        hidden_channels=cfg.model.hidden_channels, num_layers=cfg.model.num_layers,
        num_vn_slots=cfg.model.num_vn_slots, heads=cfg.model.heads,
        head_dim=cfg.model.head_dim, embed_dim=cfg.model.embed_dim,
        norm=cfg.model.norm, angular=cfg.model.angular,
    ).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    ds = BreakingBadDataset(
        root_dir=cfg.data.root_dir, split=args.split,
        val_frac=cfg.data.val_frac, test_frac=cfg.data.test_frac,
        split_seed=cfg.data.split_seed,
        input_source=cfg.data.input_source,
        decimate_to=cfg.data.decimate_to, max_vertices=cfg.data.max_vertices,
        correspondence_tol=cfg.data.correspondence_tol,
        return_meshes=False, seed=cfg.data.seed,
    )

    all_metrics, residuals, component_counts, match_distances = [], [], [], []
    skipped = 0

    for i in tqdm(range(args.num_scenes), desc=f"eval[{args.split}]", unit="scene"):
        sample = ds[i]
        if sample is None:
            skipped += 1
            continue

        clean = sample["graph"]
        diffused = sample["diffused_graph"]
        n_frag = int(clean.num_fragments)

        # Attributes the collate step normally adds; single-scene here, so
        # fragment_id is already scene-local and every fragment is scene 0.
        for g in (clean, diffused, sample.get("frac_graph"), sample.get("diff_frac_graph")):
            if g is not None:
                g.fragment_scene_id = torch.zeros(int(g.num_fragments), dtype=torch.long)

        input_graph = (sample["diff_frac_graph"] if cfg.data.input_source == "frac"
                       else diffused).to(device)
        with torch.no_grad():
            out = model(**build_model_inputs(input_graph))

        R_gt = sample["t_matrices"][:, :3, :3].transpose(-1, -2).numpy()
        R_pred = R_gt if args.oracle_rotation else out["R_pred"].cpu().numpy()

        # Fragment-local (centralized) clean geometry, and the true placement.
        verts = split_per_fragment(clean.x[:, 0:3], clean.fragment_id, n_frag)
        # Ground-truth translation = each fragment's CLEAN centroid: putting the
        # centralized clean fragment back at its centroid reassembles the object
        # exactly. Feature construction carries this through precisely so that
        # translation metrics are computable from the graph alone.
        t_gt = clean.fragment_centroid.cpu().numpy()

        # Fix the gauge: the solver determines translations only up to one
        # global offset per connected component, so both prediction and truth
        # are re-centred on their own scene centroid before comparison.
        t_gt = t_gt - t_gt.mean(axis=0, keepdims=True)

        if args.rotation_only:
            t_pred = t_gt
        else:
            emb_v = out["vertex_embedding"].cpu().numpy()
            fid = input_graph.fragment_id.cpu().numpy()
            n_in = int(input_graph.num_fragments)
            embeddings = [emb_v[fid == f] for f in range(n_in)]
            in_pts = split_per_fragment(input_graph.x[:, 0:3], input_graph.fragment_id, n_in)
            in_nrm = split_per_fragment(input_graph.x[:, 3:6], input_graph.fragment_id, n_in)

            oriented_pts = [p @ R_pred[f].T for f, p in enumerate(in_pts)]
            oriented_nrm = [n @ R_pred[f].T for f, n in enumerate(in_nrm)]

            matches = match_scene(embeddings, oriented_pts, oriented_nrm)
            if args.match_report and len(matches):
                match_distances.append(matches.score)

            res = solve_translations(matches, n_frag)
            residuals.append(res.residual_rms)
            component_counts.append(res.num_components)

            t_pred = res.translations - res.translations.mean(axis=0, keepdims=True)

        all_metrics.append(
            evaluate_assembly(R_pred, R_gt, t_pred, t_gt, verts,
                              pa_threshold=args.pa_threshold)
        )

    summary = aggregate(all_metrics)
    summary["skipped_scenes"] = skipped
    summary["mode"] = ("oracle_rotation" if args.oracle_rotation
                       else "rotation_only" if args.rotation_only else "full_assembly")
    summary["input_source"] = cfg.data.input_source
    if residuals:
        summary["solver_residual_rms_mean"] = float(np.nanmean(residuals))
        summary["fully_constrained_fraction"] = float(np.mean(np.array(component_counts) == 1))

    print("\n" + "=" * 74)
    print(f"RESULTS  ({summary['mode']}, split={args.split}, "
          f"input_source={cfg.data.input_source})")
    print("=" * 74)
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print("\nNote: RMSE_R_euler_deg is the number comparable to published "
          "Breaking Bad / GARF tables.\n      RMSE_R_geodesic_deg is a different "
          "convention -- do not mix them in one table.")

    if args.match_report and match_distances:
        d = np.concatenate(match_distances)
        qs = np.quantile(d, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95])
        print("\nembedding-distance quantiles over accepted matches "
              "(use these to set --max-distance):")
        for q, val in zip([1, 5, 10, 25, 50, 75, 95], qs):
            print(f"    p{q:<3d} = {val:.5f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")

    return summary


if __name__ == "__main__":
    main()
