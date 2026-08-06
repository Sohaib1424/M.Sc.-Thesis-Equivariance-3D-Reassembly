#!/usr/bin/env python
"""
Plot the loss history written by training (checkpoint_dir/loss_history.json).

    python scripts/plot_history.py --checkpoint-dir checkpoints --out curves.png
    python scripts/plot_history.py --checkpoint-dir checkpoints --compare checkpoints/frac
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
# Headless environments (Kaggle, Colab, most servers) have no display, and
# matplotlib's default backend errors out looking for one. Saving a PNG is this
# script's actual job, so Agg is right regardless of --show.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ["total", "rot", "rot_deg", "pos", "node", "mid", "face", "embv", "embe"]
TITLES = {
    "total": "total", "rot": "rotation (training term)",
    "rot_deg": "rotation error (degrees, geodesic)", "pos": "node position",
    "node": "node normal", "mid": "edge midpoint", "face": "face normals",
    "embv": "vertex interface embedding", "embe": "edge interface embedding",
}


def load(path: Path):
    with open(path / "loss_history.json") as f:
        return json.load(f)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--compare", type=str, nargs="*", default=[],
                   help="Additional checkpoint dirs to overlay (e.g. a frac-input run)")
    p.add_argument("--out", type=str, default="loss_curves.png")
    p.add_argument("--log-y", action="store_true")
    args = p.parse_args(argv)

    runs = [(Path(args.checkpoint_dir).name or "run", load(Path(args.checkpoint_dir)))]
    for extra in args.compare:
        runs.append((Path(extra).name, load(Path(extra))))

    present = [k for k in PANELS if any(k in h["train"] or k in h["val"] for _n, h in runs)]
    cols = 3
    rows = (len(present) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)

    for i, key in enumerate(present):
        ax = axes[i // cols][i % cols]
        for run_name, hist in runs:
            for phase, style in (("train", "-"), ("val", "--")):
                series = hist.get(phase, {}).get(key)
                if not series:
                    continue
                label = f"{run_name} {phase}" if len(runs) > 1 else phase
                ax.plot(range(len(series)), series, style, label=label, linewidth=1.4)
        ax.set_title(TITLES.get(key, key))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if args.log_y and key != "rot_deg":
            ax.set_yscale("log")
        ax.legend(fontsize=7)

    for j in range(len(present), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
