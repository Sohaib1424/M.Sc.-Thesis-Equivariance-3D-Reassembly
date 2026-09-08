#!/usr/bin/env python
"""
Plot the training/validation loss history.

    python -m scripts.plot_history --checkpoint_dir checkpoints --out curves.png

Reads `history.json`, which the trainer flushes after every epoch, so this
works on a run that is still going (or one that was killed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless by default: Kaggle/Colab have no display
import matplotlib.pyplot as plt  # noqa: E402

from vngat.training.history import History  # noqa: E402

PANELS = [
    ("total", "Total (weighted sum)"),
    ("rot_deg", "Rotation error (degrees)"),
    ("rot", "Rotation (geodesic, rad)"),
    ("pos", "Node position"),
    ("node", "Node normal"),
    ("mid", "Edge midpoint"),
    ("face", "Adjacent face normals"),
    ("emb_v", "Vertex embedding consistency"),
    ("emb_e", "Edge embedding consistency"),
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--history", type=str, default="")
    p.add_argument("--out", type=str, default="loss_curves.png")
    p.add_argument("--logy", action="store_true")
    args = p.parse_args(argv)

    path = Path(args.history) if args.history else Path(args.checkpoint_dir) / "history.json"
    history = History.load(path)
    if history.num_epochs == 0:
        raise SystemExit(f"No epochs recorded in {path}")

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, PANELS):
        for phase, style in (("train", "-"), ("val", "--")):
            values = history.data.get(phase, {}).get(key, [])
            if values:
                ax.plot(range(len(values)), values, style, label=phase, linewidth=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if args.logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    meta = history.data.get("meta", {})
    idx = len(PANELS)
    for key, title in (("lr", "Learning rate"),
                       ("train_data_seconds", "Data vs compute (s/epoch)"),
                       ("max_nodes", "Largest scene (nodes)")):
        if idx >= len(axes) or key not in meta:
            continue
        ax = axes[idx]
        ax.plot(meta[key], linewidth=1.6, label=key)
        if key == "train_data_seconds" and "train_compute_seconds" in meta:
            ax.plot(meta["train_compute_seconds"], linewidth=1.6, label="compute")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if key == "lr":
            ax.set_yscale("log")
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    fig.suptitle(f"VN-GAT training history ({history.num_epochs} epochs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
