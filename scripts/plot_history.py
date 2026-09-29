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
    ("anchor_deg", "Rotation, largest fragment as anchor (deg)"),
    ("absolute_deg", "Rotation, each object's stored frame (deg)"),
    ("rot_deg", "Rotation loss term (degrees)"),
    ("rot", "Rotation (geodesic, rad)"),
    ("pos", "Node position"),
    ("node", "Node normal"),
    ("face", "Adjacent face normals"),
    ("emb_v", "Vertex embedding consistency"),
    ("emb_e", "Edge embedding consistency"),
    ("tilt", "Tilt off the symmetry axis (deg)"),
    ("twist", "Twist about the symmetry axis (deg)"),
    ("head_cos", "Rotation-head |cos| (1 = ill-conditioned)"),
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

    cols = 4
    rows = -(-(len(PANELS) + 1 + 4) // cols)     # the panels, the category plot, four meta
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

    idx = len(PANELS)
    by_category = history.data.get("val_category", {})
    if by_category and idx < len(axes):
        # Validation only: training is drawn under `balance`, validation never is.
        ax = axes[idx]
        for name, values in sorted(by_category.items()):
            ax.plot(range(len(values)), values, linewidth=1.4, marker="o", markersize=3, label=name)
        ax.set_title("Val rotation error by category (deg)", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        idx += 1

    meta = history.data.get("meta", {})
    for key, title in (("lr", "Learning rate"),
                       ("grad_norm", "Gradient norm before clipping"),
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
        if key in ("lr", "grad_norm"):
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
