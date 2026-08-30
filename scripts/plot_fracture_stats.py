#!/usr/bin/env python3
"""
Figure for the README: how much of each fragment survives fracture-surface
extraction.

Reads the per-fragment CSV written by ``scripts.extract_fracture_surfaces`` and
renders two panels:

**Left -- what fraction of the dataset survives.** Stacked bars for vertices,
faces and edges, computed as ``sum(kept) / sum(original)`` over every fragment
at once. Size-weighted, so this is the fraction of *all geometry* that the
extractor keeps -- the number that governs batch memory and load time.

**Right -- how much fragments differ.** A histogram of per-fragment kept
percentage. The left panel is one number; this panel is the reason that number
is not the whole story. A tight distribution means extraction behaves the same
everywhere; a wide or bimodal one means fragment size or category decides how
much survives, and the aggregate hides it.

Both panels are rendered from the same CSV, so they cannot disagree.

Usage::

    python -m scripts.plot_fracture_stats --stats out/stats.csv
    python -m scripts.plot_fracture_stats --stats out/stats.csv --both-themes

``--both-themes`` writes a light and a dark variant, for the GitHub
``<picture>`` element that follows the reader's theme::

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/fracture_reduction_dark.png">
      <img alt="Fracture-surface extraction" src="docs/fracture_reduction.png">
    </picture>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.extract_fracture_surfaces import read_stats_csv  # noqa: E402

# Palette slots 1 and 2 of the project's categorical theme. Validated for
# colour-vision deficiency and for contrast against both chart surfaces
# (worst-pair CVD dE 24.7 light / 26.8 dark against an 8.0 target; normal-vision
# 33.6 / 31.8 against a 15.0 floor). Do not substitute by eye -- a "nicer" pair
# is very often one a deuteranope cannot separate.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "kept": "#2a78d6", "dropped": "#eb6834",
        "text": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19", "kept": "#3987e5", "dropped": "#d95926",
        "text": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
    },
}

QUANTITIES = [("vertices", "frac_vertices"),
              ("faces", "frac_faces"),
              ("edges", "frac_edges")]


def aggregate(frame):
    """
    Size-weighted kept fraction per quantity.

    Deliberately NOT the mean of ``kept_*_pct``: that averages percentages, so a
    200-vertex chip counts as much as a 40,000-vertex body. For "how much of the
    dataset survives", every vertex must count once, which means summing the
    counts and dividing -- not averaging the ratios. The two differ whenever
    fragment sizes vary, and on real data they differ a lot.
    """
    rows = []
    for name, frac in QUANTITIES:
        if name not in frame.columns or frac not in frame.columns:
            continue
        pair = frame[[name, frac]].dropna()
        total = float(pair[name].sum())
        if total <= 0:
            continue
        kept = float(pair[frac].sum())
        rows.append({"quantity": name, "total": total, "kept": kept,
                     "kept_pct": 100.0 * kept / total,
                     "dropped_pct": 100.0 * (1.0 - kept / total),
                     "n": int(len(pair))})
    return rows


def render(frame, out_path: Path, theme_name: str, hist_column: str, dpi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = THEMES[theme_name]
    rows = aggregate(frame)
    if not rows:
        raise SystemExit("no usable columns in the stats CSV")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.5, 4.0), dpi=dpi,
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.28})
    fig.patch.set_facecolor(t["surface"])

    # ---- left: stacked composition -------------------------------------
    ax_left.set_facecolor(t["surface"])
    labels = [r["quantity"] for r in rows]
    y = range(len(rows))
    # A 2px surface-coloured gap between the two segments, so the boundary
    # reads as a boundary rather than a colour change.
    for i, r in enumerate(rows):
        ax_left.barh(i, r["kept_pct"], color=t["kept"], height=0.55, zorder=3)
        ax_left.barh(i, r["dropped_pct"], left=r["kept_pct"] + 0.4,
                     color=t["dropped"], height=0.55, zorder=3)
        ax_left.text(r["kept_pct"] / 2, i, f"{r['kept_pct']:.1f}%",
                     ha="center", va="center", color="#ffffff",
                     fontsize=11, fontweight="bold", zorder=4)
        drop_mid = r["kept_pct"] + 0.4 + r["dropped_pct"] / 2
        ax_left.text(drop_mid, i, f"{r['dropped_pct']:.1f}%", ha="center",
                     va="center", color="#ffffff", fontsize=11,
                     fontweight="bold", zorder=4)

    ax_left.set_yticks(list(y), labels, color=t["text"], fontsize=11)
    ax_left.set_xlim(0, 100.4)
    ax_left.set_xticks([0, 25, 50, 75, 100])
    ax_left.set_xticklabels(["0", "25", "50", "75", "100%"], color=t["muted"],
                            fontsize=9)
    ax_left.invert_yaxis()
    ax_left.xaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax_left.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_left.spines[side].set_visible(False)
    ax_left.spines["bottom"].set_color(t["axis"])
    ax_left.tick_params(length=0)
    ax_left.set_title("Share of the dataset kept as fracture surface",
                      color=t["text"], fontsize=12, fontweight="bold",
                      loc="left", pad=12)
    ax_left.legend(handles=[Patch(facecolor=t["kept"], label="fracture surface"),
                            Patch(facecolor=t["dropped"], label="dropped")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                   frameon=False, fontsize=10, labelcolor=t["secondary"])

    # ---- right: per-fragment distribution -------------------------------
    ax_right.set_facecolor(t["surface"])
    series = frame[hist_column].dropna() if hist_column in frame.columns else None
    if series is None or series.empty:
        ax_right.text(0.5, 0.5, f"no {hist_column} column", ha="center",
                      va="center", color=t["muted"], transform=ax_right.transAxes)
    else:
        ax_right.hist(series, bins=50, range=(0, 100), color=t["kept"], zorder=3)
        median = float(series.median())
        ax_right.axvline(median, color=t["dropped"], linewidth=2, zorder=4)
        # Flip the label to the left of the line when the median sits far
        # enough right that the text would run off the axes.
        right_side = median < 62
        ax_right.annotate(f"median {median:.1f}%",
                          xy=(median, ax_right.get_ylim()[1] * 0.94),
                          xytext=(6 if right_side else -6, 0),
                          textcoords="offset points",
                          ha="left" if right_side else "right",
                          color=t["dropped"], fontsize=10, fontweight="bold")
        ax_right.set_xlim(0, 100)
        ax_right.set_xlabel(f"{hist_column.replace('_', ' ')} per fragment",
                            color=t["secondary"], fontsize=10)
        ax_right.set_ylabel("fragments", color=t["secondary"], fontsize=10)

    ax_right.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax_right.set_axisbelow(True)
    for side in ("top", "right"):
        ax_right.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax_right.spines[side].set_color(t["axis"])
    ax_right.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax_right.set_title("Spread across fragments", color=t["text"], fontsize=12,
                       fontweight="bold", loc="left", pad=12)

    fig.text(0.008, 0.015,
             f"{len(frame):,} fragments  ·  "
             f"{frame['scene'].nunique() if 'scene' in frame else '?'} scenes",
             color=t["muted"], fontsize=9)
    # subplots_adjust rather than tight_layout: the legend is anchored outside
    # its axes, which tight_layout warns about and lays out wrongly.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.24, wspace=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the fracture-extraction figure for the README.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stats", default="out/stats.csv",
                        help="per-fragment CSV from scripts.extract_fracture_surfaces")
    parser.add_argument("--out", default="docs/fracture_reduction.png")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--both-themes", action="store_true",
                        help="also write a _dark variant beside --out")
    parser.add_argument("--hist-column", default="kept_faces_pct")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        parser.error("pandas is required: pip install pandas")

    stats = Path(args.stats)
    if not stats.is_file():
        parser.error(f"{stats} not found -- run scripts.extract_fracture_surfaces first")

    frame = read_stats_csv(stats)
    out = Path(args.out)

    targets = [(out, args.theme)]
    if args.both_themes:
        other = "dark" if args.theme == "light" else "light"
        targets.append((out.with_name(f"{out.stem}_{other}{out.suffix}"), other))

    rows = []
    for path, theme in targets:
        rows = render(frame, path, theme, args.hist_column, args.dpi)
        print(f"wrote {path}  ({theme})")

    print()
    for r in rows:
        print(f"  {r['quantity']:9s} kept {r['kept_pct']:5.1f}%   "
              f"dropped {r['dropped_pct']:5.1f}%   "
              f"({r['kept']:,.0f} / {r['total']:,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
