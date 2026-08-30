#!/usr/bin/env python3
"""
How many attention connections does the fracture-surface graph actually create,
and how does that compare with GARF?

The design question this answers
--------------------------------
The proposed model connects every fracture-surface vertex of every fragment to
every fracture-surface vertex of every *other* fragment. That is a complete
bipartite graph between fragments, so for a scene whose fragments carry
``f_1 .. f_N`` fracture vertices the cross-fragment edge count is

    E_cross = 1/2 * [ (sum_i f_i)^2 - sum_i f_i^2 ]

which is **quadratic in the total number of fracture vertices**. Mesh
resolution therefore drives cost directly: a mesh twice as fine costs four
times as much.

GARF does not have this property. It samples a fixed M = 5000 points per object
(Poisson-disk, GARF appendix B.1) *regardless of mesh resolution*, so its
attention cost per scene is a constant. That is the comparison that matters:
not "who is bigger today" but "what happens as meshes get finer".

What GARF actually costs
------------------------
Two different modules, often conflated:

``PTv3 encoder`` (the pretrained fracture-segmentation backbone) does **not**
use all-pairs attention. It serialises points along a space-filling curve and
attends within fixed patches -- GARF Table II lists Encoder Patch Size
``[1024, 1024, 1024, 1024]``. Cost is therefore O(M * patch), not O(M^2). Only
the first (largest) stage is counted here; PTv3 downsamples between stages, so
later stages are cheaper and this is an upper bound on the per-stage cost.

``Flow-matching transformer`` (the denoiser) *does* apply global attention
across all M points -- GARF Sec. 3.2, "the global attention layer ... with
l = M". That is genuinely M(M-1)/2 undirected pairs per layer, over 6 layers.
FlashAttention reduces the *memory* of this, not the number of pairs.

Caveats
-------
Counts are undirected pairs, the unit in which "12.5 million connections" is
usually quoted. Multiply by 2 for directed message counts. This measures graph
size, not FLOPs or wall-clock: a vector-neuron channel is not the same unit of
work as a 512-dim transformer channel, so treat the ratio as a scaling
argument, not a speed prediction.

Usage::

    python -m scripts.analyze_graph_cost --stats out/stats.csv
    python -m scripts.analyze_graph_cost --stats out/stats.csv --plot docs/graph_cost.png
    python -m scripts.analyze_graph_cost --stats out/stats.csv --cap 512
    python -m scripts.analyze_graph_cost --stats out/stats.csv --scene-budget 256

``--scene-budget`` models one FIXED token budget per scene, split across
fragments -- the configuration that makes the inter-fragment pair count
independent of mesh resolution. ``--cap`` models a per-fragment cap instead,
which bounds the worst case but leaves the cost scaling with fragment count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.extract_fracture_surfaces import read_stats_csv  # noqa: E402
from scripts.plot_fracture_stats import THEMES  # noqa: E402

SCENE_KEYS = ["subset", "category", "scene", "mode"]


def garf_patch_pairs(points: int, patch: int) -> int:
    """Undirected pairs for one serialised patch-attention layer over `points`."""
    full, remainder = divmod(points, patch)
    total = full * patch * (patch - 1) // 2
    if remainder > 1:
        total += remainder * (remainder - 1) // 2
    return total


def garf_global_pairs(points: int) -> int:
    return points * (points - 1) // 2


def scene_costs(frame, cap: int | None = None, scene_budget: int | None = None):
    """Per (scene, mode): intra-fragment mesh edges and cross-fragment pairs."""
    keys = [k for k in SCENE_KEYS if k in frame.columns]
    if not keys:
        raise SystemExit("stats CSV has none of the scene-identifying columns")

    work = frame.copy()
    frac = work["frac_vertices"].to_numpy(dtype=np.float64)
    if cap:
        # Sub-sampling each fragment's fracture surface to at most `cap`
        # vertices, the obvious mitigation. Applied before the sums, because
        # capping after aggregation would be a different (and wrong) quantity.
        frac = np.minimum(frac, float(cap))
    work["_f"] = frac
    work["_f2"] = frac ** 2

    grouped = work.groupby(keys, sort=False).agg(
        fragments=("piece", "count"),
        vertices=("vertices", "sum"),
        intra_edges=("edges", "sum"),
        frac_sum=("_f", "sum"),
        frac_sq_sum=("_f2", "sum"),
    ).reset_index()

    if scene_budget:
        # A fixed per-scene budget split proportionally: every scene contributes
        # the same number of tokens, so the pair count stops depending on how
        # finely the meshes happen to be tessellated.
        from reassembly.mesh.patches import allocate_budget

        sums, squares = [], []
        for _, chunk in work.groupby(keys, sort=False):
            counts = allocate_budget(chunk["_f"].to_numpy(), scene_budget)
            sums.append(float(counts.sum()))
            squares.append(float((counts.astype(np.float64) ** 2).sum()))
        grouped["frac_sum"] = sums
        grouped["frac_sq_sum"] = squares

    grouped["cross_pairs"] = 0.5 * (grouped["frac_sum"] ** 2 - grouped["frac_sq_sum"])
    grouped["total_pairs"] = grouped["intra_edges"] + grouped["cross_pairs"]
    return grouped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare this design's graph size against GARF's attention cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stats", default="out/stats.csv")
    parser.add_argument("--garf-points", type=int, default=5000,
                        help="M, points sampled per object (GARF appendix B.1)")
    parser.add_argument("--garf-patch", type=int, default=1024,
                        help="PTv3 encoder patch size (GARF Table II)")
    parser.add_argument("--garf-fm-layers", type=int, default=6,
                        help="flow-matching transformer encoder layers")
    parser.add_argument("--cross-layers", type=int, default=2,
                        help="layers of THIS design that run cross-fragment "
                             "attention (the rest are intra-fragment only)")
    parser.add_argument("--intra-layers", type=int, default=4,
                        help="layers of THIS design that run intra-fragment "
                             "message passing over the mesh graph")
    parser.add_argument("--scene-budget", type=int, default=None,
                        help="model a FIXED per-scene token budget, split across "
                             "fragments in proportion to their fracture-vertex "
                             "count (GARF-style). Makes the inter-fragment count "
                             "a constant, independent of mesh resolution")
    parser.add_argument("--cap", type=int, default=None,
                        help="sub-sample each fragment's fracture surface to at "
                             "most this many vertices before counting")
    parser.add_argument("--plot", default=None, help="write a distribution figure here")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--out-csv", default=None, help="per-scene costs")
    args = parser.parse_args(argv)

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        parser.error("pandas is required: pip install pandas")

    stats = Path(args.stats)
    if not stats.is_file():
        parser.error(f"{stats} not found -- run scripts.extract_fracture_surfaces first")

    import pandas as pd
    frame = read_stats_csv(stats)
    for needed in ("frac_vertices", "vertices", "edges"):
        if needed not in frame.columns:
            parser.error(f"stats CSV lacks '{needed}'")

    grouped = scene_costs(frame, args.cap, args.scene_budget)
    ptv3 = garf_patch_pairs(args.garf_points, args.garf_patch)
    fm_one = garf_global_pairs(args.garf_points)
    fm_all = fm_one * args.garf_fm_layers

    cross = grouped["cross_pairs"].to_numpy()
    total = grouped["total_pairs"].to_numpy()

    def q(a, p):
        return float(np.percentile(a, p))

    print("=" * 78)
    print("GRAPH SIZE PER SCENE  (undirected pairs; x2 for directed messages)")
    print("=" * 78)
    print(f"  scenes x modes analysed      {len(grouped):,}")
    print(f"  fragments per scene          median {grouped['fragments'].median():.0f}"
          f"   max {grouped['fragments'].max():.0f}")
    print(f"  mesh vertices per scene      median {grouped['vertices'].median():,.0f}"
          f"   max {grouped['vertices'].max():,.0f}")
    if args.cap:
        print(f"  fracture vertices capped at  {args.cap:,} per fragment")
    if args.scene_budget:
        print(f"  FIXED scene token budget     {args.scene_budget:,} per scene")
    print()
    print("  this design")
    print(f"    intra-fragment mesh edges  median {grouped['intra_edges'].median():15,.0f}")
    print(f"    cross-fragment attention   median {np.median(cross):15,.0f}"
          f"   p90 {q(cross, 90):15,.0f}")
    print(f"    TOTAL                      median {np.median(total):15,.0f}"
          f"   p90 {q(total, 90):15,.0f}")
    print()
    print(f"  GARF (fixed at M={args.garf_points:,} points, independent of mesh resolution)")
    print(f"    PTv3 encoder, stage 1      {ptv3:15,.0f}   (patch {args.garf_patch:,}, NOT all-pairs)")
    print(f"    FM global attn, per layer  {fm_one:15,.0f}")
    print(f"    FM global attn, {args.garf_fm_layers} layers  {fm_all:15,.0f}")
    print()
    # Like-for-like: total connections summed over the whole stack on BOTH
    # sides. Comparing one of this design's layers against GARF's six-layer
    # total flatters this design by 6x and is the easy mistake to make here.
    stack = (grouped["intra_edges"].to_numpy() * args.intra_layers
             + cross * args.cross_layers)
    print(f"  WHOLE-STACK COMPARISON  ({args.intra_layers} intra + "
          f"{args.cross_layers} cross layers  vs  GARF's {args.garf_fm_layers} FM layers)")
    print(f"    this design                median {np.median(stack):15,.0f}"
          f"   p90 {q(stack, 90):15,.0f}")
    print(f"    GARF FM stack              {fm_all:15,.0f}")
    ratio = stack / fm_all
    print(f"    ratio                      median {np.median(ratio):8.2f}x"
          f"     p90 {q(ratio, 90):8.2f}x     max {ratio.max():8.2f}x")
    share = float((ratio > 1.0).mean() * 100)
    print(f"    {share:.1f}% of scenes cost MORE than GARF's full FM stack")
    print("=" * 78)

    # What cap would buy parity? Solved by search rather than algebra because
    # the cap interacts with the per-fragment distribution, not just its sum.
    if not args.cap and not args.scene_budget:
        print("\n  parity search: largest per-fragment fracture-vertex cap whose")
        print("  whole stack stays at or below GARF's FM stack")
        best = None
        for candidate in (4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 96, 64):
            capped = scene_costs(frame, candidate)
            med = float(np.median(
                capped["intra_edges"].to_numpy() * args.intra_layers
                + capped["cross_pairs"].to_numpy() * args.cross_layers))
            if med <= fm_all:
                best = (candidate, med)
                break
        if best:
            print(f"    cap = {best[0]:,} vertices/fragment -> median {best[1]:,.0f}"
                  f"  ({best[1] / fm_all:.2f}x GARF)")
        else:
            print("    no candidate cap reached parity; sub-sample harder or")
            print("    attend over fracture patches rather than raw vertices")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(args.out_csv, index=False)
        print(f"\nper-scene costs -> {args.out_csv}")

    if args.plot:
        _plot(total, ptv3, fm_one, fm_all, Path(args.plot), args.theme,
              args.garf_points, args.garf_fm_layers)
        print(f"figure -> {args.plot}")
    return 0


def _plot(total, ptv3, fm_one, fm_all, out: Path, theme_name, points, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme_name]
    positive = total[total > 0]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    bins = np.logspace(np.log10(max(positive.min(), 1.0)),
                       np.log10(positive.max() * 1.05), 60)
    ax.hist(positive, bins=bins, color=t["kept"], zorder=3,
            label="this design (per scene)")
    ax.set_xscale("log")

    for value, label, style in ((ptv3, f"GARF PTv3 stage 1", ":"),
                                (fm_one, "GARF FM global attn, 1 layer", "--"),
                                (fm_all, f"GARF FM global attn, {layers} layers", "-")):
        ax.axvline(value, color=t["dropped"], linewidth=2, linestyle=style, zorder=4)
        ax.annotate(label, xy=(value, ax.get_ylim()[1] * 0.97),
                    xytext=(5, 0), textcoords="offset points", rotation=90,
                    va="top", ha="left", color=t["dropped"], fontsize=8.5,
                    fontweight="bold")

    ax.set_xlabel("attention connections per scene (undirected pairs, log scale)",
                  color=t["secondary"], fontsize=10)
    ax.set_ylabel("scene x mode", color=t["secondary"], fontsize=10)
    ax.set_title("Graph size: fracture-surface cross-attention vs GARF",
                 color=t["text"], fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5,
              labelcolor=t["secondary"])
    fig.text(0.008, 0.015,
             f"GARF lines are constant: it samples M={points:,} points per object "
             f"regardless of mesh resolution.", color=t["muted"], fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.20)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
