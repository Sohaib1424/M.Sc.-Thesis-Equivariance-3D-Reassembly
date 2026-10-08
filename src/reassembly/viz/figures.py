"""
Figures from training histories and evaluation results.

Drawn from :mod:`reassembly.viz.results`. matplotlib is imported inside the
functions, so importing this module needs neither it nor a display; the
caller picks the backend (``scripts/make_figures.py`` uses ``Agg``). Every
function takes the loaded results and a :class:`FigureWriter`, which saves
each figure in every format asked for and keeps the list the index is
written from.

Style
-----
The project's validated categorical palette -- the slots
``scripts/plot_fracture_stats.py`` uses -- in a fixed order: the first run or
dataset is always blue, the second orange, and a colour follows its run or
dataset through every figure. Thin lines, hairline solid grids, text in ink
rather than in a series colour, a white background (these go on paper). At
most eight runs or datasets per figure: past that no further colour is safe.

Reading the training curves
---------------------------
Since v7 a run trains on the embedding term alone -- the total is it -- and
rotation, position, normal and face are scores of the rotations fitted from
its matches, on validation every epoch (on training too with
``--score_train``). ``best.pt`` is the epoch with the highest validation
acc@10 and is marked on every single-run curve. A run from before, with the
rotation head, logged the five terms unweighted, their weighted sum as the
total, and kept the epoch with the lowest validation error as ``best.pt``;
its figures say so.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .results import (ACCURACY_THRESHOLDS, CHANCE_GEODESIC_DEG, Evaluation, History,
                      assembly_by_type, by_piece_count, distribution, type_order)

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948")
"""Categorical slots, in order. Validated for colour-vision deficiency on the
adjacent pairs (worst CVD dE 9.1 against an 8.0 target) -- do not reorder or
substitute by eye."""
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
"""One hue, light to dark: low values recede, high values stand out."""

CHANCE_ROTATION_RAD = math.pi / 2 + 2 / math.pi
LOG_FLOOR_DEG = 0.01
"""Errors below this are drawn at it on a log axis (an exact match is ~1e-5 deg)."""
SOURCE_NAMES = {"matched": "rotations from the embedding matches",
                "network": "the rotation head (network)"}
MAX_SERIES = len(SERIES)


class FigureWriter:
    """Saves figures and tables under one folder and records what it wrote."""

    def __init__(self, root, formats: Sequence[str] = ("png", "pdf"), dpi: int = 200):
        self.root = Path(root)
        self.formats = tuple(f.lstrip(".").lower() for f in formats)
        self.dpi = dpi
        self.written: List[Tuple[str, str]] = []

    def save(self, fig, relative: str, description: str) -> None:
        import matplotlib.pyplot as plt

        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        for fmt in self.formats:
            fig.savefig(target.with_suffix("." + fmt), dpi=self.dpi, bbox_inches="tight",
                        facecolor=SURFACE)
        plt.close(fig)
        self.written.append((", ".join(f"{relative}.{fmt}" for fmt in self.formats),
                             description))

    def note(self, relative: str, description: str) -> None:
        """A non-figure file (a table) for the index."""
        self.written.append((relative, description))


# --------------------------------------------------------------------------
# Training curves, one run
# --------------------------------------------------------------------------

TERM_PANELS = (
    ("rotation", "Rotation (geodesic angle, rad)", CHANCE_ROTATION_RAD),
    ("position", "Position (mean vertex distance)", None),
    ("normal", "Vertex normals (1 - cos)", 1.0),
    ("face", "Face normals (2 x (1 - cos))", 2.0),
)
SPLITS = (("train", "training", SERIES[0]), ("val", "validation", SERIES[1]))


def plot_total_loss(history: History, writer: FigureWriter, smooth: int = 1) -> bool:
    if not (history.has("train_total") or history.has("val_total")):
        return False
    plt = _pyplot()
    with _style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _frame(fig, f"Total loss  |  {history.label}",
               "Weighted sum of the five terms; compare runs on it only when their "
               "--w_* are the same." if history.has_head else
               "The loss the step descends: the embedding term alone. The four geometric "
               "terms\nare scores of the matched rotations and never part of it.")
        for split, name, colour in SPLITS:
            _curve(ax, history, f"{split}_total", name, colour, smooth)
        _mark_best(ax, history)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        _top_legend(ax, ncol=2)
        writer.save(fig, f"training/{_slug(history.label)}/loss_total",
                    f"{history.label}: total loss, training and validation.")
    return True


def plot_loss_terms(history: History, writer: FigureWriter, smooth: int = 1) -> bool:
    """The four rotation-dependent terms, one panel each, so none hides another."""
    panels = [p for p in TERM_PANELS
              if history.has(f"train_{p[0]}") or history.has(f"val_{p[0]}")]
    if not panels:
        return False
    plt = _pyplot()
    with _style():
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True)
        _frame(fig, (f"Loss terms  |  {history.label}" if history.has_head else
                     f"Geometric scores  |  {history.label}"),
               ("Unweighted. Each term compares the predicted rotation with the truth: "
                "angle, vertex positions, vertex normals, face normals."
                if history.has_head else
                "Never trained on. Each compares the rotations fitted from the embedding "
                "matches with the truth:\nangle, vertex positions, vertex normals, face "
                "normals (anchor protocol).")
               + _best_note(history))
        for ax, (term, title, chance) in zip(axes.flat, TERM_PANELS):
            for split, name, colour in SPLITS:
                _curve(ax, history, f"{split}_{term}", name, colour, smooth)
            if chance is not None:
                _reference(ax, chance, "chance", axis="y")
            _mark_best(ax, history, label=False)
            ax.set_title(title, loc="left", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("epoch")
        _figure_legend(fig, axes.flat[0], ncol=2)
        fig.subplots_adjust(hspace=0.32, wspace=0.22)
        writer.save(fig, f"training/{_slug(history.label)}/loss_terms",
                    f"{history.label}: rotation, position, normal and face "
                    + ("terms (unweighted)" if history.has_head else
                       "scores of the matched rotations") + ", training and validation.")
    return True


def plot_embedding(history: History, writer: FigureWriter, smooth: int = 1) -> bool:
    """The embedding term beside match@1, the number that says what it achieved."""
    if not (history.has("train_embedding") or history.has("val_embedding")):
        return False
    plt = _pyplot()
    with _style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.4))
        _frame(fig, f"Embedding  |  {history.label}",
               "InfoNCE over coincident break vertices (lower is better), and match@1:\n"
               "the share of vertices whose nearest embedding is their true partner."
               + _best_note(history))
        for split, name, colour in SPLITS:
            _curve(left, history, f"{split}_embedding", name, colour, smooth)
            _curve(right, history, f"{split}_match@1", name, colour, smooth)
        left.set_title("Embedding term (InfoNCE)", loc="left", fontsize=9)
        right.set_title("match@1", loc="left", fontsize=9)
        right.set_ylim(0, 1)
        for ax in (left, right):
            ax.set_xlabel("epoch")
            _mark_best(ax, history, label=False)
        _figure_legend(fig, left, ncol=2)
        fig.subplots_adjust(wspace=0.22)
        writer.save(fig, f"training/{_slug(history.label)}/embedding",
                    f"{history.label}: embedding term and match@1, training and validation.")
    return True


def plot_rotation_error(history: History, writer: FigureWriter, smooth: int = 1) -> bool:
    """The rotation error in degrees, against chance: the matched rotations' --
    or, for a run with one, the rotation head's."""
    keys = (("train_rotation_degrees", "training, mean", SERIES[0]),
            ("val_geodesic_deg", "validation, mean", SERIES[1]),
            ("val_geodesic_median_deg", "validation, median", SERIES[2]))
    if not any(history.has(key) for key, _, _ in keys):
        return False
    source = "the rotation head" if history.has_head else "the matched rotations"
    plt = _pyplot()
    with _style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _frame(fig, f"Rotation error of {source}  |  {history.label}",
               "Each piece relative to its object's largest piece (anchor protocol). "
               "Chance: a random rotation, 126.5°.")
        for key, name, colour in keys:
            _curve(ax, history, key, name, colour, smooth)
        _reference(ax, CHANCE_GEODESIC_DEG, "chance", axis="y")
        _mark_best(ax, history)
        ax.set_ylim(0, 180)
        ax.set_yticks(range(0, 181, 30))
        ax.set_xlabel("epoch")
        ax.set_ylabel("geodesic error (deg)")
        _top_legend(ax, ncol=3)
        owner = "the rotation head's" if history.has_head else "the matched rotations'"
        writer.save(fig, f"training/{_slug(history.label)}/rotation_error",
                    f"{history.label}: {owner} mean (and validation median) error per epoch.")
    return True


def plot_accuracy(history: History, writer: FigureWriter, smooth: int = 1) -> bool:
    """Validation acc@5/10/30 and the share the matching reached, per epoch:
    the curve ``best.pt`` is chosen on (acc@10)."""
    keys = [(f"val_acc@{t:g}deg", f"acc@{t:g}", SERIES[i])
            for i, t in enumerate(ACCURACY_THRESHOLDS)]
    keys.append(("val_reached", "reached by the matching", SERIES[3]))
    if history.has_head or not any(history.has(key) for key, _, _ in keys):
        return False
    plt = _pyplot()
    with _style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _frame(fig, f"Validation accuracy  |  {history.label}",
               "Share of scored pieces whose matched rotation is within 5, 10 and 30 deg of "
               "the truth,\nand the share the matching reached at all. best.pt has the "
               "highest acc@10.")
        for key, name, colour in keys:
            _curve(ax, history, key, name, colour, smooth)
        _mark_best(ax, history)
        ax.set_ylim(0, 1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("share of pieces")
        _top_legend(ax, ncol=4)
        writer.save(fig, f"training/{_slug(history.label)}/accuracy",
                    f"{history.label}: validation acc@5/10/30 and the share reached by the "
                    f"matching, per epoch.")
    return True


def plot_category_heatmap(history: History, writer: FigureWriter, smooth: int = 5) -> bool:
    """Validation error of the rotation head per category and epoch."""
    names = history.categories()
    if not names:
        return False
    from .results import object_type

    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    rows = {}
    for name in names:
        rows.setdefault(object_type(name), []).append(history.category_series(name))
    labels = list(rows)
    matrix = np.vstack([_smooth(_column_mean(np.vstack(rows[k])), smooth) for k in labels])
    final = np.array([_last_finite(row) for row in matrix])
    order = np.argsort(final)
    matrix, labels = matrix[order], [labels[i] for i in order]
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        return False
    low = 10 * math.floor(float(finite.min()) / 10)
    high = max(10 * math.ceil(float(finite.max()) / 10), 130)
    epochs = history.epochs()
    with _style():
        height = 1.2 + 0.24 * len(labels)
        fig, ax = plt.subplots(figsize=(8.0, height))
        source = "rotation head" if history.has_head else "matched rotations"
        _frame(fig, f"Validation error by category, {source}  |  {history.label}",
               f"Mean geodesic error per epoch ({smooth}-epoch moving average), best "
               f"category at the top. The line on the scale marks chance, 126.5°.")
        cmap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap,
                          vmin=low, vmax=high,
                          extent=(epochs[0] - 0.5, epochs[-1] + 0.5, len(labels) - 0.5, -0.5))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.grid(False)
        ax.set_xlabel("epoch")
        bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.04)
        bar.set_label("geodesic error (deg)")
        bar.outline.set_visible(False)
        if low < CHANCE_GEODESIC_DEG < high:
            bar.ax.axhline(CHANCE_GEODESIC_DEG, color=INK, lw=0.9)
        owner = "the rotation head's" if history.has_head else "the matched rotations'"
        writer.save(fig, f"training/{_slug(history.label)}/val_by_category",
                    f"{history.label}: {owner} validation error per category and epoch "
                    f"(heatmap).")
    return True


# --------------------------------------------------------------------------
# Training curves, several runs (ablations)
# --------------------------------------------------------------------------

COMPARE_PANELS = (
    ("geodesic", "Rotation error, mean (deg)"),
    ("acc@10", "acc@10"),
    ("reached", "Share reached by the matching"),
    ("embedding", "Embedding term (InfoNCE)"),
    ("match@1", "match@1"),
    ("position", "Position score"),
)


def plot_comparison(histories: Sequence[History], split: str, writer: FigureWriter,
                    smooth: int = 1) -> bool:
    """One line per run, one panel per quantity -- for ablations."""
    if len(histories) < 2:
        return False
    _check_count(histories, "runs")
    plt = _pyplot()
    name = "validation" if split == "val" else "training"
    with _style():
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True)
        _frame(fig, f"Runs compared, {name}",
               "One line per run. The rotation error, acc@10, reach and position score are "
               "those of the rotations\nfitted from each run's embedding matches (a run with "
               "a rotation head shows its head's).")
        for ax, (quantity, title) in zip(axes.flat, COMPARE_PANELS):
            for index, history in enumerate(histories):
                key = _compare_key(split, quantity)
                _curve(ax, history, key, history.label, SERIES[index], smooth)
            if quantity == "geodesic":
                _reference(ax, CHANCE_GEODESIC_DEG, "chance", axis="y")
            if quantity in ("match@1", "acc@10", "reached"):
                ax.set_ylim(0, 1)
            ax.set_title(title, loc="left", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("epoch")
        _figure_legend(fig, axes.flat[0], ncol=min(len(histories), 4))
        fig.subplots_adjust(hspace=0.32, wspace=0.25)
        writer.save(fig, f"training/compare_{split}",
                    f"Every run, {name}: rotation error, acc@10, the share reached by the "
                    f"matching, the embedding term, match@1 and the position score.")
    return True


def _compare_key(split: str, quantity: str) -> str:
    if quantity == "geodesic":
        return "val_geodesic_deg" if split == "val" else "train_rotation_degrees"
    if quantity == "acc@10":
        return f"{split}_acc@10deg"
    return f"{split}_{quantity}"


# --------------------------------------------------------------------------
# Evaluation: distributions by object type
# --------------------------------------------------------------------------

def plot_errors_by_type(evaluations: Sequence[Evaluation], source: str, writer: FigureWriter,
                        kind: str = "box", scale: Optional[str] = None) -> bool:
    """
    Per object type, one box (or violin) per dataset: the spread of the
    per-piece rotation error. Median, quartiles, 5th-95th percentiles and the
    mean are drawn on both kinds.
    """
    having = [e for e in evaluations if source in e.sources()]
    if not having:
        return False
    _check_count(evaluations, "datasets")
    scale = scale or ("log" if source == "matched" else "linear")
    log = scale == "log"
    order = type_order(having, source)
    families = set().union(*(e.families for e in having))
    slots = max(sum((e.pieces["type"] == t).any() for e in having) for t in order)
    height = 0.8 / max(slots, 1)
    plt = _pyplot()

    def transform(values):
        values = np.asarray(values, dtype=np.float64)
        return np.log10(np.clip(values, LOG_FLOOR_DEG, None)) if log else values

    with _style():
        fig, ax = plt.subplots(figsize=(8.0, 1.6 + len(order) * (0.10 + 0.15 * slots)))
        what = "box plots" if kind == "box" else "violins"
        explain = ("One box per dataset: box = middle 50%, whiskers = 5th–95th percentile, "
                   "| = median, ◇ = mean." if kind == "box" else
                   "One violin per dataset (the density of the errors); inside: bar = middle "
                   "50%, line = 5th–95th percentile, | = median, ◇ = mean.")
        scale_note = (f"Log scale; errors below {LOG_FLOOR_DEG:g}° are drawn at "
                      f"{LOG_FLOOR_DEG:g}°." if log else
                      "Dashed line: chance, a random rotation (126.5°).")
        _frame(fig, f"Rotation error by object type  |  {SOURCE_NAMES[source]}",
               f"{explain}\nEach piece is scored relative to its object's largest piece. "
               f"{scale_note}")
        for row, name in enumerate(order):
            present = [(i, e) for i, e in enumerate(evaluations)
                       if any(e is h for h in having) and (e.pieces["type"] == name).any()]
            for slot, (index, evaluation) in enumerate(present):
                y = -row + ((len(present) - 1) / 2 - slot) * height
                values = evaluation.errors(source)[evaluation.pieces["type"] == name]
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                stats = distribution(values)
                colour = SERIES[index]
                if kind == "violin":
                    _violin(ax, transform(values), y, height, colour)
                    _inner_box(ax, stats, transform, y, height)
                else:
                    _box(ax, stats, transform, y, height, colour)
                ax.plot(transform([stats["mean"]]), [y], ls="none", marker="D", ms=4.2,
                        mfc=SURFACE, mec=INK, mew=0.9, zorder=5)
        ax.set_yticks([-row for row in range(len(order))])
        ax.set_yticklabels(order)
        ax.set_ylim(-len(order) + 0.5, 0.5)
        ax.grid(axis="y", visible=False)
        separator = next((row for row, name in enumerate(order) if name in families), None)
        if separator:
            ax.axhline(-separator + 0.5, color=AXIS, lw=0.8)
        _error_axis(ax, log)
        handles = _dataset_handles(evaluations, having)
        handles += _marker_handles()
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
                  frameon=False, borderaxespad=0.4, handlelength=1.4)
        writer.save(fig, f"evaluation/rotation_error_by_type_{source}_{kind}",
                    f"Per-piece rotation error by object type and dataset, "
                    f"{SOURCE_NAMES[source]} ({what}, {scale} scale).")
    return True


def _box(ax, stats, transform, y, height, colour) -> None:
    left, right = transform([stats["q1"], stats["q3"]])
    low, high = transform([stats["p5"], stats["p95"]])
    median = transform([stats["median"]])[0]
    thick = height * 0.62
    ax.plot([low, left], [y, y], color=colour, lw=1.0, zorder=3, solid_capstyle="butt")
    ax.plot([right, high], [y, y], color=colour, lw=1.0, zorder=3, solid_capstyle="butt")
    for end in (low, high):
        ax.plot([end, end], [y - thick * 0.3, y + thick * 0.3], color=colour, lw=1.0, zorder=3)
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((left, y - thick / 2), max(right - left, 1e-9), thick,
                           facecolor=_tint(colour, 0.28), edgecolor=colour, lw=1.0, zorder=3))
    ax.plot([median, median], [y - thick / 2, y + thick / 2], color=INK, lw=1.5, zorder=4,
            solid_capstyle="butt")


def _violin(ax, values, y, height, colour) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 5 or float(np.ptp(values)) <= 1e-9:
        return                                   # too few for a density: markers only
    parts = ax.violinplot([values], positions=[y], widths=height * 0.92, showextrema=False,
                          showmedians=False, showmeans=False, **_horizontal())
    for body in parts["bodies"]:
        body.set_facecolor(_tint(colour, 0.45))
        body.set_edgecolor(colour)
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
        body.set_zorder(2)


def _inner_box(ax, stats, transform, y, height) -> None:
    low, high = transform([stats["p5"], stats["p95"]])
    left, right = transform([stats["q1"], stats["q3"]])
    median = transform([stats["median"]])[0]
    ax.plot([low, high], [y, y], color=INK, lw=0.8, zorder=3, solid_capstyle="butt")
    ax.plot([left, right], [y, y], color=INK, lw=3.0, zorder=3, solid_capstyle="butt")
    ax.plot([median, median], [y - height * 0.22, y + height * 0.22], color=INK, lw=1.5,
            zorder=4, solid_capstyle="butt")


def _error_axis(ax, log: bool) -> None:
    if log:
        ticks = [0.01, 0.1, 1, 10, 100, 180]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(math.log10(LOG_FLOOR_DEG) - 0.1, math.log10(180) + 0.05)
        ax.set_xlabel("rotation error (deg, log scale)")
    else:
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("rotation error (deg)")
        _reference(ax, CHANCE_GEODESIC_DEG, None, axis="x")      # named in the subtitle


# --------------------------------------------------------------------------
# Evaluation: accuracy curve, part accuracy, piece count
# --------------------------------------------------------------------------

def plot_accuracy_curves(evaluations: Sequence[Evaluation], writer: FigureWriter) -> bool:
    """Share of pieces within each error threshold: the whole distribution, read as accuracy."""
    having = [e for e in evaluations if e.sources()]
    if not having:
        return False
    _check_count(evaluations, "datasets")
    from matplotlib.lines import Line2D

    plt = _pyplot()
    thresholds = np.logspace(math.log10(LOG_FLOOR_DEG), math.log10(180), 400)
    styles = {"matched": "-", "network": (0, (4, 2))}
    sources_drawn = []
    with _style():
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        heads = any("network" in e.sources() for e in having)
        _frame(fig, "Pieces within an error threshold",
               "Share of scored pieces whose rotation error is at most the threshold.\n"
               + ("Solid: rotations from the embedding matches; dashed: the rotation head."
                  if heads else "Rotations fitted from the embedding matches."))
        for index, evaluation in enumerate(evaluations):
            for source in evaluation.sources():
                values = evaluation.errors(source)
                values = np.sort(values[np.isfinite(values)])
                share = np.searchsorted(values, thresholds, side="right") / max(values.size, 1)
                ax.plot(thresholds, share, color=SERIES[index], ls=styles[source],
                        lw=1.6 if source == "matched" else 1.3)
                if source not in sources_drawn:
                    sources_drawn.append(source)
        ax.set_xscale("log")
        ticks = [0.01, 0.1, 1, 5, 10, 30, 100, 180]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(LOG_FLOOR_DEG, 180)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("error threshold (deg, log scale)")
        ax.set_ylabel("share of pieces")
        handles = _dataset_handles(evaluations, having, line=True)
        if len(sources_drawn) > 1:
            handles += [Line2D([], [], color=INK, ls=styles[s], lw=1.4,
                               label="matched" if s == "matched" else "network")
                        for s in ("matched", "network") if s in sources_drawn]
        _top_legend(ax, handles=handles, ncol=min(len(handles), 4))
        writer.save(fig, "evaluation/accuracy_curve",
                    "Share of pieces within each rotation-error threshold, per dataset.")
    return True


def plot_part_accuracy(evaluations: Sequence[Evaluation], writer: FigureWriter,
                       order: Optional[Sequence[str]] = None) -> bool:
    """Part accuracy per object type and dataset (scenes averaged as the benchmark does)."""
    having = [e for e in evaluations
              if np.isfinite(np.asarray(e.scenes["part_accuracy"], dtype=float)).any()]
    if not having:
        return False
    _check_count(evaluations, "datasets")
    source = "matched" if any("matched" in e.sources() for e in having) else "network"
    order = list(order or type_order(having, source))
    rows = assembly_by_type(having, order)
    if not rows:
        return False
    families = set().union(*(e.families for e in having))
    slots = max(sum(1 for r in rows if r["type"] == t) for t in order)
    height = 0.8 / max(slots, 1)
    plt = _pyplot()
    with _style():
        fig, ax = plt.subplots(figsize=(7.4, 1.6 + len(order) * (0.10 + 0.15 * slots)))
        _frame(fig, "Part accuracy by object type",
               "Share of pieces placed with Chamfer distance < 0.01, averaged per scene and "
               "then over scenes; one bar per dataset.\nRows in the order of the rotation-"
               "error figure (lowest median error first).")
        index_of = {e.label: i for i, e in enumerate(evaluations)}   # labels are unique
        for row_index, name in enumerate(order):
            present = [r for r in rows if r["type"] == name]
            for slot, record in enumerate(present):
                y = -row_index + ((len(present) - 1) / 2 - slot) * height
                value = record["part_accuracy"]
                if not np.isfinite(value):
                    continue
                colour = SERIES[index_of[record["dataset"]]]
                ax.barh(y, value, height=height * 0.78, color=colour, edgecolor=SURFACE,
                        linewidth=0.6, zorder=3)
                ax.text(value + 0.012, y, f"{value:.2f}", va="center", ha="left",
                        fontsize=6.8, color=SECONDARY)
        ax.set_yticks([-r for r in range(len(order))])
        ax.set_yticklabels(order)
        ax.set_ylim(-len(order) + 0.5, 0.5)
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", visible=False)
        separator = next((r for r, name in enumerate(order) if name in families), None)
        if separator:
            ax.axhline(-separator + 0.5, color=AXIS, lw=0.8)
        ax.set_xlabel("part accuracy")
        ax.legend(handles=_dataset_handles(evaluations, having, filled=True), loc="lower left",
                  bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, borderaxespad=0.4)
        writer.save(fig, "evaluation/part_accuracy_by_type",
                    "Part accuracy per object type and dataset.")
    return True


def plot_by_piece_count(evaluations: Sequence[Evaluation], writer: FigureWriter,
                        minimum_scenes: int = 5) -> bool:
    """How the scores move with the number of pieces in the scene."""
    having = [e for e in evaluations if len(e.scenes["type"])]
    if not having:
        return False
    _check_count(evaluations, "datasets")
    plt = _pyplot()
    tables = {id(e): by_piece_count(e) for e in having}
    bins = len(next(iter(tables.values())))
    # Only the ranges some dataset fills, so an empty "21+" leaves no gap.
    shown = [b for b in range(bins)
             if any(rows[b]["scenes"] >= minimum_scenes for rows in tables.values())]
    if not shown:
        return False
    labels = [next(iter(tables.values()))[b]["pieces"].replace("-", "–") for b in shown]
    x = np.arange(len(shown))
    source = "matched" if any("matched" in e.sources() for e in having) else "network"
    panels = (("part_accuracy", "Part accuracy", (0, 1)),
              (f"{source}_median_deg", "Median rotation error (deg, log)", None),
              ("reached", "Share placed by the matching", (0, 1)))
    with _style():
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
        _frame(fig, "Scores by the number of pieces in the scene",
               f"Ranges with fewer than {minimum_scenes} scenes are left out. Rotation errors: "
               f"{SOURCE_NAMES[source]}.")
        for ax, (key, title, limits) in zip(axes, panels):
            for index, evaluation in enumerate(evaluations):
                if id(evaluation) not in tables:
                    continue
                rows = [tables[id(evaluation)][b] for b in shown]
                values = np.array([row[key] if row["scenes"] >= minimum_scenes else np.nan
                                   for row in rows], dtype=np.float64)
                if not np.isfinite(values).any():
                    continue
                if key.endswith("_deg"):
                    values = np.clip(values, LOG_FLOOR_DEG, None)
                ax.plot(x, values, color=SERIES[index], lw=1.5, marker="o", ms=4.0,
                        mec=SURFACE, mew=1.0, label=evaluation.label)
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("pieces in the scene")
            if limits:
                ax.set_ylim(*limits)
            if key.endswith("_deg"):
                ax.set_yscale("log")
                ax.minorticks_off()
                ticks = [0.01, 0.1, 1, 10, 100]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
        _figure_legend(fig, axes[0], ncol=min(len(having), 4))
        fig.subplots_adjust(wspace=0.28)
        writer.save(fig, "evaluation/by_piece_count",
                    "Part accuracy, median rotation error and the share placed by the "
                    "matching, against pieces per scene.")
    return True


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _style():
    import matplotlib as mpl

    return mpl.rc_context({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "font.size": 8.5, "text.color": INK,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42,     # text stays text in the PDF
    })


def _frame(fig, title: str, subtitle: Optional[str] = None) -> float:
    """
    Title and explanation above everything, left-aligned, and the panels moved
    down to make room. ``subtitle`` breaks where it has ``\\n``. Returns the
    inches taken from the top, where a legend may go next.
    """
    height = fig.get_size_inches()[1]
    lines = subtitle.count("\n") + 1 if subtitle else 0
    fig.text(0.01, 1 - 0.10 / height, title, ha="left", va="top", fontsize=11,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.36 / height, subtitle, ha="left", va="top", fontsize=7.8,
                 color=SECONDARY, linespacing=1.35)
    used = 0.42 + 0.17 * lines
    fig.subplots_adjust(top=1 - (used + 0.36) / height)
    fig._reassembly_header = used
    return used


def _top_legend(ax, handles=None, ncol: int = 3) -> None:
    kwargs = {"handles": handles} if handles is not None else {}
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False,
              borderaxespad=0.3, handlelength=1.8, **kwargs)


def _figure_legend(fig, ax, ncol: int) -> None:
    """One legend for every panel, between the explanation and the panels."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    height = fig.get_size_inches()[1]
    used = getattr(fig, "_reassembly_header", 0.6)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1 - used / height),
               ncol=ncol, frameon=False, handlelength=1.8)
    rows = math.ceil(len(handles) / max(ncol, 1))
    fig.subplots_adjust(top=1 - (used + 0.28 * rows + 0.3) / height)


def _curve(ax, history: History, key: str, label: str, colour: str, smooth: int) -> bool:
    if not history.has(key):
        return False
    x, y = history.epochs(), history.series(key)
    if smooth > 1:
        ax.plot(x, y, color=colour, lw=0.7, alpha=0.3)
        y = _smooth(y, smooth)
    ax.plot(x, y, color=colour, lw=1.5, label=label)
    return True


def _best_note(history: History) -> str:
    best = history.best_epoch()
    return "" if best is None else f" Grey line: best.pt (epoch {best})."


def _mark_best(ax, history: History, label: bool = True) -> None:
    best = history.best_epoch()
    if best is None:
        return
    ax.axvline(best, color=MUTED, lw=0.8, zorder=1)
    if label:
        ax.annotate(f"best.pt (epoch {best})", xy=(best, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-3, -2), textcoords="offset points", ha="right", va="top",
                    fontsize=7, color=SECONDARY)


def _reference(ax, value: float, label: Optional[str], axis: str = "y") -> None:
    """A chance level: a muted dashed line, named at its end unless ``label`` is None."""
    if axis == "y":
        ax.axhline(value, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if label:
            ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                        fontsize=7, color=MUTED)
    else:
        ax.axvline(value, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if label:
            ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                        fontsize=7, color=MUTED)


def _dataset_handles(evaluations, having, line: bool = False, filled: bool = False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    for index, evaluation in enumerate(evaluations):
        if not any(evaluation is other for other in having):
            continue
        colour = SERIES[index]
        if line:
            handles.append(Line2D([], [], color=colour, lw=1.8, label=evaluation.label))
        else:
            handles.append(Patch(facecolor=colour if filled else _tint(colour, 0.35),
                                 edgecolor=colour, lw=1.0, label=evaluation.label))
    return handles


def _marker_handles():
    from matplotlib.lines import Line2D

    return [Line2D([], [], ls="none", marker="|", ms=8, mew=1.5, color=INK, label="median"),
            Line2D([], [], ls="none", marker="D", ms=4.2, mfc=SURFACE, mec=INK, mew=0.9,
                   label="mean")]


def _horizontal() -> Dict:
    """``vert=False`` before matplotlib 3.10, ``orientation="horizontal"`` from it."""
    import matplotlib

    parts = []
    for piece in matplotlib.__version__.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return {"orientation": "horizontal"} if tuple(parts) >= (3, 10) else {"vert": False}


def _tint(colour: str, amount: float) -> str:
    """``colour`` mixed toward white: a light wash of the same hue, opaque."""
    colour = colour.lstrip("#")
    rgb = [int(colour[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(255 - (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that skips NaN; ``window`` <= 1 returns the input."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(int(window))
    valid = np.isfinite(values)
    total = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1e-12), np.nan)


def _column_mean(stack: np.ndarray) -> np.ndarray:
    """Mean over rows, skipping NaN; NaN (no warning) where a column has none."""
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _last_finite(values: np.ndarray) -> float:
    finite = np.flatnonzero(np.isfinite(values))
    return float(values[finite[-1]]) if finite.size else np.inf


def _slug(label: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label.strip())
    return keep.strip("-") or "run"


def _check_count(items, what: str) -> None:
    if len(items) > MAX_SERIES:
        raise ValueError(f"{len(items)} {what}: at most {MAX_SERIES} fit one figure's "
                         f"colours -- split them over several calls")
