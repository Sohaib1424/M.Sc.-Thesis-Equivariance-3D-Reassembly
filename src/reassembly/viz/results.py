"""
Training histories and evaluation results, read back for figures and tables.

Numpy only -- no torch, no matplotlib -- so results copied off the training
machine can be tabulated anywhere. (A history inside a checkpoint is the one
exception: reading a ``.pt`` needs torch.)

What is read
------------
history
    ``<checkpoint_dir>/history.json``, or the ``history`` a checkpoint carries:
    one row per epoch with ``train_<x>`` and ``val_<x>`` for the loss, the
    scores and the metrics. Since v7 the network trains on the embedding term
    alone (``train_total`` is it), and rotation, position, normal and face are
    scores of the rotations fitted from its matches -- every epoch on
    validation, on training only with ``--score_train``. A run from before
    that (one with a rotation head, :attr:`History.has_head`) logged the five
    terms **unweighted** and their weighted sum as the total.
evaluation
    ``<split>_metrics*.json`` from ``python -m scripts.train --evaluate``. The
    per-fragment numbers come from its per-scene records
    (``assembly_scenes``), so the evaluation must have been assembled -- the
    default, not ``--no_assemble``. Since v7 every rotation in it is fitted
    from the matches (``rotation_head: false``); a file from before holds the
    rotation head's errors, and a matched one of those both.

Object types
------------
Everyday sorts its objects into categories (``Bottle``, ``Mug``, ...);
artifact has none. A volume-constrained copy nests the base subset inside its
own folder, so its categories read ``everyday_compressed/Bottle`` -- and an
artifact copy's reads ``artifact_compressed``. :func:`object_type` maps all of
these to one name per kind of object, so the same type lines up across every
subset: the category's last part, or for a subset without categories the
subset's family (``Artifact``).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

UNCATEGORISED = "(uncategorised)"
LOSS_TERMS = ("rotation", "position", "normal", "face", "embedding")
ROTATION_SOURCES = ("network", "matched")
CHANCE_GEODESIC_DEG = 126.48
"""Mean geodesic angle to a uniformly random rotation, pi/2 + 2/pi rad."""
ACCURACY_THRESHOLDS = (5.0, 10.0, 30.0)
PIECE_BINS: Tuple[Tuple[int, Optional[int]], ...] = (
    (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 12), (13, 20), (21, None))
"""Pieces per scene, grouped where scenes get scarce (the benchmark keeps 2-20)."""


# --------------------------------------------------------------------------
# Object types
# --------------------------------------------------------------------------

def _type_and_family(category: str, scene: str = "") -> Tuple[str, bool]:
    """``(type, whether it is a subset family rather than a category)``."""
    name = (category or "").replace("\\", "/").strip("/").rsplit("/", 1)[-1]
    if name and name != UNCATEGORISED and not name.endswith("_compressed"):
        return name, False
    family = name if name.endswith("_compressed") else \
        (scene or "").replace("\\", "/").strip("/").split("/", 1)[0]
    family = family.split("-")[-1]                # volume_constrained-artifact_compressed
    if family.endswith("_compressed"):
        family = family[:-len("_compressed")]
    return (family.capitalize() if family else "Uncategorised"), True


def object_type(category: str, scene: str = "") -> str:
    """
    The kind of object a scene shows: ``Bottle`` for ``Bottle`` and for
    ``everyday_compressed/Bottle``; for a subset without categories, its
    family -- ``Artifact`` for an ``(uncategorised)`` artifact scene and for a
    volume-constrained copy's ``artifact_compressed``.
    """
    return _type_and_family(category, scene)[0]


# --------------------------------------------------------------------------
# Histories
# --------------------------------------------------------------------------

@dataclass(eq=False)
class History:
    """One run's per-epoch log, one row per epoch (the last row of each)."""

    label: str
    path: Path
    rows: List[Dict]

    def epochs(self) -> np.ndarray:
        """1-based, as the training log prints them."""
        return np.asarray([int(row.get("epoch", i)) + 1 for i, row in enumerate(self.rows)])

    def series(self, key: str) -> np.ndarray:
        """``key`` per epoch, NaN where a row lacks it."""
        return np.asarray([_number(row.get(key)) for row in self.rows], dtype=np.float64)

    def has(self, key: str) -> bool:
        return bool(np.isfinite(self.series(key)).any())

    @property
    def has_head(self) -> bool:
        """Whether the run had the rotation head v7 removed: it logged the
        head's own diagnostics."""
        return any(self.has(key) for key in ("train_head_cos", "val_head_cos",
                                             "val_absolute_geodesic_deg"))

    def best_epoch(self) -> Optional[int]:
        """The epoch ``best.pt`` holds: the highest ``val_acc@5deg`` -- or,
        for a run with the rotation head, the lowest ``val_geodesic_deg``."""
        if self.has_head:
            values = self.series("val_geodesic_deg")
            if not np.isfinite(values).any():
                return None
            return int(self.epochs()[int(np.nanargmin(values))])
        values = self.series("val_acc@5deg")
        if not np.isfinite(values).any():
            return None
        return int(self.epochs()[int(np.nanargmax(values))])

    def categories(self) -> List[str]:
        names: List[str] = []
        for row in self.rows:
            for name in (row.get("val_by_category") or {}):
                if name not in names:
                    names.append(name)
        return names

    def category_series(self, name: str, key: str = "geodesic_deg") -> np.ndarray:
        return np.asarray([_number(((row.get("val_by_category") or {}).get(name) or {}).get(key))
                           for row in self.rows], dtype=np.float64)


def load_history(path, label: Optional[str] = None) -> History:
    """
    ``history.json`` (a list of rows), or a checkpoint whose ``history`` it
    reads -- that needs torch. An epoch cut short and re-run appears twice in
    the log; the last row of each epoch is kept.
    """
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        rows = state.get("history") or []
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("history", []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: no per-epoch history in it")
    latest: Dict[int, Dict] = {}
    for index, row in enumerate(rows):
        latest[int(row.get("epoch", index))] = row
    ordered = [latest[epoch] for epoch in sorted(latest)]
    return History(label or _default_label(path), path, ordered)


# --------------------------------------------------------------------------
# Evaluations
# --------------------------------------------------------------------------

@dataclass(eq=False)
class Evaluation:
    """
    One ``--evaluate`` result. ``pieces`` holds one entry per *scored*
    fragment (every fragment but each scene's anchor), ``scenes`` one per
    scene; both are dicts of equal-length arrays.
    """

    label: str
    path: Path
    summary: Dict
    rotations: str
    pieces: Dict[str, np.ndarray]
    scenes: Dict[str, np.ndarray]
    families: set = field(default_factory=set)

    def errors(self, source: str) -> np.ndarray:
        """Per scored fragment, in degrees; NaN where this file does not hold it."""
        return self.pieces.get(source, np.full(len(self.pieces["type"]), np.nan))

    def sources(self) -> List[str]:
        """The rotation sources whose per-fragment errors this file holds."""
        return [s for s in ROTATION_SOURCES if np.isfinite(self.errors(s)).any()]

    def types(self) -> List[str]:
        return sorted(set(self.pieces["type"].tolist()))


def load_evaluation(path, label: Optional[str] = None) -> Evaluation:
    """Read a ``<split>_metrics*.json``; per-fragment arrays need its scene records."""
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = summary.pop("assembly_scenes", None) or []
    rotations = summary.get("rotations", "network")
    columns: Dict[str, list] = {k: [] for k in
                                ("type", "scene", "pieces", "network", "matched", "reached",
                                 "part_chamfer")}
    scene_columns: Dict[str, list] = {k: [] for k in
                                      ("type", "pieces", "part_accuracy", "rmse_t", "chamfer",
                                       "part_chamfer", "geodesic_deg", "euler_rmse_deg",
                                       "matched_share", "matches", "verified_matches")}
    families = set()
    for index, record in enumerate(records):
        kind, family = _type_and_family(record.get("_category", ""), record.get("_scene", ""))
        if family:
            families.add(kind)
        fragments = _number(record.get("fragments"))
        count = int(fragments) if np.isfinite(fragments) else 0
        scene_columns["type"].append(kind)
        scene_columns["pieces"].append(count)
        for key in ("part_accuracy", "rmse_t", "chamfer", "part_chamfer", "geodesic_deg",
                    "euler_rmse_deg", "matched_share", "matches", "verified_matches"):
            scene_columns[key].append(_number(record.get(key)))

        angles = list(record.get("_scored_geodesic_deg") or [])
        anchor = int(record.get("_anchor", -1))
        scored = [f for f in range(count) if f != anchor]
        aligned = len(scored) == len(angles)
        network = record.get("_network_geodesic_deg")
        reached = record.get("_reached")
        part = record.get("_part_chamfer")
        for position, angle in enumerate(angles):
            fragment = scored[position] if aligned else None
            columns["type"].append(kind)
            columns["scene"].append(index)
            columns["pieces"].append(count)
            if rotations == "matched":
                columns["matched"].append(_number(angle))
                columns["network"].append(_number(network[position])
                                          if network and len(network) == len(angles)
                                          else np.nan)
            else:
                columns["network"].append(_number(angle))
                columns["matched"].append(np.nan)
            columns["reached"].append(
                bool(reached[fragment]) if (reached and fragment is not None
                                            and fragment < len(reached)) else np.nan)
            columns["part_chamfer"].append(
                _number(part[fragment]) if (part and fragment is not None
                                            and fragment < len(part)) else np.nan)

    pieces = {
        "type": np.asarray(columns["type"], dtype=object),
        "scene": np.asarray(columns["scene"], dtype=np.int64),
        "pieces": np.asarray(columns["pieces"], dtype=np.int64),
        "network": np.asarray(columns["network"], dtype=np.float64),
        "matched": np.asarray(columns["matched"], dtype=np.float64),
        "reached": np.asarray(columns["reached"], dtype=np.float64),
        "part_chamfer": np.asarray(columns["part_chamfer"], dtype=np.float64),
    }
    scenes = {key: np.asarray(values, dtype=object if key == "type" else np.float64)
              for key, values in scene_columns.items()}
    return Evaluation(label or _default_label(path), path, summary, rotations, pieces,
                      scenes, families)


# --------------------------------------------------------------------------
# Statistics -- the tables every figure is drawn from
# --------------------------------------------------------------------------

def distribution(values: np.ndarray) -> Dict[str, float]:
    """Count, mean, the quartiles, the 5th/95th percentiles and acc@5/10/30."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    out: Dict[str, float] = {"n": int(values.size)}
    if not values.size:
        return out
    p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    out.update(mean=float(values.mean()), median=float(median), q1=float(q1), q3=float(q3),
               p5=float(p5), p95=float(p95))
    for threshold in ACCURACY_THRESHOLDS:
        out[f"acc@{threshold:g}"] = float((values < threshold).mean())
    return out


def type_order(evaluations: Sequence[Evaluation], source: str) -> List[str]:
    """
    Every type present, best (lowest pooled median error) first, the subset
    families (``Artifact``) after the categories. One order for every figure,
    so a type sits in the same row throughout.
    """
    pooled: Dict[str, List[np.ndarray]] = {}
    families = set()
    for evaluation in evaluations:
        families |= evaluation.families
        errors = evaluation.errors(source)
        for kind in evaluation.types():
            pooled.setdefault(kind, []).append(errors[evaluation.pieces["type"] == kind])

    def median(kind: str) -> float:
        values = np.concatenate(pooled[kind]) if pooled[kind] else np.zeros(0)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.inf

    return sorted(pooled, key=lambda kind: (kind in families, median(kind), kind))


def errors_by_type(evaluations: Sequence[Evaluation], source: str) -> List[Dict]:
    """One row per (type, evaluation) that has pieces: its error distribution."""
    rows = []
    for kind in type_order(evaluations, source):
        for evaluation in evaluations:
            mask = evaluation.pieces["type"] == kind
            if not mask.any():
                continue
            stats = distribution(evaluation.errors(source)[mask])
            if stats["n"]:
                rows.append({"type": kind, "dataset": evaluation.label, "source": source,
                             **stats})
    return rows


ASSEMBLY_KEYS = ("part_accuracy", "rmse_t", "chamfer", "part_chamfer", "geodesic_deg",
                 "euler_rmse_deg", "matched_share", "matches", "verified_matches")
"""The per-scene numbers averaged per type: the benchmark's, then the matching's."""


def assembly_by_type(evaluations: Sequence[Evaluation], order: Sequence[str]) -> List[Dict]:
    """Per (type, evaluation): the benchmark's per-scene means and the scene count."""
    rows = []
    for kind in order:
        for evaluation in evaluations:
            mask = evaluation.scenes["type"] == kind
            if not mask.any():
                continue
            row = {"type": kind, "dataset": evaluation.label, "scenes": int(mask.sum())}
            for key in ASSEMBLY_KEYS:
                row[key] = _nanmean(evaluation.scenes[key][mask])
            rows.append(row)
    return rows


SCORE_KEYS = ("position", "normal", "face")
"""The geometric scores besides the rotation's, kept per category by an evaluation."""


def scores_by_type(evaluation: Evaluation) -> Dict[str, Dict[str, float]]:
    """
    The evaluation's per-category means (``by_category``: the scores, the share
    the matching reached), merged per object type -- weighted by fragments, as
    the categories' own means are -- and over every fragment as ``"(all)"``.
    Empty for a file from before the scores were kept per category, and for
    one with a rotation head, whose terms were the head's losses.
    """
    if evaluation.summary.get("rotation_head", True):
        return {}
    breakdown = evaluation.summary.get("by_category") or {}
    keys = SCORE_KEYS + ("reached",)
    merged: Dict[str, Dict[str, float]] = {}
    for category, row in breakdown.items():
        kind = object_type(category)
        weight = _number(row.get("fragments"))
        if not np.isfinite(weight) or weight <= 0:
            continue
        slot = merged.setdefault(kind, {"fragments": 0.0, **{f"_{k}": 0.0 for k in keys},
                                        **{f"_{k}_n": 0.0 for k in keys}})
        slot["fragments"] += weight
        for key in keys:
            value = _number(row.get(key))
            if np.isfinite(value):
                slot[f"_{key}"] += value * weight
                slot[f"_{key}_n"] += weight
    out = {}
    for kind, slot in merged.items():
        out[kind] = {key: (slot[f"_{key}"] / slot[f"_{key}_n"] if slot[f"_{key}_n"] else np.nan)
                     for key in keys}
    summary = evaluation.summary
    if out:
        out["(all)"] = {key: _number(summary.get(key)) for key in keys}
    return out


def by_type(evaluations: Sequence[Evaluation], order: Sequence[str]) -> List[Dict]:
    """
    Every number per (type, evaluation), one row each, and each evaluation's
    whole set as type ``"(all)"``: the per-piece rotation error (count, mean,
    median, quartiles, acc@5/10/30), the share of pieces the matching reached,
    the position, normal and face scores, and the scenes' assembly scores and
    match counts. A subset without categories (Artifact) is one type.
    """
    rows = []
    for kind in list(order) + ["(all)"]:
        for evaluation in evaluations:
            if kind == "(all)":
                piece_mask = np.ones(len(evaluation.pieces["type"]), dtype=bool)
                scene_mask = np.ones(len(evaluation.scenes["type"]), dtype=bool)
            else:
                piece_mask = evaluation.pieces["type"] == kind
                scene_mask = evaluation.scenes["type"] == kind
            if not piece_mask.any() and not scene_mask.any():
                continue
            source = "matched" if "matched" in evaluation.sources() else "network"
            stats = distribution(evaluation.errors(source)[piece_mask])
            row = {"type": kind, "dataset": evaluation.label, "pieces": stats.pop("n")}
            for key in ("mean", "median", "q1", "q3", "acc@5", "acc@10", "acc@30"):
                row[f"error_{key}" if key in ("mean", "median", "q1", "q3") else key] = \
                    stats.get(key, np.nan)
            row["reached"] = _nanmean(evaluation.pieces["reached"][piece_mask])
            scores = scores_by_type(evaluation).get(kind, {})
            for key in SCORE_KEYS:
                row[key] = scores.get(key, np.nan)
            row["scenes"] = int(scene_mask.sum())
            for key in ASSEMBLY_KEYS:
                if key == "matched_share":
                    continue
                row[key] = _nanmean(evaluation.scenes[key][scene_mask])
            rows.append(row)
    return rows


def by_piece_count(evaluation: Evaluation, bins=PIECE_BINS) -> List[Dict]:
    """Per range of pieces per scene: scenes, part accuracy, median errors, reach."""
    rows = []
    for low, high in bins:
        upper = np.inf if high is None else high
        scene_mask = (evaluation.scenes["pieces"] >= low) & (evaluation.scenes["pieces"] <= upper)
        piece_mask = (evaluation.pieces["pieces"] >= low) & (evaluation.pieces["pieces"] <= upper)
        row = {"dataset": evaluation.label, "pieces": _bin_label(low, high),
               "scenes": int(scene_mask.sum()), "scored_pieces": int(piece_mask.sum()),
               "part_accuracy": _nanmean(evaluation.scenes["part_accuracy"][scene_mask])}
        for source in ROTATION_SOURCES:
            values = evaluation.errors(source)[piece_mask]
            values = values[np.isfinite(values)]
            row[f"{source}_median_deg"] = float(np.median(values)) if values.size else np.nan
        row["reached"] = _nanmean(evaluation.pieces["reached"][piece_mask])
        rows.append(row)
    return rows


def summary_row(evaluation: Evaluation) -> Dict:
    """The headline numbers of one evaluation, as one table row."""
    summary = evaluation.summary
    info = summary.get("evaluation") or {}
    # Since v7 the top-level numbers are the matched rotations' and there is no
    # head; a file from before keeps the head's at the top level and, when
    # matched, the matched rotations' under "matched".
    head = summary.get("rotation_head", True)
    matched = summary.get("matched") or ({} if head else summary)
    network = summary if head else {}
    assembly = summary.get("assembly") or {}
    return {
        "dataset": evaluation.label,
        "file": str(evaluation.path),
        "checkpoint": f"{info.get('checkpoint', '')}"
                      + (f" (epoch {info['epoch']})" if "epoch" in info else ""),
        "rotations": evaluation.rotations,
        # A file from before v7 names no placement: the global solve was the only one.
        "placement": summary.get("placement") or "global",
        "scenes": int(len(evaluation.scenes["type"])),
        "objects": _number(info.get("objects")),
        "trained_shapes": _number(info.get("trained_shapes")),
        "scored_pieces": int(summary.get("geodesic_fragments", len(evaluation.pieces["type"]))),
        "network_mean_deg": _number(network.get("geodesic_deg")),
        "network_median_deg": _number(network.get("geodesic_median_deg")),
        "matched_mean_deg": _number(matched.get("geodesic_deg")),
        "matched_median_deg": _number(matched.get("geodesic_median_deg")),
        "matched_acc@5": _number(matched.get("acc@5deg")),
        "matched_acc@10": _number(matched.get("acc@10deg")),
        "matched_acc@30": _number(matched.get("acc@30deg")),
        "reached": _number(matched.get("reached")),
        "part_accuracy": _number(assembly.get("part_accuracy")),
        "rmse_t": _number(assembly.get("rmse_t")),
        "chamfer": _number(assembly.get("chamfer")),
        "scene_geodesic_deg": _number(assembly.get("geodesic_deg")),
        "scene_euler_rmse_deg": _number(assembly.get("euler_rmse_deg")),
        "matches": _number(assembly.get("matches")),
        "verified_matches": _number(assembly.get("verified_matches")),
        "match@1": _number(summary.get("match@1")),
        # The four scores (none for a file with a head: its terms were losses).
        **{key: _number(summary.get(key))
           for key in ("rotation", "position", "normal", "face") if not head},
        "embedding": _number(summary.get("embedding")),
    }


# --------------------------------------------------------------------------
# Writing tables
# --------------------------------------------------------------------------

def write_csv(path, rows: Sequence[Dict]) -> Path:
    """``rows`` as CSV, columns in first-seen order; floats to 6 significant digits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])
    return path


def markdown_table(rows: Sequence[Dict], columns: Sequence[Tuple[str, str, str]]) -> str:
    """``columns`` as ``(key, heading, format)``; a missing or NaN value prints ``-``."""
    lines = ["| " + " | ".join(heading for _, heading, _ in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            if isinstance(value, float) and not np.isfinite(value):
                cells.append("-")
            elif value is None or value == "":
                cells.append("-")
            else:
                cells.append(format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------

def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _bin_label(low: int, high: Optional[int]) -> str:
    if high is None:
        return f"{low}+"
    return str(low) if low == high else f"{low}-{high}"


def _cell(value):
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:.6g}"
    return value


def _default_label(path: Path) -> str:
    """The folder a result sits in (``eval/W10-artifact/...`` -> ``W10-artifact``)."""
    parent = path.resolve().parent.name
    return parent or path.stem
