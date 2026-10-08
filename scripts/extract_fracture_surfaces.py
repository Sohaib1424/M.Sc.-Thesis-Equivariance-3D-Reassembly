#!/usr/bin/env python3
"""
Walk the whole dataset and extract the fracture surface of every fragment.

Exhaustive and ordered: every scene, every fracture mode, every fragment, in
sorted order. Nothing is sampled. Re-running gives byte-identical output.

    python -m scripts.extract_fracture_surfaces --root data --workers 4

What it writes
--------------
``--stats-csv`` (default ``fracture_stats.csv``)
    One row per fragment: vertex and face counts before and after extraction,
    plus the fraction disposed. Streamed to disk as it goes, so a run that is
    killed leaves valid partial output.

``--masks-dir``
    One ``.npz`` per (scene, mode) holding the fracture mask as packed bits.
    A mask is ~1 bit per face, so the whole dataset fits in well under a
    gigabyte, and the surfaces can be rebuilt from it exactly. This is the
    format to use for training input.

``--export-dir``
    Actual ``.npz`` fracture-surface meshes. Roughly 100x the size of masks;
    intended for a subset (use with ``--limit`` or ``--max-modes``), not the
    whole dataset.

Two labelling methods
---------------------
``--method dihedral`` (default)
    Per-fragment heuristic on dihedral sharpness. Works on a fragment in
    isolation, so it is what you can compute at inference time.

``--method coincidence``
    Ground truth: a vertex is fracture exactly when another fragment has a
    vertex at the same point. Needs the whole scene, but it is exact.

``--method both``
    Computes both and scores the heuristic against the truth (precision,
    recall, IoU per fragment). Use this to pick ``--sharp-threshold``.

Resuming
--------
``--resume`` reads the existing CSV, skips any (scene, mode) already present,
and appends. Safe to interrupt at any point; combine with ``--time-budget``
to fit inside a session limit.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.data.paths import Scene, count_objects, find_scenes  # noqa: E402
from reassembly.data.scene import SceneReader  # noqa: E402
from reassembly.mesh.fracture import (  # noqa: E402
    DEFAULT_SHARP_THRESHOLD,
    fracture_agreement,
    fracture_face_mask,
    fracture_face_mask_from_vertices,
    fracture_vertex_masks,
)
from reassembly.arrays import compact_indices  # noqa: E402

CSV_FIELDS = [
    "subset", "category", "scene", "mode", "piece",
    "vertices", "faces", "edges",
    "frac_vertices", "frac_faces", "frac_edges",
    "kept_vertices_pct", "kept_faces_pct", "kept_edges_pct",
    "degenerate_faces",
    "true_frac_vertices", "precision", "recall", "iou",
]


@dataclass
class Options:
    root: Path
    method: str
    sharp_threshold: float
    masks_dir: Optional[Path]
    export_dir: Optional[Path]
    max_modes: Optional[int]
    coincidence_tol: float
    compute_edges: bool
    min_fracture_faces: int = 0


# --------------------------------------------------------------------------
# per-scene work (runs in a worker process)
# --------------------------------------------------------------------------

def process_scene(args) -> tuple[str, List[dict], str]:
    """Extract every fragment of every mode of one scene. Returns (key, rows, error)."""
    scene, opts, done_modes = args
    rows: List[dict] = []
    try:
        reader = SceneReader(scene.path)
        modes = reader.mode_names()
        if opts.max_modes is not None:
            modes = modes[: opts.max_modes]
        modes = [m for m in modes if m not in done_modes]

        for mode in modes:
            result = reader.load_mode(mode)
            if not result.fragments:
                continue
            rows.extend(_process_mode(scene, mode, result.fragments, opts))
    except Exception as exc:                                   # noqa: BLE001
        return (str(scene.path), rows, f"{type(exc).__name__}: {exc}")
    return (str(scene.path), rows, "")


def _process_mode(scene: Scene, mode: str, fragments, opts: Options) -> List[dict]:
    from reassembly.mesh.topology import count_unique_edges

    truth_vertex_masks = None
    if opts.method in ("coincidence", "both"):
        truth_vertex_masks = fracture_vertex_masks(fragments, tol=opts.coincidence_tol)

    rows, face_masks, face_counts = [], [], []
    for index, fragment in enumerate(fragments):
        vertices = np.asarray(fragment.vertices)
        faces = np.asarray(fragment.faces)
        n_vertices, n_faces = vertices.shape[0], faces.shape[0]

        heuristic = None
        if opts.method in ("dihedral", "both"):
            heuristic = fracture_face_mask(vertices, faces, opts.sharp_threshold,
                                           min_faces=opts.min_fracture_faces,
                                           has_duplicate_faces=False)
            face_mask = heuristic.face_mask
            degenerate = heuristic.degenerate_faces
        else:
            face_mask = fracture_face_mask_from_vertices(faces, truth_vertex_masks[index])
            degenerate = 0

        kept_faces = faces[face_mask]
        kept_vertices, _ = compact_indices(kept_faces, n_vertices)
        n_frac_v, n_frac_f = kept_vertices.shape[0], kept_faces.shape[0]

        n_edges = count_unique_edges(faces, n_vertices) if opts.compute_edges else -1
        n_frac_edges = count_unique_edges(kept_faces, n_vertices) if opts.compute_edges else -1

        row = {
            "subset": scene.subset, "category": scene.category, "scene": scene.name,
            "mode": mode, "piece": index,
            "vertices": n_vertices, "faces": n_faces, "edges": n_edges,
            "frac_vertices": n_frac_v, "frac_faces": n_frac_f, "frac_edges": n_frac_edges,
            "kept_vertices_pct": _pct(n_frac_v, n_vertices),
            "kept_faces_pct": _pct(n_frac_f, n_faces),
            "kept_edges_pct": _pct(n_frac_edges, n_edges) if opts.compute_edges else "",
            "degenerate_faces": degenerate,
            "true_frac_vertices": "", "precision": "", "recall": "", "iou": "",
        }

        if truth_vertex_masks is not None:
            truth_faces = fracture_face_mask_from_vertices(faces, truth_vertex_masks[index])
            row["true_frac_vertices"] = int(truth_vertex_masks[index].sum())
            if opts.method == "both":
                score = fracture_agreement(heuristic.face_mask, truth_faces)
                row["precision"] = round(score["precision"], 5)
                row["recall"] = round(score["recall"], 5)
                row["iou"] = round(score["iou"], 5)

        rows.append(row)
        face_masks.append(face_mask)
        face_counts.append(n_faces)

        if opts.export_dir is not None:
            _export_surface(opts.export_dir, scene, mode, index, vertices, kept_faces)

    if opts.masks_dir is not None and face_masks:
        _write_masks(opts.masks_dir, scene, mode, face_masks, face_counts)
    return rows


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def _scene_slug(scene: Scene) -> str:
    parts = [scene.subset, scene.category, scene.name]
    return "__".join(p.replace("/", "_") for p in parts if p)


def _write_masks(out_dir: Path, scene: Scene, mode: str, masks, counts) -> None:
    """Packed-bit fracture masks for one (scene, mode). ~1 bit per face."""
    target = out_dir / _scene_slug(scene)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / f"{mode}.npz",
        packed=np.packbits(np.concatenate(masks)),
        face_counts=np.asarray(counts, dtype=np.int64),
    )


def _export_surface(out_dir: Path, scene: Scene, mode: str, index: int,
                    vertices: np.ndarray, faces: np.ndarray) -> None:
    target = out_dir / _scene_slug(scene) / mode
    target.mkdir(parents=True, exist_ok=True)
    kept, new_faces = compact_indices(faces, vertices.shape[0])
    np.savez_compressed(
        target / f"piece_{index:03d}.npz",
        vertices=vertices[kept].astype(np.float32),
        faces=new_faces.astype(np.int32),
    )


def load_masks(path: str | Path) -> List[np.ndarray]:
    """Read a mask file written by ``--masks-dir`` back into per-fragment masks."""
    data = np.load(path)
    counts = data["face_counts"]
    flat = np.unpackbits(data["packed"])[: int(counts.sum())].astype(bool)
    return list(np.split(flat, np.cumsum(counts)[:-1]))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def scene_key(subset: str, category: str, name: str) -> tuple:
    """
    Full identity of a scene.

    The scene *name* alone is not unique: the same object name occurs in more
    than one subset, and in more than one category within Everyday. Keying
    resume state on the name alone silently marks unprocessed scenes as done.
    """
    return (subset, category, name)


def read_done(csv_path: Path) -> dict:
    """(scene identity -> set of finished modes) from an existing CSV."""
    done: dict = {}
    if not csv_path.is_file():
        return done
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = scene_key(row["subset"], row.get("category", ""), row["scene"])
            done.setdefault(key, set()).add(row["mode"])
    return done


class Progress:
    """One-line progress. No dependency on tqdm; falls back cleanly when piped."""

    def __init__(self, total: int, enabled: bool = True):
        self.total, self.done, self.rows = total, 0, 0
        self.start = time.perf_counter()
        self.enabled = enabled and sys.stderr.isatty()

    def update(self, rows: int) -> None:
        self.done += 1
        self.rows += rows
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        rate = self.done / elapsed if elapsed else 0.0
        eta = (self.total - self.done) / rate if rate else 0.0
        bar_len = 28
        filled = int(bar_len * self.done / max(self.total, 1))
        bar = "#" * filled + "." * (bar_len - filled)
        sys.stderr.write(
            f"\r[{bar}] {self.done}/{self.total} scenes  "
            f"{self.rows} fragments  {rate:.2f} sc/s  eta {_hms(eta)}   "
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustively extract fracture surfaces from Breaking Bad.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="data", help="dataset root directory")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="top-level subset names to include (default: all)")
    parser.add_argument("--method", choices=["dihedral", "coincidence", "both"],
                        default="dihedral")
    parser.add_argument("--sharp-threshold", type=float, default=DEFAULT_SHARP_THRESHOLD)
    parser.add_argument("--coincidence-tol", type=float, default=1e-6)
    parser.add_argument("--min-fracture-faces", type=int, default=0,
                        help="a fragment of a broken object always has SOME fracture "
                             "surface, so an empty mask is the heuristic failing. Set >=1 "
                             "to relax the threshold for that fragment alone until at "
                             "least this many faces are labelled")
    parser.add_argument("--stats-csv", default="fracture_stats.csv")
    parser.add_argument("--summary-csv", default=None,
                        help="also write aggregate statistics here")
    parser.add_argument("--masks-dir", default=None,
                        help="write packed-bit fracture masks here")
    parser.add_argument("--export-dir", default=None,
                        help="write fracture-surface meshes here (large)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (scenes are independent)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N scenes")
    parser.add_argument("--max-modes", type=int, default=None,
                        help="cap fracture modes per scene")
    parser.add_argument("--skip-edges", action="store_true",
                        help="omit unique-edge counts (about 25%% faster)")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing CSV, skipping finished modes")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="stop cleanly after this many hours")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be processed and exit")
    args = parser.parse_args(argv)

    scenes = find_scenes(args.root, args.subsets)
    if not scenes:
        parser.error(f"no scene directories found under {args.root!r}")
    if args.limit:
        scenes = scenes[: args.limit]

    csv_path = Path(args.stats_csv)
    done = read_done(csv_path) if args.resume else {}

    print(f"root          {Path(args.root).resolve()}")
    print(f"scenes        {len(scenes)}  ({count_objects(scenes)} distinct objects)")
    print(f"method        {args.method}"
          + (f"  (threshold {args.sharp_threshold})" if args.method != "coincidence" else ""))
    print(f"stats         {csv_path}" + ("  [resume]" if args.resume else ""))
    if args.masks_dir:
        print(f"masks         {args.masks_dir}")
    if args.export_dir:
        print(f"surfaces      {args.export_dir}")
    print(f"workers       {args.workers}")

    if args.dry_run:
        total_modes = sum(len(s.mode_dirs()) for s in scenes[:50])
        per_scene = total_modes / min(len(scenes), 50)
        print(f"\n~{per_scene:.0f} modes/scene, ~{per_scene * len(scenes):,.0f} "
              f"(scene, mode) pairs to process")
        return 0

    opts = Options(
        root=Path(args.root), method=args.method,
        sharp_threshold=args.sharp_threshold,
        min_fracture_faces=args.min_fracture_faces,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        export_dir=Path(args.export_dir) if args.export_dir else None,
        max_modes=args.max_modes, coincidence_tol=args.coincidence_tol,
        compute_edges=not args.skip_edges,
    )
    for directory in (opts.masks_dir, opts.export_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    tasks = [
        (scene, opts, done.get(scene_key(scene.subset, scene.category, scene.name), set()))
        for scene in scenes
    ]
    write_header = not (args.resume and csv_path.is_file())
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stop = {"now": False}

    def _handle(signum, frame):                                # noqa: ARG001
        stop["now"] = True
        sys.stderr.write("\ninterrupt received -- finishing current scenes\n")

    signal.signal(signal.SIGINT, _handle)

    progress = Progress(len(tasks))
    started = time.perf_counter()
    failures: List[tuple] = []
    total_rows = 0

    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for key, rows, error in _run(tasks, args.workers, stop):
            if error:
                failures.append((key, error))
            if rows:
                writer.writerows(rows)
                handle.flush()
                total_rows += len(rows)
            progress.update(len(rows))
            if args.time_budget and (time.perf_counter() - started) / 3600 >= args.time_budget:
                stop["now"] = True

    progress.close()
    elapsed = time.perf_counter() - started
    print(f"\n{total_rows} fragments in {_hms(elapsed)}"
          f"  ({total_rows / elapsed:.1f} fragments/s)" if elapsed else "")

    if failures:
        print(f"\n{len(failures)} scene(s) failed:")
        for key, error in failures[:20]:
            print(f"  {key}: {error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    if args.summary_csv:
        _write_summary(csv_path, Path(args.summary_csv))
    return 1 if failures else 0


def _run(tasks, workers: int, stop) -> Iterable:
    if workers <= 1:
        for task in tasks:
            if stop["now"]:
                break
            yield process_scene(task)
        return

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_scene, tasks, chunksize=1):
            yield result
            if stop["now"]:
                break


# Explicit dtypes for the identifier columns. Without them pandas infers types
# per chunk, and `category` is empty for subsets that have no category level --
# so on a 141 MB file one chunk infers str and another infers float-NaN, which
# is the "Columns (1) have mixed types" DtypeWarning.
#
# Default NA handling is kept, because the numeric columns legitimately contain
# empty fields (kept_edges_pct under --skip-edges; precision/recall/iou unless
# --method both) and those must parse as NaN, not as the string "". The id
# columns are then filled back to "" so that grouping on category does not
# silently drop every artifact-subset row.
STATS_ID_COLUMNS = ("subset", "category", "scene", "mode")


def read_stats_csv(path, usecols=None):
    """Read a stats CSV without dtype guessing. Shared by the plotting scripts."""
    import pandas as pd

    dtypes = {c: "string" for c in STATS_ID_COLUMNS
              if usecols is None or c in usecols}
    frame = pd.read_csv(path, dtype=dtypes, usecols=usecols)
    for column in STATS_ID_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def _write_summary(stats_csv: Path, out: Path) -> None:
    """Aggregate the per-fragment CSV into the distribution table."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("pandas not installed; skipping summary")
        return

    frame = read_stats_csv(stats_csv)
    columns = [c for c in ("vertices", "faces", "edges", "frac_vertices", "frac_faces",
                           "frac_edges", "kept_vertices_pct", "kept_faces_pct",
                           "kept_edges_pct", "precision", "recall", "iou")
               if c in frame.columns and frame[c].notna().any()]
    summary = frame[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary.insert(0, "metric", summary.index)
    summary.to_csv(out, index=False)
    print(f"\nsummary over {len(frame)} fragments -> {out}")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:11.3f}"):
        print(summary.to_string(index=False))

    _write_reduction(frame, out.with_name(out.stem + "_reduction" + out.suffix))


def _write_reduction(frame, out: Path) -> None:
    """
    How much geometry the extractor discards, dataset-wide.

    Two numbers per quantity, because they answer different questions and
    differ whenever fragment sizes vary:

    ``per-fragment mean``
        the mean of the per-fragment percentages. Every fragment counts once,
        so a 200-vertex chip weighs as much as a 40,000-vertex body. This is
        the right number for "what does extraction do to a typical fragment".

    ``dataset aggregate``
        ``1 - sum(kept) / sum(original)`` over every fragment at once, i.e.
        weighted by size. This is the right number for "how much smaller does
        my dataset get", which is the one that governs batch memory and
        loading time.

    Reporting only the mean is the common mistake: it is not the fraction of
    the dataset removed, and the two can differ by several points.
    """
    import pandas as pd

    rows = []
    for name in ("vertices", "faces", "edges"):
        kept_pct, frac = f"kept_{name}_pct", f"frac_{name}"
        if kept_pct not in frame.columns or not frame[kept_pct].notna().any():
            continue
        per_fragment = frame[kept_pct].dropna()
        row = {
            "quantity": name,
            "fragments": int(per_fragment.count()),
            "mean_reduction_pct": 100.0 - per_fragment.mean(),
            "median_reduction_pct": 100.0 - per_fragment.median(),
            "p25_reduction_pct": 100.0 - per_fragment.quantile(0.75),
            "p75_reduction_pct": 100.0 - per_fragment.quantile(0.25),
            "worst_reduction_pct": 100.0 - per_fragment.min(),
            "best_reduction_pct": 100.0 - per_fragment.max(),
        }
        if frac in frame.columns and name in frame.columns:
            total, kept_total = frame[name].sum(), frame[frac].sum()
            row["total_original"] = int(total)
            row["total_kept"] = int(kept_total)
            row["aggregate_reduction_pct"] = (
                100.0 * (1.0 - kept_total / total) if total else float("nan"))
        rows.append(row)

    if not rows:
        return

    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("REDUCTION FROM FRACTURE-SURFACE EXTRACTION")
    print("=" * 78)
    header = f"{'':10}{'per-fragment mean':>20}{'median':>12}{'dataset aggregate':>22}"
    print(header)
    for row in rows:
        aggregate = row.get("aggregate_reduction_pct")
        aggregate_text = "-" if aggregate is None else f"{aggregate:.1f}% removed"
        print(f"{row['quantity']:10}"
              f"{row['mean_reduction_pct']:>15.1f}% removed"
              f"{row['median_reduction_pct']:>11.1f}%"
              f"{aggregate_text:>22}")
    print("=" * 78)
    print("per-fragment mean: every fragment weighs the same.")
    print("dataset aggregate: weighted by size -- the fraction of all geometry removed.")
    print(f"\nreduction table -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
