#!/usr/bin/env python3
"""
Figures and tables from training histories and evaluation results, into one
folder (and a zip of it) to download.

    python -m scripts.make_figures --out figures/W10 \\
        --history W10=checkpoint/history.json \\
        --eval "Everyday=eval/W10-full/val_metrics_matched.json" \\
        --eval "Everyday VC=eval/W10-everyday-vc/val_metrics_matched.json" \\
        --eval "Artifact=eval/W10-artifact/val_metrics_matched.json" \\
        --eval "Artifact VC=eval/W10-artifact-vc/val_metrics_matched.json"

``--history`` and ``--eval`` repeat, each as ``LABEL=PATH`` (or just ``PATH``:
the label is then the folder the file is in). A history is a run's
``history.json`` or a checkpoint (``.pt``, needs torch); an evaluation is the
``<split>_metrics*.json`` that ``python -m scripts.train --evaluate`` writes.
Up to eight of each -- the order you give is the colour order, the same in
every figure.

What comes out (``<out>/README.md`` lists every file)
------------------------------------------------------
training/<run>/   total loss; the four geometric terms, one panel each;
                  the embedding term beside match@1; the rotation head's
                  error against chance; its error per category and epoch.
training/         with two or more runs: every run on one set of axes,
                  training and validation (for ablations).
evaluation/       per-piece rotation error by object type and dataset, as
                  box plots and as violins (matched rotations on a log
                  scale, the rotation head's on a linear one); the share of
                  pieces within each error threshold; part accuracy by
                  object type; scores against pieces per scene.
tables/           the numbers behind every figure, as CSV, and a summary
                  of each evaluation in CSV and Markdown.

Notes
-----
* The rotation head's per-piece spread needs ``_network_geodesic_deg`` in a
  matched evaluation's file, written from this version on; older matched
  files give the matched spread only (the head's means are in the tables).
* Per-piece numbers come from the scene records an evaluation writes when it
  assembles (the default); a ``--no_assemble`` file gives the summary only.
* Only numpy and matplotlib are needed, so the files can be copied off the
  training machine and drawn anywhere.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUMMARY_COLUMNS = (
    ("dataset", "dataset", ""),
    ("placement", "placement", ""),
    ("scenes", "scenes", "d"),
    ("scored_pieces", "pieces scored", "d"),
    ("network_mean_deg", "head mean (°)", ".2f"),
    ("network_median_deg", "head median (°)", ".2f"),
    ("matched_mean_deg", "matched mean (°)", ".2f"),
    ("matched_median_deg", "matched median (°)", ".2f"),
    ("matched_acc@5", "acc@5", ".3f"),
    ("reached", "placed by matching", ".0%"),
    ("part_accuracy", "part accuracy", ".3f"),
    ("rmse_t", "RMSE(T)", ".4f"),
    ("chamfer", "CD", ".5f"),
    ("scene_euler_rmse_deg", "Euler RMSE (°)", ".2f"),
)


def labelled(text: str) -> Tuple[Optional[str], str]:
    """``LABEL=PATH`` -> ``(LABEL, PATH)``; a bare path has no label."""
    label, separator, path = text.partition("=")
    if not separator:
        return None, text.strip()
    return (label.strip() or None), path.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", action="append", default=[], metavar="[LABEL=]PATH",
                        help="a run's history.json, or a checkpoint (.pt); repeat for "
                             "several runs")
    parser.add_argument("--eval", action="append", default=[], metavar="[LABEL=]PATH",
                        help="an evaluation's <split>_metrics*.json; repeat for several "
                             "datasets")
    parser.add_argument("--out", default="figures", help="output folder (default: figures)")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        help="file formats per figure (default: png pdf)")
    parser.add_argument("--dpi", type=int, default=200, help="for raster formats")
    parser.add_argument("--smooth", type=int, default=1,
                        help="moving-average window, in epochs, for the training curves "
                             "(1: none; the raw curve stays faintly behind)")
    parser.add_argument("--no_zip", action="store_true",
                        help="do not write <out>.zip beside the folder")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.history and not args.eval:
        print("nothing to draw: give at least one --history or --eval")
        return 2

    import matplotlib

    matplotlib.use("Agg")                  # files only; no display needed

    from reassembly.viz import figures as fg
    from reassembly.viz.results import (ROTATION_SOURCES, assembly_by_type, by_piece_count,
                                        errors_by_type, load_evaluation, load_history,
                                        markdown_table, summary_row, type_order, write_csv)

    histories = []
    for text in args.history:
        label, path = labelled(text)
        histories.append(load_history(path, label))
        print(f"history     {histories[-1].label:<20} {path}  "
              f"({len(histories[-1].rows)} epochs)")
    evaluations = []
    for text in args.eval:
        label, path = labelled(text)
        evaluations.append(load_evaluation(path, label))
        evaluation = evaluations[-1]
        sources = ", ".join(evaluation.sources()) or "summary only"
        print(f"evaluation  {evaluation.label:<20} {path}  "
              f"({len(evaluation.scenes['type'])} scenes; per piece: {sources})")
    for items, what in ((histories, "--history"), (evaluations, "--eval")):
        if len(items) > fg.MAX_SERIES:
            print(f"{len(items)} {what}: at most {fg.MAX_SERIES} fit one figure's colours")
            return 2
        seen = set()
        for item in items:                 # a label names one colour and one folder
            base, count = item.label, 1
            while item.label in seen:
                count += 1
                item.label = f"{base} ({count})"
            seen.add(item.label)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = fg.FigureWriter(out, formats=args.formats, dpi=args.dpi)
    skipped: List[str] = []

    for history in histories:
        fg.plot_total_loss(history, writer, args.smooth)
        fg.plot_loss_terms(history, writer, args.smooth)
        fg.plot_embedding(history, writer, args.smooth)
        fg.plot_rotation_error(history, writer, args.smooth)
        fg.plot_category_heatmap(history, writer, smooth=max(args.smooth, 5))
    if len(histories) > 1:
        for split in ("val", "train"):
            fg.plot_comparison(histories, split, writer, args.smooth)

    if evaluations:
        for source in ROTATION_SOURCES:
            if not any(source in e.sources() for e in evaluations):
                if source == "network" and any(e.rotations == "matched" for e in evaluations):
                    skipped.append("the rotation head's per-piece spread: these matched "
                                   "files predate _network_geodesic_deg (re-run --evaluate "
                                   "with this version to get it)")
                continue
            for kind in ("box", "violin"):
                fg.plot_errors_by_type(evaluations, source, writer, kind=kind)
            rows = errors_by_type([e for e in evaluations if source in e.sources()], source)
            write_csv(out / "tables" / f"errors_by_type_{source}.csv", rows)
            writer.note(f"tables/errors_by_type_{source}.csv",
                        f"Per type and dataset: count, mean, quartiles, 5th/95th percentiles "
                        f"and acc@5/10/30 of the per-piece error ({fg.SOURCE_NAMES[source]}).")
        fg.plot_accuracy_curves(evaluations, writer)
        fg.plot_part_accuracy(evaluations, writer)
        fg.plot_by_piece_count(evaluations, writer)

        source = ("matched" if any("matched" in e.sources() for e in evaluations)
                  else "network")
        order = type_order(evaluations, source)
        write_csv(out / "tables" / "assembly_by_type.csv", assembly_by_type(evaluations, order))
        writer.note("tables/assembly_by_type.csv",
                    "Per type and dataset: scenes, part accuracy, RMSE(T), Chamfer, per-scene "
                    "geodesic and Euler RMSE.")
        write_csv(out / "tables" / "by_piece_count.csv",
                  [row for e in evaluations for row in by_piece_count(e)])
        writer.note("tables/by_piece_count.csv",
                    "Per pieces-per-scene range and dataset: scenes, part accuracy, median "
                    "errors, share placed by the matching.")
        rows = [summary_row(e) for e in evaluations]
        write_csv(out / "tables" / "summary.csv", rows)
        (out / "tables" / "summary.md").write_text(
            markdown_table(rows, SUMMARY_COLUMNS) + "\n", encoding="utf-8")
        writer.note("tables/summary.csv, tables/summary.md",
                    "The headline numbers of every evaluation, one row each.")

    _write_index(out, args, histories, evaluations, writer, skipped)
    print(f"\nwrote {len(writer.written)} entries under {out}  (index: {out / 'README.md'})")
    for note in skipped:
        print(f"  skipped {note}")
    if not args.no_zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=str(out.parent),
                                      base_dir=out.name)
        print(f"  zipped to {archive}")
    return 0


def _write_index(out: Path, args, histories, evaluations, writer, skipped) -> None:
    lines = ["# Figures", "",
             f"Made by `python -m scripts.make_figures` on "
             f"{datetime.date.today().isoformat()} from:", ""]
    lines += [f"- run **{h.label}**: `{h.path}`" for h in histories]
    lines += [f"- dataset **{e.label}**: `{e.path}`" for e in evaluations]
    lines += ["", "Colours follow the order above, in every figure.", "",
              "| file | what it shows |", "|---|---|"]
    lines += [f"| {name} | {description} |" for name, description in writer.written]
    if skipped:
        lines += ["", "Not drawn:", ""] + [f"- {note}" for note in skipped]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
