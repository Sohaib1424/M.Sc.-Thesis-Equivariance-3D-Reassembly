#!/usr/bin/env python3
"""
Print the report ``--evaluate`` printed, from the metrics file it wrote.

    python -m scripts.report_metrics eval/W10-artifact/val_metrics_matched.json
    python -m scripts.report_metrics eval/W10-artifact eval/W10-artifact-vc
    python -m scripts.report_metrics "eval/*/val_metrics_matched.json"

A folder means every ``*_metrics*.json`` in it; a pattern is expanded here, so
it works without a shell (``subprocess.run`` with a list, Windows).

Every report is also saved beside its file as ``<split>_report.txt`` or
``<split>_report_matched.txt`` (``--no_save`` to skip) -- a few kilobytes,
where the metrics file of a full split runs to tens of megabytes. Evaluations
from this version on write that file themselves; this script is for the ones
that did not, and for reading a report again without re-running anything.

The text is built by the same function ``--evaluate`` prints with
(:func:`reassembly.training.format_evaluation`), so the numbers cannot drift
from the file. A file written before the report's header was saved prints
every number, without the checkpoint and fragment-range lines.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reassembly.training import format_evaluation  # noqa: E402


def metrics_files(arguments: List[str]) -> List[Path]:
    """Files, folders and patterns to the metrics files they name, in order, once each."""
    found: List[Path] = []
    for argument in arguments:
        if any(character in argument for character in "*?["):
            matches = [Path(p) for p in sorted(glob.glob(argument))]
        elif Path(argument).is_dir():
            matches = sorted(Path(argument).glob("*_metrics*.json"))
        else:
            matches = [Path(argument)]
        if not matches:
            print(f"[report] nothing matches {argument}")
        for path in matches:
            if path not in found:
                found.append(path)
    return found


def report_path(metrics: Path) -> Path:
    """``val_metrics_matched.json`` -> ``val_report_matched.txt``, beside it."""
    stem = metrics.stem
    name = stem.replace("_metrics", "_report", 1) if "_metrics" in stem else stem + "_report"
    return metrics.with_name(name + ".txt")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="metrics files (<split>_metrics*.json), folders holding them, "
                             "or patterns")
    parser.add_argument("--no_save", action="store_true",
                        help="print only; do not write <split>_report*.txt beside each file")
    args = parser.parse_args(argv)

    files = metrics_files(args.paths)
    failed = 0
    for path in files:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"\n== {path}\n  cannot read it: {error}")
            failed += 1
            continue
        report = format_evaluation(summary)
        print(f"\n== {path}\n{report}")
        if not args.no_save:
            target = report_path(path)
            target.write_text(report + "\n", encoding="utf-8")
            print(f"\n  saved {target}")
    if not files:
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
