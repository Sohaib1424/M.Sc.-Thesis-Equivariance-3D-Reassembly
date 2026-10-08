#!/usr/bin/env python3
"""
Choose ``--sharp-threshold`` by measurement instead of by eye.

The dihedral heuristic is a stand-in for the thing you cannot compute at
inference: which faces actually mate with another fragment. The coincidence
method *can* compute that, exactly, because Breaking Bad stores fragments in
their assembled frame. So the threshold has a ground truth to be tuned against,
and there is no reason to guess it.

For each candidate threshold this reports, over a sample of real scenes:

``empty``       fraction of fragments the heuristic labels with **no** fracture
                surface. A fragment of a broken object always has some, so any
                non-zero value here is pure heuristic failure.
``precision``   of the faces called fracture, how many really are
``recall``      of the faces that really are, how many were found
``IoU`` / ``F1``  the symmetric combined scores
``F-beta``      recall-weighted (beta = 2 by default). **This is the one to pick
                by** for graph construction: a missed fracture face is an edge
                the model never sees and cannot recover, while a spurious one is
                an edge attention can learn to down-weight. The two errors are
                not symmetric, so F1 is the wrong summary here.
``kept``        mean fraction of faces labelled -- watch this climb as the
                threshold loosens and the mask starts swallowing the smooth
                exterior

Run it on a few hundred scenes; the numbers stabilise long before the full
1,442.

    python -m scripts.tune_sharp_threshold --root data --limit 200 --workers 4

``--min-fracture-faces 1`` repeats the sweep with the per-fragment fallback
enabled, which is the honest way to see what that fallback costs in precision.

The run ends with a verdict that refuses to endorse a reading the sweep does not
support: it says so when the best threshold sits at the edge of the swept range
(the real optimum may be outside it), when precision barely responds to the
threshold (something other than the threshold is capping it), and when the best
F-beta is low enough that a learned segmenter is the better answer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.data.paths import find_scenes  # noqa: E402
from reassembly.data.scene import SceneReader  # noqa: E402
from reassembly.mesh.fracture import (  # noqa: E402
    fracture_face_mask,
    fracture_face_mask_from_vertices,
    fracture_vertex_masks,
)

# Wide on purpose. The useful range runs right up to 1.0 (where every adjacency
# counts as sharp), and a sweep that stops short reports its own last point as
# the optimum -- an answer indistinguishable from a real one. Covering both ends
# costs a few minutes and removes the need for a second run.
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999)


def _score(scene_path, thresholds, tol, max_modes, min_faces):
    """Per-threshold tallies for one scene. Runs in a worker process."""
    reader = SceneReader(scene_path)
    modes = reader.mode_names()[:max_modes]
    acc = {t: dict(tp=0, fp=0, fn=0, kept=0.0, empty=0, n=0) for t in thresholds}

    for mode in modes:
        fragments = reader.load_mode(mode).fragments
        if len(fragments) < 2:
            continue
        truth_vertices = fracture_vertex_masks(fragments, tol=tol)

        for i, fragment in enumerate(fragments):
            V = np.asarray(fragment.vertices)
            F = np.asarray(fragment.faces)
            if F.shape[0] == 0:
                continue
            truth = fracture_face_mask_from_vertices(F, truth_vertices[i])
            for t in thresholds:
                pred = fracture_face_mask(V, F, t, has_duplicate_faces=False,
                                          min_faces=min_faces).face_mask
                a = acc[t]
                a["tp"] += int(np.count_nonzero(pred & truth))
                a["fp"] += int(np.count_nonzero(pred & ~truth))
                a["fn"] += int(np.count_nonzero(~pred & truth))
                a["kept"] += float(pred.mean())
                a["empty"] += int(not pred.any())
                a["n"] += 1
    return acc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep --sharp-threshold against coincidence ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", default="data")
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=200, help="scenes to sample")
    parser.add_argument("--max-modes", type=int, default=3,
                        help="fracture modes per scene (they are highly correlated, "
                             "so a few is plenty)")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--coincidence-tol", type=float, default=1e-6)
    parser.add_argument("--min-fracture-faces", type=int, default=0,
                        help="enable the per-fragment fallback during the sweep")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="F-beta weighting. beta>1 favours recall, which is what\n"
                             "graph construction wants: a missed edge is unrecoverable,\n"
                             "a spurious one is only noise")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)

    thresholds = tuple(args.thresholds or DEFAULT_THRESHOLDS)
    scenes = find_scenes(args.root, args.subsets)
    if not scenes:
        parser.error(f"no scene directories found under {args.root!r}")
    # Evenly spaced through the sorted list rather than the first N, so the
    # sample is not one category.
    if args.limit and args.limit < len(scenes):
        step = len(scenes) / args.limit
        scenes = [scenes[int(i * step)] for i in range(args.limit)]

    print(f"root        {Path(args.root).resolve()}")
    print(f"scenes      {len(scenes)}  (<= {args.max_modes} modes each)")
    print(f"thresholds  {', '.join(str(t) for t in thresholds)}")
    print(f"fallback    min_fracture_faces={args.min_fracture_faces}")
    print()

    totals = {t: dict(tp=0, fp=0, fn=0, kept=0.0, empty=0, n=0) for t in thresholds}
    tasks = [(str(s.path), thresholds, args.coincidence_tol, args.max_modes,
              args.min_fracture_faces) for s in scenes]

    done = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_score, *task) for task in tasks]
            for fut in futures:
                _merge(totals, fut.result())
                done += 1
                if done % 25 == 0:
                    print(f"  [{done}/{len(tasks)}]")
    else:
        for task in tasks:
            _merge(totals, _score(*task))
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(tasks)}]")

    beta = args.beta
    rows = []
    print()
    print("=" * 88)
    print(f"{'threshold':>10} {'empty%':>8} {'precision':>10} {'recall':>8}"
          f" {'IoU':>7} {'F1':>7} {'F' + str(beta):>7} {'kept%':>8}")
    print("=" * 88)
    for t in thresholds:
        a = totals[t]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        b2 = beta * beta
        fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)
                 if b2 * precision + recall else 0.0)
        empty = 100.0 * a["empty"] / a["n"] if a["n"] else 0.0
        kept = 100.0 * a["kept"] / a["n"] if a["n"] else 0.0
        rows.append(dict(threshold=t, fragments=a["n"], empty_pct=empty,
                         precision=precision, recall=recall, iou=iou, f1=f1,
                         fbeta=fbeta, kept_pct=kept))
        print(f"{t:>10} {empty:>8.2f} {precision:>10.4f} {recall:>8.4f}"
              f" {iou:>7.4f} {f1:>7.4f} {fbeta:>7.4f} {kept:>8.2f}")
    print("=" * 88)

    if rows:
        _verdict(rows, thresholds, beta)

    if args.out_csv:
        import csv as _csv
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")
    return 0


def _verdict(rows, thresholds, beta):
    """
    Say what the numbers mean, and refuse to endorse a reading the sweep does
    not actually support.
    """
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_fb = max(rows, key=lambda r: r["fbeta"])

    print(f"\nbest F1  at --sharp-threshold {best_f1['threshold']}"
          f"   (F1 {best_f1['f1']:.4f}, precision {best_f1['precision']:.3f},"
          f" recall {best_f1['recall']:.3f})")
    print(f"best F{beta}  at --sharp-threshold {best_fb['threshold']}"
          f"   (F{beta} {best_fb['fbeta']:.4f}, precision {best_fb['precision']:.3f},"
          f" recall {best_fb['recall']:.3f})")

    print("\nFor building an attention graph, prefer the F"
          f"{beta} choice over the F1 one.")
    print("  A missed fracture face is an edge the model never sees, and no amount")
    print("  of training recovers it. A spurious one is an edge attention can learn")
    print(f"  to down-weight. F{beta} weights recall accordingly. (If you use")
    print("  mode='sample', precision matters more than this rule suggests: false")
    print("  positives consume slots in a fixed token budget.)")

    warnings = []

    lo, hi = thresholds[0], thresholds[-1]
    if best_fb["threshold"] in (lo, hi) and len(rows) >= 3:
        # An optimum at the edge only matters if the metric is still *moving*
        # there. If the last three points agree to within 0.5% the curve has
        # flattened, the edge is a plateau rather than a cut-off, and extending
        # the sweep would find nothing.
        tail = [r["fbeta"] for r in rows[-3:]] if best_fb["threshold"] == hi \
            else [r["fbeta"] for r in rows[:3]]
        if max(tail) - min(tail) > 0.005:
            warnings.append(
                f"The best threshold sits at the EDGE of the swept range "
                f"({lo}-{hi}) and F{beta} is still climbing there, so the real "
                f"optimum is probably outside it. Re-run with --thresholds "
                f"extended in that direction.")

    precisions = [r["precision"] for r in rows]
    if max(precisions) - min(precisions) < 0.10:
        warnings.append(
            f"Precision is nearly flat across the whole sweep "
            f"({min(precisions):.3f}-{max(precisions):.3f}). The threshold is not "
            f"what is limiting it -- some faces are being labelled for a reason no "
            f"threshold controls (the rim effect: a flat interface meeting the "
            f"fragment's outer wall makes genuinely sharp edges there). Tuning "
            f"will not fix that ceiling; a learned segmenter would.")

    if any(r["empty_pct"] > 0 for r in rows):
        worst = max(r["empty_pct"] for r in rows)
        warnings.append(
            f"Up to {worst:.2f}% of fragments get an EMPTY mask at some threshold, "
            f"which is impossible for a real fragment. Pass --min-fracture-faces 1 "
            f"to the extraction run; it relaxes only the affected fragments.")

    if best_fb["fbeta"] < 0.80:
        warnings.append(
            f"Best F{beta} is {best_fb['fbeta']:.3f}. Below ~0.8 the graph is being "
            f"built on labels that are wrong for a large minority of faces, and "
            f"every downstream result inherits that. Consider training a small "
            f"segmenter on the coincidence labels, as GARF does.")

    if warnings:
        print("\n" + "-" * 88)
        for w in warnings:
            print(f"! {w}\n")
    else:
        print("\nNo caveats: the optimum is interior, precision responds to the "
              "threshold, and no fragment comes out empty.")


def _merge(totals, part):
    for t, a in part.items():
        for key in a:
            totals[t][key] += a[key]


if __name__ == "__main__":
    raise SystemExit(main())
