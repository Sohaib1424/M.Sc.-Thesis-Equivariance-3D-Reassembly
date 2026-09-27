#!/usr/bin/env python3
"""
How does training error move with the number of distinct OBJECTS?

    python -m scripts.scaling_sweep --root_dir data --counts 2 8 32 --seeds 0 1 2 \\
        --steps_per_object 400 --checkpoint_dir runs/scaling

Each (count, seed) is a separate ``python -m scripts.train`` run on the first
``count`` training objects (``--max_objects``, spread across the catalogue),
and the sweep reads their histories. Any other training flag
(``--num_gpus``, ``--batch_size``, ``--hidden_channels`` ...) is passed through
to every run. Steps are optimizer steps, as in ``--steps_per_epoch``.

The two confounds this exists to avoid
--------------------------------------
1. **Equal total budget.** Giving every run the same number of steps gives each
   object 1/N of them, so error rises with N by arithmetic alone.
   ``--steps_per_object`` is held fixed instead and total steps grow with N --
   expensive at large N, which is the honest price of the question.
2. **A wall-clock cap.** A cap silently truncates exactly the large-N runs and
   reintroduces (1). None is set unless you pass ``--time_budget_hours``, and a run
   that fell short of its budget is marked TRUNCATED and excluded from the
   verdict rather than reported as comparable.

The learning rate is constant by default (``--lr_schedule constant``): a decaying
schedule reaches its floor at different points in runs of different lengths.

Reading the result
------------------
Read the between-seed spread first: in the run this was built for, it (73.8 deg
at N=8) dwarfed the difference between counts (7.1 deg), so one seed per point
would have supported whichever conclusion it landed on. Then, per count, the
best over seeds of ``best_train_deg`` -- at a constant rate the curves
oscillate, so the last epoch is a poor estimate of what a run reached.

* flat or falling with N -> a shared shape-to-frame rule is being learned;
* rising steadily with N -> objects interfere;
* pinned near 126.5 deg for N > 2 -> a fragment does not determine its
  orientation once the object is unknown.

This reads TRAINING error on purpose: the question is capacity, not
generalisation. Validation is kept to ``--val_steps`` x ``--batch_size`` scenes
(8 by default) to stay cheap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.config_flags import (add_config_arguments, config_flags,  # noqa: E402
                                  config_from_args)

# Set by the sweep itself, per run; never passed through from the command line:
# the flags, and the Config fields they set.
_OWNED_FLAGS = {"max_objects", "steps_per_epoch", "epochs", "seed", "out_dir", "resume",
                "lr_warmup_epochs", "val_steps", "max_hours"}
_OWNED = {"max_objects", "steps_per_epoch", "epochs", "seed", "out_dir", "resume",
          "resume_from", "warmup_fraction", "limit_val", "max_hours"}


def _slope(values) -> float:
    """Least-squares slope, degrees per epoch."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x, mean_y = (n - 1) / 2.0, sum(values) / n
    denominator = sum((x - mean_x) ** 2 for x in range(n)) or 1.0
    return sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values)) / denominator


def run_one(count: int, seed: int, args, passthrough) -> dict:
    out_dir = Path(args.out_dir) / f"objects{count:04d}_seed{seed}"
    total = args.steps_per_object * count
    epochs = max(4, min(args.max_epochs, -(-total // args.steps_per_epoch)))
    warmup = (args.lr_warmup_epochs if args.lr_warmup_epochs is not None
              else 0.03 * epochs)
    command = [
        sys.executable, "-m", "scripts.train", *passthrough,
        "--max_objects", str(count), "--steps_per_epoch", str(args.steps_per_epoch),
        "--epochs", str(epochs), "--seed", str(seed), "--checkpoint_dir", str(out_dir),
        "--resume", "none", "--val_steps", str(args.val_steps),
        "--lr_warmup_epochs", repr(warmup),
    ]
    if args.max_hours:
        command += ["--time_budget_hours", str(args.max_hours)]
    print(f"\n{'=' * 72}\n  {count} objects | seed {seed} | {epochs} epochs x "
          f"{args.steps_per_epoch} steps ({epochs * args.steps_per_epoch / count:.0f} "
          f"per object)\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=False)

    history_file = out_dir / "history.json"
    if not history_file.is_file():
        return {"objects": count, "seed": seed, "error": "no history written"}
    history = json.loads(history_file.read_text())
    curve = [row["train_rotation_degrees"] for row in history
             if row.get("train_rotation_degrees") is not None]
    if not curve:
        return {"objects": count, "seed": seed, "error": "no rotation error recorded"}
    # Optimizer steps, the unit of --steps_per_object ("steps" counts passes).
    steps = sum(row.get("train_steps", row.get("steps", 0)) for row in history
                if not row.get("partial"))
    achieved = steps / count
    tail = curve[-max(1, len(curve) // 10):]
    return {
        "objects": count, "seed": seed, "epochs": len(curve),
        "requested_steps_per_object": args.steps_per_object,
        "steps_per_object": round(achieved, 1),
        "truncated": achieved < 0.95 * args.steps_per_object,
        "first_train_deg": curve[0], "best_train_deg": min(curve),
        "final_train_deg": sum(tail) / len(tail),
        "tail_slope_deg_per_epoch": _slope(curve[-max(2, len(curve) // 4):]),
    }


def summarise(results, seeds) -> None:
    print(f"\n{'=' * 72}\n  SCALING SUMMARY   (chance 126.47 deg)\n{'=' * 72}")
    print(f"  {'objects':>8} {'seed':>5} {'steps/obj':>10} {'first':>8} {'best':>8} "
          f"{'final':>8} {'slope':>7}  flags")
    for r in results:
        if "error" in r:
            print(f"  {r['objects']:>8} {r['seed']:>5}  {r['error']}")
            continue
        flags = (["TRUNCATED"] if r["truncated"] else []) + (
            ["still descending"] if r["tail_slope_deg_per_epoch"] < -0.3 else [])
        print(f"  {r['objects']:>8} {r['seed']:>5} {r['steps_per_object']:>10.0f} "
              f"{r['first_train_deg']:>8.2f} {r['best_train_deg']:>8.2f} "
              f"{r['final_train_deg']:>8.2f} {r['tail_slope_deg_per_epoch']:>+7.2f}  "
              f"{', '.join(flags)}")
    good = [r for r in results if "error" not in r and not r["truncated"]]
    truncated = [r for r in results if r.get("truncated")]
    if truncated:
        print(f"\n  INVALID: {len(truncated)} run(s) stopped short of the per-object budget "
              f"and are left out below -- rerun them without --time_budget_hours.")
    by_count = {}
    for r in good:
        by_count.setdefault(r["objects"], []).append(r["best_train_deg"])
    spreads = [max(v) - min(v) for v in by_count.values() if len(v) > 1]
    if spreads:
        print(f"\n  between-seed spread at one count: {min(spreads):.1f} to {max(spreads):.1f} "
              f"deg -- a difference across counts smaller than this is not measurable here.")
    elif len(seeds) == 1:
        print("\n  one seed per count: there is no estimate of run-to-run noise yet. "
              "Pass --seeds 0 1 2 before concluding anything.")
    if by_count:
        print(f"\n  {'objects':>8} {'best of seeds':>14} {'median':>8}")
        for count in sorted(by_count):
            values = sorted(by_count[count])
            print(f"  {count:>8} {values[0]:>14.2f} {values[len(values) // 2]:>8.2f}")
    falling = [r for r in good if r["tail_slope_deg_per_epoch"] < -0.3]
    if falling:
        print(f"\n  CAUTION: {len(falling)} run(s) were still descending at the end -- lower "
              f"bounds on progress, not ceilings. Raise --steps_per_object.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps_per_object", type=int, default=400)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--sweep_summary", default="",
                        help="where to write the JSON summary (default: "
                             "<checkpoint_dir>/summary.json)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    # The sweep's own defaults for what it owns. A decaying rate reaches its
    # floor at different points in runs of different lengths, so constant.
    args.steps_per_epoch = args.steps_per_epoch or 20
    args.out_dir = args.out_dir or "runs/scaling"
    args.val_steps = args.val_steps if args.val_steps is not None else 4
    args.lr_schedule = args.lr_schedule or "constant"
    config = config_from_args(argparse.Namespace(
        **{k: v for k, v in vars(args).items() if k not in _OWNED_FLAGS}))
    passthrough = config_flags(config, fields={f for f in vars(config)} - _OWNED)

    results = [run_one(count, seed, args, passthrough)
               for count in args.counts for seed in args.seeds]
    summarise(results, args.seeds)
    out = Path(args.sweep_summary or Path(args.out_dir) / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
