#!/usr/bin/env python
"""
Object-count scaling sweep.

    python -m scripts.scaling_sweep --root_dir data --counts 2 8 32 \
        --steps_per_object 400 --seeds 0 1 2

THE TWO CONFOUNDS THIS SCRIPT EXISTS TO AVOID
---------------------------------------------
1. EQUAL TOTAL BUDGET. Giving every run the same number of optimiser steps
   means each object receives 1/N of the updates, so error degrades with N by
   arithmetic alone. `--steps_per_object` is the control instead, and total
   steps scale with N -- expensive at large N, which is the honest cost of the
   question.

2. A WALL-CLOCK CAP. `--time_budget_hours` defaults to 0 (no cap) because a cap
   silently truncates exactly the large-N runs and reintroduces (1). A 0.5h cap
   once gave 8 objects 43% and 32 objects 12% of the requested budget. Runs
   that fall short are flagged INVALID in the summary rather than reported as
   if they were comparable.

READING THE RESULT
------------------
Read the between-seed spread FIRST, then the best across seeds at each count.
In the run this script was built for, that spread (73.8 deg at N=8) dwarfed the
difference between counts (7.1 deg), so a single seed per point would have
supported whichever conclusion it happened to land on.

With per-object budget fixed, N is the only deliberate variable:

  * best error FLAT or IMPROVING in N -> a shared shape-to-frame mapping is
    being learned; any plateau at larger N is an optimisation problem.
  * best error RISING steadily with N -> objects are interfering.
  * best error pinned near 126 for N > 2 -> structural: a fragment does not
    determine its own canonical orientation once the object is unknown.

Report `best_train_deg` alongside `final_train_deg`: at a constant learning
rate these runs oscillate by tens of degrees, so the last epoch is a poor
estimate of what a run reached.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(count: int, args, seed: int = 0) -> dict:
    out_dir = Path(args.out_dir) / f"scale_{count:04d}_seed{seed}"
    # Steps scale with object count so every object gets the same number of
    # updates in every run.
    total_steps = args.steps_per_object * count
    epochs = min(args.max_epochs, max(4, total_steps // args.steps_per_epoch))

    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", args.config,
        "--root_dir", args.root_dir,
        "--checkpoint_dir", str(out_dir),
        "--tag", f"scale{count}s{seed}",
        "--resume", "none",
        "--max_scenes", str(count),
        "--data_subsets", args.data_subsets,
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--val_steps", "1",
        "--epochs", str(epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        # Fixed rate: a plateau scheduler would decay by different amounts in
        # different runs and confound the comparison.
        "--lr_schedule", "constant",
        "--lr_warmup_epochs", str(args.lr_warmup_epochs),
        "--num_gpus", str(args.num_gpus),
        "--save_every", str(epochs),
        "--seed", str(seed),
    ]
    if args.time_budget_hours > 0:
        cmd += ["--time_budget_hours", str(args.time_budget_hours)]

    print(f"\n{'=' * 70}\n  {count} objects | seed {seed} | {epochs} epochs | "
          f"{epochs * args.steps_per_epoch} steps "
          f"({epochs * args.steps_per_epoch / count:.0f} per object)\n{'=' * 70}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)

    history = out_dir / "history.json"
    if not history.is_file():
        return {"objects": count, "seed": seed, "error": "no history written"}
    data = json.loads(history.read_text())
    deg = data.get("train", {}).get("rot_deg", [])
    if not deg:
        return {"objects": count, "seed": seed, "error": "no rot_deg recorded"}

    tail = deg[-max(1, len(deg) // 10):]
    # Slope over the last quarter: says whether the run flattened or was simply
    # cut off. Without it a mid-descent number reads as a ceiling.
    tail_q = deg[-max(2, len(deg) // 4):]
    xs = list(range(len(tail_q)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(tail_q) / len(tail_q)
    denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, tail_q)) / denom
    achieved = len(deg) * args.steps_per_epoch / count

    return {
        "objects": count,
        "seed": seed,
        "requested_steps_per_object": args.steps_per_object,
        "steps_per_object": round(achieved, 1),
        "truncated": achieved < 0.95 * args.steps_per_object,
        "first_train_deg": deg[0],
        "best_train_deg": min(deg),
        "final_train_deg": sum(tail) / len(tail),
        "tail_slope_deg_per_epoch": slope,
        "epochs": len(deg),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="data")
    p.add_argument("--config", default="configs/kaggle_2xt4_full.yaml")
    p.add_argument("--data_subsets", default="everyday_compressed")
    p.add_argument("--out_dir", default="/kaggle/working/scaling")
    p.add_argument("--counts", type=int, nargs="+", default=[2, 8, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Repeat every count with these seeds. Between-seed spread has been "
                        "measured at 73.8 deg for a fixed object count, which is larger "
                        "than the effect being measured, so one seed per point cannot "
                        "separate the hypotheses.")
    p.add_argument("--steps_per_object", type=int, default=400)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--steps_per_epoch", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_warmup_epochs", type=int, default=5)
    p.add_argument("--num_gpus", type=int, default=2)
    p.add_argument("--time_budget_hours", type=float, default=0.0,
                   help="0 = no cap. A cap silently truncates the large-N runs.")
    p.add_argument("--out", default="/kaggle/working/scaling/summary.json")
    args = p.parse_args(argv)

    results = [run_one(c, args, s) for c in args.counts for s in args.seeds]

    print(f"\n{'=' * 70}\n  SCALING SUMMARY\n{'=' * 70}")
    print(f"  {'objects':>8} {'seed':>5} {'steps/obj':>10} {'first':>9} {'best':>9} "
          f"{'final':>9} {'tail slope':>11}  flags")
    for r in results:
        if "error" in r:
            print(f"  {r['objects']:>8} {r['seed']:>5} {r['error']}")
            continue
        flags = []
        if r["truncated"]:
            flags.append(f"TRUNCATED ({r['steps_per_object']:.0f} of "
                         f"{r['requested_steps_per_object']} steps/obj)")
        if r["tail_slope_deg_per_epoch"] < -0.3:
            flags.append("still descending")
        print(f"  {r['objects']:>8} {r['seed']:>5} {r['steps_per_object']:>10.0f} "
              f"{r['first_train_deg']:>9.2f} {r['best_train_deg']:>9.2f} "
              f"{r['final_train_deg']:>9.2f} {r['tail_slope_deg_per_epoch']:>+11.2f}  "
              f"{'; '.join(flags)}")

    print("\n  reference: chance 126.47 deg")
    good = [r for r in results if "error" not in r]

    truncated = [r for r in good if r["truncated"]]
    if truncated:
        print(f"\n  INVALID: {len(truncated)} run(s) were truncated before reaching the "
              f"requested per-object budget. Object count and budget are confounded here "
              f"-- re-run with --time_budget_hours 0 before reading anything from it.")

    by_count: dict = {}
    for r in good:
        by_count.setdefault(r["objects"], []).append(r["best_train_deg"])
    spreads = [max(v) - min(v) for v in by_count.values() if len(v) > 1]
    if spreads:
        print(f"\n  between-seed spread at a fixed object count: "
              f"{min(spreads):.1f} to {max(spreads):.1f} deg")
        print("  Any difference across counts smaller than this is not measurable "
              "with these repeats.")
        print(f"\n  {'objects':>8} {'best of seeds':>15} {'median':>9}")
        for n in sorted(by_count):
            v = sorted(by_count[n])
            print(f"  {n:>8} {min(v):>15.2f} {v[len(v) // 2]:>9.2f}")
    elif len(args.seeds) == 1:
        print("\n  NOTE: one seed per point, so there is no estimate of run-to-run noise. "
              "Pass --seeds 0 1 2 before drawing conclusions.")

    still_falling = [r for r in good if r["tail_slope_deg_per_epoch"] < -0.3]
    if still_falling:
        counts = ", ".join(str(r["objects"]) for r in still_falling)
        print(f"\n  CAUTION: {len(still_falling)} run(s) were still descending at the cutoff "
              f"({counts} objects). Those numbers are lower bounds on progress, not "
              f"ceilings -- raise --steps_per_object before treating them as limits.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
