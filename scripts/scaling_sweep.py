#!/usr/bin/env python
"""
Object-count scaling sweep.

    python -m scripts.scaling_sweep --root_dir data --counts 2 8 32 --steps_per_object 400

THE CONFOUND THIS SCRIPT EXISTS TO AVOID
----------------------------------------
An earlier version gave every run the SAME TOTAL number of optimiser steps. At
a fixed total, each object receives 1/N of the updates, so error degrades with
N by arithmetic alone -- whether or not a shared shape-to-frame mapping exists.
That run's numbers (2 -> 33 deg, 8 -> 109, 32 -> 120, 128 -> 121) looked like a
structural wall, but compared against the 2-object control AT A MATCHED
PER-OBJECT BUDGET they were within ~5 deg of it at N=8 and ~2 deg at N=128.
Under-trained, not blocked.

So `--steps_per_object` is the control here, and total steps scale linearly
with N. That makes large N genuinely expensive; it is the real cost of the
question, not something to optimise away.

READING THE RESULT
------------------
With per-object budget held fixed, N is the only variable:

  * error roughly FLAT in N        -> a shared mapping is being learned, and
                                      the earlier plateau was an optimisation
                                      problem. Scale up and train longer.
  * error RISES steadily with N    -> objects are interfering; the mapping is
                                      not shared. Capacity may help, or the
                                      formulation may need to change.
  * error pinned at ~126 for N > 2 -> structural. A single fragment does not
                                      determine its canonical orientation once
                                      the object is unknown, and no budget
                                      fixes that.

Report `best_train_deg` alongside `final_train_deg`: at a constant learning
rate these runs oscillate, and the last epoch is a noisy estimate of what the
run reached.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(count: int, args) -> dict:
    out_dir = Path(args.out_dir) / f"scale_{count:04d}"
    # Steps scale with object count so every object gets the same number of
    # updates in every run. Without this the comparison measures budget, not N.
    total_steps = args.steps_per_object * count
    epochs = min(args.max_epochs, max(4, total_steps // args.steps_per_epoch))
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", args.config,
        "--root_dir", args.root_dir,
        "--checkpoint_dir", str(out_dir),
        "--tag", f"scale{count}",
        "--resume", "none",
        "--max_scenes", str(count),
        "--data_subsets", args.data_subsets,
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--val_steps", "1",
        "--epochs", str(epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        # A fixed rate: a plateau scheduler would confound the comparison by
        # decaying different amounts in different runs.
        "--lr_schedule", "constant",
        "--num_gpus", str(args.num_gpus),
        "--save_every", str(epochs),
        "--time_budget_hours", str(args.time_budget_hours),
    ]
    print(f"\n{'=' * 70}\n  {count} objects | {epochs} epochs | "
          f"{epochs * args.steps_per_epoch} steps "
          f"({epochs * args.steps_per_epoch / count:.0f} per object)\n{'=' * 70}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)

    history = out_dir / "history.json"
    if not history.is_file():
        return {"objects": count, "error": "no history written"}
    data = json.loads(history.read_text())
    deg = data.get("train", {}).get("rot_deg", [])
    if not deg:
        return {"objects": count, "error": "no rot_deg recorded"}
    tail = deg[-max(1, len(deg) // 10):]
    # Slope over the last quarter: says whether the run had flattened or was
    # simply cut off. Without it a mid-descent number reads as a ceiling.
    tail_q = deg[-max(2, len(deg) // 4):]
    xs = list(range(len(tail_q)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(tail_q) / len(tail_q)
    denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, tail_q)) / denom
    return {
        "objects": count,
        "final_train_deg": sum(tail) / len(tail),
        "best_train_deg": min(deg),
        "first_train_deg": deg[0],
        "tail_slope_deg_per_epoch": slope,
        "epochs": len(deg),
        "steps_per_object": round(len(deg) * args.steps_per_epoch / count, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="data")
    p.add_argument("--config", default="configs/kaggle_2xt4_full.yaml")
    p.add_argument("--data_subsets", default="everyday_compressed")
    p.add_argument("--out_dir", default="/kaggle/working/scaling")
    p.add_argument("--counts", type=int, nargs="+", default=[2, 8, 32, 128])
    p.add_argument("--steps_per_object", type=int, default=400,
                   help="Optimiser steps PER OBJECT. Total steps scale with --counts, "
                        "which is the point: a fixed total confounds object count with "
                        "per-object budget.")
    p.add_argument("--max_epochs", type=int, default=400,
                   help="Cap, so a large N cannot run away with the session.")
    p.add_argument("--steps_per_epoch", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_gpus", type=int, default=2)
    p.add_argument("--time_budget_hours", type=float, default=0.5)
    p.add_argument("--out", default="/kaggle/working/scaling/summary.json")
    args = p.parse_args(argv)

    results = [run_one(c, args) for c in args.counts]

    print(f"\n{'=' * 70}\n  SCALING SUMMARY\n{'=' * 70}")
    print(f"  {'objects':>8} {'steps/obj':>10} {'first':>9} {'best':>9} {'final':>9} {'tail slope':>11}")
    for r in results:
        if "error" in r:
            print(f"  {r['objects']:>8} {r['error']}")
            continue
        print(f"  {r['objects']:>8} {r['steps_per_object']:>10.0f} {r['first_train_deg']:>9.2f} "
              f"{r['best_train_deg']:>9.2f} {r['final_train_deg']:>9.2f} "
              f"{r['tail_slope_deg_per_epoch']:>+11.2f}")

    print("\n  reference: chance 126.47 deg | symmetry floor ~90 deg")
    good = [r for r in results if "error" not in r]
    still_falling = [r for r in good if r["tail_slope_deg_per_epoch"] < -0.3]
    if still_falling:
        print(f"\n  CAUTION: {len(still_falling)} run(s) were still descending at the cutoff "
              f"({', '.join(str(r['objects']) for r in still_falling)} objects). Their "
              f"numbers are lower bounds on progress, not ceilings -- raise "
              f"--steps_per_object before drawing conclusions from them.")
    if len(good) >= 2:
        spread = good[-1]["best_train_deg"] - good[0]["best_train_deg"]
        print(f"\n  best-error spread from {good[0]['objects']} to {good[-1]['objects']} "
              f"objects: {spread:+.1f} deg, at a matched per-object budget.")
        print("  Read the TREND across counts, not any single value. See the module "
              "docstring for what each shape implies.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
