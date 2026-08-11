#!/usr/bin/env python
"""
Object-count scaling sweep -- the experiment that decides what to do next.

    python -m scripts.scaling_sweep --root_dir data --counts 2 8 32 128

Trains the same model, for the same number of optimiser steps, on an
increasing number of distinct objects, and reports the TRAINING rotation error
each run reaches. Validation is ignored on purpose: the question is whether the
model can FIT n objects at all, not whether it generalises.

WHY THIS IS THE DECIDING EXPERIMENT
-----------------------------------
Two hypotheses explain a model that reaches 11.8 deg on 2 objects and sits at
chance on 439, and they call for opposite responses:

  CAPACITY / OPTIMISATION -- error degrades smoothly with object count
  (11.8 -> ~30 -> ~60 -> ~90). More parameters, more steps and a better
  schedule will keep buying progress, so spend the compute.

  STRUCTURAL -- error collapses to chance by 8 or 32 objects. Then no amount
  of training helps, because a single fragment does not determine its own
  canonical orientation once the object is unknown. Most of the Everyday
  subset is surfaces of revolution (bottles, bowls, cups, vases, rings), for
  which the azimuth about the symmetry axis is genuinely unidentifiable from
  one fragment, and a per-fragment canonicaliser floors near 90 deg geodesic
  regardless of capacity. The answer would then be to change the formulation
  -- iterative refinement, where partially-assembled fragments make relative
  orientation meaningful -- not to train longer.

Each run is short by design. Read the trend, not any single number.
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
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        # A fixed rate: a plateau scheduler would confound the comparison by
        # decaying different amounts in different runs.
        "--lr_schedule", "constant",
        "--num_gpus", str(args.num_gpus),
        "--save_every", str(args.epochs),
        "--time_budget_hours", str(args.time_budget_hours),
    ]
    print(f"\n{'=' * 70}\n  {count} objects\n{'=' * 70}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)

    history = out_dir / "history.json"
    if not history.is_file():
        return {"objects": count, "error": "no history written"}
    data = json.loads(history.read_text())
    deg = data.get("train", {}).get("rot_deg", [])
    if not deg:
        return {"objects": count, "error": "no rot_deg recorded"}
    tail = deg[-max(1, len(deg) // 10):]
    return {
        "objects": count,
        "final_train_deg": sum(tail) / len(tail),
        "best_train_deg": min(deg),
        "first_train_deg": deg[0],
        "epochs": len(deg),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="data")
    p.add_argument("--config", default="configs/kaggle_2xt4_full.yaml")
    p.add_argument("--data_subsets", default="everyday_compressed")
    p.add_argument("--out_dir", default="/kaggle/working/scaling")
    p.add_argument("--counts", type=int, nargs="+", default=[2, 8, 32, 128])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps_per_epoch", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_gpus", type=int, default=2)
    p.add_argument("--time_budget_hours", type=float, default=0.5)
    p.add_argument("--out", default="/kaggle/working/scaling/summary.json")
    args = p.parse_args(argv)

    results = [run_one(c, args) for c in args.counts]

    print(f"\n{'=' * 70}\n  SCALING SUMMARY\n{'=' * 70}")
    print(f"  {'objects':>8} {'first':>9} {'best':>9} {'final':>9}")
    for r in results:
        if "error" in r:
            print(f"  {r['objects']:>8} {r['error']}")
            continue
        print(f"  {r['objects']:>8} {r['first_train_deg']:>9.2f} "
              f"{r['best_train_deg']:>9.2f} {r['final_train_deg']:>9.2f}")

    print("\n  reference: chance 126.47 deg | symmetry floor ~90 deg | 2-object run reached 11.8")
    good = [r for r in results if "error" not in r]
    if len(good) >= 2:
        first, last = good[0], good[-1]
        if last["final_train_deg"] > 115:
            print("\n  READ: training error is back at chance by "
                  f"{last['objects']} objects. That points at the structural "
                  "hypothesis -- more capacity or steps will not fix it.")
        elif last["final_train_deg"] < 95:
            print("\n  READ: still fitting well at "
                  f"{last['objects']} objects. Capacity/optimisation story -- "
                  "buy more steps and scale the model.")
        else:
            print("\n  READ: degrading smoothly and approaching the ~90 deg symmetry "
                  "floor. Consistent with the azimuth being unidentifiable; check the "
                  "tilt/twist split in scripts.evaluate to confirm.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
