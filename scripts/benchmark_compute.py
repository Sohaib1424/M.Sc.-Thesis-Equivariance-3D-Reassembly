#!/usr/bin/env python
"""
Measure the numbers the thesis's central claim rests on: throughput, memory,
and cost-per-epoch, on the hardware you actually have.

    python scripts/benchmark_compute.py --config configs/kaggle_t4x2.yaml
    python scripts/benchmark_compute.py --config configs/default.yaml --sweep-budget

The claim being tested is "competitive accuracy without datacenter GPUs, in
substantially less training time". That has two halves, and only one of them is
measured by a loss curve:

  * ACCURACY comes from ``scripts/evaluate.py`` (GARF-comparable metrics).
  * COST comes from here -- parameters, peak memory, seconds per step, scenes
    per second, and the extrapolated wall-clock hours for a full run.

GARF's published reference point is 4x H100 for 72 hours. The honest comparison
is total GPU-hours and the class of hardware, both stated explicitly, plus a
note that the two systems are not trained on identical schedules. Reporting
"our model is smaller" without wall-clock numbers is not the claim this thesis
is making.

``--sweep-budget`` traces the accuracy/compute knob directly: throughput and
memory as a function of the vertex budget, which is what makes the
decimation-vs-accuracy trade-off a curve rather than an anecdote.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from reassembly.config import Config
from reassembly.models.vn_gat_model import VNGATModel
from reassembly.utils.memory import AmpContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_memory import synthetic_scene  # noqa: E402


def time_steps(model, batch, amp: AmpContext, optimizer, device, steps=12, warmup=3):
    """Median seconds per optimizer step. Median, not mean: the first few steps
    include allocator warm-up and cuDNN autotuning, and one slow outlier
    shifts a mean enough to change conclusions."""
    times = []
    for i in range(steps + warmup):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            out = model(**batch)
        loss = out["R_pred"].float().square().sum() + \
            out["vertex_embedding"].float().square().sum()
        amp.backward(loss)
        amp.step(optimizer, grad_clip=5.0, parameters=model.parameters())

        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_point(cfg, nodes, fragments, batch_scenes, device):
    model = VNGATModel(
        hidden_channels=cfg.model.hidden_channels, num_layers=cfg.model.num_layers,
        num_vn_slots=cfg.model.num_vn_slots, heads=cfg.model.heads,
        head_dim=cfg.model.head_dim, embed_dim=cfg.model.embed_dim,
        norm=cfg.model.norm, gradient_checkpointing=cfg.model.gradient_checkpointing,
        angular=cfg.model.angular,
    ).to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
    amp = AmpContext(cfg.train.amp, device_type=device.type, dtype=cfg.train.amp_dtype)

    # One "batch" of B scenes is a single concatenated graph, which is exactly
    # how collation presents it -- so B scenes of N nodes is one graph of B*N.
    batch = synthetic_scene(nodes * batch_scenes, fragments * batch_scenes, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        sec = time_steps(model, batch, amp, optimizer, device)
        peak = (torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else float("nan"))
        status = "ok"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        sec, peak, status = float("nan"), float("nan"), "OOM"

    params = model.num_parameters()
    del model, batch, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(seconds_per_step=sec, peak_gib=peak, status=status, params=params)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sweep-budget", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()
    device = torch.device(args.device)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    world = max(cfg.train.num_gpus, 1)
    print(f"device: {gpu_name} x{world}   torch {torch.__version__}")
    print(f"model : hidden={cfg.model.hidden_channels} layers={cfg.model.num_layers} "
          f"heads={cfg.model.heads} slots={cfg.model.num_vn_slots} "
          f"ckpt={cfg.model.gradient_checkpointing} amp={cfg.train.amp}")
    print()

    budgets = ([2000, 5000, 10000, 20000, 40000] if args.sweep_budget
               else [cfg.data.decimate_to or 20000])

    rows = []
    header = (f"{'vertices/scene':>15} {'batch':>6} {'s/step':>9} {'scenes/s':>9} "
              f"{'peak GiB':>9} {'h/epoch':>9} {'status':>7}")
    print(header)
    print("-" * len(header))

    for nodes in budgets:
        r = run_point(cfg, nodes, args.fragments, cfg.train.batch_size, device)
        sec = r["seconds_per_step"]
        scenes_per_s = (cfg.train.batch_size / sec) if sec == sec else float("nan")
        # One logged epoch = steps_per_epoch + val_steps, PER RANK; ranks run
        # concurrently, so wall-clock per epoch is the per-rank time.
        steps = cfg.train.steps_per_epoch + cfg.train.val_steps
        hours = (sec * steps / 3600.0) if sec == sec else float("nan")
        print(f"{nodes:>15,} {cfg.train.batch_size:>6} {sec:>9.4f} {scenes_per_s:>9.2f} "
              f"{r['peak_gib']:>9.3f} {hours:>9.3f} {r['status']:>7}")
        rows.append(dict(vertices=nodes, batch=cfg.train.batch_size, **r,
                         scenes_per_second=scenes_per_s, hours_per_epoch=hours))

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = ok[-1]
        total_h = best["hours_per_epoch"] * cfg.train.epochs
        gpu_hours = total_h * world
        print()
        print(f"parameters                     : {ok[0]['params']:,}")
        print(f"projected wall-clock, {cfg.train.epochs} epochs: "
              f"{total_h:.2f} h on {world}x {gpu_name}")
        print(f"projected total GPU-hours      : {gpu_hours:.2f}")
        print(f"GARF reference                 : 4x H100 x 72 h = 288 H100-GPU-hours")
        print()
        print("NOTE: this measures COMPUTE COST only, on synthetic graphs of the given\n"
              "size -- it excludes data loading (real mesh work, often the actual\n"
              "bottleneck; the training loop prints a data= vs compute= split per epoch)\n"
              "and says nothing about accuracy. Pair it with scripts/evaluate.py, and\n"
              "state both halves together in any comparison table.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"gpu": gpu_name, "world_size": world,
                       "config": cfg.to_dict(), "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
