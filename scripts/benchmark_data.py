#!/usr/bin/env python
"""
Measure where data-loading time actually goes, and whether it will bottleneck
training.

    python -m scripts.benchmark_data --root_dir data --num_scenes 40 --num_workers 2

Prints a per-stage breakdown for single-scene loading, then the throughput of
a real DataLoader. The number to watch is the last line: with prefetching, an
epoch costs `max(data_time / workers, compute_time)`, not their sum, so data
loading only hurts once it exceeds compute.

This exists because throughput assumptions are the easiest thing in this
project to get wrong by an order of magnitude -- measure before planning a
multi-day run.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()


from vngat.data.correspondence import compute_scene_correspondence  # noqa: E402
from vngat.data.dataset import BreakingBadDataset  # noqa: E402
from vngat.data.features import get_features  # noqa: E402
from vngat.data.io import load_random_scene  # noqa: E402
from vngat.data.mesh_ops import extract_fractures  # noqa: E402
from vngat.data.splits import get_random_directory  # noqa: E402
from vngat.utils.env import dataloader_worker_init, seed_everything  # noqa: E402


def stage_breakdown(args) -> None:
    print("=== per-stage cost of one scene (median over samples) ===")
    timings = {"load_random_scene": [], "get_features": [], "correspondence": [],
               "extract_fractures": []}
    sizes = []
    for _ in range(args.num_scenes):
        scene_dir = get_random_directory(
            args.root_dir, split="train", max_scenes=args.max_scenes,
        )
        t = time.perf_counter()
        meshes = load_random_scene(str(scene_dir), fracture_pattern=args.fracture_pattern or None)
        timings["load_random_scene"].append(time.perf_counter() - t)
        if len(meshes) < 2:
            continue

        t = time.perf_counter()
        frags = [get_features(m) for m in meshes]
        timings["get_features"].append(time.perf_counter() - t)

        t = time.perf_counter()
        compute_scene_correspondence(meshes)
        timings["correspondence"].append(time.perf_counter() - t)

        t = time.perf_counter()
        _ = [extract_fractures(m) for m in meshes]
        timings["extract_fractures"].append(time.perf_counter() - t)

        sizes.append((sum(f.num_nodes for f in frags), sum(f.num_edges for f in frags), len(meshes)))

    total = 0.0
    for name, values in timings.items():
        if not values:
            continue
        med = statistics.median(values)
        note = "  (only when input_source=frac)" if name == "extract_fractures" else ""
        print(f"  {name:<22} {med * 1000:8.1f} ms{note}")
        if name != "extract_fractures":
            total += med
    print(f"  {'--> full-mesh total':<22} {total * 1000:8.1f} ms/scene "
          f"({1 / max(total, 1e-9):.1f} scenes/s single-threaded)")

    if sizes:
        nodes = [s[0] for s in sizes]
        edges = [s[1] for s in sizes]
        frags = [s[2] for s in sizes]
        print(f"\n  scene size: nodes  median {statistics.median(nodes):,.0f}  max {max(nodes):,}")
        print(f"              edges  median {statistics.median(edges):,.0f}  max {max(edges):,}")
        print(f"              frags  median {statistics.median(frags):.0f}  max {max(frags)}")


def loader_throughput(args) -> None:
    """
    Sequential throughput of the real dataset object.

    There is no framework DataLoader here: the TF branch builds batches
    directly, so this measures exactly what training will pay. Multiply by
    `num_workers` for the parallel figure -- the stages are independent per
    scene.
    """
    import time

    print(f"\n=== dataset throughput (batch_size={args.batch_size}) ===")
    dataset = BreakingBadDataset(
        root_dir=args.root_dir, split="train", max_scenes=args.max_scenes,
        fracture_pattern=args.fracture_pattern or None, input_source=args.input_source,
        nominal_length=args.num_scenes)
    start = time.perf_counter()
    scenes = 0
    for i in range(args.num_scenes // max(args.batch_size, 1)):
        for _ in range(args.batch_size):
            dataset[scenes]
            scenes += 1
    elapsed = time.perf_counter() - start
    print(f"  {scenes} scenes in {elapsed:.1f}s  ->  {scenes / max(elapsed, 1e-9):.2f} scenes/s")
    print(f"  {elapsed / max(scenes, 1) * args.batch_size:.3f} s per batch of {args.batch_size},"
          f" single-threaded")
    print(f"  with {args.num_workers} workers, expect roughly "
          f"{elapsed / max(scenes, 1) * args.batch_size / max(args.num_workers, 1):.3f} s/batch")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--num_scenes", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--fracture_pattern", type=str, default="fractured_")
    p.add_argument("--input_source", type=str, default="full", choices=["full", "frac"])
    p.add_argument("--skip_stages", action="store_true")
    args = p.parse_args(argv)

    seed_everything(0)
    if not args.skip_stages:
        stage_breakdown(args)
    loader_throughput(args)


if __name__ == "__main__":
    main()
