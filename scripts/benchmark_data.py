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

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vngat.config import Config  # noqa: E402
from vngat.data.correspondence import compute_scene_correspondence  # noqa: E402
from vngat.data.dataset import BreakingBadDataset, collate_fn, dataset_kwargs  # noqa: E402
from vngat.data.features import get_features  # noqa: E402
from vngat.data.graph import merge_fragments  # noqa: E402
from vngat.data.io import load_scene  # noqa: E402
from vngat.data.mesh_ops import extract_fractures  # noqa: E402
from vngat.utils.env import dataloader_worker_init, seed_everything  # noqa: E402


def _config(args) -> Config:
    """The data definition: a training YAML if given, then the flags."""
    cfg = Config.from_yaml(args.config) if args.config else Config()
    for name in ("root_dir", "data_subsets", "split_source", "max_scenes",
                 "fracture_pattern", "input_source"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    return cfg


def stage_breakdown(args) -> None:
    import numpy as np

    print("=== per-stage cost of one scene (median over samples) ===")
    cfg = _config(args)
    objects = BreakingBadDataset(split="train", **dataset_kwargs(cfg)).objects
    timings = {"load_scene": [], "get_features": [], "correspondence": [],
               "extract_fractures": []}
    sizes = []
    delta_norms = []
    rng = np.random.default_rng(0)
    for _ in range(args.num_scenes):
        entry = objects[int(rng.integers(len(objects)))]
        scene_dir, mode = entry.modes[int(rng.integers(len(entry.modes)))]
        t = time.perf_counter()
        meshes = load_scene(scene_dir, mode)
        timings["load_scene"].append(time.perf_counter() - t)
        if len(meshes) < 2:
            continue

        t = time.perf_counter()
        frags = [get_features(m) for m in meshes]
        scene = merge_fragments(frags, cfg.normalize_mode)
        timings["get_features"].append(time.perf_counter() - t)
        # How long an edge is next to the unit face normals it travels with.
        delta_norms.append(float(scene.edge_attr[:, 2].norm(dim=-1).median()))

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
    if delta_norms:
        print(f"\n  edge relative position |p_u - p_v| ({cfg.normalize_mode}-normalised): "
              f"median {statistics.median(delta_norms):.4f}")
        print("  (the two face normals it travels with have norm 1; the layer has to learn")
        print("   any gain it needs between the two, so a very small value is worth knowing)")


def loader_throughput(args) -> None:
    print(f"\n=== DataLoader throughput (num_workers={args.num_workers}, "
          f"batch_size={args.batch_size}) ===")
    dataset = BreakingBadDataset(split="train", nominal_length=args.num_scenes,
                                 **dataset_kwargs(_config(args)))
    kwargs = dict(
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, drop_last=True,
        worker_init_fn=dataloader_worker_init, pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    loader = DataLoader(dataset, **kwargs)

    n_batches = n_scenes = 0
    start = time.perf_counter()
    for batch in loader:
        n_batches += 1
        n_scenes += batch["num_scenes"]
    elapsed = time.perf_counter() - start

    print(f"  {n_batches} batches / {n_scenes} scenes in {elapsed:.1f}s")
    print(f"  {n_scenes / max(elapsed, 1e-9):.2f} scenes/s   "
          f"{n_batches / max(elapsed, 1e-9):.2f} batches/s")
    print(f"  {elapsed / max(n_batches, 1):.3f} s per batch of {args.batch_size}")
    print("\n  With prefetching an epoch costs max(data, compute), not data + compute.")
    print("  Data loading only bottlenecks once the per-batch time above exceeds")
    print("  the per-batch compute time reported in the training table.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="Training YAML to take the data definition from.")
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--data_subsets", type=str, default=None)
    p.add_argument("--split_source", type=str, default=None, choices=["hash", "official"])
    p.add_argument("--num_scenes", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--fracture_pattern", type=str, default=None)
    p.add_argument("--input_source", type=str, default=None, choices=["full", "frac"])
    p.add_argument("--skip_stages", action="store_true")
    args = p.parse_args(argv)

    seed_everything(0)
    if not args.skip_stages:
        stage_breakdown(args)
    loader_throughput(args)


if __name__ == "__main__":
    main()
