#!/usr/bin/env python3
"""
Where the data pipeline's time goes -- measured on this machine, on the real data.

    python -m scripts.benchmark_data --root_dir /kaggle/input/breaking-bad --scenes 24
    python -m scripts.benchmark_data --root_dir ... --scenes 24 --throughput 30 --num_workers 2

Three measurements, all medians over real training scenes:

1. **Per stage, per scene** -- reading and decompressing the mesh, the fracture
   labels, the coincidence clusters, and ``build_scene`` (tokens, topology,
   features), so a slow epoch can be traced to the stage that makes it slow.
2. **Where the batch is finished, three ways** -- the perturbed copy and the
   cross-fragment pair lists made in the loader's worker (the old path), the
   copy made on the device, and both made on the device (the default on a
   GPU). Reported as loader time per batch, bytes per batch, and the device's
   own time to finish the batch -- on a GPU, synchronised, so the number is
   what the training step actually pays. This is the measurement behind
   ``Config.perturb_on_device``; run it on the GPU the training will use.
3. **Loader throughput** -- a real ``DataLoader`` with ``--num_workers`` processes,
   batches per second. With prefetching an epoch costs
   ``max(loading, compute)``, not the sum: loading only slows training once a
   batch takes longer to build than to train on.

Every training flag is accepted (``--tokens_per_scene``, ``--label_method``,
...), so the measured pipeline is the one the run will use.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.config_flags import add_config_arguments, config_from_args  # noqa: E402


def _median_ms(values) -> str:
    return f"{1000 * statistics.median(values):8.1f} ms" if values else "       -   "


def _bytes(batch) -> int:
    import torch

    return sum(v.element_size() * v.numel() for v in batch._asdict().values()
               if torch.is_tensor(v))


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def stage_breakdown(dataset, indices, config):
    """Time each stage of building one scene. Returns (items, fixed seconds):
    the usable item indices, and the per-scene time of the stages that do not
    depend on where the batch is finished (read, labels, clusters)."""
    from reassembly.data.features import build_scene

    timings = {"read + decompress": [], "fracture labels": [],
               "coincidence clusters": [], "build_scene": []}
    usable, sizes = [], []
    for i in indices:
        start = time.perf_counter()
        meshes = dataset.meshes(i)
        timings["read + decompress"].append(time.perf_counter() - start)
        if len(meshes) < 2:
            continue
        start = time.perf_counter()
        masks = dataset.labels(meshes)
        timings["fracture labels"].append(time.perf_counter() - start)
        start = time.perf_counter()
        cluster = dataset.clusters(meshes)
        timings["coincidence clusters"].append(time.perf_counter() - start)
        start = time.perf_counter()
        sample = build_scene(
            [np.asarray(m.vertices, dtype=np.float64) for m in meshes],
            [np.asarray(m.faces) for m in meshes], masks,
            rng=np.random.default_rng(i), normalize_mode=config.normalize_mode,
            token_mode=config.token_mode, token_metric=config.token_metric,
            tokens_per_scene=config.tokens_per_scene, cluster=cluster,
            key=dataset.key(i), perturb=True)
        timings["build_scene"].append(time.perf_counter() - start)
        usable.append(i)
        sizes.append((sum(len(f.target_vertices) for f in sample.fragments),
                      len(sample.fragments)))

    print("=== per stage, one scene (median) ===")
    total = 0.0
    for name, values in timings.items():
        print(f"  {name:<22}{_median_ms(values)}")
        total += statistics.median(values) if values else 0.0
    print(f"  {'total':<22}{1000 * total:8.1f} ms   "
          f"(~{1 / max(total, 1e-9):.1f} scenes/s per worker process, before collating)")
    if sizes:
        vertices = [s[0] for s in sizes]
        fragments = [s[1] for s in sizes]
        print(f"  scene size: vertices median {int(np.median(vertices)):,}  max {max(vertices):,}"
              f"   fragments median {int(np.median(fragments))}  max {max(fragments)}")
    fixed = sum(statistics.median(timings[k]) for k in
                ("read + decompress", "fracture labels", "coincidence clusters")
                if timings[k])
    return usable, fixed


def device_side(dataset, usable, fixed: float, config, device: str) -> None:
    """
    Loader time, bytes and device time per batch, three ways.

    Loader time = the fixed stages (read, labels, clusters: identical in all
    three) + ``build_scene`` + ``collate``, which are what differ.
    """
    import torch

    from reassembly.data.features import build_scene, collate, complete_batch

    size = config.batch_size
    groups = [usable[k:k + size] for k in range(0, len(usable) - size + 1, size)]
    if not groups:
        print("\n(not enough scenes for one batch -- raise --scenes)")
        return
    inputs = {}
    for i in usable:
        meshes = dataset.meshes(i)
        inputs[i] = ([np.asarray(m.vertices, dtype=np.float64) for m in meshes],
                     [np.asarray(m.faces) for m in meshes], dataset.labels(meshes),
                     dataset.clusters(meshes))

    ways = [("before: copy + pairs in the loader", True, True),
            ("copy on the device", False, True),
            ("copy + pairs on the device", False, False)]
    print(f"\n=== where the batch is finished (batches of {size}, device {device}) ===")
    base = None
    for name, perturb, pairs in ways:
        loader_s, sizes, device_s = [], [], []
        for group in groups:
            start = time.perf_counter()
            samples = [build_scene(*inputs[i][:3], rng=np.random.default_rng(i),
                                   normalize_mode=config.normalize_mode,
                                   token_mode=config.token_mode,
                                   token_metric=config.token_metric,
                                   tokens_per_scene=config.tokens_per_scene,
                                   cluster=inputs[i][3], key=dataset.key(i),
                                   perturb=perturb)
                       for i in group]
            batch = collate(samples, pairs=pairs)
            loader_s.append(time.perf_counter() - start + fixed * len(group))
            sizes.append(_bytes(batch))
            moved = type(batch)(**{k: (v.to(device) if torch.is_tensor(v) else v)
                                   for k, v in batch._asdict().items()})
            _sync(device)
            start = time.perf_counter()
            complete_batch(moved)
            _sync(device)
            device_s.append(time.perf_counter() - start)
        row = (statistics.median(loader_s), statistics.median(sizes),
               statistics.median(device_s))
        base = base or row
        print(f"  {name:<36} loader {1000 * row[0]:8.1f} ms ({100 * row[0] / base[0]:5.1f}%)"
              f"   {row[1] / 1e6:8.2f} MB ({100 * row[1] / base[1]:5.1f}%)"
              f"   device {1000 * row[2]:7.2f} ms")
    if not device.startswith("cuda"):
        print("  (no GPU: the device column is CPU time. On a CPU run that is the wrong trade,")
        print("   which is why CPU training keeps the pair lists in the loader's workers.)")


def throughput(dataset, config, batches: int) -> None:
    from reassembly.training import _loader

    import torch

    loader = _loader(dataset, config, True, 0, 1, 0,
                     pairs_in_worker=not torch.cuda.is_available())
    print(f"\n=== loader throughput: {config.workers} worker(s), "
          f"{config.batch_size} scene(s) per batch (--micro_batch_scenes) ===")
    start, count, scenes = time.perf_counter(), 0, 0
    for batch, _dropped in loader:
        count += 1
        scenes += 0 if batch is None else batch.num_scenes
        if count >= batches:
            break
    elapsed = time.perf_counter() - start
    print(f"  {count} batches / {scenes} scenes in {elapsed:.1f} s -> "
          f"{count / max(elapsed, 1e-9):.2f} batches/s ({elapsed / max(count, 1):.2f} s/batch)")
    print("  Loading only slows training once s/batch here exceeds the training step's own"
          " time (preflight step 7 measures that).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=24,
                        help="training scenes to time, spread across the split")
    parser.add_argument("--throughput", type=int, default=20,
                        help="batches to pull through a real DataLoader (0 to skip)")
    parser.add_argument("--device", default=None,
                        help="where to finish batches (default: cuda if available)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    import torch

    from reassembly.training import BreakingBadScenes

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = BreakingBadScenes(config, "train", epoch_seed=0)
    indices = np.linspace(0, len(dataset) - 1, min(args.scenes, len(dataset))).round().astype(int)
    print(f"{len(dataset):,} training scenes; timing {len(indices)} spread across the split\n")
    usable, fixed = stage_breakdown(
        dataset, [int(i) for i in dict.fromkeys(indices.tolist())], config)
    device_side(dataset, usable, fixed, config, device)
    if args.throughput:
        throughput(dataset, config, args.throughput)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
