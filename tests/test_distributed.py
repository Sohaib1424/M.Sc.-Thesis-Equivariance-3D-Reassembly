"""
Two and three real processes (gloo, CPU), each one "GPU".

What these pin, against a single-process reference computed step by step:

* **the step is the exact fragment-weighted mean over every GPU** -- including
  when GPUs contribute different numbers of fragments, and when one of them
  drops a batch (unusable, too large, out of memory on every attempt);
* **the replicas stay bit-identical**, which the old path broke by rescaling
  each GPU's gradient by its *own* fragment count after the all-reduce;
* **no GPU can leave another waiting** -- the old path's rank-local skips made
  one GPU call the gradient all-reduce fewer times than its peer, so the last
  one of the epoch never completed. Every run here has a timeout, and a hang
  fails the test instead of stalling the suite;
* **a stop on one GPU stops all of them at the same step**;
* **the epoch summary covers every GPU's data**, and validation scores each
  scene exactly once.

``train()`` itself is run end to end on two and three CPU processes at the
bottom of the file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist                     # noqa: E402
import torch.multiprocessing as mp                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

TIMEOUT = 240.0

# GPU -> the fragment counts of the scenes it draws, one scene per batch. The
# GPUs hold different totals on purpose: equal weighting per GPU is exactly
# what goes wrong when they do.
LAYOUTS = {
    0: [3, 2, 4, 2],
    1: [5, 2, 3, 3],
    2: [2, 4, 2, 5],
}


def _scene(seed: int, fragments: int):
    import trimesh

    from reassembly.data.features import build_scene
    from reassembly.data.transforms import random_rotations

    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(fragments):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0 - 0.1 * i)
        v = np.asarray(mesh.vertices).copy() + [2.2 * i, 0.0, 0.0]
        v[rng.choice(len(v), 8, replace=False)] *= 1.2
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), 12, replace=False)] = True
        vertices.append(v)
        faces.append(np.asarray(mesh.faces))
        masks.append(mask)
    total = sum(len(v) for v in vertices)
    cluster = np.full(total, -1, np.int64)
    cluster[:12] = np.arange(12) // 3
    return build_scene(vertices, faces, masks,
                       rotations=random_rotations(fragments, np.random.default_rng(seed + 1)),
                       tokens_per_scene=24, cluster=cluster, key=f"scene{seed}")


def _scenes(rank: int):
    return [_scene(100 * rank + i, n) for i, n in enumerate(LAYOUTS[rank])]


def _config(**overrides):
    from reassembly.training import Config

    base = dict(channels=8, heads=2, head_dim=2, embedding_dim=4,
                schedule=("intra", "cross", "intra"), workers=0, batch_size=1,
                accumulate=2, grad_clip=1e9, lr=0.1, warmup_fraction=0.0,
                min_lr_fraction=1.0, steps_per_epoch=0)
    base.update(overrides)
    return Config(**base)


class _Listed(torch.utils.data.Dataset):
    def __init__(self, items, unusable=()):
        self.items, self.unusable = items, set(unusable)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from reassembly.training import Skipped

        if i in self.unusable:
            return Skipped(f"unusable{i}", "made unusable by the test")
        return self.items[i]


def _loader(items, unusable=()):
    from reassembly.training import _collate_samples

    return torch.utils.data.DataLoader(_Listed(items, unusable), batch_size=1,
                                       collate_fn=_collate_samples)


def _run(world: int, target, *args):
    """Spawn ``world`` processes on ``target``; fail, not hang, past TIMEOUT."""
    context = mp.spawn(target, args=(world, *args), nprocs=world, join=False)
    deadline = time.time() + TIMEOUT
    while not context.join(timeout=1.0):
        if time.time() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(f"{world} processes did not finish in {TIMEOUT:.0f}s -- "
                        f"a GPU is waiting on a collective its peer never called")


def _join(rank: int, world: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)


# --------------------------------------------------------------------------
# The step, on every GPU
# --------------------------------------------------------------------------

def _train_one_epoch(rank, world, init_file, out_dir, case):
    """One GPU: seed differently, broadcast, run one real epoch, save weights."""
    import unittest.mock as mock

    import reassembly.training as training
    from reassembly import distributed as rd

    _join(rank, world, init_file)
    try:
        config = _config(max_vertices_per_batch=(1 if case.get("skip_all") == rank else 0))
        torch.manual_seed(1000 + rank)                 # different on purpose
        model = training.build_model(config)
        rd.broadcast_parameters(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

        unusable = case.get("unusable", {}).get(rank, ())
        oom_scene = case.get("oom", {}).get(rank)
        real = training._forward

        def forward(model_, batch, criterion, config_):
            if oom_scene is not None and f"scene{oom_scene}" in batch.scene_keys:
                raise torch.cuda.OutOfMemoryError("simulated, every attempt")
            return real(model_, batch, criterion, config_)

        with mock.patch.object(training, "_forward", forward):
            summary, step, stopped = training.run_epoch(
                model, _loader(_scenes(rank), unusable), training.build_criterion(config),
                config, optimizer=optimizer, step=0, total_steps=100, label="train",
                show_progress=False, distributed=True)
        differ = rd.parameters_differ(model)
        torch.save({"state": model.state_dict(), "summary": summary, "step": step,
                    "stopped": stopped, "differ": differ},
                   Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _reference(world, case):
    """
    The same epoch in one process, the slow and obvious way: for each step, the
    gradient summed over every GPU's contributing batches, divided by their
    total fragment count, then plain SGD.
    """
    import reassembly.training as training

    config = _config()
    torch.manual_seed(1000)                            # rank 0's initialisation
    model = training.build_model(config)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    criterion = training.build_criterion(config)
    scenes = {r: _scenes(r) for r in range(world)}
    positions = len(LAYOUTS[0])
    for start in range(0, positions, config.accumulate):
        optimizer.zero_grad(set_to_none=True)
        fragments = 0
        for rank in range(world):
            if case.get("skip_all") == rank:
                continue
            for position in range(start, min(start + config.accumulate, positions)):
                if position in case.get("unusable", {}).get(rank, ()):
                    continue
                if case.get("oom", {}).get(rank) == 100 * rank + position:
                    continue
                batch, _ = training._collate_samples([scenes[rank][position]])
                loss, _, _ = training._forward(model, batch, criterion, config)
                (loss * batch.num_fragments).backward()
                fragments += batch.num_fragments
        if fragments:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= fragments
            optimizer.step()
    return initial, model.state_dict()


CASES = {
    "everything contributes": {},
    "one GPU has an unusable scene": {"unusable": {1: (0,)}},
    "one GPU runs out of memory on every attempt": {"oom": {0: 1}},
    "one GPU skips every batch": {"skip_all": 1},
}


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("name", list(CASES))
def test_the_step_is_the_global_mean_and_the_replicas_stay_identical(tmp_path, world, name):
    case = CASES[name]
    _run(world, _train_one_epoch, str(tmp_path / "init"), str(tmp_path), case)
    results = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False)
               for r in range(world)]

    assert all(r["differ"] is None for r in results), results[0]["differ"]
    for other in results[1:]:
        for key, value in results[0]["state"].items():
            assert torch.equal(value, other["state"][key]), f"replicas differ in {key}"

    # Against the reference, relative to how far the weights actually moved:
    # two steps at lr 0.1 amplify float32 round-off from the first step into
    # ~1e-4 of the update by the second (the one-step comparison agrees to
    # ~1e-7). A wrong weighting -- per-GPU means, or a GPU's own fragment count
    # applied after the all-reduce -- is off by tens of percent of the update.
    initial, reference = _reference(world, case)
    for key, value in reference.items():
        moved = float((value - initial[key]).abs().max())
        off = float((results[0]["state"][key] - value).abs().max())
        assert off <= 1e-3 * moved + 1e-6, (
            f"{name}: {key} is not the global fragment-weighted mean step "
            f"(off by {off:.3g} on an update of {moved:.3g})")

    # Every GPU reports the same, whole-epoch summary.
    summaries = [r["summary"] for r in results]
    assert all(s["fragments"] == summaries[0]["fragments"] for s in summaries)
    expected = sum(n for r in range(world) for n in LAYOUTS[r])
    if name == "everything contributes":
        assert summaries[0]["fragments"] == expected
    assert all(r["step"] == len(LAYOUTS[0]) for r in results), "step counts batches"


# --------------------------------------------------------------------------
# Stopping
# --------------------------------------------------------------------------

class _TriggerAfter:
    """A stop signal that fires on one GPU only, after ``batches`` forwards."""

    def __init__(self, batches):
        self.batches, self.seen = batches, 0

    @property
    def triggered(self):
        return self.seen >= self.batches


def _stop_on_one(rank, world, init_file, out_dir):
    import unittest.mock as mock

    import reassembly.training as training
    from reassembly import distributed as rd

    _join(rank, world, init_file)
    try:
        config = _config(accumulate=1)
        torch.manual_seed(0)
        model = training.build_model(config)
        rd.broadcast_parameters(model)
        signal = _TriggerAfter(2 if rank == world - 1 else 10 ** 9)
        real = training._forward

        def counting(model_, batch, criterion, config_):
            signal.seen += 1
            return real(model_, batch, criterion, config_)

        with mock.patch.object(training, "_forward", counting):
            _summary, step, stopped = training.run_epoch(
                model, _loader(_scenes(rank)), training.build_criterion(config), config,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1), step=0,
                total_steps=100, label="train", show_progress=False,
                stop_signal=signal, distributed=True)
        (Path(out_dir) / f"rank{rank}.json").write_text(
            json.dumps({"step": step, "stopped": stopped}))
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 3])
def test_a_stop_on_one_gpu_stops_every_gpu_at_the_same_step(tmp_path, world):
    """
    The stop signal used to be installed on rank 0 only, and the time budget
    was checked rank by rank: a GPU that stopped left the others blocked in
    their next all-reduce. Now each GPU only votes, at the step.
    """
    _run(world, _stop_on_one, str(tmp_path / "init"), str(tmp_path))
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]
    assert all(r["stopped"] for r in results)
    assert len({r["step"] for r in results}) == 1
    assert results[0]["step"] == 2


# --------------------------------------------------------------------------
# The gradient all-reduce itself
# --------------------------------------------------------------------------

def _sum_gradients(rank, world, init_file, out_dir):
    from reassembly import distributed as rd

    _join(rank, world, init_file)
    try:
        a = torch.nn.Parameter(torch.zeros(3))
        b = torch.nn.Parameter(torch.zeros(2))
        c = torch.nn.Parameter(torch.zeros(4))
        a.grad = torch.full((3,), float(rank + 1))
        if rank == 0:
            b.grad = torch.ones(2)          # only one GPU produced it
        rd.sum_gradients([a, b, c])
        torch.save({"a": a.grad, "b": b.grad, "c": c.grad}, Path(out_dir) / f"g{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_a_gradient_no_gpu_produced_stays_absent(tmp_path, world):
    """
    ``None`` must mean the same on every GPU: AdamW skips a parameter whose
    grad is None -- no decay, no moment update -- so a GPU that stepped a
    parameter its peer skipped would drift from it for the rest of the run.
    """
    _run(world, _sum_gradients, str(tmp_path / "init"), str(tmp_path))
    for r in range(world):
        g = torch.load(tmp_path / f"g{r}.pt")
        assert torch.equal(g["a"], torch.full((3,), float(sum(range(1, world + 1)))))
        assert torch.equal(g["b"], torch.ones(2)), "one GPU's gradient reaches all"
        assert g["c"] is None, "nobody's gradient stays nobody's"


# --------------------------------------------------------------------------
# train() end to end, on CPU processes
# --------------------------------------------------------------------------

def _mug_dataset(root: Path, objects: int = 6, modes: int = 2) -> Path:
    from test_training_recovery import _dataset

    return _dataset(root, objects=objects, modes=modes)


def _train_config(root: Path, out: Path, **overrides):
    from reassembly.training import Config

    base = dict(root=str(root), out_dir=str(out), channels=8, heads=2, head_dim=2,
                embedding_dim=4, workers=0, tokens_per_scene=16, batch_size=1,
                epochs=2, modes_per_scene=None, schedule=("intra", "cross"),
                check_init=False, val_frac=0.34, test_frac=0.0,
                steps_per_epoch=3, checkpoint_every_minutes=0.0)
    base.update(overrides)
    return Config(**base)


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 3])
def test_train_runs_end_to_end_on_several_processes(tmp_path, world, monkeypatch):
    """
    The whole launcher -- spawn, broadcast, fixed-length epochs, the per-step
    reductions, sharded validation gathered to one summary, rank-0-only
    checkpoints -- on ``world`` CPU processes. Then a resume, which must pick
    up at the next epoch with the step count it left at.
    """
    from reassembly.training import train

    monkeypatch.setenv("MASTER_PORT", str(__import__("reassembly.distributed",
                                                      fromlist=["free_port"]).free_port()))
    root = _mug_dataset(tmp_path / "data")
    out = tmp_path / "out"
    history = train(_train_config(root, out, devices=world, epochs=1))
    assert len(history) == 1
    row = history[0]
    assert row["gpus"] == world
    assert row["steps"] == 3 and row["step"] == 3
    assert row["train_steps"] == 3, "one optimizer step per batch at accumulate=1"
    # Validation: every val scene once, whichever GPU scored it.
    from reassembly.training import BreakingBadScenes

    val = BreakingBadScenes(_train_config(root, out), "val")
    assert row["val_batches"] + row["val_skipped"] == len(val)
    assert np.isfinite(row["val_geodesic_deg"])
    assert (out / "last.pt").exists() and (out / "best.pt").exists()

    monkeypatch.setenv("MASTER_PORT", str(__import__("reassembly.distributed",
                                                      fromlist=["free_port"]).free_port()))
    history = train(_train_config(root, out, devices=world, epochs=2))
    assert [h["epoch"] for h in history] == [0, 1]
    assert history[1]["step"] == 6
