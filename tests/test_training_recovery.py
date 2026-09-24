"""
Two silent failures in the epoch loop, pinned.

Both were live in this codebase. Neither raises, neither shows up in a loss
curve, and both corrupt the run in a way that looks like a model that will not
learn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

import reassembly.training as training
from reassembly.training import BreakingBadScenes, Config, build_criterion


def _dataset(root: Path, objects: int = 6, modes: int = 2) -> Path:
    from scipy.sparse import identity, save_npz

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])
    for index in range(objects):
        directory = (root / "everyday_compressed" / "everyday_compressed"
                     / "Mug" / f"mug_{index}")
        directory.mkdir(parents=True)
        mesh.export(directory / "compressed_mesh.obj")
        save_npz(directory / "compressed_data.npz",
                 identity(len(vertices), format="csr"))
        for mode in range(modes):
            pieces = 2 + mode
            mode_dir = directory / f"fractured_{mode}"
            mode_dir.mkdir()
            labels = np.floor((theta + np.pi) / (2 * np.pi) * pieces)
            np.save(mode_dir / "compressed_fracture.npy",
                    np.clip(labels, 0, pieces - 1).astype(np.int64))
    return root


@pytest.fixture
def root(tmp_path):
    return _dataset(tmp_path / "data")


def _config(root, **kwargs):
    defaults = dict(
        root=str(root), out_dir=str(Path(root).parent / "out"),
        channels=16, heads=4, head_dim=4, embedding_dim=8, workers=0,
        tokens_per_scene=32, batch_size=1, accumulate=1, epochs=1, modes_per_scene=None,
        schedule=("intra",), check_init=False, val_frac=0.34, test_frac=0.0,
        steps_per_epoch=0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _model_and_loader(config, batches=None):
    torch.manual_seed(0)
    model = training.build_model(config)
    dataset = BreakingBadScenes(config, "train")
    if batches is not None:
        dataset.items = dataset.items[:batches * config.batch_size]
    loader = training._loader(dataset, config, shuffle=False, rank=0, world=1,
                              epoch=0)
    return model, loader


# --------------------------------------------------------------------------
# 1. A non-finite loss must not reach backward()
# --------------------------------------------------------------------------

def test_a_nonfinite_loss_never_reaches_backward(root, monkeypatch):
    """
    The check used to run *after* ``scaled.backward()``, where it was
    decorative: by the time it fired, NaN was already in ``.grad`` for every
    parameter, and on a syncing micro-batch DDP had already all-reduced it to
    the peer. The counter said "skipped" while the optimizer state said
    otherwise -- and Adam's moments never recover from NaN, so the run
    continues for hours producing nothing.
    """
    config = _config(root)
    model, loader = _model_and_loader(config, batches=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    real = training._forward

    def poisoned(model_, batch, criterion, config_):
        loss, report, R = real(model_, batch, criterion, config_)
        return loss * float("nan"), report, R

    monkeypatch.setattr(training, "_forward", poisoned)
    summary, _step, _stopped = training.run_epoch(
        model, loader, build_criterion(config), config, optimizer=optimizer,
        device="cpu", label="train", show_progress=False,
    )

    assert summary["nonfinite"] == 3, "every batch should have been caught"
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
        assert torch.isfinite(parameter).all(), f"{name} was updated with NaN"


def test_a_nonfinite_batch_is_counted_and_named(root, monkeypatch):
    """A count alone hides which samples fail. If the same ones fail every
    epoch they are effectively excluded from training, and that has to be
    visible rather than inferred from a total that never moves."""
    config = _config(root)
    model, loader = _model_and_loader(config, batches=2)
    real = training._forward
    calls = {"n": 0}

    def sometimes(model_, batch, criterion, config_):
        loss, report, R = real(model_, batch, criterion, config_)
        calls["n"] += 1
        return (loss * float("inf") if calls["n"] == 1 else loss), report, R

    monkeypatch.setattr(training, "_forward", sometimes)
    summary, _s, _t = training.run_epoch(
        model, loader, build_criterion(config), config,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device="cpu", label="train", show_progress=False,
    )
    assert summary["nonfinite"] == 1
    assert any(name.startswith("nonfinite:") for name in summary["dropped_names"])


# --------------------------------------------------------------------------
# 2. An OOM must leave the accumulation group as if the batch never existed
# --------------------------------------------------------------------------

class _Positions(torch.utils.data.Dataset):
    """The real dataset, with chosen positions made unusable."""

    def __init__(self, dataset, unusable=()):
        self.dataset, self.unusable = dataset, set(unusable)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if i in self.unusable:
            return training.Skipped(f"unusable{i}", "made unusable by the test")
        return self.dataset[i]


def _gradients_at_every_step(config, items, oom=(), unusable=()):
    """
    Run one epoch over ``items`` and return the gradient the optimizer was
    handed at every step, in order.

    Captured at ``clip_grad_norm_``, which the loop calls right after dividing
    by the contributing fragments and before the step -- so this is exactly the
    quantity the normalisation is supposed to produce. ``oom`` positions run out
    of memory on every attempt, the checkpointed retry included.
    """
    import unittest.mock as mock

    torch.manual_seed(0)
    model = training.build_model(config)
    dataset = BreakingBadScenes(config, "train")
    dataset.items = [dataset.items[i] for i in items]
    doomed = {f"{dataset.catalog.objects[dataset.items[p][0]].key}/{dataset.items[p][1]}"
              for p in oom}
    loader = torch.utils.data.DataLoader(
        _Positions(dataset, unusable), batch_size=config.batch_size,
        collate_fn=training._collate_samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    captured = []
    real_forward = training._forward
    real_clip = torch.nn.utils.clip_grad_norm_

    def maybe_oom(model_, batch, criterion, config_):
        if doomed & set(batch.scene_keys):
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_forward(model_, batch, criterion, config_)

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append([None if p.grad is None else p.grad.detach().clone()
                         for p in parameters])
        return real_clip(parameters, *args, **kwargs)

    with mock.patch.object(training, "_forward", maybe_oom), \
         mock.patch.object(torch.nn.utils, "clip_grad_norm_", capture):
        summary, _s, _t = training.run_epoch(
            model, loader, build_criterion(config), config, optimizer=optimizer,
            device="cpu", label="train", show_progress=False)
    return captured, summary


def _identical(a, b):
    """
    The same gradients bit for bit, with ``None`` in the same places.

    Exact, which the one thread in the test below makes possible. With two,
    on a busy machine the CPU sums round differently from run to run, and
    AdamW then enlarges the difference: it divides each gradient by its own
    size, so rounding in a nearly-zero one (this model has a parameter whose
    whole gradient is ~3e-7) becomes a difference in the step itself. Measured
    under load: the second step's gradients differed by up to 2.5e-4 of their
    norm in 5 runs of 8 on two threads, and were bitwise equal in 8 of 8 on
    one. No tolerance separates that from a real difference, and the old one
    (1e-5 of each tensor's own norm) failed on it intermittently.
    """
    compared = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is y
            continue
        compared += 1
        assert torch.equal(x, y)
    assert compared > 0, "no gradients were compared"


def test_an_oom_mid_group_is_exactly_a_batch_that_never_existed(root):
    """
    THE regression guard, for both halves of the old failure.

    The old handler zeroed the whole accumulation group on an OOM -- throwing
    away the micro-batches before it -- and forgot to reset the fragment count,
    so the next step divided by fragments whose gradients were gone. Now each
    micro-batch's gradient is computed with the running total set aside, so a
    failure discards that micro-batch alone.

    The layout: ``accumulate=2`` over four batches, the one at position 1
    running out of memory on every attempt (the retry with checkpointing
    included)::

        group A   batch 0 contributes, batch 1 fails -> step on batch 0 alone
        group B   batches 2 and 3 -> step on both

    which must be *identical*, step for step, to the same epoch with batch 1
    simply unusable -- and group A must equal a run over batch 0 alone.
    """
    config = _config(root, accumulate=2)
    # One thread, so the comparison can be exact (see `_identical`).
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        after_oom, summary = _gradients_at_every_step(config, [0, 1, 2, 3], oom=[1])
        never_there, _ = _gradients_at_every_step(config, [0, 1, 2, 3], unusable=[1])
        alone, _ = _gradients_at_every_step(config, [0])
    finally:
        torch.set_num_threads(threads)

    assert summary["oom"] == 1 and summary["oom_recovered"] == 0
    assert len(after_oom) == len(never_there) == 2, "one step per group"
    for a, b in zip(after_oom, never_there):
        _identical(a, b)
    _identical(after_oom[0], alone[0])


def test_the_oom_is_counted_and_named_by_scene(root):
    """
    A batch that fails even with checkpointing is counted and named -- by the
    scenes in it, not by its position, so the same scene failing in the next
    epoch is recognisably the same failure. One that fits on the retry is
    counted separately, because it was kept.
    """
    config = _config(root, accumulate=2)
    _steps, summary = _gradients_at_every_step(config, [0, 1, 2, 3], oom=[1])
    assert summary["oom"] == 1
    named = [n for n in summary["dropped_names"] if n.startswith("OOM:")]
    assert len(named) == 1 and "mug_" in named[0], named
    assert list(summary["failures"].values())[0] == (1, "out of memory")

    torch.manual_seed(0)
    model = training.build_model(config)
    dataset = BreakingBadScenes(config, "train")
    dataset.items = dataset.items[:4]
    loader = training._loader(dataset, config, shuffle=False, rank=0, world=1,
                              epoch=0)
    import unittest.mock as mock

    real = training._forward
    seen = {"n": 0}

    def once(model_, batch, criterion, config_):
        seen["n"] += 1
        if seen["n"] == 2:                      # the second batch, first attempt only
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(model_, batch, criterion, config_)

    with mock.patch.object(training, "_forward", once):
        summary, _s, _t = training.run_epoch(
            model, loader, build_criterion(config), config,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            device="cpu", label="train", show_progress=False,
        )
    assert summary["oom"] == 0 and summary["oom_recovered"] == 1
    assert summary["batches"] == 4, "the recovered batch is kept"


# --------------------------------------------------------------------------
# 3. Resuming a partial epoch, and remembering what failed
# --------------------------------------------------------------------------

def test_a_restarted_epoch_goes_back_to_the_step_it_began_at(root):
    """
    A partial epoch is re-run from its start, so its step count must go back to
    where the epoch began. It used to continue from the mid-epoch count, and
    every interrupted session pushed the learning-rate schedule ahead of an
    uninterrupted run by the part of the epoch that was repeated.
    """
    from reassembly.training import save_checkpoint

    config = _config(root, epochs=2, steps_per_epoch=2)
    torch.manual_seed(0)
    model = training.build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = Path(config.out_dir)
    save_checkpoint(out / "last.pt", model, optimizer, config, epoch=1, step=3,
                    history=[{"epoch": 0, "step": 2}], best=float("inf"),
                    completed=False, epoch_step=2)
    history = training.train(config)
    assert history[-1]["epoch"] == 1
    assert history[-1]["step"] == 4, "restarted from step 2, plus 2 batches"


def test_repeat_offenders_are_named_across_epochs_and_sessions(tmp_path):
    from reassembly.training import (_record_offenders, _repeat_offenders,
                                     save_checkpoint)

    offenders = {}
    _record_offenders(offenders, 0, {"failures": {"a/m1": (1, "out of memory"),
                                                  "b/m2": (1, "out of memory")}})
    assert _repeat_offenders(offenders) == [], "one failure is not a pattern"
    _record_offenders(offenders, 1, {"failures": {"a/m1": (1, "non-finite loss")}},
                      {"failures": {"c/m3": (2, "out of memory")}})
    repeat = _repeat_offenders(offenders)
    assert [key for key, _ in repeat] == ["a/m1"]
    assert repeat[0][1]["epochs"] == [0, 1] and repeat[0][1]["count"] == 2

    # The tally travels in the checkpoint, so it outlives the session.
    config = _config(tmp_path)
    model = training.build_model(config)
    save_checkpoint(tmp_path / "last.pt", model, None, config, epoch=1, step=10,
                    history=[], best=1.0, offenders=offenders)
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert state["offenders"] == offenders


# --------------------------------------------------------------------------
# 4. Evaluation assembles, and scores the data the model was trained on
# --------------------------------------------------------------------------

def test_evaluate_assembles_the_prediction_and_scores_it(root, capsys):
    import dataclasses

    config = _config(root, epochs=1, steps_per_epoch=2)
    training.train(config)
    capsys.readouterr()
    summary = training.evaluate(config, checkpoint="last.pt", split="val")
    assembly = summary["assembly"]
    for key in ("rmse_t", "chamfer", "part_accuracy", "geodesic_deg", "matches"):
        assert key in assembly and np.isfinite(assembly[key]), key
    assert 0.0 <= assembly["part_accuracy"] <= 1.0
    assert len(summary["assembly_scenes"]) == summary["batches"]
    out = capsys.readouterr().out
    assert "part accuracy" in out and "RMSE(T)" in out
    assert (Path(config.out_dir) / "val_metrics.json").exists()

    # The flags disagree with the checkpoint on a data setting: the checkpoint wins,
    # out loud -- unless asked otherwise.
    other = dataclasses.replace(config, tokens_per_scene=8)
    training.evaluate(other, checkpoint="last.pt", split="val", assemble=False)
    out = capsys.readouterr().out
    assert "using the checkpoint's --tokens_per_scene 32" in out
    assert "rotation only" in out
    training.evaluate(other, checkpoint="last.pt", split="val", assemble=False,
                      data_from_checkpoint=False)
    assert "tokens_per_scene" not in capsys.readouterr().out


def test_the_time_budget_finishes_the_epoch_it_runs_out_in(root, capsys):
    """
    The budget is read between epochs, as in Thesis 1: the epoch it runs out in
    is trained to its last step, validated in full and saved as complete, and
    the next session starts at the next epoch. It used to stop at the next
    optimizer step, leaving half an epoch to be run again.
    """
    import dataclasses

    config = _config(root, epochs=3, steps_per_epoch=3, max_hours=1e-9)
    history = training.train(config)
    assert [row["epoch"] for row in history] == [0], "stops after the first epoch"
    assert history[0]["partial"] == 0 and history[0]["steps"] == 3
    val = BreakingBadScenes(config, "val")
    assert history[0]["val_batches"] + history[0]["val_skipped"] == len(val)
    state = torch.load(Path(config.out_dir) / "last.pt", map_location="cpu",
                       weights_only=False)
    assert state["completed"] and state["epoch"] == 0
    assert "which was finished, validated and saved" in capsys.readouterr().out

    resumed = training.train(dataclasses.replace(config, max_hours=1.0))
    assert [row["epoch"] for row in resumed] == [0, 1, 2]


def test_last_pt_carries_the_best_so_far(root):
    """
    last.pt used to be written before `best` was updated, so it carried the
    previous best. A run resumed from it then treated its next epoch as a new
    best even when it was worse -- and overwrote best.pt with it.
    """
    config = _config(root, epochs=1, steps_per_epoch=1)
    training.train(config)
    out = Path(config.out_dir)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert np.isfinite(last["best"]) and last["best"] == best["best"]
