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
        tokens_per_scene=32, batch_size=1, epochs=1, modes_per_scene=None,
        schedule=("intra",), check_init=False, val_frac=0.34, test_frac=0.0,
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

def _gradient_at_the_last_step(config, items, oom_at=None):
    """
    Run one epoch over ``items`` and return the gradients the optimizer was
    handed at the final step.

    Captured at ``clip_grad_norm_``, which the loop calls immediately after the
    group normalisation and before the step -- so this is exactly the quantity
    the normalisation is supposed to produce.
    """
    import unittest.mock as mock

    torch.manual_seed(0)
    model = training.build_model(config)
    dataset = BreakingBadScenes(config, "train")
    dataset.items = [dataset.items[i] for i in items]
    loader = training._loader(dataset, config, shuffle=False, rank=0, world=1,
                              epoch=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    captured, seen = {}, {"n": 0}
    real_forward = training._forward
    real_clip = torch.nn.utils.clip_grad_norm_

    def maybe_oom(model_, batch, criterion, config_):
        index = seen["n"]
        seen["n"] += 1
        if index == oom_at:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_forward(model_, batch, criterion, config_)

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured["grads"] = [None if p.grad is None else p.grad.detach().clone()
                             for p in parameters]
        return real_clip(parameters, *args, **kwargs)

    with mock.patch.object(training, "_forward", maybe_oom), \
         mock.patch.object(torch.nn.utils, "clip_grad_norm_", capture):
        training.run_epoch(model, loader, build_criterion(config), config,
                           optimizer=optimizer, device="cpu", label="train",
                           show_progress=False)
    return captured.get("grads")


def test_an_oom_mid_group_does_not_shrink_the_next_step(root):
    """
    THE regression guard.

    ``zero_grad`` in the OOM handler discards the whole accumulation group, but
    ``group_weight`` was only cleared at the optimizer step. So after an OOM
    that landed *after* an earlier micro-batch had already contributed, the
    next step divided the surviving gradients by a denominator that still
    counted the fragments whose gradients had just been thrown away.

    The layout matters and is why this is not the obvious test: the OOM has to
    fall in the MIDDLE of a group. An OOM on a group's first micro-batch is
    raised before anything is added to ``group_weight``, so the bug does not
    show. With ``accumulate=2`` over four batches and the failure at index 1:

        group A   batch 0 contributes, batch 1 OOMs -> everything discarded
        group B   batches 2 and 3 -> the only optimizer step

    correct:  divide by fragments(2) + fragments(3)
    buggy:    divide by fragments(0) + fragments(2) + fragments(3)

    which is a step roughly a third too small, silently, on exactly the batches
    large enough to run out of memory in the first place.
    """
    config = _config(root, accumulate=2)
    after_oom = _gradient_at_the_last_step(config, items=[0, 1, 2, 3], oom_at=1)
    # The same group B, with the failed group never having existed.
    reference = _gradient_at_the_last_step(config, items=[2, 3], oom_at=None)

    assert after_oom is not None and reference is not None, "no step was taken"
    compared = 0
    for a, b in zip(after_oom, reference):
        if a is None or b is None:
            assert a is b
            continue
        compared += 1
        assert torch.allclose(a, b, atol=1e-6), (
            "the step after a mid-group OOM is not normalised by the surviving "
            "fragments alone -- group_weight leaked past the zero_grad"
        )
    assert compared > 0, "no gradients were compared"


def test_the_oom_is_counted_and_named(root):
    config = _config(root, accumulate=2)
    torch.manual_seed(0)
    model = training.build_model(config)
    dataset = BreakingBadScenes(config, "train")
    dataset.items = dataset.items[:4]
    loader = training._loader(dataset, config, shuffle=False, rank=0, world=1,
                              epoch=0)

    import unittest.mock as mock

    real = training._forward
    seen = {"n": 0}

    def maybe_oom(model_, batch, criterion, config_):
        seen["n"] += 1
        if seen["n"] == 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(model_, batch, criterion, config_)

    with mock.patch.object(training, "_forward", maybe_oom):
        summary, _s, _t = training.run_epoch(
            model, loader, build_criterion(config), config,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            device="cpu", label="train", show_progress=False,
        )
    assert summary["oom"] == 1
    assert any(name.startswith("OOM:") for name in summary["dropped_names"])
