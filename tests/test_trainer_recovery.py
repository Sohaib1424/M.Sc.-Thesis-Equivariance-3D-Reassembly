"""
The OOM recovery ladder in `_run_micro_step`.

Reproduces, on CPU and without DDP, the failure that only showed up on two
GPUs during a real out-of-memory retry: a single-use context manager
(`DistributedDataParallel.no_sync()` returns one) entered a second time by the
retry loop.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import pytest
import torch

from conftest import make_scene, random_rotation
from vngat.config import Config
from vngat.losses.composite import CompositeLoss
from vngat.models.vn_gat import VNGATModel
from vngat.training import trainer as T


@contextmanager
def single_use_sync():
    """
    Stand-in for `DDP.no_sync()`.

    `@contextmanager` produces a `_GeneratorContextManager`, which deletes its
    args/kwds/func on `__enter__` -- so entering the SAME instance twice raises
    AttributeError. That is exactly the object DDP hands back, and exactly what
    the retry loop must not reuse.
    """
    yield


def _micro(scene):
    return {
        "target": scene, "input": None,
        "rot": random_rotation(scene.num_fragments),
        "trans": torch.zeros(scene.num_fragments, 3),
        "scene_dirs": ["synthetic"], "num_scenes": 1,
    }


def _setup():
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2, embed_dim=4)
    cfg = Config(amp=False, device="cpu")
    scaler = T.make_scaler(torch.device("cpu"), False)
    return model, CompositeLoss(), cfg, scaler


def test_single_use_context_manager_really_is_single_use():
    """Pins the assumption the fix rests on, so it cannot rot silently."""
    ctx = single_use_sync()
    with ctx:
        pass
    with pytest.raises(AttributeError):
        with ctx:
            pass


def test_oom_retry_rebuilds_the_sync_context(monkeypatch):
    """
    First attempt raises an out-of-memory error; the ladder must retry with
    gradient checkpointing and enter a FRESH sync context.

    Against the buggy version this fails with
    "'_GeneratorContextManager' object has no attribute 'args'".
    """
    model, loss_fn, cfg, scaler = _setup()
    scene = make_scene(seed=11)
    original = T._forward_loss
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(*args, **kwargs)

    monkeypatch.setattr(T, "_forward_loss", flaky)

    losses, recovered, used_fallback = T._run_micro_step(
        model, model, loss_fn, _micro(scene), torch.device("cpu"), cfg,
        True, scaler, single_use_sync, scale=1.0,
    )
    assert recovered is True
    assert used_fallback is False           # the scene itself succeeded on retry
    assert calls["n"] == 2
    assert torch.isfinite(losses["total"]).all()


def test_second_oom_falls_back_to_the_placeholder_step(monkeypatch):
    """
    Both real attempts fail -> substitute the placeholder scene so this rank
    still performs one backward pass and stays in lockstep with its peers.
    """
    model, loss_fn, cfg, scaler = _setup()
    scene = make_scene(seed=12)
    original = T._forward_loss
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(*args, **kwargs)

    monkeypatch.setattr(T, "_forward_loss", flaky)

    losses, recovered, used_fallback = T._run_micro_step(
        model, model, loss_fn, _micro(scene), torch.device("cpu"), cfg,
        True, scaler, single_use_sync, scale=1.0,
    )
    assert recovered and used_fallback
    assert calls["n"] == 3
    assert torch.isfinite(losses["total"]).all()


def test_grad_checkpointing_flag_is_restored(monkeypatch):
    """A retry must not leave checkpointing permanently on."""
    model, loss_fn, cfg, scaler = _setup()
    model.grad_checkpointing = False
    original = T._forward_loss
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return original(*args, **kwargs)

    monkeypatch.setattr(T, "_forward_loss", flaky)
    T._run_micro_step(model, model, loss_fn, _micro(make_scene(seed=13)),
                      torch.device("cpu"), cfg, True, scaler, single_use_sync, scale=1.0)
    assert model.grad_checkpointing is False


def test_non_oom_errors_are_not_retried(monkeypatch):
    """A genuine bug must surface immediately, not be masked by the ladder."""
    model, loss_fn, cfg, scaler = _setup()
    calls = {"n": 0}

    def broken(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("self (Half) and source (Float) must have the same scalar type")

    monkeypatch.setattr(T, "_forward_loss", broken)
    with pytest.raises(RuntimeError, match="scalar type"):
        T._run_micro_step(model, model, loss_fn, _micro(make_scene(seed=14)),
                          torch.device("cpu"), cfg, True, scaler, nullcontext, scale=1.0)
    assert calls["n"] == 1


def test_oom_detection_recognises_both_spellings():
    assert T._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert T._is_oom(RuntimeError("CUDA error: out of memory"))
    assert not T._is_oom(RuntimeError("shape mismatch"))
