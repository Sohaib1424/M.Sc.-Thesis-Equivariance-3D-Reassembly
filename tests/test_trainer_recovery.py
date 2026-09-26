"""
The micro-step recovery ladder (`_run_micro_step`) and the step weighting
(`_finish_step`).

Reproduces, on CPU and without DDP, failures that only showed up on two GPUs
during real runs: a single-use context manager (`DistributedDataParallel.
no_sync()` returns one) entered a second time by the retry loop; a retried
scene whose partial gradient was counted twice; a finite loss with a
non-finite gradient poisoning the whole step.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import pytest
import torch

from conftest import make_scene
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


def _micro(scene, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(scene.num_fragments, 3, 3, generator=g, dtype=torch.float64))
    q[:, :, 0] *= torch.linalg.det(q).sign().unsqueeze(-1)
    return {
        "target": scene, "input": None,
        "rot": q.float(),
        "scene_dirs": ["synthetic"], "categories": ["Cat"], "num_scenes": 1,
    }


def _setup():
    torch.manual_seed(0)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4)
    cfg = Config(amp=False, device="cpu", grad_clip=0.0)
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

    result = T._run_micro_step(
        model, model, loss_fn, _micro(scene), torch.device("cpu"), cfg,
        True, scaler, single_use_sync, scale=1.0,
    )
    assert result.recovered is True
    assert result.status == "ok"             # the scene itself succeeded on retry
    assert calls["n"] == 2
    assert torch.isfinite(result.losses["total"]).all()


def test_second_oom_skips_the_scene_and_keeps_the_accumulation(monkeypatch):
    """
    Both attempts fail -> the scene is SKIPPED. The previous version substituted
    a placeholder scene and back-propagated its real loss, adding a meaningless
    gradient; rank symmetry never needed it, because every real micro-step runs
    under no_sync and the one all-reduce is fired by `_sync_gradients`.
    """
    model, loss_fn, cfg, scaler = _setup()
    for p in model.parameters():                     # "earlier micro-steps"
        p.grad = torch.full_like(p, 0.25)
    calls = {"n": 0}

    def oom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(T, "_forward_loss", oom)
    result = T._run_micro_step(
        model, model, loss_fn, _micro(make_scene(seed=12)), torch.device("cpu"), cfg,
        True, scaler, single_use_sync, scale=1.0,
    )
    assert result.status == "oom" and result.recovered
    assert calls["n"] == 2
    assert all(torch.equal(p.grad, torch.full_like(p, 0.25)) for p in model.parameters())


def test_oom_retry_does_not_count_the_scene_twice(monkeypatch):
    """
    An OOM part way through a backward has already written part of the scene's
    gradient. The retry must start from the accumulation as it was, not add the
    scene a second time on top of the partial one.
    """
    model, loss_fn, cfg, scaler = _setup()
    micro = _micro(make_scene(seed=15), seed=3)

    for p in model.parameters():
        p.grad = None
    clean = T._run_micro_step(model, model, loss_fn, micro, torch.device("cpu"), cfg,
                              True, scaler, nullcontext, scale=1.0)
    reference = [p.grad.clone() for p in model.parameters()]
    assert clean.status == "ok"

    for p in model.parameters():
        p.grad = None
    merge = T._merge_grads
    calls = {"n": 0}

    def fail_after_backward(params, stash):
        # The backward has fully written this scene's gradient when this runs:
        # the worst case of "part way through".
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return merge(params, stash)

    monkeypatch.setattr(T, "_merge_grads", fail_after_backward)
    retried = T._run_micro_step(model, model, loss_fn, micro, torch.device("cpu"), cfg,
                                True, scaler, nullcontext, scale=1.0)
    assert retried.status == "ok" and retried.recovered
    for p, ref in zip(model.parameters(), reference):
        assert torch.allclose(p.grad, ref, atol=1e-6), "the retried scene was counted twice"


def test_finite_loss_with_non_finite_gradient_is_dropped():
    model, loss_fn, cfg, scaler = _setup()
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.5)
    handle = model.rotation_head.map.weight.register_hook(lambda g: g * float("nan"))
    try:
        result = T._run_micro_step(model, model, loss_fn, _micro(make_scene(seed=16)),
                                   torch.device("cpu"), cfg, True, scaler, nullcontext, scale=1.0)
    finally:
        handle.remove()
    assert result.status == "nonfinite"
    assert torch.isfinite(result.losses["total"])
    assert all(torch.equal(p.grad, torch.full_like(p, 0.5)) for p in model.parameters())


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


# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------
def test_resolve_device_adds_a_missing_index():
    """
    `torch.device("cuda")` has no index and `torch.cuda.set_device` rejects it.
    The single-GPU path passed the config string straight through, so
    `--num_gpus 1` died with "Expected a torch.device with a specified index".
    """
    from vngat.training.distributed import resolve_device

    if torch.cuda.is_available():
        assert resolve_device("cuda", rank=0) == torch.device("cuda:0")
        assert resolve_device("cuda", rank=1) == torch.device("cuda:1")
        assert resolve_device("cuda:1", rank=0) == torch.device("cuda:1")  # explicit wins
    else:
        assert resolve_device("cuda") == torch.device("cpu")   # graceful fallback


def test_resolve_device_passes_cpu_through():
    from vngat.training.distributed import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolved_device_is_accepted_by_set_device():
    """The end-to-end property that actually broke."""
    from vngat.training.distributed import resolve_device

    device = resolve_device("cuda", rank=0)
    if device.type == "cuda":
        torch.cuda.set_device(device)          # must not raise
        assert torch.cuda.current_device() == device.index


def test_non_finite_loss_is_skipped_not_propagated(monkeypatch):
    """
    A NaN loss must never reach backward. Poisoned weights are unrecoverable --
    every later forward is NaN -- so the run would keep going for hours
    producing nothing.
    """
    model, loss_fn, cfg, scaler = _setup()
    original = T._forward_loss

    def nan_loss(*args, **kwargs):
        out = original(*args, **kwargs)
        out["total"] = out["total"] * float("nan")
        return out

    monkeypatch.setattr(T, "_forward_loss", nan_loss)
    result = T._run_micro_step(
        model, model, loss_fn, _micro(make_scene(seed=21)), torch.device("cpu"),
        cfg, True, scaler, nullcontext, scale=1.0,
    )
    assert result.status == "nonfinite"
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


# --------------------------------------------------------------------------
# Step weighting
# --------------------------------------------------------------------------
def _reference_gradient(model, loss_fn, micros, weights):
    """d/dtheta of sum_i w_i L_i, computed directly."""
    model.zero_grad(set_to_none=True)
    total = 0.0
    for micro, w in zip(micros, weights):
        scene = T.prepare_scene(micro, torch.device("cpu"))
        total = total + w * T._forward_loss(model, loss_fn, scene, torch.device("cpu"), False)["total"]
    total.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return grads


def _accumulate(model, loss_fn, cfg, scaler, micros, drop=()):
    frags = [m["target"].num_fragments for m in micros]
    rank = float(sum(frags))
    contributed = 0.0
    model.zero_grad(set_to_none=True)
    for i, (micro, nf) in enumerate(zip(micros, frags)):
        if i in drop:
            continue
        result = T._run_micro_step(model, model, loss_fn, micro, torch.device("cpu"), cfg,
                                   True, scaler, nullcontext, scale=nf / rank)
        assert result.status == "ok"
        contributed += nf
    return rank, contributed


def test_step_gradient_is_the_fragment_weighted_mean():
    """A 3-fragment and a 2-fragment scene weigh 3:2, not 1:1."""
    model, loss_fn, cfg, scaler = _setup()
    micros = [_micro(make_scene(((10, 21), (7, 15), (13, 29)), seed=31), seed=1),
              _micro(make_scene(((9, 17), (6, 11)), seed=32), seed=2)]
    reference = _reference_gradient(model, loss_fn, micros, [3 / 5, 2 / 5])
    rank, contributed = _accumulate(model, loss_fn, cfg, scaler, micros)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    outcome, norm = T._finish_step(model, loss_fn, optimizer, scaler, torch.device("cpu"),
                                   cfg, rank, contributed)
    assert outcome == "stepped" and norm > 0
    for p, ref in zip(model.parameters(), reference):
        assert torch.allclose(p.grad, ref, rtol=1e-4, atol=1e-6)


def test_a_dropped_micro_batch_shrinks_the_denominator_not_the_step():
    model, loss_fn, cfg, scaler = _setup()
    micros = [_micro(make_scene(((10, 21), (7, 15), (13, 29)), seed=33), seed=4),
              _micro(make_scene(((9, 17), (6, 11)), seed=34), seed=5)]
    reference = _reference_gradient(model, loss_fn, micros[:1], [1.0])
    rank, contributed = _accumulate(model, loss_fn, cfg, scaler, micros, drop=(1,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    T._finish_step(model, loss_fn, optimizer, scaler, torch.device("cpu"), cfg, rank, contributed)
    for p, ref in zip(model.parameters(), reference):
        assert torch.allclose(p.grad, ref, rtol=1e-4, atol=1e-6)


def test_a_step_with_no_data_does_not_move_the_weights():
    """AdamW moves weights on momentum and decay alone; a step with nothing
    behind it must not happen at all."""
    model, loss_fn, cfg, scaler = _setup()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    outcome, _ = T._finish_step(model, loss_fn, optimizer, scaler, torch.device("cpu"),
                                cfg, 5.0, 0.0)
    assert outcome == "empty"
    assert all(torch.equal(a, p) for a, p in zip(before, model.parameters()))


def test_non_finite_parameter_is_detected():
    model, _, _, _ = _setup()
    from vngat.training.trainer import _first_non_finite_parameter

    assert _first_non_finite_parameter(model) is None
    with torch.no_grad():
        model.rotation_head.map.weight[0, 0] = float("nan")
    assert _first_non_finite_parameter(model) == "rotation_head.map.weight"
