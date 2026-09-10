"""Checkpoint policy, history bookkeeping, and resume."""
from __future__ import annotations

import torch

from vngat.training.checkpoint import BEST_NAME, ROLLING_NAME, CheckpointManager
from vngat.training.history import History


def _tiny():
    model = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    return model, opt, sched


def test_replaces_when_better_on_both(tmp_path):
    m = CheckpointManager(str(tmp_path), save_every=1)
    model, opt, sched = _tiny()
    m.save(model, opt, sched, None, 0, 1.0, 1.0, {}, {})
    wrote = m.save(model, opt, sched, None, 1, 0.5, 0.5, {}, {})
    assert wrote["rolling"] is True


def test_keeps_the_old_one_when_it_is_better_on_both(tmp_path):
    m = CheckpointManager(str(tmp_path), save_every=1)
    model, opt, sched = _tiny()
    m.save(model, opt, sched, None, 0, 0.5, 0.5, {}, {})
    wrote = m.save(model, opt, sched, None, 1, 0.9, 0.9, {}, {})
    assert wrote["rolling"] is False
    assert torch.load(tmp_path / ROLLING_NAME, weights_only=False)["epoch"] == 0


def test_replaces_when_only_one_metric_improves(tmp_path):
    """Strictly "better on BOTH" is what protects the stored checkpoint; a
    single improving metric is not enough to protect the OLD one."""
    m = CheckpointManager(str(tmp_path), save_every=1)
    model, opt, sched = _tiny()
    m.save(model, opt, sched, None, 0, 1.0, 1.0, {}, {})
    assert m.save(model, opt, sched, None, 1, 0.5, 1.5, {}, {})["rolling"] is True


def test_best_is_tracked_separately(tmp_path):
    m = CheckpointManager(str(tmp_path), save_every=1)
    model, opt, sched = _tiny()
    m.save(model, opt, sched, None, 0, 1.0, 0.4, {}, {})       # best val 0.4
    m.save(model, opt, sched, None, 1, 0.2, 0.9, {}, {})       # rolling replaced
    best = torch.load(tmp_path / BEST_NAME, weights_only=False)
    assert best["epoch"] == 0
    assert torch.load(tmp_path / ROLLING_NAME, weights_only=False)["epoch"] == 1


def test_resume_restores_optimizer_and_counters(tmp_path):
    m = CheckpointManager(str(tmp_path), save_every=1)
    model, opt, sched = _tiny()
    for _ in range(3):
        model(torch.randn(2, 3)).sum().backward()
        opt.step()
    m.save(model, opt, sched, None, 7, 0.3, 0.2, {"train": {"total": [1, 2]}}, {"tag": "t"})

    model2, opt2, sched2 = _tiny()
    m2 = CheckpointManager(str(tmp_path), save_every=1)
    state = m2.load(tmp_path / ROLLING_NAME, model2, opt2, sched2)
    assert state["epoch"] == 7
    assert m2.best_val == 0.2
    assert torch.allclose(model.weight, model2.weight)


def test_should_save_respects_interval_and_final_epoch():
    m = CheckpointManager("/tmp/vngat_ckpt_test", save_every=10)
    assert m.should_save(9, 100) and not m.should_save(8, 100)
    assert m.should_save(99, 100)


def test_history_truncation_realigns_after_resume():
    h = History()
    for i in range(10):
        h.append("train", {"total": float(i)})
        h.append("val", {"total": float(i)})
    h.truncate_to(4)
    assert h.num_epochs == 4
    assert h.data["val"]["total"] == [0.0, 1.0, 2.0, 3.0]


def test_history_survives_a_round_trip(tmp_path):
    h = History()
    h.append("train", {"total": 1.0, "rot": 0.5})
    h.append_meta(lr=1e-3)
    h.save(tmp_path / "history.json")
    loaded = History.load(tmp_path / "history.json")
    assert loaded.data == h.data


def test_history_load_tolerates_a_truncated_file(tmp_path):
    (tmp_path / "history.json").write_text('{"train": {"tot')
    assert History.load(tmp_path / "history.json").num_epochs == 0


def test_restart_schedule_spans_only_the_remaining_epochs():
    """
    A resumed cosine must anneal over the epochs that REMAIN, not the run total.

    Without `span`, resuming at epoch 125 with --epochs 250 would spread the
    anneal over 250 epochs while only 125 are left, so the rate would still be
    at half its base when the session ended.
    """
    from vngat.config import Config
    from vngat.training.trainer import build_scheduler

    model = torch.nn.Linear(3, 3)
    cfg = Config(lr=2e-4, lr_min=1e-5, epochs=250, lr_schedule="cosine")
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched, _ = build_scheduler(cfg, opt, span=125)

    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-9
    for _ in range(125):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - cfg.lr_min) < 1e-7, "did not reach the floor"


def test_cosine_does_not_rise_after_its_span():
    """Guards the periodicity trap: past the horizon the rate must stay down."""
    from vngat.config import Config
    from vngat.training.trainer import build_scheduler

    model = torch.nn.Linear(3, 3)
    cfg = Config(lr=1e-3, lr_min=1e-5, epochs=40, lr_schedule="cosine")
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched, _ = build_scheduler(cfg, opt, span=40)
    for _ in range(120):                      # three times the horizon
        sched.step()
    assert opt.param_groups[0]["lr"] <= cfg.lr_min * 1.01


def test_restart_schedule_overrides_the_checkpoints_base_lr():
    """
    Regression guard for a silent failure.

    `LambdaLR` takes its base rates from `param_groups["initial_lr"]` and only
    writes that key when absent. `optimizer.load_state_dict` restores it from
    the checkpoint, so setting `lr` alone is undone on the scheduler's first
    step -- a restart requesting 2e-4 ran at the checkpoint's 5e-4 while the log
    reported 2e-4.
    """
    from vngat.config import Config
    from vngat.training.trainer import build_scheduler

    model = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=5e-4)
    # simulate a resumed optimiser: initial_lr already present, from the old run
    for group in opt.param_groups:
        group["initial_lr"] = 5e-4

    cfg = Config(lr=2e-4, lr_min=2e-5, epochs=60, lr_schedule="cosine")
    for group in opt.param_groups:          # what the restart block does
        group["lr"] = cfg.lr
        group["initial_lr"] = cfg.lr
    sched, _ = build_scheduler(cfg, opt, span=23)

    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-12
    sched.step()
    # one step in, still ~2e-4 -- NOT the 4.98e-4 the stale base would give
    assert opt.param_groups[0]["lr"] < 2.1e-4, opt.param_groups[0]["lr"]
    for _ in range(22):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - cfg.lr_min) < 1e-7


def test_cosine_horizon_survives_a_checkpoint_round_trip():
    """
    Regression guard for a SILENT schedule change.

    `LambdaLR.state_dict()` serialises the lambda's `__dict__` only when the
    lambda is not a plain function. With a closure the horizon was never written
    to the checkpoint, so on resume it reverted to `cfg.epochs` while
    `last_epoch` continued -- a 23-epoch anneal silently became a 60-epoch one
    and ended at 1.4e-4 instead of 2e-5, with nothing in the log to show it.
    """
    from vngat.config import Config
    from vngat.training.trainer import build_scheduler

    model = torch.nn.Linear(3, 3)
    cfg = Config(lr=2e-4, lr_min=2e-5, epochs=60, lr_schedule="cosine")

    opt_a = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched_a, _ = build_scheduler(cfg, opt_a, span=23)     # the original run
    for _ in range(4):
        sched_a.step()
    saved = sched_a.state_dict()
    assert saved["lr_lambdas"][0] is not None, "the horizon must be serialised"
    assert saved["lr_lambdas"][0]["t_max"] == 23

    # resume: a NEW process builds with the default span (= cfg.epochs = 60)
    opt_b = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched_b, _ = build_scheduler(cfg, opt_b)
    sched_b.load_state_dict(saved)
    assert sched_b.lr_lambdas[0].t_max == 23, "horizon reverted to cfg.epochs"
    assert abs(opt_a.param_groups[0]["lr"] - opt_b.param_groups[0]["lr"]) < 1e-12

    for _ in range(19):                                   # finish the 23 epochs
        sched_b.step()
    assert abs(opt_b.param_groups[0]["lr"] - cfg.lr_min) < 1e-8


def test_cosine_clamps_and_never_rises_after_its_horizon():
    from vngat.config import Config
    from vngat.training.trainer import build_scheduler

    model = torch.nn.Linear(3, 3)
    cfg = Config(lr=1e-3, lr_min=1e-5, epochs=40, lr_schedule="cosine")
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched, _ = build_scheduler(cfg, opt, span=40)
    for _ in range(120):                                  # three times the horizon
        sched.step()
    assert opt.param_groups[0]["lr"] <= cfg.lr_min * 1.01


def _write_stub_checkpoint(path, config):
    torch.save({"epoch": 39, "train_loss": 4.8, "val_loss": 5.4, "best_val": 5.4,
                "history": {}, "config": config}, path)


def test_resume_adopts_the_checkpoints_schedule_with_no_flags(tmp_path):
    """
    THE property this exists for: `--resume auto --checkpoint_dir X` continues a
    run exactly, on any platform, without reconstructing --lr / --epochs /
    --lr_schedule by hand. Rebuilding them from memory is how a 23-epoch anneal
    silently became a 60-epoch one.
    """
    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(lr=2e-4, lr_min=2e-5, lr_schedule="cosine",
                                      epochs=60, hidden_channels=128, num_layers=6))
    cfg = Config(lr=1e-3, lr_min=1e-5, lr_schedule="plateau", epochs=400,
                 hidden_channels=128, num_layers=6)
    object.__setattr__(cfg, "_explicit", frozenset())
    adopt_checkpoint_config(cfg, ckpt, is_main=False)

    assert cfg.lr == 2e-4 and cfg.lr_min == 2e-5
    assert cfg.lr_schedule == "cosine" and cfg.epochs == 60


def test_an_explicit_flag_still_wins(tmp_path):
    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(lr=2e-4, epochs=60, hidden_channels=128))
    cfg = Config(lr=2e-4, epochs=80, hidden_channels=128)
    object.__setattr__(cfg, "_explicit", frozenset({"epochs"}))
    adopt_checkpoint_config(cfg, ckpt, is_main=False)
    assert cfg.epochs == 80, "an explicitly typed flag must override the checkpoint"


def test_architecture_mismatch_fails_readably(tmp_path):
    """Better than a shape error deep inside load_state_dict."""
    import pytest as _pytest

    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(hidden_channels=128, num_layers=6))
    cfg = Config(hidden_channels=64, num_layers=6)
    object.__setattr__(cfg, "_explicit", frozenset({"hidden_channels"}))
    with _pytest.raises(SystemExit, match="Architecture mismatch"):
        adopt_checkpoint_config(cfg, ckpt, is_main=False)


def test_resume_restores_the_data_definition(tmp_path):
    """
    The worst silent failure available: dropping `--split_source official` on a
    resume would revert to `hash`, a DIFFERENT train/val split. Validation
    objects leak into training and every number afterwards is meaningless, with
    nothing in the log to say so.
    """
    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(
        split_source="official", data_subsets="everyday_compressed",
        steps_per_epoch=80, val_steps=8, max_scenes=0, input_source="full"))
    cfg = Config()                       # defaults: hash, both subsets, 50 steps
    object.__setattr__(cfg, "_explicit", frozenset())
    adopt_checkpoint_config(cfg, ckpt, is_main=False)

    assert cfg.split_source == "official"
    assert cfg.data_subsets == "everyday_compressed"
    assert cfg.steps_per_epoch == 80 and cfg.val_steps == 8


def test_resume_restores_the_architecture(tmp_path):
    """A resume must not need the architecture re-typed; only an EXPLICIT
    mismatch is an error."""
    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(hidden_channels=128, num_layers=6, num_vn_slots=12))
    cfg = Config()                       # defaults: 64 / 4 / 8
    object.__setattr__(cfg, "_explicit", frozenset())
    adopt_checkpoint_config(cfg, ckpt, is_main=False)
    assert (cfg.hidden_channels, cfg.num_layers, cfg.num_vn_slots) == (128, 6, 12)


def test_platform_flags_are_never_restored(tmp_path):
    """batch_size is per RANK, so 16 on two GPUs and 32 on one are the same
    effective batch. Restoring it would break moving between machines."""
    from vngat.config import Config
    from vngat.training.trainer import _RESTORED_FIELDS

    for name in ("batch_size", "num_gpus", "num_workers", "save_every",
                 "time_budget_hours", "checkpoint_dir", "root_dir",
                 "grad_checkpointing", "device", "tag"):
        assert name not in _RESTORED_FIELDS, name
