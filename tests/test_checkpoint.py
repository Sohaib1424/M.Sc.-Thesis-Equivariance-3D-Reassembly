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
