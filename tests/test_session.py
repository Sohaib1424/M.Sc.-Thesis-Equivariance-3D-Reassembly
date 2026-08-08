"""Surviving capped sessions: budgets, signals, and history continuity.

A ~6-day run inside 12-hour sessions is about a dozen restarts. Each of these
tests covers a way that used to lose work or corrupt the record.
"""
import json
import signal

import pytest

from reassembly.training.session import SessionLimit, merge_history


# ------------------------------------------------------------------ budgets
def test_no_budget_never_stops():
    s = SessionLimit(None)
    try:
        stop, _reason = s.should_stop(600.0)
        assert not stop
    finally:
        s.restore()


def test_stops_when_the_next_epoch_would_not_fit():
    """The point of the budget: an epoch that gets killed at the cap is a whole
    epoch of wasted compute, every session."""
    s = SessionLimit(1.0)                     # 1 hour
    try:
        s.start -= 3300                       # pretend 55 minutes have passed
        stop, reason = s.should_stop(600.0)   # a 10-minute epoch will not fit
        assert stop
        assert "next epoch" in reason
    finally:
        s.restore()


def test_continues_when_the_next_epoch_fits():
    s = SessionLimit(1.0)
    try:
        s.start -= 600                        # 10 minutes in
        stop, _ = s.should_stop(60.0)         # a 1-minute epoch fits easily
        assert not stop
    finally:
        s.restore()


def test_safety_margin_is_applied():
    """Epochs are not uniform -- scenes differ in size. Being 10% optimistic
    about the next one costs a killed epoch."""
    s = SessionLimit(1.0, safety_margin=1.5)
    try:
        s.start -= 3600 - 700                 # 700 s left
        stop, _ = s.should_stop(500.0)        # 500 * 1.5 = 750 > 700
        assert stop
    finally:
        s.restore()


def test_exhausted_budget_stops_regardless():
    s = SessionLimit(0.5)
    try:
        s.start -= 3600
        stop, reason = s.should_stop(1.0)
        assert stop and "exhausted" in reason
    finally:
        s.restore()


# ------------------------------------------------------------------ signals
def test_signal_requests_a_stop():
    """Kaggle sends SIGTERM before killing the process."""
    s = SessionLimit(None)
    try:
        signal.raise_signal(signal.SIGTERM)
        stop, reason = s.should_stop(1.0)
        assert stop and "SIGTERM" in reason
    finally:
        s.restore()


def test_handlers_are_restored():
    original = signal.getsignal(signal.SIGTERM)
    s = SessionLimit(None)
    assert signal.getsignal(signal.SIGTERM) is not original
    s.restore()
    assert signal.getsignal(signal.SIGTERM) is original


# ------------------------------------------------------------------ history
def test_history_is_trimmed_to_the_resumed_epoch():
    """last.pt can be older than the newest history entry when save_every > 1.
    Appending on top would fold the curve back on itself."""
    existing = {
        "train": {"total": [9, 8, 7, 6, 5], "rot": [5, 4, 3, 2, 1]},
        "val": {"total": [9, 8, 7, 6, 5]},
    }
    out = merge_history(existing, start_epoch=3)
    assert out["train"]["total"] == [9, 8, 7]
    assert out["train"]["rot"] == [5, 4, 3]
    assert out["val"]["total"] == [9, 8, 7]


def test_history_from_nothing_is_well_formed():
    out = merge_history({}, start_epoch=0)
    assert out == {"train": {}, "val": {}}
    out = merge_history(None, start_epoch=5)
    assert set(out) == {"train", "val"}


def test_history_survives_a_json_round_trip():
    """It is written to disk every epoch and read back on resume."""
    existing = {"train": {"total": [1.5, 1.2]}, "val": {"total": [1.6, 1.3]}}
    restored = merge_history(json.loads(json.dumps(existing)), start_epoch=2)
    assert restored["train"]["total"] == [1.5, 1.2]


def test_resuming_further_back_discards_the_later_entries():
    existing = {"train": {"total": list(range(30))}, "val": {"total": list(range(30))}}
    out = merge_history(existing, start_epoch=20)
    assert len(out["train"]["total"]) == 20
    assert out["train"]["total"][-1] == 19


# ------------------------------------------------------------------- rng
def test_rng_capture_and_restore_round_trip():
    torch = pytest.importorskip("torch", reason="torch not installed")
    import random

    import numpy as np

    from reassembly.training.session import load_rng_state, rng_state

    state = rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))

    random.random(); np.random.random(); torch.rand(1)      # advance all three

    assert load_rng_state(state)
    got = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert got == pytest.approx(expected)


def test_missing_rng_state_is_not_fatal():
    from reassembly.training.session import load_rng_state
    assert load_rng_state(None) is False
    assert load_rng_state({"python": "nonsense"}) is False
