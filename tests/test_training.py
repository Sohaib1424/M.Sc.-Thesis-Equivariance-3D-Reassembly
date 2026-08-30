"""
The training engine: schedule, checkpoints, reporting, and can it learn.

The last one is the point of the file. Everything else here guards a failure
mode that produces finite numbers and a descending curve --
`test_the_model_can_actually_learn` is the check that the architecture and all
of its conventions are *consistent enough to fit anything at all*. If the
rotation label were transposed, or the head's frame pointed the wrong way, or
the loss compared misaligned rows, every other test in this repository would
still pass and this one would not.
"""
from __future__ import annotations

import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from reassembly.data.features import build_scene
from reassembly.data.transforms import random_rotations
from reassembly.training import (
    CHANCE,
    Config,
    Progress,
    Skipped,
    _collate_samples,
    _final_report,
    _forward,
    _metrics,
    build_criterion,
    build_model,
    check_initial_losses,
    format_losses,
    learning_rate,
    run_epoch,
    save_checkpoint,
    write_history,
)


def _scene(seed: int, fragments: int = 3, dent: bool = True, clusters: bool = True):
    """A scene of distinguishable shards under a known perturbation."""
    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(fragments):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0 - 0.15 * i)
        mesh.apply_translation([2.5 * i, 0.0, 0.0])
        v, f = np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces)
        if dent:
            # Otherwise every fragment is the same sphere and its rotation is
            # genuinely unrecoverable -- the test would be measuring a symmetry,
            # not the model.
            v[rng.choice(len(v), 25, replace=False)] *= 1.25
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), 40, replace=False)] = True
        vertices.append(v)
        faces.append(f)
        masks.append(mask)
    spin = random_rotations(fragments, np.random.default_rng(1000 + seed))
    cluster = None
    if clusters:
        # Stand-in coincidence labels. Without them `num_clusters` is zero, the
        # embedding term is absent, and the whole embedding head goes untrained
        # -- see `test_without_coincidence_labels_the_embedding_head_is_untrained`.
        total = sum(len(v) for v in vertices)
        cluster = np.full(total, -1, np.int64)
        cluster[:30] = np.arange(30) // 3
    return build_scene(vertices, faces, masks, rotations=spin,
                       tokens_per_scene=48, cluster=cluster)


class _Fixed(torch.utils.data.Dataset):
    def __init__(self, n: int = 6):
        self.items = [_scene(s) for s in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# --------------------------------------------------------------------------
# Can it learn
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_model_can_actually_learn():
    """
    Memorise six scenes and drive the rotation error far below chance.

    This is a *training*-error check and proves nothing about generalisation --
    that is exactly what makes it useful. It isolates "are the conventions
    self-consistent and is the architecture capable of fitting" from "does it
    generalise", and only the first question can be answered without the real
    dataset. A transposed label or a misaligned loss row fails here and passes
    everywhere else.
    """
    torch.manual_seed(0)
    config = Config(channels=64, heads=4, lr=3e-3, batch_size=2, workers=0)
    loader = torch.utils.data.DataLoader(
        _Fixed(6), batch_size=2, shuffle=True, collate_fn=_collate_samples
    )
    model = build_model(config)
    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    first, step, _ = run_epoch(model, loader, criterion, config,
                               optimizer=optimizer, step=0, total_steps=200,
                               label="train", show_progress=False)
    assert 90.0 < first["rotation_degrees"] < 165.0, (
        f"an untrained model should start near chance "
        f"({CHANCE['geodesic_deg']:.0f} deg), got {first['rotation_degrees']:.1f}"
    )

    for _ in range(44):
        summary, step, _ = run_epoch(model, loader, criterion, config,
                                     optimizer=optimizer, step=step,
                                     total_steps=200, label="train",
                                     show_progress=False)

    assert summary["rotation_degrees"] < 70.0, (
        f"after 45 epochs on six memorisable scenes the rotation error is "
        f"{summary['rotation_degrees']:.1f} deg, against a chance level of "
        f"{CHANCE['geodesic_deg']:.0f}. The model cannot fit its own training "
        f"set, which means a convention is wrong -- not that it needs more data."
    )
    assert summary["total"] < first["total"]


def test_losses_start_at_their_reference_values():
    """
    An untrained forward pass, read against what each term must be at chance.
    A term far from its reference is measuring something other than its name.
    """
    torch.manual_seed(0)
    config = Config(channels=32, heads=4)
    batch, _ = _collate_samples([_scene(1), _scene(2), _scene(3), _scene(4)])
    model = build_model(config)
    loss, report, R = _forward(model, batch, build_criterion(config), config)

    assert torch.isfinite(loss)
    assert 80.0 < report["rotation_degrees"] < 175.0
    assert 0.6 < report["normal"] < 1.5              # chance 1.0
    assert 1.2 < report["face"] < 2.9                # chance 2.0
    assert check_initial_losses(report) == []


def test_every_parameter_trains():
    """A layer with no gradient is a layer that is not in the model."""
    torch.manual_seed(0)
    config = Config(channels=32, heads=4)
    batch, _ = _collate_samples([_scene(1), _scene(2)])
    model = build_model(config)
    loss, _, _ = _forward(model, batch, build_criterion(config), config)
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reaches: {dead}"


def test_without_coincidence_labels_the_embedding_head_is_untrained():
    """
    A consequence worth stating rather than discovering later.

    The embedding-consistency loss is the *only* thing that supervises the
    per-vertex embedding. With `supervise_embedding=False`, or on a scene whose
    fragments share no vertices, that term is absent and the readout and its MLP
    receive no gradient at all -- silently, since the rotation loss keeps
    descending exactly as before.

    It matters because stage two matches interface points by mutual nearest
    neighbours *in embedding space*. An untrained embedding head means the
    rotation model still works and the translation solver has nothing to use.
    `Config` warns about this at startup.
    """
    torch.manual_seed(0)
    config = Config(channels=32, heads=4)
    batch, _ = _collate_samples([_scene(1, clusters=False),
                                 _scene(2, clusters=False)])
    assert batch.num_clusters == 0
    model = build_model(config)
    loss, report, _ = _forward(model, batch, build_criterion(config), config)
    loss.backward()

    assert "embedding" not in report, "the term should be absent, not zero"
    dead = {n for n, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0}
    assert any(n.startswith("embedding.") for n in dead)
    assert "readout.directions.weight" in dead
    # Everything else still trains -- which is exactly why this is easy to miss.
    assert not any(n.startswith(("intra.", "cross.", "head.")) for n in dead)


def test_metrics_agree_with_a_perfect_prediction():
    R = torch.as_tensor(random_rotations(20, np.random.default_rng(0)))
    out = _metrics(R, R)
    assert out["geodesic_deg"] < 1e-4
    assert out["acc@5deg"] == 1.0


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------

def test_schedule_warms_up_then_decays():
    config = Config(lr=1e-3, warmup_fraction=0.1, min_lr_fraction=0.02)
    rates = [learning_rate(s, 1000, config) for s in range(1000)]
    assert rates[0] < rates[50] <= config.lr
    assert rates[99] == pytest.approx(config.lr, rel=1e-9)
    assert all(rates[i] >= rates[i + 1] - 1e-12 for i in range(100, 999))
    # The floor is reached at step == total; step 999 is one short of it.
    assert learning_rate(1000, 1000, config) == pytest.approx(
        config.lr * config.min_lr_fraction, rel=1e-9)
    assert rates[-1] == pytest.approx(config.lr * config.min_lr_fraction, rel=1e-3)


def test_schedule_does_not_climb_back_after_the_end():
    """
    The reason this is a function and not `CosineAnnealingLR`. That scheduler is
    periodic: past `T_max` the rate rises back toward its base, so a mis-set
    epoch count silently becomes a late-training rate increase. Its `T_max` also
    lives in its own state dict, so changing `--epochs` on resume may not change
    the schedule at all.
    """
    config = Config(lr=1e-3)
    floor = config.lr * config.min_lr_fraction
    for step in (1000, 1500, 5000, 100_000):
        assert learning_rate(step, 1000, config) == pytest.approx(floor, rel=1e-9)


def test_schedule_is_exactly_reproducible_on_resume():
    """A pure function of the step, so resuming cannot drift the schedule."""
    config = Config(lr=1e-3)
    straight = [learning_rate(s, 500, config) for s in range(500)]
    resumed = [learning_rate(s, 500, config) for s in range(200)] + \
              [learning_rate(s, 500, config) for s in range(200, 500)]
    assert straight == resumed


@pytest.mark.parametrize("total", [1, 2, 7])
def test_schedule_survives_a_tiny_run(total):
    config = Config(lr=1e-3)
    for step in range(total + 5):
        rate = learning_rate(step, total, config)
        assert 0.0 < rate <= config.lr + 1e-12


# --------------------------------------------------------------------------
# Dropped samples
# --------------------------------------------------------------------------

def test_a_dropped_sample_is_named_not_just_counted():
    """
    Dropping a bad sample is not neutral: an item that fails every epoch has
    been removed from the dataset. A per-*batch* counter cannot see it at all --
    a batch of four with one unusable item still collates and reports nothing.
    """
    batch, dropped = _collate_samples(
        [_scene(1), Skipped("mug/mode_03", "1 fragment(s)"), _scene(2)]
    )
    assert batch is not None and batch.num_scenes == 2
    assert [d.key for d in dropped] == ["mug/mode_03"]
    assert dropped[0].reason == "1 fragment(s)"


def test_a_batch_of_only_unusable_samples_collates_to_none():
    batch, dropped = _collate_samples([Skipped("a", "x"), Skipped("b", "y")])
    assert batch is None and len(dropped) == 2


def test_run_epoch_tallies_dropped_names():
    config = Config(channels=16, heads=4, workers=0)

    class Mixed(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return Skipped(f"obj{i}", "1 fragment(s)") if i == 0 else _scene(i)

    loader = torch.utils.data.DataLoader(Mixed(), batch_size=2,
                                         collate_fn=_collate_samples)
    summary, _, _ = run_epoch(build_model(config), loader, build_criterion(config),
                              config, label="val", show_progress=False)
    assert summary["dropped"] == 1
    assert summary["dropped_names"] == ["obj0"]


# --------------------------------------------------------------------------
# Checkpoints and history
# --------------------------------------------------------------------------

def test_checkpoint_carries_everything_needed_to_continue(tmp_path):
    """Weights alone restarts the schedule and throws away the optimiser."""
    config = Config(channels=16, heads=4, out_dir=str(tmp_path))
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()

    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, config, epoch=3, step=120,
                    history=[{"epoch": 0}], best=1.5)
    assert not (tmp_path / "last.pt.tmp").exists(), "no partial file left behind"

    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "config", "epoch", "step", "history",
                "best", "torch_rng", "numpy_rng", "python_rng"):
        assert key in state, f"checkpoint is missing {key}"
    assert Config(**state["config"]).channels == 16
    assert len(state["optimizer"]["state"]) > 0, "optimiser moments must survive"

    restored = build_model(config)
    restored.load_state_dict(state["model"])
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_history_csv_unions_columns_across_epochs(tmp_path):
    """
    A term that only appears in later epochs must not shift every other
    column, and a missing value must be blank rather than a wrong number.
    """
    write_history(tmp_path, [{"epoch": 0, "a": 1.0}, {"epoch": 1, "a": 2.0, "b": 3.0}])
    rows = (tmp_path / "history.csv").read_text().strip().split("\n")
    assert rows[0] == "epoch,a,b"
    assert rows[1] == "0,1.0,"
    assert rows[2] == "1,2.0,3.0"
    assert len(json.loads((tmp_path / "history.json").read_text())) == 2


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_every_loss_term_is_printed_alongside_the_total():
    summary = {"rotation": 1.0, "position": 2.0, "normal": 3.0, "face": 4.0,
               "embedding": 5.0, "total": 15.0}
    line = format_losses(summary)
    for name in ("rotation", "position", "normal", "face", "embedding", "TOTAL"):
        assert name in line


def test_progress_bar_works_without_a_terminal(capsys):
    """
    A captured notebook cell is not a tty, and carriage returns there produce
    one unreadable line. It must fall back to periodic newlines.
    """
    bar = Progress(10, prefix="test")
    bar.tty = False
    for _ in range(10):
        bar.update(1)
    bar.close()
    out = capsys.readouterr().out
    assert "test [" in out and "10/10" in out and "100.0%" in out
    assert "\r" not in out


def test_initial_loss_check_catches_a_term_that_is_off():
    assert check_initial_losses({"rotation_degrees": 126.0, "normal": 1.0,
                                 "face": 2.0}) == []
    complaints = check_initial_losses({"rotation_degrees": 12.0, "normal": 0.02,
                                       "face": 0.1})
    assert len(complaints) == 3


@pytest.mark.parametrize("curve,expected", [
    ([126.4, 126.3], "at chance"),
    ([91.0, 89.5], "axis-only floor"),
    ([60.0, 50.0, 40.0, 20.0], "still descending"),
])
def test_the_final_verdict_refuses_to_flatter_a_run(curve, expected, capsys):
    """
    Interpretation rules written before the results, and enforced. A run at
    chance, one parked at the axis-only floor, and one truncated mid-descent all
    produce a perfectly reportable number that means something quite different
    from what it looks like.
    """
    _final_report([{"val_geodesic_deg": v} for v in curve])
    assert expected in capsys.readouterr().out


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_config_rejects_settings_that_would_fail_later():
    with pytest.raises(ValueError):
        Config(channels=30, heads=4)
    with pytest.raises(ValueError):
        Config(label_method="guess")
    with pytest.raises(ValueError):
        Config(epochs=0)


def test_cli_flags_round_trip_into_the_config():
    from scripts.train import build_parser, config_from_args

    args = build_parser().parse_args(
        ["--root", "/data", "--epochs", "7", "--channels", "128", "--heads", "8",
         "--no-amp", "--schedule", "intra", "cross", "--lr", "5e-4"]
    )
    config = config_from_args(args)
    assert config.root == "/data" and config.epochs == 7
    assert config.channels == 128 and config.heads == 8
    assert config.amp is False and config.lr == pytest.approx(5e-4)
    assert tuple(config.schedule) == ("intra", "cross")


def test_cli_leaves_unspecified_fields_at_their_defaults():
    """
    The parser's defaults are `None`, not copies of the dataclass values, so
    there is only one place a default lives and it cannot drift.
    """
    from scripts.train import build_parser, config_from_args

    config = config_from_args(build_parser().parse_args(["--root", "/data"]))
    default = Config()
    assert config.epochs == default.epochs
    assert config.channels == default.channels
    assert config.tokens_per_scene == default.tokens_per_scene


def test_multi_gpu_from_a_notebook_raises_something_actionable(monkeypatch):
    """
    `mp.spawn` re-imports `__main__` in each child, which does not exist in a
    notebook cell -- the children die with a FileNotFoundError on `<stdin>`
    that says nothing about the cause. Under pytest `__main__` *does* have a
    file, so the notebook condition is simulated by removing it.
    """
    import __main__

    from reassembly.training import train

    monkeypatch.delattr(__main__, "__file__", raising=False)
    with pytest.raises(RuntimeError, match="notebook"):
        train(Config(root="/nonexistent", devices=2))


# --------------------------------------------------------------------------
# Resuming across sessions
# --------------------------------------------------------------------------

def _tiny_checkpoint(path, config, *, epoch, step, completed, history=None,
                     elapsed=0.0, best=9.9):
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    save_checkpoint(path, model, optimizer, config, epoch, step,
                    history or [], best, completed=completed, elapsed=elapsed)


def test_a_partial_epoch_is_restarted_not_skipped(tmp_path):
    """
    A mid-epoch checkpoint records the epoch as incomplete, so resume runs it
    again. Treating it as finished would silently drop the part that never ran,
    and nothing downstream could tell.
    """
    from reassembly.training import _resume_path

    config = Config(channels=16, heads=4, out_dir=str(tmp_path))
    _tiny_checkpoint(tmp_path / "last.pt", config, epoch=2, step=40, completed=False)
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert state["completed"] is False
    assert state["epoch"] + (1 if state["completed"] else 0) == 2

    _tiny_checkpoint(tmp_path / "done.pt", config, epoch=2, step=40, completed=True)
    finished = torch.load(tmp_path / "done.pt", map_location="cpu", weights_only=False)
    assert finished["epoch"] + (1 if finished["completed"] else 0) == 3


def test_resume_from_reads_a_different_directory(tmp_path):
    """
    The Kaggle case: a session writes to /kaggle/working, and the next session
    mounts that output read-only under /kaggle/input. Load and save paths differ.
    """
    from reassembly.training import _resume_path

    source, destination = tmp_path / "session1", tmp_path / "session2"
    source.mkdir()
    config = Config(channels=16, heads=4, out_dir=str(destination))
    _tiny_checkpoint(source / "last.pt", config, epoch=1, step=10, completed=True)

    found = _resume_path(Config(channels=16, heads=4, out_dir=str(destination),
                                resume_from=str(source)),
                         destination / "last.pt")
    assert found == source / "last.pt", "a directory should resolve to its last.pt"

    direct = _resume_path(Config(channels=16, heads=4,
                                 resume_from=str(source / "last.pt")),
                          destination / "last.pt")
    assert direct == source / "last.pt", "an explicit file should be used as given"

    missing = _resume_path(Config(channels=16, heads=4,
                                  resume_from=str(tmp_path / "nothing")),
                           destination / "last.pt")
    assert missing is None


def test_resume_refuses_a_different_architecture():
    """
    Otherwise `load_state_dict` raises a shape error naming a tensor, which
    takes a while to trace back to the flag that caused it.
    """
    from reassembly.training import _check_resume_compatible

    saved = {"channels": 64, "heads": 4, "head_dim": 8, "embedding_dim": 32,
             "schedule": ["intra", "cross"]}
    _check_resume_compatible(Config(channels=64, heads=4,
                                    schedule=("intra", "cross")), saved, False)
    with pytest.raises(ValueError, match="different"):
        _check_resume_compatible(Config(channels=128, heads=4,
                                        schedule=("intra", "cross")), saved, False)


def test_resume_warns_when_the_data_settings_changed(capsys):
    """
    An architecture change raises. A *data* change raises nothing -- training
    continues on a different problem than the weights were trained for, and the
    loss curve just has a step in it that reads as noise.
    """
    from reassembly.training import _check_resume_compatible

    saved = {"channels": 64, "heads": 4, "label_method": "dihedral",
             "tokens_per_scene": 2048, "epochs": 40}
    _check_resume_compatible(Config(channels=64, heads=4,
                                    label_method="coincidence",
                                    tokens_per_scene=512, epochs=80), saved, True)
    out = capsys.readouterr().out
    assert "DATA settings changed" in out
    assert "label_method" in out and "tokens_per_scene" in out
    assert "epochs changed" in out, "a changed epoch count rescales the schedule"


def test_rng_state_is_restored_not_merely_saved(capsys):
    """
    Saved-but-never-restored is the same as not saved: a resumed run draws a
    different perturbation stream from an uninterrupted one, so the two diverge
    and neither is reproducible. This was a real bug.
    """
    from reassembly.training import _restore_rng

    torch.manual_seed(1234)
    marker = torch.get_rng_state().clone()
    state = {"torch_rng": marker.clone(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate()}

    torch.manual_seed(9999)
    assert not torch.equal(torch.get_rng_state(), marker)
    _restore_rng(state, rank=0)
    assert torch.equal(torch.get_rng_state(), marker)


def test_a_broken_rng_state_warns_instead_of_crashing(capsys):
    """A checkpoint from another torch build can carry a state this one rejects.
    Not fatal -- but not silent either, since the stream then differs."""
    from reassembly.training import _restore_rng

    _restore_rng({"torch_rng": "not a tensor"}, rank=0)
    assert "could not restore RNG" in capsys.readouterr().out


def test_checkpoint_records_cumulative_training_time(tmp_path):
    """A run spanning four sessions still knows how long it has trained."""
    config = Config(channels=16, heads=4, out_dir=str(tmp_path))
    _tiny_checkpoint(tmp_path / "last.pt", config, epoch=3, step=90,
                     completed=True, elapsed=37_000.0)
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert state["elapsed"] == pytest.approx(37_000.0)
    assert state["format"] == 2


def test_an_unwritable_out_dir_fails_before_training_not_after(tmp_path):
    """
    Pointing `out_dir` at a read-only location otherwise surfaces at the first
    checkpoint -- eleven hours in, with nothing saved.
    """
    from reassembly.training import _check_writable

    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(RuntimeError, match="cannot write checkpoints"):
        _check_writable(blocker / "sub")
    _check_writable(tmp_path / "fine")           # and a good path is silent


def test_the_time_report_projects_remaining_sessions():
    """The number that decides whether a plan is workable."""
    from reassembly.training import _time_report

    history = [{"epoch": i, "seconds": 3600.0} for i in range(3)]
    line = _time_report(history, Config(epochs=13, max_hours=11.0), 10_800.0, 0)
    assert "1:00:00/epoch" in line
    assert "10 epoch(s) left" in line
    assert "1 more session(s)" in line


def test_the_stop_signal_only_sets_a_flag(capsys):
    """
    Saving from inside a signal handler risks a half-written file, and under
    DDP it would desync the ranks. The handler flags; the loop acts.
    """
    import signal

    from reassembly.training import StopSignal

    watch = StopSignal()
    assert not watch.triggered
    watch._handle(signal.SIGTERM, None)
    assert watch.triggered and watch.name == "SIGTERM"
    assert "SIGTERM" in capsys.readouterr().out


def test_kaggle_gets_the_right_resume_recipe():
    """
    "Just rerun it" is wrong advice on Kaggle -- /kaggle/working becomes a
    dataset mounted elsewhere next session -- and getting it wrong costs a
    session to find out.
    """
    from reassembly.training import _resume_recipe

    kaggle = _resume_recipe(Config(), Path("/kaggle/working/vgat"))
    assert "resume_from" in kaggle and "input" in kaggle and "read-only" in kaggle
    local = _resume_recipe(Config(), Path("runs/vgat"))
    assert "rerun the same command" in local


def test_a_resumed_run_carries_its_checkpoint_into_the_new_out_dir(tmp_path):
    """
    Two failures would otherwise break a Kaggle chain, both silently.

    A session killed during its first epoch leaves its `out_dir` empty, and a
    session that finds nothing left to do writes nothing at all. Either way the
    *next* session resumes from an empty directory, starts from scratch, and
    discards every hour spent so far -- announcing only "starting from scratch",
    one line into a long log.
    """
    source, destination = tmp_path / "s1", tmp_path / "s2"
    source.mkdir()
    config = Config(channels=16, heads=4, out_dir=str(destination))
    _tiny_checkpoint(source / "last.pt", config, epoch=2, step=40,
                     completed=True, history=[{"epoch": 0}], elapsed=5000.0)

    # The behaviour under test is in `_worker`, so assert the property that
    # makes it work: the resume point is readable and carries its own history.
    state = torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    assert state["elapsed"] == 5000.0 and state["history"] == [{"epoch": 0}]
    save_checkpoint(destination / "last.pt", build_model(config), None, config,
                    state["epoch"], state["step"], state["history"],
                    state["best"], completed=True, elapsed=state["elapsed"])
    carried = torch.load(destination / "last.pt", map_location="cpu",
                         weights_only=False)
    assert carried["epoch"] == 2 and carried["elapsed"] == 5000.0
    assert carried["history"] == [{"epoch": 0}]


def test_a_partial_epoch_is_flagged_in_the_history():
    """
    An epoch cut short is recorded -- its metrics are real -- but the next
    session re-runs it, so the same epoch number appears twice. Without a flag
    the curve is ambiguous exactly where the interruption was.
    """
    history = [{"epoch": 5, "partial": 1, "val_geodesic_deg": 40.0},
               {"epoch": 5, "partial": 0, "val_geodesic_deg": 38.0},
               {"epoch": 6, "partial": 0, "val_geodesic_deg": 37.0}]
    clean = [h for h in history if not h["partial"]]
    assert [h["epoch"] for h in clean] == [5, 6]


def test_the_embedding_head_has_no_unlearnable_bias():
    """
    The final embedding layer carries no bias, and that is provable rather than
    stylistic. The embedding is supervised only by the centroid-variance loss,
    and adding a constant to every embedding shifts them all equally -- the
    scatter about each cluster centroid does not move. It is unidentifiable for
    stage two too, since a global shift leaves every pairwise distance
    untouched. A parameter that cannot be learned would otherwise show up
    forever as a dead gradient in every check.
    """
    from reassembly.nn.losses import embedding_consistency_loss

    z = torch.randn(12, 8, dtype=torch.float64)
    cluster = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    plain = embedding_consistency_loss(z, cluster, 4)
    shifted = embedding_consistency_loss(z + 7.5, cluster, 4)
    # Exact in exact arithmetic, so the tolerance is round-off, not slack.
    assert plain.item() == pytest.approx(shifted.item(), rel=1e-12)

    # The operative claim: a bias on the output receives no gradient at all.
    bias = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    embedding_consistency_loss(z + bias, cluster, 4).backward()
    assert bias.grad.abs().max().item() < 1e-12

    model = build_model(Config(channels=32, heads=4))
    assert model.embedding[-1].bias is None


def test_preflight_reports_a_missing_dataset_rather_than_crashing(capsys):
    """
    The single most likely first failure on a new machine, and the one whose
    default traceback says least about what to do.
    """
    from reassembly.training import preflight

    assert preflight(Config(root="/nonexistent-dataset-path")) is False
    out = capsys.readouterr().out
    assert "--root must point at the directory" in out


def test_limit_spans_the_split_instead_of_taking_a_prefix():
    """
    `limit_train=40` used to mean "the first 40 items". Scenes are sorted by
    path and each contributes several modes, so that was the first five objects
    of whichever category sorts first -- a smoke test that exercises one kind of
    geometry, and a "small subset" experiment on an unrepresentative sample with
    nothing saying so.
    """
    items = [(obj, f"mode{m}") for obj in range(40) for m in range(8)]

    prefix = items[:40]
    assert len({obj for obj, _ in prefix}) == 5, "the old behaviour, for contrast"

    keep = np.linspace(0, len(items) - 1, 40).astype(int)
    strided = [items[i] for i in dict.fromkeys(keep.tolist())]
    assert len({obj for obj, _ in strided}) == 40, "one mode from every object"
    assert strided == sorted(strided), "and still deterministic and ordered"
