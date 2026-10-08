"""
The Hugging Face Hub mirror (``reassembly.hub``): off unless all three
settings are given, pulled before a resume into an empty folder, pushed after
every epoch, and never allowed to stop the run or to leak the token.

``huggingface_hub`` is replaced by a stand-in that keeps the "repository" in a
local folder, so these run offline and check what was actually sent.
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reassembly.hub import FILES, HubSync

TOKEN = "hf_not-a-real-token-123"
REPO = "someone/reassembly-run"


class FakeHub:
    """Records every call; the repository is the folder ``remote``."""

    def __init__(self, remote: Path):
        self.remote = remote
        self.calls = []
        self.fail_create = False
        self.fail_push = False
        self.on_upload = None          # called with the folder at upload time

    def module(self) -> types.ModuleType:
        hub = self

        class HfApi:
            def __init__(self, token=None):
                hub.calls.append(("HfApi", token))

            def create_repo(self, repo_id, *, private=None, exist_ok=False, **_):
                if hub.fail_create:
                    raise RuntimeError("401 Client Error: Unauthorized")
                hub.calls.append(("create_repo", repo_id, private, exist_ok))
                hub.remote.mkdir(parents=True, exist_ok=True)

            def upload_folder(self, *, folder_path, repo_id, commit_message=None,
                              allow_patterns=None, **_):
                if hub.fail_push:
                    raise ConnectionError("network unreachable")
                if hub.on_upload:
                    hub.on_upload(Path(folder_path))
                hub.calls.append(("upload_folder", repo_id, str(folder_path),
                                  commit_message, tuple(allow_patterns)))
                for name in allow_patterns:
                    if (Path(folder_path) / name).is_file():
                        shutil.copy(Path(folder_path) / name, hub.remote / name)

        def snapshot_download(repo_id, *, token=None, allow_patterns=None,
                              local_dir=None, **_):
            hub.calls.append(("snapshot_download", repo_id, token,
                              tuple(allow_patterns), str(local_dir)))
            for name in allow_patterns:
                if (hub.remote / name).is_file():
                    shutil.copy(hub.remote / name, Path(local_dir) / name)
            return str(local_dir)

        module = types.ModuleType("huggingface_hub")
        module.HfApi, module.snapshot_download = HfApi, snapshot_download
        return module

    def named(self, name):
        return [call for call in self.calls if call[0] == name]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    hub = FakeHub(tmp_path / "remote")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    return hub


# --------------------------------------------------------------------------
# When it is on
# --------------------------------------------------------------------------

def test_off_unless_all_three_are_given(tmp_path, fake, capsys):
    assert not HubSync().start(tmp_path, resume=True)
    assert capsys.readouterr().out == ""                 # as before: silent
    for partial in [(REPO, "", ""), (REPO, str(tmp_path), ""), ("", str(tmp_path), TOKEN)]:
        assert not HubSync(*partial).start(tmp_path, resume=True)
        assert "all needed" in capsys.readouterr().out
    assert fake.calls == []


def test_the_folder_must_be_the_runs_own(tmp_path, fake, capsys):
    hub = HubSync(REPO, str(tmp_path / "elsewhere"), TOKEN)
    assert not hub.start(tmp_path / "run", resume=True) and not hub.enabled
    assert "is not the run's folder" in capsys.readouterr().out
    # The same folder spelled differently is the same folder.
    (tmp_path / "run").mkdir()
    assert HubSync(REPO, str(tmp_path / "run" / ".." / "run"), TOKEN).start(
        tmp_path / "run", resume=False)


def test_a_missing_library_or_a_bad_token_turns_it_off(tmp_path, fake, monkeypatch, capsys):
    fake.fail_create = True
    assert not HubSync(REPO, str(tmp_path), TOKEN).start(tmp_path, resume=False)
    assert "cannot reach or create" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)        # not installed
    assert not HubSync(REPO, str(tmp_path), TOKEN).start(tmp_path, resume=False)
    assert "not installed" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Pull, push
# --------------------------------------------------------------------------

def test_start_makes_a_private_repo_and_pulls_into_an_empty_folder(tmp_path, fake):
    fake.remote.mkdir()
    for name in ("last.pt", "history.json"):
        (fake.remote / name).write_text(f"remote {name}")
    run = tmp_path / "run"
    hub = HubSync(REPO, str(run), TOKEN)
    assert hub.start(run, resume=True) and hub.enabled
    assert fake.named("create_repo") == [("create_repo", REPO, True, True)]
    assert fake.named("snapshot_download") == [
        ("snapshot_download", REPO, TOKEN, FILES, str(run))]
    assert (run / "last.pt").read_text() == "remote last.pt"


def test_a_folder_that_has_last_pt_is_never_overwritten(tmp_path, fake):
    fake.remote.mkdir()
    (fake.remote / "last.pt").write_text("remote, older")
    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_text("local, newer")
    assert HubSync(REPO, str(run), TOKEN).start(run, resume=True)
    assert fake.named("snapshot_download") == []
    assert (run / "last.pt").read_text() == "local, newer"
    # A fresh start does not pull either.
    (run / "last.pt").unlink()
    assert HubSync(REPO, str(run), TOKEN).start(run, resume=False)
    assert fake.named("snapshot_download") == []


def test_push_is_one_commit_and_a_failure_does_not_stop_anything(tmp_path, fake, capsys):
    hub = HubSync(REPO, str(tmp_path), TOKEN)
    assert not hub.push("before start")                   # off until started
    hub.start(tmp_path, resume=False)
    (tmp_path / "last.pt").write_text("weights")
    assert hub.push("epoch 3")
    assert fake.named("upload_folder") == [
        ("upload_folder", REPO, str(tmp_path), "epoch 3", FILES)]
    fake.fail_push = True
    assert not hub.push("epoch 4")                        # warned, not raised
    assert "failed, continuing" in capsys.readouterr().out


def test_the_token_is_never_printed(tmp_path, fake, capsys):
    hub = HubSync(REPO, str(tmp_path), TOKEN)
    hub.start(tmp_path, resume=True)
    hub.push("epoch 1")
    fake.fail_push = True
    hub.push("epoch 2")
    assert TOKEN not in capsys.readouterr().out + repr(hub)


# --------------------------------------------------------------------------
# Through the command line and the training loop
# --------------------------------------------------------------------------

def test_the_flags_are_not_config_fields():
    """The token must not reach a checkpoint, and checkpoints are uploaded."""
    import dataclasses

    from reassembly.training import Config
    from scripts.train import build_parser, config_from_args

    args = build_parser().parse_args(["--hf_repo_id", REPO, "--hf_local_dir", "./ck",
                                      "--hf_token", TOKEN, "--checkpoint_dir", "./ck"])
    assert (args.hf_repo_id, args.hf_local_dir, args.hf_token) == (REPO, "./ck", TOKEN)
    config = config_from_args(args)
    assert TOKEN not in repr(dataclasses.asdict(config))
    assert not {f.name for f in dataclasses.fields(Config)} & {"hf_repo_id", "hf_local_dir",
                                                               "hf_token"}


def test_training_pushes_every_epoch_and_a_fresh_machine_resumes_from_the_hub(
        tmp_path, fake, capsys):
    from test_split_integration import _config, _dataset

    import reassembly.training as training

    root = _dataset(tmp_path / "data")
    out = tmp_path / "out"
    config = _config(root, out_dir=str(out), epochs=2, steps_per_epoch=1)

    seen = []
    fake.on_upload = lambda folder: seen.append(sorted(p.name for p in folder.iterdir()))
    training.train(config, hub=HubSync(REPO, str(out), TOKEN))
    pushes = fake.named("upload_folder")
    assert [call[3].split(":")[0] for call in pushes] == ["epoch 1", "epoch 2"]
    # Each push came after that epoch's files were written.
    assert all({"last.pt", "history.json", "history.csv"} <= set(names) for names in seen)
    assert TOKEN.encode() not in (out / "last.pt").read_bytes()

    # A fresh machine: the local folder is gone, the repository has the run.
    shutil.rmtree(out)
    capsys.readouterr()
    more = _config(root, out_dir=str(out), epochs=3, steps_per_epoch=1)
    history = training.train(more, hub=HubSync(REPO, str(out), TOKEN))
    printed = capsys.readouterr().out
    assert "[hub] pulled" in printed and "resumed from" in printed
    assert [row["epoch"] for row in history] == [0, 1, 2]


def test_training_without_the_mirror_never_touches_the_hub(tmp_path, fake):
    from test_split_integration import _config, _dataset

    import reassembly.training as training

    root = _dataset(tmp_path / "data")
    training.train(_config(root, out_dir=str(tmp_path / "out"), epochs=1, steps_per_epoch=1))
    training.train(_config(root, out_dir=str(tmp_path / "out2"), epochs=1, steps_per_epoch=1),
                   hub=HubSync())
    assert fake.calls == []


def test_the_epoch_a_session_stops_on_is_pushed_too(tmp_path, fake):
    """The time budget ends a session with a break; the push comes before it,
    or the session's last epoch would exist only on the machine being lost."""
    from test_split_integration import _config, _dataset

    import reassembly.training as training

    root = _dataset(tmp_path / "data")
    out = tmp_path / "out"
    config = _config(root, out_dir=str(out), epochs=3, steps_per_epoch=1, max_hours=1e-9)
    history = training.train(config, hub=HubSync(REPO, str(out), TOKEN))
    assert len(history) == 1                               # stopped for time
    assert [call[3].split(":")[0] for call in fake.named("upload_folder")] == ["epoch 1"]
