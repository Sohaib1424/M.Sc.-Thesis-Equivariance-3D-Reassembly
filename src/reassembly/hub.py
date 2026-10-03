"""
Hugging Face Hub mirroring of a run's files -- optional.

On only when all three are given: ``--hf_repo_id``, ``--hf_local_dir`` and
``--hf_token``. With none of them nothing here runs and training is exactly
what it was; with some but not all, one line says so and training goes on
without the mirror.

When on:

* at startup, a run resuming with ``--resume auto`` into a folder that has no
  ``last.pt`` yet (a fresh machine) first pulls the files from the repository,
  so the resume finds them. A folder that already has ``last.pt`` is never
  overwritten: it is at least as new as the last upload;
* after every epoch, once that epoch's files are written, they are pushed in
  one commit.

``--hf_local_dir`` must be the run's own folder (``--checkpoint_dir``): that is
where :func:`reassembly.training.train` writes ``last.pt``, ``best.pt``,
``history.json``, ``history.csv`` and ``offenders.json``, so it is what is
pushed and pulled into.

The token lives in memory only. It is not a :class:`~reassembly.training.Config`
field, so it is never written into a checkpoint -- which matters, because the
checkpoints are themselves uploaded -- and it is never printed.

Every Hub call is guarded: a failed pull or push, or a missing
``huggingface_hub``, prints a warning and training continues with its local
files. Losing the mirror must never cost the run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

FILES = ("last.pt", "best.pt", "history.json", "history.csv", "offenders.json")


class HubSync:
    """The three settings, and whether the mirror is on for this run."""

    def __init__(self, repo_id: str = "", local_dir: str = "", token: str = ""):
        self.repo_id = (repo_id or "").strip()
        self.local_dir = (local_dir or "").strip()
        self._token = (token or "").strip()
        self.enabled = False

    def __repr__(self) -> str:                    # never the token
        return (f"HubSync(repo_id={self.repo_id!r}, local_dir={self.local_dir!r}, "
                f"enabled={self.enabled})")

    @property
    def requested(self) -> bool:
        """All three settings given."""
        return bool(self.repo_id and self.local_dir and self._token)

    @property
    def partly_given(self) -> bool:
        return bool(self.repo_id or self.local_dir or self._token) and not self.requested

    # ------------------------------------------------------------------
    def start(self, run_dir, resume: bool) -> bool:
        """
        Turn the mirror on if the settings are usable, creating the repository
        (private) if it does not exist; then pull, if ``resume`` and
        ``run_dir`` has no ``last.pt``. Returns whether the mirror is on.
        """
        self.enabled = False
        if not self.requested:
            if self.partly_given:
                print("[hub] off: --hf_repo_id, --hf_local_dir and --hf_token are all "
                      "needed to mirror to the Hugging Face Hub")
            return False
        if Path(self.local_dir).resolve() != Path(run_dir).resolve():
            print(f"[hub] off: --hf_local_dir {self.local_dir} is not the run's folder "
                  f"{run_dir} (--checkpoint_dir), where the files are written")
            return False
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("[hub] off: huggingface_hub is not installed (pip install huggingface_hub)")
            return False
        try:
            HfApi(token=self._token).create_repo(self.repo_id, private=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 -- the mirror must not stop the run
            print(f"[hub] off: cannot reach or create {self.repo_id}: "
                  f"{type(exc).__name__}: {exc}")
            return False
        self.enabled = True
        print(f"[hub] mirroring {', '.join(FILES)} to {self.repo_id} after every epoch")
        if resume and not (Path(self.local_dir) / "last.pt").is_file():
            self.pull()
        return True

    def pull(self) -> List[str]:
        """The files in ``FILES`` from the repository into ``local_dir``; the
        names that arrived. Never raises."""
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.repo_id, token=self._token,
                              allow_patterns=list(FILES), local_dir=self.local_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[hub] nothing pulled from {self.repo_id}: {type(exc).__name__}: {exc}")
            return []
        pulled = _present(self.local_dir, FILES)
        print(f"[hub] pulled {', '.join(pulled) if pulled else 'nothing'} from {self.repo_id}")
        return pulled

    def push(self, message: str) -> bool:
        """The files in ``FILES`` from ``local_dir`` to the repository, in one
        commit. Never raises; returns whether it worked."""
        if not self.enabled:
            return False
        started = time.perf_counter()
        try:
            from huggingface_hub import HfApi

            HfApi(token=self._token).upload_folder(
                folder_path=self.local_dir, repo_id=self.repo_id,
                commit_message=message, allow_patterns=list(FILES),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [hub] push to {self.repo_id} failed, continuing with local files: "
                  f"{type(exc).__name__}: {exc}")
            return False
        print(f"  [hub] pushed {', '.join(_present(self.local_dir, FILES))} to "
              f"{self.repo_id} in {time.perf_counter() - started:.1f} s")
        return True


def _present(folder, names: Sequence[str]) -> List[str]:
    return [name for name in names if (Path(folder) / name).is_file()]


__all__ = ["FILES", "HubSync"]
