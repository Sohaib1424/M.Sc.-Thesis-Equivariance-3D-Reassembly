"""
Checkpointing.

POLICY (exactly as specified for this project)
----------------------------------------------
* A rolling checkpoint is written every `save_every` epochs and REPLACES the
  previous one -- UNLESS the previous one is better on BOTH validation and
  training loss, in which case the older file is kept and the new one is
  discarded. "Better on both" is deliberately strict: a run that improves
  training loss while validation loss worsens is overfitting, and a run that
  improves validation while training worsens is usually noise, so neither
  alone is allowed to overwrite a genuinely better earlier state.
* `best.pt` is maintained separately and updated whenever validation improves,
  so the strict rolling policy can never lose the best model.
* Every checkpoint is a FULL resumable state -- model, optimiser, scheduler,
  AMP scaler, epoch counter, best-so-far metrics, RNG states, and the config
  it was produced with -- because a Kaggle run is resumed across sessions and
  restoring weights alone silently restarts the LR schedule and the optimiser
  moments.

Writes are atomic (temp file + `os.replace`): a session killed mid-write
otherwise leaves a truncated file that fails to load, which is the worst
possible time to discover it.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.progress import write
from .drive import DriveSync

ROLLING_NAME = "checkpoint.pt"
BEST_NAME = "best.pt"
HISTORY_NAME = "history.json"


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 10,
        drive: Optional[DriveSync] = None,
        tag: str = "vngat",
    ):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.drive = drive
        self.tag = tag
        self.stored_train: float = float("inf")
        self.stored_val: float = float("inf")
        self.best_val: float = float("inf")

    # -- naming ---------------------------------------------------------
    def _remote(self, name: str) -> str:
        return f"{self.tag}_{name}"

    def local(self, name: str) -> Path:
        return self.dir / name

    # -- saving ---------------------------------------------------------
    def should_save(self, epoch: int, total_epochs: int) -> bool:
        return ((epoch + 1) % self.save_every == 0) or (epoch + 1 == total_epochs)

    def is_improvement(self, train_loss: float, val_loss: float) -> bool:
        """
        Replace the stored checkpoint unless the stored one beats the new one
        on BOTH metrics.
        """
        stored_better_on_both = (self.stored_val <= val_loss) and (self.stored_train <= train_loss)
        return not stored_better_on_both

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        epoch: int,
        train_loss: float,
        val_loss: float,
        history: Dict,
        config: Dict,
        force: bool = False,
    ) -> Dict[str, bool]:
        """Returns {'rolling': bool, 'best': bool} -- what was actually written."""
        state = {
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": self.best_val,
            "history": history,
            "config": config,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        wrote = {"rolling": False, "best": False}

        if force or self.is_improvement(train_loss, val_loss):
            _atomic_save(state, self.local(ROLLING_NAME))
            self.stored_train, self.stored_val = train_loss, val_loss
            wrote["rolling"] = True
        else:
            write(
                f"  [ckpt] kept the previous checkpoint "
                f"(stored train={self.stored_train:.4f}/val={self.stored_val:.4f} beats "
                f"current train={train_loss:.4f}/val={val_loss:.4f} on both)"
            )

        if val_loss < self.best_val:
            self.best_val = val_loss
            state["best_val"] = val_loss
            _atomic_save(state, self.local(BEST_NAME))
            wrote["best"] = True

        _write_history(self.local(HISTORY_NAME), history)

        if self.drive is not None and self.drive.enabled:
            if wrote["rolling"]:
                self.drive.upload(str(self.local(ROLLING_NAME)), self._remote(ROLLING_NAME))
            if wrote["best"]:
                self.drive.upload(str(self.local(BEST_NAME)), self._remote(BEST_NAME))
            self.drive.upload(str(self.local(HISTORY_NAME)), self._remote(HISTORY_NAME))

        return wrote

    # -- loading --------------------------------------------------------
    def locate(self, resume: str) -> Optional[Path]:
        """Resolve `resume` to a concrete file, pulling from Drive if needed."""
        if resume in ("", "none", "None"):
            return None
        if resume != "auto":
            path = Path(resume)
            return path if path.is_file() else None

        local = self.local(ROLLING_NAME)
        if self.drive is not None and self.drive.enabled:
            # Prefer the remote copy: after a session restart the local
            # directory is empty, and if it is NOT empty the remote is at worst
            # the same epoch (it is uploaded immediately after every local save).
            if self.drive.download(self._remote(ROLLING_NAME), str(local)):
                write(f"  [ckpt] pulled {self._remote(ROLLING_NAME)} from Google Drive")
                self.drive.download(self._remote(HISTORY_NAME), str(self.local(HISTORY_NAME)))
        return local if local.is_file() else None

    def load(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str = "cpu",
        restore_rng: bool = True,
    ) -> Dict:
        state = torch.load(path, map_location=map_location, weights_only=False)
        _unwrap(model).load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])

        self.stored_train = float(state.get("train_loss", float("inf")))
        self.stored_val = float(state.get("val_loss", float("inf")))
        self.best_val = float(state.get("best_val", self.stored_val))

        if restore_rng and state.get("rng"):
            try:
                rng = state["rng"]
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:  # noqa: BLE001
                write(f"  [ckpt] RNG state not restored ({type(exc).__name__}); continuing")
        return state


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _atomic_save(state: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _write_history(path: Path, history: Dict) -> None:
    import json

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(history, fh)
    os.replace(tmp, path)
