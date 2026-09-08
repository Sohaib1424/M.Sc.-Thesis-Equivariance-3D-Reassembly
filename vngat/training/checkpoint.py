"""
Checkpointing, TensorFlow.

POLICY, unchanged from the PyTorch branch:
  * `last.pt`-equivalent (`last/`) is written EVERY save and is what `resume`
    reads. It exists because the rolling checkpoint below can FREEZE -- in one
    real run it stuck at epoch 69 while training continued to epoch 100, so
    resuming would silently have discarded 31 epochs.
  * `checkpoint/` is replaced every `save_every` epochs UNLESS the stored one is
    better on BOTH train and validation loss. Deliberately strict: improving
    train while validation worsens is overfitting, and the reverse is usually
    noise, so neither alone may overwrite a genuinely better earlier state.
  * `best/` is updated whenever validation improves, so the strict rolling
    policy can never lose the best model.

Weights go through Keras; the surrounding state (epoch, losses, history, config,
optimizer variables) goes to a sidecar `.npz` + `.json`, so a checkpoint stays
loadable without reconstructing the exact optimizer object.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..utils.progress import write

ROLLING, BEST, LAST = "checkpoint", "best", "last"
HISTORY_NAME = "history.json"


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, save_every: int = 10,
                 drive=None, tag: str = "vngat"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.drive = drive
        self.tag = tag
        self.stored_train = self.stored_val = self.best_val = float("inf")

    def local(self, name: str) -> Path:
        return self.dir / name

    def should_save(self, epoch: int, total_epochs: int) -> bool:
        return ((epoch + 1) % self.save_every == 0) or (epoch + 1 == total_epochs)

    def is_improvement(self, train_loss: float, val_loss: float) -> bool:
        """Replace unless the stored checkpoint beats the new one on BOTH."""
        return not ((self.stored_val <= val_loss) and (self.stored_train <= train_loss))

    def _write(self, model, optimizer, name: str, meta: Dict) -> None:
        d = self.local(name)
        d.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(d / "model.weights.h5"))
        if optimizer is not None:
            try:
                np.savez(d / "optimizer.npz",
                         **{f"v{i}": v.numpy() for i, v in enumerate(optimizer.variables)})
            except Exception as exc:  # noqa: BLE001
                write(f"  [ckpt] optimizer state not saved ({type(exc).__name__}); "
                      f"weights are still resumable")
        # Atomic: a session killed mid-write otherwise leaves a truncated file
        # that fails to load, at the worst possible moment.
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, d / "state.json")

    def save(self, model, optimizer, epoch: int, train_loss: float, val_loss: float,
             history: Dict, config: Dict, force: bool = False) -> Dict[str, bool]:
        # best_val is updated BEFORE the state is written, so BOTH files record
        # it. Otherwise resuming from the rolling checkpoint restores
        # best_val = inf and the first validation of the next session always
        # looks like a new best -- destroying the previous session's best model.
        improved_best = val_loss < self.best_val
        if improved_best:
            self.best_val = val_loss
        meta = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                    best_val=self.best_val, history=history, config=config)

        wrote = {"last": True, "rolling": False, "best": False}
        self._write(model, optimizer, LAST, meta)      # always current
        if force or self.is_improvement(train_loss, val_loss):
            self._write(model, optimizer, ROLLING, meta)
            self.stored_train, self.stored_val = train_loss, val_loss
            wrote["rolling"] = True
        if improved_best:
            self._write(model, optimizer, BEST, meta)
            wrote["best"] = True

        tmp = self.local(HISTORY_NAME + ".tmp")
        tmp.write_text(json.dumps(history))
        os.replace(tmp, self.local(HISTORY_NAME))
        return wrote

    def locate(self, resume: str) -> Optional[Path]:
        if resume in ("", "none", "None"):
            return None
        if resume != "auto":
            p = Path(resume)
            return p if (p / "state.json").is_file() else None
        # Prefer `last`: it is always current, whereas the rolling checkpoint
        # may have been held back by the replace-unless-better policy.
        for name in (LAST, ROLLING):
            if (self.local(name) / "state.json").is_file():
                return self.local(name)
        return None

    def load(self, path: Path, model, optimizer=None) -> Dict[str, Any]:
        meta = json.loads((path / "state.json").read_text())
        model.load_weights(str(path / "model.weights.h5"))
        opt_file = path / "optimizer.npz"
        if optimizer is not None and opt_file.is_file():
            try:
                data = np.load(opt_file)
                for i, v in enumerate(optimizer.variables):
                    key = f"v{i}"
                    if key in data:
                        v.assign(data[key])
            except Exception as exc:  # noqa: BLE001
                write(f"  [ckpt] optimizer state not restored ({type(exc).__name__}); "
                      f"the LR schedule and moments restart")
        self.stored_train = float(meta.get("train_loss", float("inf")))
        self.stored_val = float(meta.get("val_loss", float("inf")))
        self.best_val = float(meta.get("best_val", self.stored_val))
        return meta
