"""
Training/validation loss history.

Every component of the composite loss is recorded for BOTH phases, every
epoch, and flushed to disk (and to Drive) immediately -- not at the end of the
run. A 12-hour session that gets killed at hour 11.9 must still leave a
plottable record behind.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class History:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, List[float]]] = {"train": {}, "val": {}, "meta": {}}

    # ------------------------------------------------------------------
    def append(self, phase: str, metrics: Dict[str, float]) -> None:
        bucket = self.data.setdefault(phase, {})
        for key, value in metrics.items():
            bucket.setdefault(key, []).append(float(value))

    def append_meta(self, **values: float) -> None:
        bucket = self.data.setdefault("meta", {})
        for key, value in values.items():
            bucket.setdefault(key, []).append(float(value))

    @property
    def num_epochs(self) -> int:
        return len(self.data.get("train", {}).get("total", []))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return self.data

    @classmethod
    def from_dict(cls, data: Dict) -> "History":
        hist = cls()
        if data:
            hist.data = data
            hist.data.setdefault("train", {})
            hist.data.setdefault("val", {})
            hist.data.setdefault("meta", {})
        return hist

    @classmethod
    def load(cls, path: str | Path) -> "History":
        path = Path(path)
        if not path.is_file():
            return cls()
        try:
            with open(path) as fh:
                return cls.from_dict(json.load(fh))
        except (json.JSONDecodeError, OSError):
            return cls()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.data, fh)

    def truncate_to(self, num_epochs: int) -> None:
        """
        Drop entries past `num_epochs`.

        Needed on resume: a checkpoint written at epoch 20 may be accompanied
        by a history file that ran on to epoch 27 before the session died.
        Appending to that would misalign every curve against the epoch axis.
        """
        for phase in ("train", "val", "meta"):
            for key, values in self.data.get(phase, {}).items():
                self.data[phase][key] = values[:num_epochs]
