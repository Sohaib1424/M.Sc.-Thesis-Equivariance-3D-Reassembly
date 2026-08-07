"""
Surviving a capped session.

Kaggle stops a notebook at 12 hours. A ~6-day training run is therefore about a
dozen restarts, and every one of them is a chance to lose work or silently
corrupt the record of what happened. Four things have to hold:

1. **Stop before being killed.** A process that is terminated mid-epoch loses
   that epoch. Predicting whether the next epoch fits in the remaining budget,
   and stopping cleanly if it does not, costs one partial epoch at most --
   once, at the end -- instead of once per session.

2. **Save enough to actually continue.** Model weights alone are not enough.
   Optimizer moments, the scheduler's position, the AMP scale factor and the
   RNG state all affect the next step. Restoring only weights restarts the
   optimizer from zero momentum, which shows up as a visible bump in the loss
   at every session boundary -- and twelve of those bumps look exactly like a
   model that will not converge.

3. **Keep one history, not twelve.** The loss curve has to span sessions, and
   it has to stay consistent when a run resumes from an older checkpoint than
   the newest history entry.

4. **Do the same thing on every rank.** Under DDP, one rank deciding to stop
   while the others continue hangs the job at the next collective. Every
   stop decision is voted on.
"""
from __future__ import annotations

import signal
import time
from typing import Dict, List, Optional, Tuple


class SessionLimit:
    """Decides when to stop, and why.

    Handles two independent reasons: a wall-clock budget, and a termination
    signal from the platform. Both resolve at an epoch boundary, where saving
    is cheap and every rank is already synchronised.
    """

    def __init__(self, budget_hours: Optional[float], main: bool = True,
                 safety_margin: float = 1.15):
        self.budget = budget_hours * 3600.0 if budget_hours else None
        self.start = time.monotonic()
        self.main = main
        self.safety_margin = safety_margin
        self.signalled: Optional[str] = None
        self._previous: Dict[int, object] = {}
        self._install()

    def _install(self) -> None:
        def handler(signum, _frame):
            # Only a flag is set here. Saving a checkpoint from inside a signal
            # handler races with whatever the main thread is doing to the same
            # tensors, and under DDP it would also mean one rank writing while
            # the others are mid-collective.
            self.signalled = signal.Signals(signum).name

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):
                # No handler outside the main thread; harmless.
                pass

    def restore(self) -> None:
        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> Optional[float]:
        return None if self.budget is None else self.budget - self.elapsed

    def should_stop(self, last_epoch_seconds: float) -> Tuple[bool, str]:
        """Called at an epoch boundary. Returns ``(stop, reason)``."""
        if self.signalled:
            return True, f"received {self.signalled}"

        if self.budget is None:
            return False, ""

        remaining = self.remaining
        if remaining <= 0:
            return True, f"time budget exhausted ({self.budget / 3600:.2f} h)"

        # Stop if the next epoch probably will not finish. The margin covers
        # epoch-to-epoch variation -- scenes differ in size, so epochs are not
        # uniform, and being 10% optimistic here costs a whole killed epoch.
        needed = last_epoch_seconds * self.safety_margin
        if last_epoch_seconds > 0 and needed > remaining:
            return True, (f"next epoch needs ~{needed / 60:.1f} min, "
                          f"{remaining / 60:.1f} min left in the budget")
        return False, ""

    def describe(self) -> str:
        if self.budget is None:
            return f"elapsed {self.elapsed / 3600:.2f} h (no budget set)"
        return (f"elapsed {self.elapsed / 3600:.2f} h of "
                f"{self.budget / 3600:.2f} h "
                f"({self.remaining / 60:.0f} min left)")


def merge_history(existing: Dict[str, Dict[str, List[float]]],
                  start_epoch: int) -> Dict[str, Dict[str, List[float]]]:
    """Trim a loaded history so it lines up with the epoch being resumed from.

    These can disagree. ``last.pt`` is written every ``save_every`` epochs while
    the history file is written every epoch, so a run resumed from an older
    checkpoint would otherwise append epoch 20 after an existing epoch 29 and
    produce a curve that folds back on itself. Truncating to ``start_epoch``
    makes the history match the weights actually being restored.
    """
    trimmed: Dict[str, Dict[str, List[float]]] = {}
    for phase, series in (existing or {}).items():
        trimmed[phase] = {k: list(v)[:start_epoch] for k, v in series.items()}
    for phase in ("train", "val"):
        trimmed.setdefault(phase, {})
    return trimmed


def rng_state() -> Dict:
    """Capture RNG state so resuming continues the stream rather than
    replaying it. Without this, every session revisits the same scenes in the
    same order with the same augmentations -- twelve sessions would show the
    model a twelfth of the variety it should see."""
    import random

    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: Optional[Dict]) -> bool:
    """Returns whether the stream was restored. Never raises: losing the RNG
    position is a small cost, and it must not be able to block a resume."""
    if not state:
        return False
    try:
        import random

        import numpy as np
        import torch

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu()
                            if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            devices = torch.cuda.device_count()
            saved = state["cuda"]
            if len(saved) == devices:
                torch.cuda.set_rng_state_all(saved)
        return True
    except Exception:                                        # noqa: BLE001
        # A checkpoint from a different device count or torch version, or an
        # environment without torch at all, should not block a resume.
        return False
