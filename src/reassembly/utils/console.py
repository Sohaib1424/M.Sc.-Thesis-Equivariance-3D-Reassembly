"""
Console output: warning filters and the live training display.

Two things live here because both are about making a long run readable.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

from tqdm.auto import tqdm


def quiet_third_party_warnings() -> None:
    """Silence known-noisy warnings from dependencies, and nothing else.

    Deliberately targeted rather than a blanket ``ignore``. A global filter
    would also hide warnings this project raises on purpose -- the degenerate
    fragment notices, the decimation budget notices, the fragment-count
    mismatch guard -- and those are the ones worth seeing.

    Suppressed:

    * ``torch.jit.script is deprecated`` -- emitted from inside
      torch_geometric at import time. Nothing here calls it.
    * ``The given NumPy array is not writable`` -- the project's own cause of
      this was fixed at source (``data/features.py`` now takes a writable copy
      of ``mesh.edges_unique``); the filter covers the same warning arriving
      from library internals.
    * ``torch.cuda.amp.* is deprecated`` -- version-shim churn in torch's AMP
      API, already handled in ``utils/memory.py``.
    """
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r".*torch\.jit\.script.*",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message=r".*given NumPy array is not writable.*",
    )
    # Two calls rather than one loop. `warnings.filterwarnings` asserts
    # `isinstance(category, type)`, so a TUPLE of categories raises
    # "category must be a class" -- and a loop to work around that introduces
    # an indented block, which is easy to lose when patching this by hand.
    # Every statement in this function sits at one indentation level on
    # purpose.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r".*torch\.(cuda\.)?amp\..*is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message=r".*torch\.(cuda\.)?amp\..*is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message=r".*TypedStorage is deprecated.*",
    )


class MetricTable:
    """A single progress bar with running metrics in its postfix.

    Deliberately ONE bar and nothing else.

    An earlier version stacked three tqdm bars -- a real one plus two whose
    ``bar_format`` was just ``{desc}`` -- to pin column headers above live
    numbers. That relies on ANSI cursor movement between lines, and Kaggle's
    notebook subprocess reports ``isatty() == True`` while not actually
    supporting it. The result was a header printed once, one new line per
    update, and the metric row bleeding into the end of the bar line.

    Detecting the difference is not reliably possible from inside the process,
    so this does not try. One bar rewrites one line, which every terminal and
    every notebook handles. The aligned table is printed once per epoch by
    :func:`format_epoch_line`, where there is no cursor trickery involved.
    """

    #: Short postfix keys -- the full names do not fit on one line alongside
    #: the bar, and an overflowing line wraps, which looks like the bar is
    #: broken.
    SHORT = {"total": "tot", "rot": "rot", "pos": "pos", "node": "nrm",
             "mid": "mid", "face": "fac", "embv": "ev", "embe": "ee",
             "rot_deg": "deg"}

    def __init__(
        self,
        iterable,
        columns: Sequence[str],
        desc: str = "",
        total: Optional[int] = None,
        disable: bool = False,
        width: int = 8,
        log_every: int = 30,
    ):
        self.columns = list(columns)
        self.width = width
        self.disable = disable
        self.bar = tqdm(iterable, desc=desc, total=total, disable=disable,
                        leave=False, unit="batch", dynamic_ncols=True)

    def __iter__(self):
        return iter(self.bar)

    def set_desc(self, text: str) -> None:
        self.bar.set_description(text)

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        if self.disable:
            return
        postfix = {}
        for name in self.columns:
            value = metrics.get(name)
            if value is None:
                continue
            key = self.SHORT.get(name, name)
            if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
                postfix[key] = f"{value:.1e}"
            else:
                postfix[key] = f"{value:.3f}"
        self.bar.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def format_epoch_line(epoch: int, train: Dict[str, float], val: Dict[str, float],
                      columns: Sequence[str], lr: float, seconds: float,
                      width: int = 8) -> str:
    """One compact block per finished epoch, kept aligned with the live table."""
    def row(label, m):
        cells = "".join(
            f"{m.get(c, float('nan')):>{width}.3f}" for c in columns
        )
        return f"  {label:<7}{cells}"

    head = "  " + f"{'':<7}" + "".join(f"{c:>{width}}" for c in columns)
    return (
        f"[epoch {epoch:04d}]  lr={lr:.2e}  {seconds:.1f}s\n"
        f"{head}\n{row('train', train)}\n{row('val', val)}"
    )
