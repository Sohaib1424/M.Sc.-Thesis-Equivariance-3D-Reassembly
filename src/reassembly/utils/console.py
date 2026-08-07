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


def _is_interactive_terminal() -> bool:
    """Can the terminal rewrite a line in place?

    This is the difference between a progress bar and 150 lines of spam.
    `!python script.py` in a notebook runs a SUBPROCESS whose stdout is a pipe,
    not a TTY. tqdm still emits carriage returns, but the notebook renders each
    one as a new line -- and the multi-line cursor movement that stacks a
    header above live numbers does not work at all, so the header prints once
    and every update lands underneath it.

    When there is no TTY, printing a periodic status line is the only output
    that stays readable.
    """
    import os
    import sys

    if os.environ.get("REASSEMBLY_FORCE_BAR"):
        return True
    for stream in (sys.stderr, sys.stdout):
        try:
            if stream is not None and stream.isatty():
                return True
        except Exception:                                    # noqa: BLE001
            pass
    return False


class MetricTable:
    """Live training metrics, in whichever form the terminal can actually show.

    **Interactive terminal** -- a progress bar with an aligned table under it,
    rewritten in place:

        E000 train  67%|######7   | 100/150 [00:45<00:22,  2.21batch/s]
              total     rot     pos    node     mid    face    embv    embe     deg
              9.374   5.857   0.154   0.951   0.153   1.902   0.001   0.000  123.37

    **Piped output** (`!python ...` in a notebook, nohup, a log file) -- a
    header once, then one aligned row at intervals:

        step      total     rot     pos    node     mid    face    embv    embe     deg
         30/150    9.374   5.857   0.154   0.951   0.153   1.902   0.001   0.000  123.37
         60/150    8.911   5.512   0.149   0.938   0.150   1.874   0.001   0.000  119.02

    Same columns either way, so a log and a terminal read the same.
    """

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
        self.iterable = iterable
        self.desc = desc
        self.log_every = max(1, log_every)
        self.interactive = _is_interactive_terminal() and not disable
        self._step = 0
        self._last = {}
        self._header_written = False

        self.total = total
        if self.total is None:
            try:
                self.total = len(iterable)
            except TypeError:
                self.total = None

        self.bar = self.header = self.values = None
        if disable:
            return

        if self.interactive:
            self.bar = tqdm(iterable, desc=desc, total=self.total, leave=False,
                            unit="batch", position=0, dynamic_ncols=True)
            try:
                self.header = tqdm(total=0, position=1, bar_format="{desc}",
                                   leave=False, dynamic_ncols=True)
                self.values = tqdm(total=0, position=2, bar_format="{desc}",
                                   leave=False, dynamic_ncols=True)
                self.header.set_description_str(self._row(self.columns, "step"))
                self.values.set_description_str(self._row(["-"] * len(self.columns), ""))
            except Exception:                                # noqa: BLE001
                for extra in (self.header, self.values):
                    if extra is not None:
                        extra.close()
                self.header = self.values = None

    def _row(self, cells: Sequence[str], label: str) -> str:
        return f"  {label:>9}" + "".join(f"{c:>{self.width}}" for c in cells)

    def __iter__(self):
        if self.bar is not None:
            for item in self.bar:
                self._step += 1
                yield item
        else:
            for item in self.iterable:
                self._step += 1
                yield item
                self._maybe_log()

    def _maybe_log(self) -> None:
        if self.disable or self.interactive or not self._last:
            return
        if self._step % self.log_every and self._step != self.total:
            return
        if not self._header_written:
            print(self._row(self.columns, "step"), flush=True)
            self._header_written = True
        label = f"{self._step}/{self.total}" if self.total else str(self._step)
        print(self._row(self._format(self._last), label), flush=True)

    def _format(self, metrics: Dict[str, float]):
        cells = []
        for name in self.columns:
            value = metrics.get(name)
            if value is None:
                cells.append("-")
            elif abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
                cells.append(f"{value:.1e}")
            else:
                cells.append(f"{value:.3f}")
        return cells

    def set_desc(self, text: str) -> None:
        # Only meaningful for a redrawable bar; in a pipe it would print a line
        # per phase change, which is four lines per batch of pure noise.
        if self.bar is not None:
            self.bar.set_description(text)

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        if self.disable:
            return
        self._last = dict(metrics)
        if self.values is not None:
            self.values.set_description_str(self._row(self._format(metrics), ""))
        elif self.bar is not None:
            self.bar.set_postfix(dict(zip(self.columns, self._format(metrics))),
                                 refresh=False)

    def close(self) -> None:
        for extra in (self.values, self.header):
            if extra is not None:
                extra.close()
        if self.bar is not None:
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
