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
    # One call per category: warnings.filterwarnings asserts
    # isinstance(category, type), so a tuple raises "category must be a class".
    for category in (FutureWarning, DeprecationWarning):
        warnings.filterwarnings(
            "ignore", category=category,
            message=r".*torch\.(cuda\.)?amp\..*is deprecated.*",
        )
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message=r".*TypedStorage is deprecated.*",
    )


class MetricTable:
    """A progress bar with an aligned metric table underneath it.

        E000 train  67%|######7   | 100/150 [00:45<00:22,  2.21batch/s]
              total     rot     pos    node     mid    face    embv    embe     deg
              9.374   5.857   0.154   0.951   0.153   1.902   0.001   0.000  123.37

    Implemented as three stacked ``tqdm`` bars: the real one, plus two whose
    ``bar_format`` is just ``{desc}`` so they render as static lines that can
    be rewritten in place. That is the only way to keep column headers pinned
    above live numbers without reprinting the header on every update.

    Falls back to a single bar with an inline postfix if the extra lines
    cannot be created -- some notebook frontends only render one bar per cell,
    and a broken layout should degrade rather than raise.
    """

    def __init__(
        self,
        iterable,
        columns: Sequence[str],
        desc: str = "",
        total: Optional[int] = None,
        disable: bool = False,
        width: int = 8,
    ):
        self.columns = list(columns)
        self.width = width
        self.disable = disable
        self._fallback = False

        self.bar = tqdm(iterable, desc=desc, total=total, disable=disable,
                        leave=False, unit="batch", position=0, dynamic_ncols=True)
        self.header = None
        self.values = None
        if disable:
            return

        try:
            self.header = tqdm(total=0, position=1, bar_format="{desc}",
                               leave=False, dynamic_ncols=True)
            self.values = tqdm(total=0, position=2, bar_format="{desc}",
                               leave=False, dynamic_ncols=True)
            self.header.set_description_str(self._row(self.columns))
            self.values.set_description_str(self._row(["-"] * len(self.columns)))
        except Exception:                                  # noqa: BLE001
            self._fallback = True
            for extra in (self.header, self.values):
                if extra is not None:
                    extra.close()
            self.header = self.values = None

    def _row(self, cells: Sequence[str]) -> str:
        return "  " + "".join(f"{c:>{self.width}}" for c in cells)

    def __iter__(self):
        return iter(self.bar)

    def set_desc(self, text: str) -> None:
        self.bar.set_description(text)

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        if self.disable:
            return
        cells = []
        for name in self.columns:
            value = metrics.get(name)
            if value is None:
                cells.append("-")
            elif abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
                cells.append(f"{value:.1e}")
            else:
                cells.append(f"{value:.3f}")
        if self._fallback or self.values is None:
            self.bar.set_postfix(dict(zip(self.columns, cells)), refresh=False)
        else:
            self.values.set_description_str(self._row(cells))

    def close(self) -> None:
        for extra in (self.values, self.header):
            if extra is not None:
                extra.close()
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
