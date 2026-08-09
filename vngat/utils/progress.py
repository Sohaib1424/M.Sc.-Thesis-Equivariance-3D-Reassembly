"""
Progress display.

Two deliberate choices, both driven by how Kaggle actually renders output:

1. ONE progress bar, reused for every phase of every epoch, never nested.
   Stacked/nested tqdm bars redraw with carriage returns that Kaggle's output
   cell does not handle -- you end up with hundreds of orphaned bar lines.

2. Per-epoch results go to a fixed-width TABLE printed with plain `print`
   (not through the bar), so the scroll-back is a readable log rather than
   the residue of a redrawn line.

`tqdm` is treated as optional: on a non-TTY stream (which is exactly what a
Kaggle notebook subprocess gets) it is driven with `ascii=True` and a slow
`mininterval` so it emits a bounded number of lines instead of thousands.
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional

try:  # pragma: no cover - trivial import guard
    from tqdm.auto import tqdm as _tqdm

    _HAVE_TQDM = True
except ImportError:  # pragma: no cover
    _HAVE_TQDM = False


def is_tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class NullBar:
    """Stand-in with tqdm's interface, for non-main ranks / no tqdm."""

    def __init__(self, iterable: Optional[Iterable] = None, **_: object) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable or ())

    def update(self, _n: int = 1) -> None: ...
    def set_description(self, _d: str) -> None: ...
    def set_postfix_str(self, _s: str) -> None: ...
    def reset(self, total: Optional[int] = None) -> None: ...
    def close(self) -> None: ...
    def write(self, msg: str) -> None:
        print(msg, flush=True)


def make_bar(total: int, desc: str, enabled: bool = True):
    """A single reusable bar. Call `.reset(total=...)` between phases rather
    than creating a new one."""
    if not enabled or not _HAVE_TQDM:
        return NullBar(total=total)
    tty = is_tty()
    return _tqdm(
        total=total,
        desc=desc,
        leave=True,
        unit="scene",
        dynamic_ncols=tty,
        ncols=None if tty else 100,
        ascii=not tty,
        # On a non-TTY (Kaggle subprocess) every refresh is a NEW line, so
        # refresh rarely. On a TTY, refresh often enough to feel live.
        mininterval=0.3 if tty else 10.0,
        file=sys.stderr,
    )


def write(msg: str) -> None:
    """Print without corrupting an active bar."""
    if _HAVE_TQDM:
        _tqdm.write(msg, file=sys.stderr)
    else:  # pragma: no cover
        print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Epoch results table
# --------------------------------------------------------------------------
_COLUMNS = [
    ("epoch", 6),
    ("phase", 6),
    ("total", 9),
    ("rot", 8),
    ("deg", 7),
    ("pos", 8),
    ("norm", 8),
    ("mid", 8),
    ("face", 8),
    ("emb_v", 8),
    ("emb_e", 8),
    ("data_s", 8),
    ("comp_s", 8),
]


def table_header() -> str:
    head = "".join(name.rjust(width) for name, width in _COLUMNS)
    return head + "\n" + "-" * len(head)


def table_row(epoch: int, phase: str, metrics: dict, data_s: float, comp_s: float) -> str:
    def fmt(key: str, width: int) -> str:
        value = metrics.get(key)
        if value is None:
            return "-".rjust(width)
        return f"{value:.4f}".rjust(width)

    cells = [
        str(epoch).rjust(6),
        phase.rjust(6),
        fmt("total", 9),
        fmt("rot", 8),
        fmt("rot_deg", 7),
        fmt("pos", 8),
        fmt("node", 8),
        fmt("mid", 8),
        fmt("face", 8),
        fmt("emb_v", 8),
        fmt("emb_e", 8),
        f"{data_s:.1f}".rjust(8),
        f"{comp_s:.1f}".rjust(8),
    ]
    return "".join(cells)
