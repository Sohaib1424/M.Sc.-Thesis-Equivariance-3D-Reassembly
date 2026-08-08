"""Console helpers.

These exist because `quiet_third_party_warnings()` shipped broken: it passed a
TUPLE of categories to `warnings.filterwarnings`, which asserts
`isinstance(category, type)`. Every entry-point script calls it at import time,
so the failure was immediate and total -- `AssertionError: category must be a
class` before a single line of real work.

Nothing caught it because the static checks IMPORT modules, and this function
is only ever CALLED. A test that calls it is the whole fix.
"""
import warnings

import pytest

from reassembly.utils.console import (
    MetricTable, format_epoch_line, quiet_third_party_warnings,
)


def test_quiet_third_party_warnings_runs():
    quiet_third_party_warnings()


def test_quiet_third_party_warnings_is_idempotent():
    """Several scripts may call it, and a notebook re-runs cells freely."""
    for _ in range(3):
        quiet_third_party_warnings()


def test_every_filter_category_is_a_class():
    """The exact failure: filterwarnings requires a class, not a tuple."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        quiet_third_party_warnings()
        for entry in warnings.filters:
            category = entry[2]
            assert isinstance(category, type), f"{category!r} is not a class"


def test_targeted_not_blanket():
    """A global 'ignore' would also hide this project's own warnings -- the
    degenerate-fragment notices and the decimation budget notices are exactly
    the ones worth seeing."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.resetwarnings()
        quiet_third_party_warnings()
        warnings.warn("decimation budget unreachable for this scene", UserWarning)
        assert len(seen) == 1, "a project warning was suppressed"


def test_known_noise_is_actually_suppressed():
    with warnings.catch_warnings(record=True) as seen:
        warnings.resetwarnings()
        quiet_third_party_warnings()
        warnings.warn("The given NumPy array is not writable...", UserWarning)
        warnings.warn("torch.jit.script is deprecated", DeprecationWarning)
        assert len(seen) == 0


def test_epoch_line_is_column_aligned():
    cols = ("total", "rot", "pos")
    text = format_epoch_line(3, {"total": 1.0, "rot": 2.0, "pos": 3.0},
                             {"total": 1.5, "rot": 2.5, "pos": 3.5},
                             cols, lr=3e-4, seconds=12.3)
    lines = text.splitlines()
    assert "epoch 0003" in lines[0]
    # header, train and val rows must all be the same width
    assert len(lines[1]) == len(lines[2]) == len(lines[3])
    assert "train" in lines[2] and "val" in lines[3]


def test_epoch_line_survives_missing_metrics():
    """An epoch that processed zero usable batches reports NaN; formatting it
    must not raise on top of that."""
    text = format_epoch_line(0, {}, {}, ("total", "rot"), lr=1e-4, seconds=1.0)
    assert "nan" in text.lower()


# --------------------------------------------------------------------- output
def test_piped_output_logs_rows_instead_of_redrawing(capsys):
    """`!python script.py` in a notebook has no TTY. tqdm's in-place redraw
    becomes one new line per update, and the stacked header/value bars do not
    work at all -- the header prints once and every update lands beneath it."""
    cols = ("total", "rot")
    table = MetricTable(range(20), cols, log_every=5)
    assert not table.interactive, "no TTY under pytest capture"

    for i, _ in enumerate(table):
        table.update_metrics({"total": 1.0 + i, "rot": 2.0 + i})
    table.close()

    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 5, f"expected header + 4 rows, got {len(lines)}"
    assert "total" in lines[0] and "rot" in lines[0]
    assert "5/20" in lines[1]
    assert "20/20" in lines[-1]


def test_piped_output_writes_the_header_once(capsys):
    table = MetricTable(range(30), ("total",), log_every=10)
    for _ in table:
        table.update_metrics({"total": 1.0})
    table.close()
    out = capsys.readouterr().out
    assert out.count("step") == 1


def test_columns_line_up(capsys):
    cols = ("total", "rot", "pos")
    table = MetricTable(range(10), cols, log_every=5)
    for _ in table:
        table.update_metrics({"total": 1.0, "rot": 22.5, "pos": 0.0001})
    table.close()
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len({len(l) for l in lines}) == 1, "rows are not the same width"


def test_disabled_table_prints_nothing(capsys):
    table = MetricTable(range(10), ("total",), disable=True, log_every=1)
    for _ in table:
        table.update_metrics({"total": 1.0})
    table.close()
    assert capsys.readouterr().out == ""


def test_set_desc_is_silent_when_piped(capsys):
    """Four phase changes per batch would be pure noise in a log."""
    table = MetricTable(range(5), ("total",), log_every=100)
    for _ in table:
        table.set_desc("fwd")
        table.set_desc("bwd")
    table.close()
    assert "fwd" not in capsys.readouterr().out
