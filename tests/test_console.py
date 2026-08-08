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
# These replace an earlier set that tested a stacked multi-bar layout. That
# layout was removed because Kaggle's notebook subprocess reports
# `isatty() == True` while not supporting the cursor movement it needs, so the
# header printed once and every update landed underneath it. One bar, rewritten
# in place, is what every terminal and notebook handles.
def test_uses_exactly_one_bar():
    table = MetricTable(range(10), ("total", "rot"))
    assert hasattr(table, "bar")
    assert not hasattr(table, "header"), "the stacked layout should be gone"
    assert not hasattr(table, "values")
    table.close()


def test_iteration_yields_every_item():
    table = MetricTable(range(7), ("total",))
    assert list(table) == list(range(7))
    table.close()


def test_metrics_go_into_the_bar_postfix_with_short_keys():
    """Full column names do not fit on one line beside the bar; an overflowing
    line wraps and looks like the bar is broken."""
    table = MetricTable(range(3), ("total", "rot", "rot_deg"))
    table.update_metrics({"total": 9.374, "rot": 5.857, "rot_deg": 123.37})
    postfix = table.bar.postfix
    table.close()

    assert postfix is not None
    text = postfix if isinstance(postfix, str) else str(postfix)
    for short in ("tot", "rot", "deg"):
        assert short in text, f"{short} missing from {text!r}"
    assert "9.374" in text


def test_small_and_large_values_use_scientific_notation():
    table = MetricTable(range(3), ("embv", "total"))
    table.update_metrics({"embv": 4.7e-5, "total": 12345.0})
    text = str(table.bar.postfix)
    table.close()
    assert "e-" in text
    assert "e+" in text or "1.2e" in text


def test_missing_metrics_are_omitted_not_crashed():
    table = MetricTable(range(3), ("total", "rot", "pos"))
    table.update_metrics({"total": 1.0})          # rot and pos absent
    table.close()


def test_set_desc_updates_the_bar():
    table = MetricTable(range(3), ("total",), desc="E000 train")
    table.set_desc("E000 train bwd")
    assert "bwd" in table.bar.desc
    table.close()


def test_disabled_table_produces_no_output(capsys):
    table = MetricTable(range(10), ("total",), disable=True)
    for _ in table:
        table.update_metrics({"total": 1.0})
    table.close()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_works_as_a_context_manager():
    with MetricTable(range(4), ("total",)) as table:
        for _ in table:
            table.update_metrics({"total": 1.0})


def test_epoch_table_is_where_the_columns_live():
    """The aligned table moved to once-per-epoch, printed by
    format_epoch_line, where no cursor trickery is involved."""
    cols = ("total", "rot", "pos")
    text = format_epoch_line(7, {"total": 1.0, "rot": 2.0, "pos": 3.0},
                             {"total": 1.5, "rot": 2.5, "pos": 3.5},
                             cols, lr=3e-4, seconds=12.3)
    lines = text.splitlines()
    assert len(lines) == 4
    assert len(lines[1]) == len(lines[2]) == len(lines[3])
    for name in cols:
        assert name in lines[1]
