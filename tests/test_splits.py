"""Deterministic hash-based splitting."""
import numpy as np
import pytest

from reassembly.data.splits import SPLITS, assign_split


def _names(n=20000):
    return [f"shape_{i:06d}" for i in range(n)]


def test_deterministic_across_calls():
    for name in _names(500):
        assert assign_split(name, 0.1, 0.1, 0) == assign_split(name, 0.1, 0.1, 0)


def test_splits_are_disjoint_and_exhaustive():
    buckets = {s: set() for s in SPLITS}
    for name in _names():
        buckets[assign_split(name, 0.1, 0.1, 0)].add(name)
    assert buckets["train"].isdisjoint(buckets["val"])
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["val"].isdisjoint(buckets["test"])
    assert sum(len(v) for v in buckets.values()) == 20000


@pytest.mark.parametrize("val_frac,test_frac", [(0.1, 0.1), (0.2, 0.05), (0.15, 0.15)])
def test_fractions_are_approximately_respected(val_frac, test_frac):
    names = _names()
    counts = {s: 0 for s in SPLITS}
    for name in names:
        counts[assign_split(name, val_frac, test_frac, 0)] += 1
    n = len(names)
    assert abs(counts["val"] / n - val_frac) < 0.02
    assert abs(counts["test"] / n - test_frac) < 0.02
    assert abs(counts["train"] / n - (1 - val_frac - test_frac)) < 0.02


def test_seed_changes_the_partition():
    a = [assign_split(n, 0.1, 0.1, 0) for n in _names(2000)]
    b = [assign_split(n, 0.1, 0.1, 1) for n in _names(2000)]
    assert a != b


def test_not_salted_by_process_hash_seed():
    """Must not depend on PYTHONHASHSEED: DDP ranks are separate processes, and
    a per-process salt would give each rank a different train/val split."""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, %r)
        from reassembly.data.splits import assign_split
        print(",".join(assign_split(f"s{i}", .1, .1, 0) for i in range(50)))
    """) % str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src")
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, env=env).stdout.strip())
    assert len(outs) == 1


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        assign_split("x", 0.6, 0.6, 0)
