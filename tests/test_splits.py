"""Deterministic hash-based splits."""
from __future__ import annotations

from collections import Counter

from vngat.data.splits import assign_split


def test_split_is_deterministic():
    names = [f"shape_{i}" for i in range(500)]
    first = [assign_split(n, 0.1, 0.1, 0) for n in names]
    second = [assign_split(n, 0.1, 0.1, 0) for n in names]
    assert first == second


def test_split_proportions_are_approximately_right():
    names = [f"shape_{i:06d}" for i in range(20000)]
    counts = Counter(assign_split(n, 0.1, 0.1, 0) for n in names)
    total = sum(counts.values())
    assert abs(counts["val"] / total - 0.1) < 0.01
    assert abs(counts["test"] / total - 0.1) < 0.01
    assert abs(counts["train"] / total - 0.8) < 0.02


def test_splits_are_disjoint_and_total():
    names = [f"shape_{i}" for i in range(3000)]
    assigned = {n: assign_split(n, 0.15, 0.15, 3) for n in names}
    buckets = {s: {n for n, v in assigned.items() if v == s} for s in ("train", "val", "test")}
    assert sum(len(b) for b in buckets.values()) == len(names)
    assert not (buckets["train"] & buckets["val"])
    assert not (buckets["train"] & buckets["test"])
    assert not (buckets["val"] & buckets["test"])


def test_seed_changes_the_partition():
    names = [f"shape_{i}" for i in range(1000)]
    a = [assign_split(n, 0.1, 0.1, 0) for n in names]
    b = [assign_split(n, 0.1, 0.1, 1) for n in names]
    assert a != b
