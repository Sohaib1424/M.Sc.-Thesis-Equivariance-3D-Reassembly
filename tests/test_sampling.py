"""
The weighted, distributed, reproducible sampler.

The combination is the point. Stacking ``WeightedRandomSampler`` inside
``DistributedSampler`` gives each rank its own weighted draw with no guarantee
the ranks pull the same number of batches -- which under DDP is not a
statistical wrinkle but a hang, because the rank that finishes first waits
forever in the gradient all-reduce.
"""
from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from reassembly.data.sampling import WeightedDistributedSampler


def _all_ranks(weights, world, **kwargs):
    samplers = [WeightedDistributedSampler(weights, num_replicas=world, rank=r,
                                           **kwargs)
                for r in range(world)]
    return samplers, [list(s) for s in samplers]


def test_ranks_are_equal_length_and_disjoint_slices_of_one_draw():
    """
    The property that keeps DDP alive, and the reason the draw is made once.

    Every rank must pull the same number of batches. Drawing per-rank would
    also mean the ranks see statistically different epochs, which is a quieter
    problem but still one nobody wants to discover from a curve.
    """
    weights = [1.0, 5.0, 2.0, 0.5, 3.0, 1.5, 4.0]
    _samplers, draws = _all_ranks(weights, world=3, seed=11)

    assert len({len(d) for d in draws}) == 1, "ranks pulled different lengths"
    combined = [index for step in zip(*draws) for index in step]
    single = list(WeightedDistributedSampler(weights, num_replicas=1, rank=0,
                                             seed=11))
    # Interleaving the three strided slices reconstructs the single-rank draw
    # exactly, which is the same statement as "one draw, sharded".
    assert combined[:len(single)] == single[:len(combined)]


def test_length_is_padded_up_so_no_rank_is_short():
    weights = [1.0] * 10
    samplers, draws = _all_ranks(weights, world=4, seed=0)
    assert all(len(d) == 3 for d in draws)        # 10 -> 12, 3 each
    assert all(len(s) == 3 for s in samplers)


def test_the_draw_is_reproducible_and_moves_with_the_epoch():
    """
    Deterministic given (seed, epoch), and *different* across epochs. Without
    ``set_epoch`` every epoch is the same draw, so the effective dataset
    collapses to one sample of it -- which looks exactly like a model that
    stopped generalising.
    """
    weights = [1.0, 2.0, 3.0, 4.0]
    a = WeightedDistributedSampler(weights, seed=5)
    b = WeightedDistributedSampler(weights, seed=5)
    assert list(a) == list(b)

    a.set_epoch(1)
    assert list(a) != list(b)
    b.set_epoch(1)
    assert list(a) == list(b)


def test_frequencies_follow_the_weights():
    """A weight twice as large must be drawn about twice as often."""
    weights = [1.0, 2.0, 4.0, 8.0]
    sampler = WeightedDistributedSampler(weights, num_samples=200_000, seed=3)
    counts = Counter(sampler)
    total = sum(counts.values())
    expected = [w / sum(weights) for w in weights]
    for index, share in enumerate(expected):
        assert counts[index] / total == pytest.approx(share, abs=0.005)


def test_a_zero_weight_item_is_never_drawn():
    weights = [0.0, 1.0, 1.0]
    assert 0 not in set(WeightedDistributedSampler(weights, num_samples=5000,
                                                   seed=1))


def test_without_replacement_covers_every_item_once():
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    drawn = list(WeightedDistributedSampler(weights, seed=2, replacement=False))
    assert sorted(drawn) == list(range(len(weights)))


@pytest.mark.parametrize("weights,message", [
    ([], "no weights"),
    ([0.0, 0.0], "sum to zero"),
    ([1.0, -1.0], "non-negative"),
])
def test_degenerate_weights_are_rejected_loudly(weights, message):
    """
    Each of these would otherwise produce a silently wrong epoch: an empty
    split, a sampler that cannot draw, or a negative weight that
    ``torch.multinomial`` interprets in its own way.
    """
    with pytest.raises(ValueError, match=message):
        WeightedDistributedSampler(weights)


def test_rank_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        WeightedDistributedSampler([1.0, 1.0], num_replicas=2, rank=2)


def test_epoch_length_matches_the_unbalanced_run():
    """
    A balanced run and an unbalanced one must differ in *composition*, not in
    how many gradient steps each got -- otherwise the comparison confounds the
    two and the balanced run's advantage (or deficit) is partly just budget.
    """
    weights = [1.0, 9.0, 1.0, 1.0, 1.0]
    assert len(WeightedDistributedSampler(weights, seed=0)) == len(weights)
