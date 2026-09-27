"""
Samplers.

Three, each for a combination torch does not provide:

* :class:`WeightedDistributedSampler` -- weighted *and* distributed *and*
  reproducible, for a full pass over a balanced training set.
* :class:`EpochSampler` -- a **fixed number** of draws per rank per epoch,
  weighted or not, for fixed-length epochs (``Config.steps_per_epoch``).
* :class:`ShardSampler` -- every index exactly once across the ranks, with no
  padding, for validation.

``WeightedRandomSampler`` is not distributed, ``DistributedSampler`` is not
weighted and pads by repeating samples, and stacking them gives each rank a
different weighted draw with no guarantee the ranks pull the same number of
batches. For training that last point is not a statistical wrinkle: every rank
must reach every optimizer step, so the training samplers here always give
every rank the same count.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler


class WeightedDistributedSampler(Sampler[int]):
    """
    Draw indices with replacement from ``weights``, shard the result by rank.

    The draw is made **once, identically on every rank**, from a generator
    seeded by ``(seed, epoch)``; each rank then takes a strided slice of it.
    That ordering matters. Drawing per-rank from per-rank generators is the
    obvious implementation and it is wrong twice: the ranks see statistically
    different epochs, and nothing makes their lengths equal.

    Slicing with ``[rank::world]`` rather than in contiguous blocks keeps each
    rank's slice an unbiased sample of the same draw, which matters when the
    weights are skewed enough that the top of a sorted draw looks nothing like
    the bottom. The draw is not sorted here, but nothing downstream promises it
    never will be.

    ``num_samples`` defaults to the dataset length rounded up to a multiple of
    the world size, so every rank runs the same number of steps and an epoch
    stays the size it was without balancing -- the comparison between a
    balanced and an unbalanced run should differ in *composition*, not in how
    many gradient steps each got.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_replicas: int = 1,
        rank: int = 0,
        num_samples: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
        weight_tensor = torch.as_tensor(list(weights), dtype=torch.double)
        if weight_tensor.numel() == 0:
            raise ValueError("no weights: the split is empty")
        if bool((weight_tensor < 0).any()):
            raise ValueError("sampling weights must be non-negative")
        if float(weight_tensor.sum()) <= 0.0:
            raise ValueError("sampling weights sum to zero")

        self.weights = weight_tensor
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.replacement = replacement
        self.epoch = 0

        total = num_samples if num_samples is not None else weight_tensor.numel()
        # Round UP to a multiple of the world size. Rounding down would drop
        # samples; leaving it ragged would desynchronise the ranks.
        self.total_size = int(
            -(-total // num_replicas) * num_replicas
        )
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Change the draw. Must be called every epoch, as for
        ``DistributedSampler`` -- without it every epoch is the same draw and
        the effective dataset shrinks to one sample of it."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + 977 * self.epoch)
        if self.replacement:
            drawn = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=generator
            )
        else:
            count = min(self.total_size, self.weights.numel())
            drawn = torch.multinomial(
                self.weights, count, replacement=False, generator=generator
            )
            if drawn.numel() < self.total_size:
                # Pad by repeating the head of the same draw, the way
                # DistributedSampler does, so the ranks stay equal-length.
                pad = self.total_size - drawn.numel()
                drawn = torch.cat([drawn, drawn[:pad]])
        return iter(drawn[self.rank::self.num_replicas].tolist())


class EpochSampler(Sampler[int]):
    """
    Exactly ``per_rank`` indices per rank per epoch, however large the dataset.

    This is what makes an epoch a fixed amount of *training* rather than a pass
    over whatever the data happens to be. With ``modes_per_scene``, balancing,
    ``limit_train`` and the split all changing how many items there are, "one
    pass" has no stable meaning: it is 1,600 steps on one setting and 900 on the
    next, and the validation curve, the checkpoint cadence and the time per
    epoch all move with it. A fixed count keeps them put.

    Without ``weights``: seeded permutations of the whole dataset, concatenated
    until the epoch is full, so every item is drawn ``floor`` or ``ceil`` of
    ``total / len`` times and none is starved. With ``weights``: a multinomial
    draw with replacement, as :class:`WeightedDistributedSampler` does.

    Either way the draw is made **once, identically on every rank** and sliced
    ``[rank::world]``, so the ranks see disjoint parts of one epoch and every
    rank gets the same number of indices -- which every rank needs, because
    every rank must reach every optimizer step.
    """

    def __init__(
        self,
        num_items: int,
        per_rank: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if num_items < 1:
            raise ValueError("EpochSampler needs at least one item")
        if per_rank < 1:
            raise ValueError("per_rank must be >= 1")
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
        self.num_items = int(num_items)
        self.per_rank = int(per_rank)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.weights = None
        if weights is not None:
            tensor = torch.as_tensor(list(weights), dtype=torch.double)
            if tensor.numel() != self.num_items:
                raise ValueError(
                    f"{tensor.numel()} weights for {self.num_items} items")
            if bool((tensor < 0).any()) or float(tensor.sum()) <= 0.0:
                raise ValueError("sampling weights must be non-negative and not all zero")
            self.weights = tensor

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.per_rank

    def __iter__(self) -> Iterator[int]:
        total = self.per_rank * self.num_replicas
        generator = torch.Generator()
        generator.manual_seed(self.seed + 977 * self.epoch)
        if self.weights is not None:
            drawn = torch.multinomial(self.weights, total, replacement=True,
                                      generator=generator)
        else:
            rounds = -(-total // self.num_items)
            drawn = torch.cat([torch.randperm(self.num_items, generator=generator)
                               for _ in range(rounds)])[:total]
        return iter(drawn[self.rank::self.num_replicas].tolist())


class ShardSampler(Sampler[int]):
    """
    Every index exactly once across the ranks, in order, with no padding.

    For validation. ``DistributedSampler(drop_last=False)`` pads the last round
    by *repeating* samples so every rank gets the same count, which is right for
    training and wrong for measurement: the repeated scenes are counted twice
    and the mean moves. Validation contains no collective operation, so the
    ranks do not need equal counts -- their results are gathered once, at the
    end, and every scene is counted once.
    """

    def __init__(self, num_items: int, num_replicas: int = 1, rank: int = 0) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
        self.indices = list(range(rank, int(num_items), int(num_replicas)))

    def set_epoch(self, epoch: int) -> None:     # noqa: D401 -- sampler protocol
        """Validation is the same every epoch."""

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
