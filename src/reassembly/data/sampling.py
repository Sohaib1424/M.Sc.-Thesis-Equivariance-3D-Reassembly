"""
Samplers.

One class, because the combination it provides does not exist in torch:
weighted *and* distributed *and* reproducible. ``WeightedRandomSampler`` is not
distributed, ``DistributedSampler`` is not weighted, and stacking them gives
each rank a different weighted draw with no guarantee the ranks pull the same
number of batches -- which under DDP is not a statistical wrinkle but a hang,
since the rank that finishes first waits forever in the gradient all-reduce.
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
