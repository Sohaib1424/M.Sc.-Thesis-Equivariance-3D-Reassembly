"""
Segment reductions over a flat index vector.

Every graph operation here is a scatter: messages arrive as a flat ``(E, ...)``
tensor with a ``(E,)`` destination index, and have to be reduced per
destination. This module keeps that logic in one place, on plain
``torch.Tensor`` methods -- no ``torch_scatter``, deliberately. That package
needs a compiler and a matching CUDA toolkit at install time, which is a real
obstacle on the Windows machine this project trains from, and every operation
needed here exists natively as ``scatter_reduce_`` or ``index_add_``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _expand(index: Tensor, like: Tensor) -> Tensor:
    """``(E,)`` index broadcast to every trailing axis of ``like``."""
    shape = (index.shape[0],) + (1,) * (like.dim() - 1)
    return index.view(shape).expand_as(like)


def segment_sum(src: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Sum ``src`` rows into ``num_segments`` buckets. Empty buckets are zero."""
    out = src.new_zeros((num_segments,) + src.shape[1:])
    return out.index_add_(0, index, src)


def segment_mean(src: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Mean per bucket. Empty buckets are zero, not NaN."""
    total = segment_sum(src, index, num_segments)
    count = torch.zeros(num_segments, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return total / count.clamp(min=1.0).view((-1,) + (1,) * (src.dim() - 1))


def segment_max(src: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Max per bucket. Empty buckets are zero rather than ``-inf``."""
    out = src.new_full((num_segments,) + src.shape[1:], float("-inf"))
    out.scatter_reduce_(0, _expand(index, src), src, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """
    Softmax over the entries sharing a destination.

    The max subtraction is the usual overflow guard, and it is not optional
    here: attention logits are inner products of unnormalised vector features,
    which grow with channel count.

    The denominator cannot be zero. Every entry's own bucket contains its own
    ``exp(logit - max) > 0``, and the largest contributes exactly 1, so the sum
    is at least 1 for any bucket an entry actually indexes. Buckets with no
    entries are never divided by -- which is what makes an isolated node, or a
    fragment that contributed no tokens, produce a zero message instead of NaN.
    """
    peak = segment_max(logits, index, num_segments)
    weights = torch.exp(logits - peak[index])
    total = segment_sum(weights, index, num_segments)
    return weights / total[index]


def counts_to_ptr(counts: Tensor) -> Tensor:
    """
    ``[3, 0, 2] -> [0, 3, 3, 5]``, GARF's ``cu(l)``.

    A ``ptr`` names segment *boundaries* and is what variable-length attention
    kernels take; a ``batch`` vector names each element's segment and is what
    scatter ops take. Both are kept because converting between them per layer
    is wasted work, and because a zero-length segment is visible in ``ptr``
    (two equal entries) but invisible in ``batch`` -- which is exactly the
    fragment-with-no-fracture-surface case that must not silently vanish.
    """
    zero = counts.new_zeros(1)
    return torch.cat([zero, torch.cumsum(counts, dim=0)])


def ptr_to_batch(ptr: Tensor) -> Tensor:
    """Inverse of :func:`counts_to_ptr`: ``[0, 3, 3, 5] -> [0, 0, 0, 2, 2]``."""
    counts = ptr[1:] - ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=ptr.device), counts
    )


def batch_to_counts(batch: Tensor, num_segments: Optional[int] = None) -> Tensor:
    """Elements per segment. ``num_segments`` must be given if the tail is empty."""
    n = int(batch.max().item()) + 1 if num_segments is None else num_segments
    return torch.bincount(batch, minlength=n)[:n]
