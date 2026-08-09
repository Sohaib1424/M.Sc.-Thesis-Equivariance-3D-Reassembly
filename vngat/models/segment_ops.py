"""
Segment (scatter) primitives -- the small amount of PyTorch Geometric this
project actually used, written directly against core torch.

`segment_softmax` is the load-bearing one. It is what lets every attention
stage in this model normalise over a variable-size neighbourhood WITHOUT ever
materialising a dense (queries x keys) logit matrix.

Why that matters here, concretely: a Breaking Bad scene runs to ~17k vertices
and up to ~86 fragments. The virtual-node stages attend between every vertex
and every virtual node. Dense-masked, that is a (17k, 8*86) logit tensor per
head per stage per layer -- and it is computed, masked (a second allocation),
and softmaxed (a third), all retained for backward. At batch size 2 with 4
layers that alone runs to well over 10 GB and is the direct cause of the
out-of-memory crash on a 16 GB T4.

Every one of those masks was block-diagonal: a vertex may only attend to its
own fragment's slots. Segment attention exploits that by only ever forming the
pairs that are actually allowed -- (vertex, slot) rather than
(vertex, every slot in the batch) -- which turns the cost from
O(N * F * K) into O(N * K). `tests/test_segment_ops.py` asserts it produces
numerically identical weights to the dense masked version it replaces.
"""
from __future__ import annotations

import torch


_HALF = (torch.float16, torch.bfloat16)


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Accumulate in float32 whenever the data is half precision.

    NOT optional under AMP. These reductions run over WHOLE FRAGMENTS -- the
    virtual-node pooling and the final per-fragment mean sum thousands of
    values each. float16 has ~11 bits of mantissa, so once a running sum
    reaches ~2048 its ulp exceeds 1 and further contributions of order 1 are
    silently discarded; for the attention aggregation, where each of ~5000
    terms is ~2e-4, almost every term is lost below the running sum's ulp. The
    result is a pooled feature that is quietly wrong, feeding straight into the
    rotation head.

    Only the ACCUMULATOR is widened. Results are returned in the caller's dtype,
    so nothing downstream changes size and the AMP memory saving is kept.
    """
    return torch.float32 if dtype in _HALF else dtype


def segment_max(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Per-segment maximum over dim 0.

    src: (M, ...), index: (M,) in [0, num_segments). Returns (num_segments, ...).
    Segments with no members keep 0; callers only ever gather back the
    segments that do have members.
    """
    acc = _accum_dtype(src.dtype)
    out = torch.zeros((num_segments, *src.shape[1:]), dtype=acc, device=src.device)
    if src.numel() == 0:
        return out.to(src.dtype)
    work = src.to(acc)
    if hasattr(out, "index_reduce_"):
        out = out.index_reduce_(0, index, work, "amax", include_self=False)
    else:  # pragma: no cover - torch < 1.13
        expanded = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(work)
        out = out.scatter_reduce_(0, expanded, work, reduce="amax", include_self=False)
    return out.to(src.dtype)


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Per-segment sum over dim 0. src: (M, ...), index: (M,)."""
    acc = _accum_dtype(src.dtype)
    out = torch.zeros((num_segments, *src.shape[1:]), dtype=acc, device=src.device)
    if src.numel() == 0:
        return out.to(src.dtype)
    out = out.index_add_(0, index, src.to(acc))
    return out.to(src.dtype)


def segment_mean(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Per-segment mean over dim 0, with empty segments mapped to 0."""
    total = segment_sum(src, index, num_segments)
    # Integer bincount rather than an index_add_ of ones: counting thousands of
    # members in float16 hits exactly the accumulation limit described above,
    # so the DIVISOR would be wrong too.
    counts = torch.bincount(index, minlength=num_segments).clamp_min(1)
    shape = (num_segments,) + (1,) * (src.dim() - 1)
    return total / counts.reshape(shape).to(total.dtype)


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Softmax over dim 0 within each segment.

    logits: (M, ...) -- every trailing dimension is normalised independently
    (so a (M, K, H) tensor gives one distribution per (slot, head) pair).
    index: (M,) segment id per row.

    Computed in float32 under AMP and cast back: torch's own autocast policy
    puts `softmax` on the float32 list for exactly this reason, and a
    hand-written one has to do the same. The logit tensor is small (one scalar
    per pair per head), so the widened copy costs almost nothing.

    The max shift is detached: softmax is exactly invariant to it, and
    detaching keeps `index_reduce_`'s amax out of the autograd graph entirely.
    """
    if logits.numel() == 0:
        return logits
    out_dtype = logits.dtype
    work = logits.to(_accum_dtype(out_dtype))
    maxima = segment_max(work.detach(), index, num_segments)
    exp = (work - maxima.index_select(0, index)).exp()
    denom = segment_sum(exp, index, num_segments).index_select(0, index)
    return (exp / denom.clamp_min(torch.finfo(exp.dtype).tiny)).to(out_dtype)


def blockwise_softmax(logits: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
    """
    Softmax over the LAST dimension, restricted to entries sharing a block id.

    logits: (Q, ..., Q) square in the first and last dims;
    block_id: (Q,).

    Used only for the virtual-node global stage, where the population is
    fragments-of-one-scene (hundreds, not tens of thousands), so a dense
    within-block matrix is genuinely small. Masking with a large negative
    constant rather than -inf keeps rows whose block is a singleton from
    producing NaN.
    """
    mask = block_id.unsqueeze(0) == block_id.unsqueeze(-1)          # (Q, Q)
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(1)
    neg = torch.finfo(logits.dtype).min / 4
    masked = logits.masked_fill(~mask, neg)
    return torch.softmax(masked, dim=-1)
