"""
Segment scatter ops, and the proof that segment attention is numerically
identical to the dense masked attention it replaces (i.e. the memory fix is
exact, not an approximation).
"""
from __future__ import annotations

import torch

from vngat.models.segment_ops import (
    blockwise_softmax, segment_max, segment_mean, segment_softmax, segment_sum,
)


def test_segment_sum_and_mean():
    src = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    index = torch.tensor([0, 0, 1, 2])
    assert torch.allclose(segment_sum(src, index, 3), torch.tensor([[3.0], [3.0], [4.0]]))
    assert torch.allclose(segment_mean(src, index, 3), torch.tensor([[1.5], [3.0], [4.0]]))


def test_segment_mean_handles_empty_segments():
    src = torch.tensor([[5.0]])
    index = torch.tensor([2])
    out = segment_mean(src, index, 4)
    assert torch.allclose(out, torch.tensor([[0.0], [0.0], [5.0], [0.0]]))
    assert not torch.isnan(out).any()


def test_segment_max():
    src = torch.tensor([1.0, 7.0, 3.0, -2.0])
    index = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(segment_max(src, index, 2), torch.tensor([7.0, 3.0]))


def test_segment_softmax_normalises_within_segments():
    logits = torch.randn(50)
    index = torch.randint(0, 6, (50,))
    alpha = segment_softmax(logits, index, 6)
    sums = segment_sum(alpha, index, 6)
    present = torch.bincount(index, minlength=6) > 0
    assert torch.allclose(sums[present], torch.ones(int(present.sum())), atol=1e-5)


def test_segment_softmax_equals_dense_masked_softmax():
    """
    The exactness claim behind the VRAM fix.

    Dense version: build an (N, S) logit matrix, mask out disallowed pairs
    with -inf, softmax over N. Segment version: only ever form the allowed
    pairs. They must agree to floating-point noise.
    """
    torch.manual_seed(0)
    N, S = 400, 9
    index = torch.randint(0, S, (N,))
    logits = torch.randn(N, 4)                      # 4 heads

    alpha_segment = segment_softmax(logits, index, S)

    dense = torch.full((S, N, 4), float("-inf"))
    dense[index, torch.arange(N)] = logits
    alpha_dense_full = torch.softmax(dense, dim=1)
    alpha_dense = alpha_dense_full[index, torch.arange(N)]

    assert torch.allclose(alpha_segment, alpha_dense, atol=1e-6)


def test_segment_softmax_is_shift_invariant():
    logits = torch.randn(200, 2)
    index = torch.randint(0, 5, (200,))
    a = segment_softmax(logits, index, 5)
    b = segment_softmax(logits + 1000.0, index, 5)
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.isnan(b).any()


def test_segment_softmax_gradients_flow():
    logits = torch.randn(60, 3, requires_grad=True)
    index = torch.randint(0, 4, (60,))
    segment_softmax(logits, index, 4).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_blockwise_softmax_respects_blocks():
    logits = torch.randn(8, 2, 8)
    block = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    alpha = blockwise_softmax(logits, block)
    assert torch.allclose(alpha.sum(-1), torch.ones(8, 2), atol=1e-5)
    cross = block.unsqueeze(0) != block.unsqueeze(-1)
    assert float(alpha.transpose(1, 2)[cross.unsqueeze(-1).expand(8, 8, 2)].abs().max()) < 1e-6


def test_blockwise_softmax_singleton_block_is_not_nan():
    logits = torch.randn(3, 1, 3)
    alpha = blockwise_softmax(logits, torch.tensor([0, 1, 2]))
    assert not torch.isnan(alpha).any()
    assert torch.allclose(alpha.sum(-1), torch.ones(3, 1), atol=1e-5)
