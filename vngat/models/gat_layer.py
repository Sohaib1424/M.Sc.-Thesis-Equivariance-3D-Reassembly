"""
VN-GAT: equivariant multi-head graph attention along the real mesh edges.

This is the SAME layer that `E:\\My Thesis Work` uses
(`src/reassembly/nn/gat.py`): identical structure, parameterisation, scaling
and initialisation. It is built here on this package's own Vector Neuron
primitives, so the two implementations differ only in where each adds its
small numerical guard inside the shared LeakyReLU and LayerNorm (an epsilon of
1e-6 to 1e-8). Loaded with the same weights, the two agree to ~1e-6 in float64;
`docs/v5-changes.md` records the measurement.

Message passing is fragment-internal by construction -- mesh edges never cross
a fragment boundary. Cross-fragment communication is the virtual nodes' job
(`virtual_nodes.py`).

What each directed edge j -> i carries
--------------------------------------
Three equivariant vectors (see `vngat.data.features`): the two adjacent face
normals `n1, n2` in a canonical left/right order, and `p_j - p_i`, the relative
position of the neighbour. No invariant scalar -- the edge length the previous
layer used as an attention bias is gone, as is the midpoint.

The layer
---------
For a destination vertex i with incoming edges from sources j::

    q_i     = W_q x_i                                   (per head, head_dim dirs)
    k_ij    = W_ks x_j + W_kd x_i + W_ke e_ij           (source, destination, edge)
    a_ij    = softmax_j( <q_i, k_ij> / sqrt(3 head_dim) )
    m_ij    = W_vs x_j + W_ve e_ij
    out_i   = act( norm( W_o ( sum_j a_ij m_ij  +  W_self x_i ) ) )

and the block adds a residual, `x_i + out_i`.

Three differences from the layer this replaces, each measured before the
switch was made:

  * THE SCORE READS THE EDGE. The old key depended on the source vertex only,
    so changing every face normal left the attention weights exactly unchanged
    -- the normals reached the message but never decided whom to listen to.
    Here the key includes the destination and the edge.

  * A VERTEX KEEPS ITS OWN STATE. The old layer had no self term and no
    residual: an isolated vertex came out as exactly zero, and a connected one
    kept its own features only through the attention weights. `W_self x_i` and
    the block's residual fix both. (PyTorch Geometric's GATConv adds self-loops
    by default for the same reason.)

  * SCORES HAVE THEIR OWN WIDTH. Query/key width is `heads * head_dim`,
    independent of the hidden width, so the score capacity is a separate
    decision from the feature capacity.

Why attention stays equivariant
-------------------------------
The SCORE decides how much to listen and must not change when the fragment is
rotated, so it is built only from inner products of co-rotating vectors. The
MESSAGE is what is heard and must rotate with the fragment, so it is a
bias-free linear map of vector features. An invariant scalar times an
equivariant vector is equivariant, and so is the sum. That is the whole
argument; it breaks the moment a score reads a raw coordinate or a message
acquires a bias.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .segment_ops import at_least_float32, segment_softmax, segment_sum
from .vn_layers import VNLeakyReLU, VNLinear, make_norm


class VNGraphAttention(nn.Module):
    """
    One multi-head equivariant attention layer over `edge_index`.

    `edge_index` is (2, E) with row 0 the SOURCE and row 1 the DESTINATION,
    PyTorch Geometric's convention. Heads are formed on the invariant side:
    each head projects the vector features onto `head_dim` learned directions
    and scores with the inner products of those. The message channels are
    split into `heads` contiguous blocks and block h is weighted by head h.
    """

    def __init__(
        self,
        channels: int,
        edge_channels: int = 3,
        heads: int = 4,
        head_dim: int = 8,
        negative_slope: float = 0.2,
        checkpoint: bool = False,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(
                f"channels ({channels}) must be divisible by heads ({heads}): the "
                f"message width is split across heads, not replicated per head."
            )
        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.per_head = channels // heads

        # Score and message sides, split by input rather than concatenated.
        # `VNLinear` has no bias, so `W [a; b; c] == W_a a + W_b b + W_c c`
        # exactly, and the split form never materialises the (E, 2C + 3, 3)
        # concatenation. Each projection runs on the N nodes and is then
        # gathered to the E edges -- one sixth of the work on a triangle mesh.
        # `fan_in` keeps the initialisation identical to the single wide layer.
        key_fan = channels + channels + edge_channels
        self.query = VNLinear(channels, heads * head_dim)
        self.key_src = VNLinear(channels, heads * head_dim, fan_in=key_fan)
        self.key_dst = VNLinear(channels, heads * head_dim, fan_in=key_fan)
        self.key_edge = VNLinear(edge_channels, heads * head_dim, fan_in=key_fan)
        value_fan = channels + edge_channels
        self.value_src = VNLinear(channels, channels, fan_in=value_fan)
        self.value_edge = VNLinear(edge_channels, channels, fan_in=value_fan)
        # The vertex's own contribution, OUTSIDE the softmax: a vertex with no
        # edges still has an output, and the layer can learn to ignore its
        # neighbourhood entirely.
        self.self_loop = VNLinear(channels, channels)
        self.out = VNLinear(channels, channels)
        self.norm = make_norm(norm, channels)
        self.act = VNLeakyReLU(channels, negative_slope=negative_slope)
        # 3 spatial components per projected direction, head_dim of them.
        self.scale = 1.0 / math.sqrt(3 * head_dim)
        # Recompute the score in the backward pass instead of storing it. Off by
        # default: the model-level `grad_checkpointing` already wraps whole
        # layers, and the OOM ladder turns that on for a scene that needs it.
        self.checkpoint = bool(checkpoint)

    def _scores(self, x: torch.Tensor, edge_attr: Optional[torch.Tensor],
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        q = self.query(x).index_select(0, dst)
        k = self.key_src(x).index_select(0, src) + self.key_dst(x).index_select(0, dst)
        if edge_attr is not None:
            k = k + self.key_edge(edge_attr)
        q = q.reshape(-1, self.heads, self.head_dim, 3)
        k = k.reshape(-1, self.heads, self.head_dim, 3)
        # float32 for the SCORE, and specifically for the elementwise product:
        # autocast promotes `.sum` but not the product feeding it, which
        # overflows float16 once both operands pass ~256 -- and the
        # max-subtracting softmax then turns that inf into NaN. One scalar per
        # edge per head, so widening it costs nothing.
        return (at_least_float32(q) * at_least_float32(k)).sum(dim=(-2, -1)) * self.scale

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:          (N, C, 3)
        edge_index: (2, E) directed; row 0 = source j, row 1 = destination i
        edge_attr:  (E, Ce, 3) equivariant edge vectors
        Returns (N, C, 3).
        """
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        if self.checkpoint and torch.is_grad_enabled():
            logits = torch.utils.checkpoint.checkpoint(
                self._scores, x, edge_attr, src, dst, use_reentrant=False)
        else:
            logits = self._scores(x, edge_attr, src, dst)

        alpha = segment_softmax(logits, dst, n)                          # (E, heads)
        messages = self.value_src(x).index_select(0, src)                # (E, C, 3)
        if edge_attr is not None:
            messages = messages + self.value_edge(edge_attr)
        # Each head's weight broadcast over the channel block it owns. Cast to
        # the value dtype so the widest tensor here stays in half precision
        # under autocast (the softmax comes back float32).
        weight = alpha.to(messages.dtype).repeat_interleave(self.per_head, dim=1)
        aggregated = segment_sum(messages * weight.unsqueeze(-1), dst, n)

        return self.act(self.norm(self.out(aggregated + self.self_loop(x))))


class VNGraphAttentionBlock(nn.Module):
    """Attention plus a residual connection -- the unit the model stacks."""

    def __init__(self, channels: int, edge_channels: int = 3, heads: int = 4,
                 head_dim: int = 8, negative_slope: float = 0.2,
                 checkpoint: bool = False, norm: str = "layer") -> None:
        super().__init__()
        self.attention = VNGraphAttention(
            channels, edge_channels, heads, head_dim, negative_slope, checkpoint, norm,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.attention(x, edge_index, edge_attr)
