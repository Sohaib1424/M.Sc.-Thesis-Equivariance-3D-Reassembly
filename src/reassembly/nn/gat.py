"""
Equivariant graph attention along real mesh edges.

Intra-fragment message passing. No graph is constructed here: a fragment's
mesh already *is* a graph, its vertices are the nodes and its edges are the
edges, and every edge carries three equivariant vectors: the two adjacent face
normals in the canonical order that :mod:`reassembly.mesh.orientation` fixes,
plus ``p_source - p_destination``, the relative position of the neighbour.

That third channel is not decoration. A message here is a linear map of the
*source* node's features and the edge's, so without it the message can describe
what the neighbour looks like but not which way it lies.

Why attention stays equivariant
-------------------------------
The two halves of an attention layer have different jobs and different
constraints:

* the **score** decides *how much* to listen, and must be a scalar that does
  not change when the fragment is rotated -- so it is built only from inner
  products of vector features, which are invariant;
* the **message** is *what* is heard, and must rotate with the fragment -- so
  it is a bias-free linear map of vector features, which is equivariant.

An invariant scalar times an equivariant vector is equivariant, so their
product is too, and so is the sum over edges. That is the whole argument. It
breaks the moment a score is allowed to read a raw coordinate, or a message is
allowed a bias.

Isolated nodes
--------------
A vertex with no incident edges is never a destination in the softmax, so it is
never divided by an empty sum. It comes out as its own self-transform, which is
the sensible answer rather than NaN. Breaking Bad meshes do contain isolated
vertices after cell extraction, so this is not a theoretical case.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from .segment import segment_softmax, segment_sum
from .vn import VNLayerNorm, VNLeakyReLU, VNLinear


class VNGraphAttention(nn.Module):
    """
    One multi-head equivariant attention layer over ``edge_index``.

    ``edge_index`` is ``(2, E)`` with row 0 the source and row 1 the
    destination, matching PyTorch Geometric's convention so the mesh's
    directed-edge arrays drop straight in.

    Heads are formed on the *invariant* side only: each head projects the
    vector features onto ``head_dim`` learned directions and scores with the
    inner products of those, which is the vector-feature analogue of a
    dot-product score. The message channels are shared across heads and gated
    per head, which keeps the parameter count near a plain VN layer.
    """

    def __init__(
        self,
        channels: int,
        edge_channels: int = 3,
        heads: int = 4,
        head_dim: int = 8,
        negative_slope: float = 0.2,
        checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.per_head = channels // heads

        # Score and message sides, split by input rather than concatenated.
        #
        # `VNLinear` has no bias, so `W [a;b;c] == W_a a + W_b b + W_c c`
        # exactly -- and the split form never materialises the concatenation.
        # That matters a great deal here: on a batch of 81k vertices and 488k
        # directed edges, `cat([x_src, x_dst, edge_attr])` alone is 768 MB, and
        # `cat([x_src, edge_attr])` another 393 MB, per layer, held for the
        # backward pass. Splitting removes both, and lets each projection be
        # applied to the N *nodes* and then gathered to the E edges instead of
        # the other way round -- one sixth of the work on a triangle mesh.
        # Measured on that batch: 2661 MB -> 1500 MB of activations per layer.
        #
        # `fan_in` keeps the initialisation identical to the single wide layer.
        key_fan = channels + channels + edge_channels
        self.query = VNLinear(channels, heads * head_dim)
        self.key_src = VNLinear(channels, heads * head_dim, fan_in=key_fan)
        self.key_dst = VNLinear(channels, heads * head_dim, fan_in=key_fan)
        self.key_edge = VNLinear(edge_channels, heads * head_dim, fan_in=key_fan)
        value_fan = channels + edge_channels
        self.value_src = VNLinear(channels, channels, fan_in=value_fan)
        self.value_edge = VNLinear(edge_channels, channels, fan_in=value_fan)
        # The node's own contribution, outside the softmax -- so a node with no
        # edges still has an output, and so the layer can learn to ignore its
        # neighbourhood entirely.
        self.self_loop = VNLinear(channels, channels)
        self.out = VNLinear(channels, channels)
        self.norm = VNLayerNorm(channels)
        self.act = VNLeakyReLU(channels, negative_slope)
        # 3 spatial components per projected direction, head_dim of them.
        self.scale = 1.0 / math.sqrt(3 * head_dim)
        # Off by default: unlike the cross layer, where recomputation buys 10x
        # for a trivially cheap indexing op, here it re-runs the projections
        # themselves. Worth it only when memory is the binding constraint.
        self.checkpoint = bool(checkpoint)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        ``x`` ``(N, C, 3)``, ``edge_index`` ``(2, E)``, ``edge_attr``
        ``(E, Ce, 3)``. Returns ``(N, C, 3)``.
        """
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        def scores(x_, edge_attr_):
            # Project on the N nodes, then gather to the E edges. The other
            # order -- gather x to (E, C, 3) and project there -- is what the
            # concatenated form required, and costs six times as much on a
            # triangle mesh.
            q = self.query(x_)[dst].view(-1, self.heads, self.head_dim, 3)
            k = self.key_src(x_)[src] + self.key_dst(x_)[dst]
            if edge_attr_ is not None:
                k = k + self.key_edge(edge_attr_)
            k = k.view(-1, self.heads, self.head_dim, 3)
            # Invariant: both factors rotate, so the inner product does not.
            return torch.sum(q * k, dim=(-2, -1)) * self.scale        # (E, heads)

        if self.checkpoint and torch.is_grad_enabled():
            logits = torch.utils.checkpoint.checkpoint(
                scores, x, edge_attr, use_reentrant=False)
        else:
            logits = scores(x, edge_attr)

        alpha = segment_softmax(logits, dst, n)                       # (E, heads)
        messages = self.value_src(x)[src]                             # (E, C, 3)
        if edge_attr is not None:
            messages = messages + self.value_edge(edge_attr)
        # Broadcast each head's weight across the channels it owns.
        weight = alpha.repeat_interleave(self.per_head, dim=1)        # (E, C)
        aggregated = segment_sum(messages * weight.unsqueeze(-1), dst, n)

        return self.act(self.norm(self.out(aggregated + self.self_loop(x))))


class VNGraphAttentionBlock(nn.Module):
    """Attention plus a residual connection, the unit the backbone stacks."""

    def __init__(self, channels: int, edge_channels: int = 3, heads: int = 4,
                 head_dim: int = 8, negative_slope: float = 0.2,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.attention = VNGraphAttention(
            channels, edge_channels, heads, head_dim, negative_slope, checkpoint
        )

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        return x + self.attention(x, edge_index, edge_attr)
