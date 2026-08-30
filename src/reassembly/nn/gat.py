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
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.per_head = channels // heads

        # Score side: project to `heads * head_dim` directions, inner-product them.
        self.query = VNLinear(channels, heads * head_dim)
        self.key = VNLinear(channels + channels + edge_channels, heads * head_dim)
        # Message side: source features and the edge's three vectors.
        self.value = VNLinear(channels + edge_channels, channels)
        # The node's own contribution, outside the softmax -- so a node with no
        # edges still has an output, and so the layer can learn to ignore its
        # neighbourhood entirely.
        self.self_loop = VNLinear(channels, channels)
        self.out = VNLinear(channels, channels)
        self.norm = VNLayerNorm(channels)
        self.act = VNLeakyReLU(channels, negative_slope)
        # 3 spatial components per projected direction, head_dim of them.
        self.scale = 1.0 / math.sqrt(3 * head_dim)

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

        x_src, x_dst = x[src], x[dst]
        key_parts = [x_src, x_dst]
        value_parts = [x_src]
        if edge_attr is not None:
            key_parts.append(edge_attr)
            value_parts.append(edge_attr)

        q = self.query(x_dst).view(-1, self.heads, self.head_dim, 3)
        k = self.key(torch.cat(key_parts, dim=-2)).view(-1, self.heads, self.head_dim, 3)
        # Invariant: both factors rotate, so the inner product does not.
        logits = torch.sum(q * k, dim=(-2, -1)) * self.scale          # (E, heads)

        alpha = segment_softmax(logits, dst, n)                       # (E, heads)
        messages = self.value(torch.cat(value_parts, dim=-2))         # (E, C, 3)
        # Broadcast each head's weight across the channels it owns.
        weight = alpha.repeat_interleave(self.per_head, dim=1)        # (E, C)
        aggregated = segment_sum(messages * weight.unsqueeze(-1), dst, n)

        return self.act(self.norm(self.out(aggregated + self.self_loop(x))))


class VNGraphAttentionBlock(nn.Module):
    """Attention plus a residual connection, the unit the backbone stacks."""

    def __init__(self, channels: int, edge_channels: int = 3, heads: int = 4,
                 head_dim: int = 8, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.attention = VNGraphAttention(
            channels, edge_channels, heads, head_dim, negative_slope
        )

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        return x + self.attention(x, edge_index, edge_attr)
