"""
Invariant embedding heads for the interface-consistency losses (and, at
inference, for mutual-nearest-neighbour correspondence discovery feeding the
translation solver).

These embeddings MUST be invariant, not the raw equivariant backbone features.
Two fragments meeting at the same physical interface point but currently
sitting at different, uncorrelated orientations -- exactly the situation before
alignment -- produce different equivariant features at that point purely
because of their current orientations, with nothing to do with the geometry
differing. Asking equivariant features to agree there is a geometrically
impossible constraint; asking invariant projections of them to agree is the
intended one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vn_layers import VNInvariant


class InvariantEmbeddingHead(nn.Module):
    """Gram-matrix reduction (with an equivariant bottleneck) + a small MLP."""

    def __init__(self, in_channels: int, embed_dim: int, hidden_dim: int = 64, bottleneck: int = 16):
        super().__init__()
        self.invariant = VNInvariant(in_channels, bottleneck=bottleneck)
        self.mlp = nn.Sequential(
            nn.Linear(self.invariant.out_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.invariant(x))


def symmetric_edge_features(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """
    The edge head's input, as a function that is SYMMETRIC in the edge's two
    endpoints: `h[u] + h[v]` and `n1 + n2`, stacked as vector channels.

    An undirected mesh edge has no canonical endpoint order -- `edges_unique`
    just sorts by vertex index -- and two fragments meeting at the same
    physical edge have unrelated local numbering. Anything that depends on the
    order makes their two embeddings disagree for a bookkeeping reason, and the
    consistency loss would spend capacity undoing it.

    Of the edge's own channels (n1, n2, p_u - p_v), reversing the edge swaps
    the two normals and negates the relative position, so the relative
    position has no symmetric linear form and the normals contribute their sum.
    """
    src, dst = edge_index[0], edge_index[1]
    pooled = node_features.index_select(0, src) + node_features.index_select(0, dst)
    normals = (edge_attr[:, 0] + edge_attr[:, 1]).unsqueeze(1).to(pooled.dtype)
    return torch.cat([pooled, normals], dim=1)
