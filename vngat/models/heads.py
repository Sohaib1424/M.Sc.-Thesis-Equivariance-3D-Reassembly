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
    edge_vec: torch.Tensor,
) -> torch.Tensor:
    """
    Build the edge head's input as a function that is SYMMETRIC in the edge's
    two endpoints: `h[u] + h[v]` concatenated with the edge's own vectors.

    The original concatenated `[h[src], h[dst], edge_vec]`. An undirected mesh
    edge has no canonical endpoint order -- `edges_unique` just sorts by vertex
    index -- so a concatenation makes the embedding depend on that arbitrary
    index ordering. Two fragments meeting at the same physical interface edge
    have unrelated local vertex numbering, so their two embeddings would
    disagree for a purely bookkeeping reason, and the consistency loss would
    spend capacity learning to undo it. Summing removes the dependence
    exactly, at no cost in information about the pair.
    """
    src, dst = edge_index[0], edge_index[1]
    pooled = node_features.index_select(0, src) + node_features.index_select(0, dst)
    return torch.cat([pooled, edge_vec], dim=1)
