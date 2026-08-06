"""
The full VN-GAT orientation model.

    input projection ([pos, normal] -> hidden vector channels)
      -> num_layers x ( VNGATLayer , VirtualNodeCommunicationBlock )
      -> per-fragment equivariant mean pool
      -> 2 vector channels -> Gram-Schmidt -> TRANSPOSE -> R_pred
      -> (in parallel) invariant embedding heads for vertices and edges

Scope, restated because it is easy to lose: this network predicts ROTATION
only. Translation is recovered afterwards by the classical solver in
``reassembly.assembly.translation``. The invariant embeddings are what that
solver matches on -- their payoff is the translation stage, not the rotation
loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .angular import AngularBlock
from .vn_gat import VNGATLayer
from .vn_layers import VNInvariant, VNLinear, rotation_6d_to_matrix
from .virtual_nodes import VirtualNodeCommunicationBlock, scatter_mean_vectors


class InvariantEmbeddingHead(nn.Module):
    """Gram-matrix reduction + MLP -> a fixed-size rotation-invariant embedding.

    These MUST be invariant rather than the raw equivariant backbone features.
    Two fragments meeting at the same physical interface point but currently
    sitting at unrelated orientations -- exactly the situation before alignment
    -- produce different equivariant features there purely because of their
    poses, with nothing about the geometry differing. Asking equivariant
    features to agree across fragments is a geometrically impossible
    constraint; asking invariant ones to agree is the intended one.
    """

    def __init__(self, in_channels: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.invariant = VNInvariant(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.invariant(x))


class VNGATModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 16,
        num_layers: int = 3,
        num_vn_slots: int = 8,
        heads: int = 4,
        head_dim: Optional[int] = None,
        embed_dim: int = 32,
        edge_vec_channels: int = 3,
        edge_scalar_channels: int = 1,
        norm: str = "layer",
        gradient_checkpointing: bool = False,
        angular: bool = False,
        max_triplets_per_node: int = 24,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_channels = hidden_channels

        self.input_proj = VNLinear(2, hidden_channels)      # [pos, normal] -> hidden

        self.mesh_layers = nn.ModuleList([
            VNGATLayer(
                hidden_channels, hidden_channels,
                edge_vec_channels=edge_vec_channels,
                edge_scalar_channels=edge_scalar_channels,
                heads=heads, head_dim=head_dim, norm=norm,
            )
            for _ in range(num_layers)
        ])
        self.vn_blocks = nn.ModuleList([
            VirtualNodeCommunicationBlock(
                hidden_channels, num_slots=num_vn_slots, heads=heads,
                head_dim=head_dim, norm=norm,
            )
            for _ in range(num_layers)
        ])
        self.angular_blocks = nn.ModuleList([
            AngularBlock(hidden_channels, edge_vec_channels,
                         max_triplets_per_node=max_triplets_per_node)
            for _ in range(num_layers)
        ]) if angular else None

        self.rotation_head = VNLinear(hidden_channels, 2)   # -> 2 vectors for Gram-Schmidt

        self.vertex_embed_head = InvariantEmbeddingHead(hidden_channels, embed_dim)
        # Edge embeddings pool both endpoint node features plus the edge's own
        # input vector channels (midpoint, two face normals), so they carry
        # learned context AND raw local geometry.
        self.edge_embed_head = InvariantEmbeddingHead(
            2 * hidden_channels + edge_vec_channels, embed_dim
        )

    def _stage(self, i, h, edge_index, edge_scalar, edge_vec,
               fragment_id, num_fragments, fragment_scene_id):
        h = self.mesh_layers[i](h, edge_index, edge_scalar, edge_vec)
        if self.angular_blocks is not None:
            h = self.angular_blocks[i](h, edge_index, edge_vec)
        return self.vn_blocks[i](h, fragment_id, num_fragments,
                                 fragment_scene_id=fragment_scene_id)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scalar: torch.Tensor,
        edge_vec: torch.Tensor,
        fragment_id: torch.Tensor,
        num_fragments: int,
        fragment_scene_id: Optional[torch.Tensor] = None,
        forward_edge_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        x                 ``(N, 2, 3)`` input [pos, normal] vector channels
        edge_index        ``(2, 2E)`` bidirectional mesh edges
        edge_scalar       ``(2E, S)``, edge_vec ``(2E, V, 3)``, aligned with edge_index
        fragment_id       ``(N,)`` fragment of each vertex, unique across the batch
        num_fragments     explicit count (NOT ``fragment_id.max()+1``: an empty
                          fragment never appears in ``fragment_id``)
        fragment_scene_id ``(num_fragments,)`` scene of each fragment; required
                          whenever a batch holds more than one scene
        forward_edge_mask ``(2E,)`` marks one direction of each undirected edge,
                          so edge embeddings are computed once per undirected
                          edge. Must be the explicit mask, never a positional
                          half-slice -- after merging fragments the layout is
                          per-fragment [fwd; bwd] blocks, not [all fwd; all bwd].
        """
        h = self.input_proj(x)

        for i in range(len(self.mesh_layers)):
            if self.gradient_checkpointing and self.training:
                # Recompute this stage's activations during backward instead of
                # storing them. This is the single biggest VRAM lever in the
                # model: the per-edge message tensor is ~18*N*hidden floats per
                # layer and is otherwise alive from forward until backward.
                # use_reentrant=False is required for DDP compatibility.
                h = cp.checkpoint(
                    self._stage, i, h, edge_index, edge_scalar, edge_vec,
                    fragment_id, num_fragments, fragment_scene_id,
                    use_reentrant=False,
                )
            else:
                h = self._stage(i, h, edge_index, edge_scalar, edge_vec,
                                fragment_id, num_fragments, fragment_scene_id)

        # Rotation path in float32 regardless of autocast: Gram-Schmidt
        # normalizes vectors that can be small, and fp16 there costs real
        # precision on a quantity every geometry loss is then multiplied by.
        with torch.autocast(device_type=h.device.type, enabled=False):
            pooled = scatter_mean_vectors(h.float(), fragment_id, num_fragments)
            rot_vecs = self.rotation_head(pooled)                  # (F, 2, 3)
            R_pred = rotation_6d_to_matrix(rot_vecs[:, 0], rot_vecs[:, 1])

        vertex_embedding = self.vertex_embed_head(h)

        if forward_edge_mask is not None:
            src = edge_index[0][forward_edge_mask]
            dst = edge_index[1][forward_edge_mask]
            edge_vec_use = edge_vec[forward_edge_mask]
        else:
            src, dst, edge_vec_use = edge_index[0], edge_index[1], edge_vec

        edge_input = torch.cat([h[src], h[dst], edge_vec_use], dim=1)
        edge_embedding = self.edge_embed_head(edge_input)

        return {
            "node_features": h,
            "R_pred": R_pred,
            "vertex_embedding": vertex_embedding,
            "edge_embedding": edge_embedding,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
