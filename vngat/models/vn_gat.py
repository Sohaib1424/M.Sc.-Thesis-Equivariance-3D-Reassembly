"""
Full VN-GAT orientation model.

    input projection ([pos, normal] -> hidden vector channels)
      -> num_layers x (VNGATLayer, VirtualNodeBlock)
      -> per-fragment equivariant mean pool
      -> Gram-Schmidt head  -> R_pred  (see `predict_rotation` for the transpose)
      -> invariant vertex / edge embedding heads (parallel branch)

Scope, stated explicitly: this network predicts ROTATION only. Given a
fragment in its scattered state it outputs the rotation that undoes the
scattering. Translation is recovered afterwards by the classical solver in
`vngat/assembly/translation.py`, which is what turns per-fragment orientations
into an assembled scene.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .gat_layer import VNGATLayer
from .heads import InvariantEmbeddingHead, symmetric_edge_features
from .segment_ops import segment_mean
from .virtual_nodes import VirtualNodeBlock
from .vn_layers import VNLinear, predict_rotation


class VNGATModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_vn_slots: int = 8,
        heads: int = 4,
        embed_dim: int = 32,
        edge_vec_channels: int = 3,
        edge_scalar_channels: int = 1,
        gram_bottleneck: int = 16,
        norm: str = "layer",
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = VNLinear(2, hidden_channels)

        self.mesh_layers = nn.ModuleList([
            VNGATLayer(
                hidden_channels, hidden_channels,
                edge_vec_channels=edge_vec_channels,
                edge_scalar_channels=edge_scalar_channels,
                heads=heads, norm=norm,
            )
            for _ in range(num_layers)
        ])
        self.vn_blocks = nn.ModuleList([
            VirtualNodeBlock(hidden_channels, num_slots=num_vn_slots, heads=heads, norm=norm)
            for _ in range(num_layers)
        ])

        self.rotation_head = VNLinear(hidden_channels, 2)

        self.vertex_embed_head = InvariantEmbeddingHead(
            hidden_channels, embed_dim, bottleneck=gram_bottleneck
        )
        self.edge_embed_head = InvariantEmbeddingHead(
            hidden_channels + edge_vec_channels, embed_dim, bottleneck=gram_bottleneck
        )

    # ------------------------------------------------------------------
    @staticmethod
    def symmetrise_edges(edge_index: torch.Tensor, edge_len: torch.Tensor, edge_vec: torch.Tensor):
        """
        Undirected storage -> the directed pairs message passing needs.

        The reverse copy swaps the two adjacent-face-normal slots so that
        "first normal" keeps a consistent meaning relative to the direction of
        travel along the edge. Layout is [all forward; all backward], so the
        first E rows always correspond 1:1 with the stored undirected edges --
        no interleaving, no mask.
        """
        rev_index = edge_index.flip(0)
        rev_vec = edge_vec[:, [0, 2, 1], :]
        return (
            torch.cat([edge_index, rev_index], dim=1),
            torch.cat([edge_len, edge_len], dim=0),
            torch.cat([edge_vec, rev_vec], dim=0),
        )

    def forward(
        self,
        node_vec: torch.Tensor,
        edge_index: torch.Tensor,
        edge_len: torch.Tensor,
        edge_vec: torch.Tensor,
        node_frag: torch.Tensor,
        num_fragments: int,
        frag_scene: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        node_vec:   (N, 2, 3) [centralised position, vertex normal]
        edge_index: (2, E) UNDIRECTED
        edge_len:   (E, 1) invariant, edge_vec: (E, 3, 3) equivariant
        node_frag:  (N,) global fragment id
        frag_scene: (F,) scene id per fragment
        """
        di_index, di_len, di_vec = self.symmetrise_edges(edge_index, edge_len, edge_vec)

        h = self.input_proj(node_vec)
        for mesh_layer, vn_block in zip(self.mesh_layers, self.vn_blocks):
            if self.grad_checkpointing and self.training:
                h = cp.checkpoint(mesh_layer, h, di_index, di_len, di_vec, use_reentrant=False)
                h = cp.checkpoint(vn_block, h, node_frag, num_fragments, frag_scene,
                                  use_reentrant=False)
            else:
                h = mesh_layer(h, di_index, di_len, di_vec)
                h = vn_block(h, node_frag, num_fragments, frag_scene)

        pooled = segment_mean(h, node_frag, num_fragments)      # (F, hidden, 3), equivariant
        rot_vecs = self.rotation_head(pooled)                   # (F, 2, 3)
        R_pred = predict_rotation(rot_vecs[:, 0], rot_vecs[:, 1])

        vertex_embedding = self.vertex_embed_head(h)
        edge_embedding = self.edge_embed_head(
            symmetric_edge_features(h, edge_index, edge_vec)
        )

        return {
            "node_features": h,
            "R_pred": R_pred,
            "vertex_embedding": vertex_embedding,
            "edge_embedding": edge_embedding,
        }

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
