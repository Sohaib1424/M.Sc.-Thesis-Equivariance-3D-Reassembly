"""
Full VN-GAT orientation model.

    input projection ([pos, normal] -> hidden vector channels)
      -> scale gate (log fragment size, as an invariant per-channel gain)
      -> num_layers x (VNGraphAttentionBlock over mesh edges, VirtualNodeBlock)
      -> per-fragment equivariant mean pool
      -> Gram-Schmidt head  -> R_pred  (see `predict_rotation` for the transpose)
      -> invariant vertex / edge embedding heads (parallel branch)

The mesh-edge layer is the same `VNGraphAttention` that `E:\\My Thesis Work`
uses (edge-aware keys, a self-loop outside the softmax, a residual block, its
own score width `head_dim`). Cross-fragment communication stays with this
project's virtual nodes.

Scope, stated explicitly: this network predicts ROTATION only. Given a fragment
in its scattered state it outputs the rotation that undoes the scattering.
Translation is recovered afterwards by the classical solver in
`vngat/assembly/translation.py`.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .gat_layer import VNGraphAttentionBlock
from .heads import InvariantEmbeddingHead, symmetric_edge_features
from .segment_ops import segment_mean
from .virtual_nodes import VirtualNodeBlock
from .vn_layers import VNLinear, VNScaleGate, predict_rotation

# (n1, n2, delta) on the stored copy -> (n2, n1, -delta) on the reverse copy.
_REVERSE_ORDER = [1, 0, 2]
_REVERSE_SIGN = (1.0, 1.0, -1.0)


class VNGATModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_vn_slots: int = 8,
        heads: int = 4,
        head_dim: int = 8,
        embed_dim: int = 32,
        edge_channels: int = 3,
        gram_bottleneck: int = 16,
        norm: str = "layer",
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.edge_channels = edge_channels
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = VNLinear(2, hidden_channels)
        # Fragment size joins as an invariant GATE, never as a third axis of a
        # (C, 3) tensor -- a size has no direction, so there is no legal slot.
        # Its invariant description is an 8-channel Gram (64 numbers per vertex):
        # the gate needs a coarse summary, and it runs on every vertex.
        self.scale_gate = VNScaleGate(hidden_channels, scalar_features=1)

        self.mesh_layers = nn.ModuleList([
            VNGraphAttentionBlock(hidden_channels, edge_channels, heads=heads,
                                  head_dim=head_dim, norm=norm)
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
        # h_u + h_v plus ONE edge vector, n1 + n2: the only symmetric
        # combination of the edge channels (see `symmetric_edge_features`).
        self.edge_embed_head = InvariantEmbeddingHead(
            hidden_channels + 1, embed_dim, bottleneck=gram_bottleneck
        )

    # ------------------------------------------------------------------
    @staticmethod
    def symmetrise_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Undirected storage -> the directed pairs message passing needs.

        The stored copy u -> v carries (n1, n2, p_u - p_v). The reverse copy
        v -> u carries (n2, n1, p_v - p_u): the relative position is negated
        because it is always "source minus destination", and the normals swap
        because the canonical order is decided by a triple product with the
        edge direction, which flips sign when the direction does. Layout is
        [all forward; all backward], so the first E rows always correspond 1:1
        with the stored undirected edges.
        """
        sign = edge_attr.new_tensor(_REVERSE_SIGN).view(1, -1, 1)
        reverse = edge_attr[:, _REVERSE_ORDER] * sign
        return (
            torch.cat([edge_index, edge_index.flip(0)], dim=1),
            torch.cat([edge_attr, reverse], dim=0),
        )

    def forward(
        self,
        node_vec: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_frag: torch.Tensor,
        num_fragments: int,
        frag_scene: Optional[torch.Tensor] = None,
        frag_log_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        node_vec:       (N, 2, 3) [normalised centred position, vertex normal]
        edge_index:     (2, E) UNDIRECTED, u < v
        edge_attr:      (E, 3, 3) [n1, n2, p_u - p_v], equivariant
        node_frag:      (N,) global fragment id
        frag_scene:     (F,) scene id per fragment
        frag_log_scale: (F, 1) log(world radius) per fragment, invariant
        """
        di_index, di_attr = self.symmetrise_edges(edge_index, edge_attr)

        h = self.input_proj(node_vec)
        if frag_log_scale is not None:
            h = self.scale_gate(h, frag_log_scale.index_select(0, node_frag))

        for mesh_layer, vn_block in zip(self.mesh_layers, self.vn_blocks):
            if self.grad_checkpointing and self.training:
                h = cp.checkpoint(mesh_layer, h, di_index, di_attr, use_reentrant=False)
                h = cp.checkpoint(vn_block, h, node_frag, num_fragments, frag_scene,
                                  use_reentrant=False)
            else:
                h = mesh_layer(h, di_index, di_attr)
                h = vn_block(h, node_frag, num_fragments, frag_scene)

        pooled = segment_mean(h, node_frag, num_fragments)      # (F, hidden, 3), equivariant
        rot_vecs = self.rotation_head(pooled)                   # (F, 2, 3)
        R_pred = predict_rotation(rot_vecs[:, 0], rot_vecs[:, 1])

        # Conditioning of the Gram-Schmidt head, as an invariant scalar.
        #
        # The frame orthogonalises channel 2 against channel 1, so when the two
        # are collinear the residual vanishes and both the frame and its
        # gradient become ill-conditioned. A run that stalls with |cos| near 1
        # is stuck there; a healthy one stays well below.
        with torch.no_grad():
            a1 = rot_vecs[:, 0].float()
            a2 = rot_vecs[:, 1].float()
            n1 = a1.norm(dim=-1).clamp_min(1e-12)
            n2 = a2.norm(dim=-1).clamp_min(1e-12)
            head_cos = ((a1 * a2).sum(-1) / (n1 * n2)).abs().mean()

        vertex_embedding = self.vertex_embed_head(h)
        edge_embedding = self.edge_embed_head(
            symmetric_edge_features(h, edge_index, edge_attr)
        )

        return {
            "node_features": h,
            "R_pred": R_pred,
            "head_cos": head_cos,
            "vertex_embedding": vertex_embedding,
            "edge_embedding": edge_embedding,
        }

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
