"""
VN-GAT: equivariant multi-head attention over the physical mesh graph
(fragment-internal message passing; real mesh edges never cross a fragment
boundary, by construction).

For each directed edge j -> i:

    alpha_ij = softmax_j( <q_i, k_j> / sqrt(3 * head_dim) + b_ij )
    msg_ij   = v_j + VNLinear(edge_vec_ij)
    out_i    = sum_j alpha_ij * msg_ij

``<q_i, k_j>`` is a sum of per-channel inner products, hence invariant, so the
softmax weights are invariant; weighting equivariant vectors by invariant
scalars and summing keeps the result equivariant. ``b_ij`` comes from the
edge's scalar (length) feature through a plain ``nn.Linear``, which is safe
precisely because that input has no xyz axis to protect.

MEMORY: WHY head_dim = out_channels // heads
--------------------------------------------
The dominant tensor in this layer is the per-edge message,
``(2E, heads, head_dim, 3)``. On a triangle mesh ``2E ~ 6N``, so this is
``~18 * N * heads * head_dim`` floats, alive through the backward pass.

The original sizing gave every head the FULL width
(``VNLinear(in, heads * out_channels)`` with ``head_dim = out_channels``), then
concatenated to ``heads * out_channels`` and projected back down. That is
``heads`` times more message memory and message FLOPs than standard multi-head
attention, which splits a fixed width across heads. At heads=4 that is a
straight 4x overhead, on the single largest tensor in the model, for no
representational gain -- exactly the "looks like it saves compute but does
not" pattern this project has hit before.

Splitting instead of replicating: same output width, same parameter count in
the projections, 4x less activation memory and 4x fewer message FLOPs.
``head_dim`` is still overridable if you want the old behaviour for an
ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .vn_layers import VNLeakyReLU, VNLinear, make_norm


class VNGATLayer(MessagePassing):
    """Equivariant multi-head graph attention on vector-list node features.

    Node features ``x``: ``(N, C_in, 3)``.
    Edge features, aligned with ``edge_index``:
        ``edge_scalar``: ``(E, S)``  invariant scalars (edge length)
        ``edge_vec``:    ``(E, V, 3)`` vector channels (midpoint, 2 face normals)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_vec_channels: int = 3,
        edge_scalar_channels: int = 1,
        heads: int = 4,
        head_dim: Optional[int] = None,
        negative_slope: float = 0.2,
        norm: str = "layer",
    ):
        super().__init__(aggr="add", node_dim=0)

        if head_dim is None:
            if out_channels % heads != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible by heads ({heads}), "
                    f"or pass head_dim explicitly."
                )
            head_dim = out_channels // heads

        self.heads = heads
        self.head_dim = head_dim
        self.out_channels = out_channels
        inner = heads * head_dim

        self.lin_q = VNLinear(in_channels, inner)
        self.lin_k = VNLinear(in_channels, inner)
        self.lin_v = VNLinear(in_channels, inner)
        self.lin_edge_vec = VNLinear(edge_vec_channels, inner)
        self.edge_scalar_to_bias = nn.Linear(edge_scalar_channels, heads)

        self.lin_out = VNLinear(inner, out_channels)
        self.norm = make_norm(norm, out_channels)
        self.act = VNLeakyReLU(out_channels, negative_slope=negative_slope)

        self.attn_scale = (head_dim * 3) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scalar: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)
        q = self.lin_q(x).view(N, self.heads, self.head_dim, 3)
        k = self.lin_k(x).view(N, self.heads, self.head_dim, 3)
        v = self.lin_v(x).view(N, self.heads, self.head_dim, 3)
        edge_bias = self.edge_scalar_to_bias(edge_scalar)          # (E, heads)

        # edge_vec is projected inside message(), not here: passing the small
        # (E, V, 3) tensor through propagate and expanding it late means the
        # large projected copy exists only where it is consumed.
        out = self.propagate(
            edge_index, q=q, k=k, v=v, edge_vec=edge_vec, edge_bias=edge_bias, size=(N, N)
        )                                                          # (N, heads, head_dim, 3)

        out = out.reshape(N, self.heads * self.head_dim, 3)
        return self.act(self.norm(self.lin_out(out)))

    def message(self, q_i, k_j, v_j, edge_vec, edge_bias, index, ptr, size_i):
        edge_v = self.lin_edge_vec(edge_vec).view(-1, self.heads, self.head_dim, 3)
        msg = v_j + edge_v                                          # equivariant + equivariant

        logits = (q_i * k_j).sum(dim=(-1, -2)) * self.attn_scale + edge_bias   # (E, heads)
        alpha = softmax(logits, index, ptr, size_i)                            # invariant
        return alpha.unsqueeze(-1).unsqueeze(-1) * msg
