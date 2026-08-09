"""
VN-GAT: equivariant multi-head attention over the real mesh graph.

Message passing is fragment-internal by construction -- mesh edges never cross
a fragment boundary. Cross-fragment communication is the virtual nodes' job
(`virtual_nodes.py`).

For each directed edge j -> i:

    alpha_ij = softmax_j( <q_i, k_j> / sqrt(C_h * 3) + b_ij )
    msg_ij   = v_j + VNLinear(edge_vec_ij)
    out_i    = sum_j alpha_ij * msg_ij

`<q_i, k_j>` sums over the head's channel and xyz axes, so it is
rotation-invariant; `b_ij` comes from the edge's invariant scalar (its length)
through a plain `nn.Linear`, which is safe precisely because that input has no
xyz axis to protect. An invariant-weighted sum of equivariant vectors is
equivariant, so the layer is.

MEMORY: PER-HEAD CHANNEL SPLITTING
----------------------------------
The original layer projected to `heads * out_channels` channels, i.e. it gave
EVERY head the full hidden width, multiplying the model's widest tensors by
`heads`. Those tensors are per-directed-edge -- (2E, heads, C, 3) with
2E ~ 190k at batch size 2 -- and there are five of them live per layer
(q_i, k_j, v_j, the edge projection, and the weighted message), all retained
for backward. At hidden=64/heads=4 that is ~3.6 GB per layer, ~14 GB over four
layers, from this layer alone.

Standard multi-head practice, used here, splits the width across heads instead
(`C_h = out_channels // heads`), so `heads * C_h == out_channels`: same
parameter count, same expressiveness, a factor of `heads` less activation
memory. Combined with mixed precision this layer costs ~180 MB per layer at
batch size 2 instead of ~3.6 GB.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .segment_ops import segment_softmax, segment_sum
from .vn_layers import VNLeakyReLU, VNLinear, make_norm


class VNGATLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_vec_channels: int = 3,
        edge_scalar_channels: int = 1,
        heads: int = 4,
        negative_slope: float = 0.2,
        norm: str = "layer",
    ):
        super().__init__()
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads}) so the "
                f"hidden width can be split across heads rather than replicated per head."
            )
        self.heads = heads
        self.out_channels = out_channels
        self.head_channels = out_channels // heads

        self.lin_q = VNLinear(in_channels, out_channels)
        self.lin_k = VNLinear(in_channels, out_channels)
        self.lin_v = VNLinear(in_channels, out_channels)
        self.lin_edge_vec = VNLinear(edge_vec_channels, out_channels)
        self.edge_scalar_to_bias = nn.Linear(edge_scalar_channels, heads)

        self.lin_out = VNLinear(out_channels, out_channels)
        self.norm = make_norm(norm, out_channels)
        self.act = VNLeakyReLU(out_channels, negative_slope=negative_slope)

        self.attn_scale = (self.head_channels * 3) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scalar: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:           (N, C_in, 3)
        edge_index:  (2, E) directed; row 0 = source j, row 1 = destination i
        edge_scalar: (E, S) invariant
        edge_vec:    (E, V, 3) equivariant
        """
        N = x.shape[0]
        H, Ch = self.heads, self.head_channels
        src, dst = edge_index[0], edge_index[1]

        q = self.lin_q(x).reshape(N, H, Ch, 3)
        k = self.lin_k(x).reshape(N, H, Ch, 3)
        v = self.lin_v(x).reshape(N, H, Ch, 3)
        edge_v = self.lin_edge_vec(edge_vec).reshape(-1, H, Ch, 3)

        q_i = q.index_select(0, dst)
        k_j = k.index_select(0, src)
        logits = (q_i * k_j).sum(dim=(-1, -2)) * self.attn_scale          # (E, H) invariant
        logits = logits + self.edge_scalar_to_bias(edge_scalar)
        alpha = segment_softmax(logits, dst, N)                           # (E, H)

        # `.to(v.dtype)`: autocast promotes the `.sum()` above to float32, so
        # `alpha` comes back float32 and would promote `msg` with it -- and
        # `msg` is (2E, heads, C_h, 3), the widest tensor in the layer. Casting
        # the weights, not the values, keeps the big one in half precision.
        msg = (v.index_select(0, src) + edge_v) * alpha.to(v.dtype)[..., None, None]
        out = segment_sum(msg, dst, N).reshape(N, self.out_channels, 3)

        out = self.lin_out(out)
        out = self.norm(out)
        return self.act(out)
