"""
VN-GAT: equivariant multi-head attention over the real mesh graph, TensorFlow.

Message passing is fragment-internal by construction -- mesh edges never cross a
fragment boundary. Cross-fragment communication is the virtual nodes' job.

For each directed edge j -> i:

    alpha_ij = softmax_j( <q_i, k_j> / sqrt(C_h * 3) + b_ij )
    msg_ij   = v_j + VNLinear(edge_vec_ij)
    out_i    = sum_j alpha_ij * msg_ij

`<q_i, k_j>` sums over the head's channel and xyz axes, so it is
rotation-invariant; `b_ij` comes from the edge's invariant scalar (its length)
through a plain Dense layer, which is safe precisely because that input has no
xyz axis to protect. An invariant-weighted sum of equivariant vectors is
equivariant, so the layer is.

HEADS SPLIT THE WIDTH, THEY DO NOT REPLICATE IT. Giving every head the full
hidden width multiplies the model's widest tensors -- per-directed-edge, ~190k
rows at batch size 2, five of them live per layer -- by `heads`. Splitting
(C_h = out_channels // heads) has the same parameter count and expressiveness
for a factor of `heads` less activation memory.
"""
from __future__ import annotations

import tensorflow as tf

from .segment_ops import at_least_float32, segment_softmax, segment_sum
from .vn_layers import VNLeakyReLU, VNLinear, make_norm


class VNGATLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels: int, heads: int = 4,
                 negative_slope: float = 0.2, norm: str = "layer", **kw):
        super().__init__(**kw)
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads}) so the "
                f"hidden width can be split across heads rather than replicated per head.")
        self.out_channels, self.heads = out_channels, heads
        self.head_channels = out_channels // heads
        self.attn_scale = (self.head_channels * 3) ** -0.5

        self.lin_q = VNLinear(out_channels, dtype=self.dtype)
        self.lin_k = VNLinear(out_channels, dtype=self.dtype)
        self.lin_v = VNLinear(out_channels, dtype=self.dtype)
        self.lin_edge_vec = VNLinear(out_channels, dtype=self.dtype)
        self.edge_scalar_to_bias = tf.keras.layers.Dense(heads, dtype=self.dtype)
        self.lin_out = VNLinear(out_channels, dtype=self.dtype)
        self.norm = make_norm(norm, dtype=self.dtype)
        self.act = VNLeakyReLU(negative_slope, dtype=self.dtype)

    def call(self, x, edge_index, edge_scalar, edge_vec):
        """
        x:           (N, C_in, 3)
        edge_index:  (2, E) directed; row 0 = source j, row 1 = destination i
        edge_scalar: (E, S) invariant
        edge_vec:    (E, V, 3) equivariant
        """
        n = tf.shape(x)[0]
        H, Ch = self.heads, self.head_channels
        src, dst = edge_index[0], edge_index[1]

        q = tf.reshape(self.lin_q(x), [-1, H, Ch, 3])
        k = tf.reshape(self.lin_k(x), [-1, H, Ch, 3])
        v = tf.reshape(self.lin_v(x), [-1, H, Ch, 3])
        edge_v = tf.reshape(self.lin_edge_vec(edge_vec), [-1, H, Ch, 3])

        q_i, k_j = tf.gather(q, dst), tf.gather(k, src)
        # float32 for the SCORE, and specifically for the elementwise product.
        # Reductions get promoted by mixed-precision policies but the product
        # `q_i * k_j` does not: in float16 it overflows to inf once both
        # operands pass ~256 (256^2 > 65504), and `segment_softmax` then
        # subtracts the per-segment max for stability -- inf - inf = NaN. The
        # score is one scalar per edge per head, so widening it is cheap; the
        # (E, H, C_h, 3) value tensors stay in half precision.
        logits = tf.reduce_sum(at_least_float32(q_i) * at_least_float32(k_j),
                               axis=[-1, -2]) * self.attn_scale
        logits = logits + at_least_float32(self.edge_scalar_to_bias(edge_scalar))
        alpha = segment_softmax(logits, dst, n)

        msg = (tf.gather(v, src) + edge_v) * tf.cast(alpha, v.dtype)[..., None, None]
        out = tf.reshape(segment_sum(msg, dst, n), [-1, self.out_channels, 3])
        return self.act(self.norm(self.lin_out(out)))
