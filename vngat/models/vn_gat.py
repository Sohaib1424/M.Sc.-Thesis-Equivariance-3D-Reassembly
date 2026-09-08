"""
Full VN-GAT orientation model, TensorFlow.

    input projection ([pos, normal] -> hidden vector channels)
      -> num_layers x (VNGATLayer, VirtualNodeBlock)
      -> per-fragment equivariant mean pool
      -> Gram-Schmidt head -> R_pred   (see `predict_rotation` for the transpose)
      -> invariant vertex / edge embedding heads (parallel branch)

Scope, stated explicitly: this network predicts ROTATION only. Translation is
recovered afterwards by the classical solver in `vngat/assembly/translation.py`,
because per-fragment centralisation cancels translation exactly and the rotation
network cannot see it even in principle.
"""
from __future__ import annotations

import tensorflow as tf

from .gat_layer import VNGATLayer
from .heads import InvariantEmbeddingHead, symmetric_edge_features
from .segment_ops import segment_mean
from .virtual_nodes import VirtualNodeBlock
from .vn_layers import VNLinear, predict_rotation


class VNGATModel(tf.keras.Model):
    def __init__(self, hidden_channels: int = 64, num_layers: int = 4,
                 num_vn_slots: int = 8, heads: int = 4, embed_dim: int = 32,
                 edge_vec_channels: int = 3, gram_bottleneck: int = 16,
                 norm: str = "layer", **kw):
        super().__init__(**kw)
        if hidden_channels % heads:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads}): "
                f"the hidden width is split across heads, not replicated.")
        self.hidden_channels, self.num_layers = hidden_channels, num_layers
        self.input_proj = VNLinear(hidden_channels, dtype=self.dtype)
        self.mesh_layers = [VNGATLayer(hidden_channels, heads=heads, norm=norm,
                                       dtype=self.dtype) for _ in range(num_layers)]
        self.vn_blocks = [VirtualNodeBlock(hidden_channels, num_slots=num_vn_slots,
                                           heads=heads, norm=norm, dtype=self.dtype)
                          for _ in range(num_layers)]
        self.rotation_head = VNLinear(2, dtype=self.dtype)
        self.vertex_embed_head = InvariantEmbeddingHead(
            embed_dim, bottleneck=gram_bottleneck, dtype=self.dtype)
        self.edge_embed_head = InvariantEmbeddingHead(
            embed_dim, bottleneck=gram_bottleneck, dtype=self.dtype)

    @staticmethod
    def symmetrise_edges(edge_index, edge_len, edge_vec):
        """
        Undirected storage -> the directed pairs message passing needs.

        The reverse copy swaps the two adjacent-face-normal slots so "first
        normal" keeps a consistent meaning relative to the direction of travel.
        Layout is [all forward; all backward], so the first E rows correspond
        1:1 with the stored undirected edges -- no interleaving, no mask, and
        the whole "which copy is the forward one" bug class is impossible.
        """
        rev_index = tf.reverse(edge_index, axis=[0])
        rev_vec = tf.gather(edge_vec, [0, 2, 1], axis=1)
        return (tf.concat([edge_index, rev_index], axis=1),
                tf.concat([edge_len, edge_len], axis=0),
                tf.concat([edge_vec, rev_vec], axis=0))

    def call(self, node_vec, edge_index, edge_len, edge_vec, node_frag,
             *, num_fragments, frag_scene=None, training=False):
        di_index, di_len, di_vec = self.symmetrise_edges(edge_index, edge_len, edge_vec)

        h = self.input_proj(node_vec)
        for mesh_layer, vn_block in zip(self.mesh_layers, self.vn_blocks):
            h = mesh_layer(h, di_index, di_len, di_vec)
            h = vn_block(h, node_frag, num_fragments=num_fragments, frag_scene=frag_scene)

        pooled = segment_mean(h, node_frag, num_fragments)      # (F, hidden, 3) equivariant
        rot_vecs = self.rotation_head(pooled)                   # (F, 2, 3)
        a1, a2 = rot_vecs[:, 0], rot_vecs[:, 1]
        R_pred = predict_rotation(a1, a2)

        # Conditioning of the Gram-Schmidt head, as an invariant scalar. The
        # frame orthogonalises channel 2 against channel 1, so when the two are
        # collinear the orthogonal residual vanishes and both the frame and its
        # gradient become ill-conditioned. Reported, never optimised: real runs
        # sat at |cos| ~ 0.8-0.9 while learning nothing, and dropped to ~0.2
        # exactly when they started making progress.
        a1f, a2f = tf.stop_gradient(a1), tf.stop_gradient(a2)
        n1 = tf.maximum(tf.norm(a1f, axis=-1), 1e-12)
        n2 = tf.maximum(tf.norm(a2f, axis=-1), 1e-12)
        head_cos = tf.reduce_mean(tf.abs(tf.reduce_sum(a1f * a2f, -1) / (n1 * n2)))

        return {
            "node_features": h,
            "R_pred": R_pred,
            "head_cos": head_cos,
            "vertex_embedding": self.vertex_embed_head(h),
            "edge_embedding": self.edge_embed_head(
                symmetric_edge_features(h, edge_index, edge_vec)),
        }

    def num_parameters(self) -> int:
        return int(sum(int(tf.size(w)) for w in self.trainable_weights))
