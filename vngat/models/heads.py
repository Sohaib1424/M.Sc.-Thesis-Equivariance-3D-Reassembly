"""
Invariant embedding heads, TensorFlow.

These MUST be invariant, not the raw equivariant backbone features. Two
fragments meeting at the same physical interface point but sitting at
uncorrelated orientations -- exactly the situation before alignment -- produce
different equivariant features there purely because of pose. Asking those to
agree is geometrically impossible; asking invariant projections of them to
agree is the intended constraint.
"""
from __future__ import annotations

import tensorflow as tf

from .vn_layers import VNInvariant


class InvariantEmbeddingHead(tf.keras.layers.Layer):
    """Gram-matrix reduction (with an equivariant bottleneck) plus a small MLP."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, bottleneck: int = 16, **kw):
        super().__init__(**kw)
        self.invariant = VNInvariant(bottleneck, dtype=self.dtype)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu", dtype=self.dtype),
            tf.keras.layers.Dense(embed_dim, dtype=self.dtype),
        ])

    def call(self, x):
        return self.mlp(self.invariant(x))


def symmetric_edge_features(node_features, edge_index, edge_vec):
    """
    Edge-head input that is SYMMETRIC in the edge's two endpoints:
    `h[u] + h[v]` concatenated with the edge's own vectors.

    Concatenating `[h[src], h[dst]]` would make the embedding depend on
    `edges_unique`'s arbitrary index ordering. Two fragments meeting at the same
    physical interface edge have unrelated local vertex numbering, so their
    embeddings would disagree for a purely bookkeeping reason and the
    consistency loss would spend capacity undoing it. Summing removes the
    dependence exactly, at no cost in information about the pair.
    """
    pooled = tf.gather(node_features, edge_index[0]) + tf.gather(node_features, edge_index[1])
    return tf.concat([pooled, edge_vec], axis=1)
