"""
Cross-fragment communication through K virtual nodes per fragment, TensorFlow.

Real mesh edges never join two fragments, so without this the network could only
reason about one fragment at a time -- and orientation is only determined
relative to the rest of the object.

Three stages:
  1. Upward pooling  : each fragment's vertices -> that fragment's K slots.
  2. Exchange        : invariant descriptors of same-scene slots attend to each
                       other, gating each fragment's own equivariant vectors.
  3. Downward cast   : each vertex attends back to its own fragment's slots.

PER-FRAGMENT EQUIVARIANCE -- WHY STAGE 2 EXCHANGES INVARIANTS ONLY
------------------------------------------------------------------
Diffusion rotates EVERY FRAGMENT INDEPENDENTLY, so the property the model needs
is

    G_f(A_1 x_1, ..., A_F x_F) = G_f(x_1, ..., x_F) A_f^T,

i.e. fragment f's output must be equivariant to ITS OWN rotation and INVARIANT
to every other fragment's. That is what lets a single learned canonicalisation,
G_f(clean) = I, produce the target A_f^T for every draw.

Letting slots of different fragments attend to each other as VECTORS destroys
it. An attention logit <q from fragment a, k from fragment b> becomes
<A_a q, A_b k>, which is invariant only when A_a = A_b. Such a block is
equivariant to a GLOBAL rotation of the whole scene but not to the per-fragment
rotations training actually applies -- a distinction invisible to a global-only
equivariance check while quietly removing the guarantee.

So everything crossing a fragment boundary here is invariant: slots reduce to
rotation-invariant descriptors, those attend to each other, and the resulting
invariant context produces a per-channel GATE rescaling each fragment's own
equivariant slot vectors. Invariant scalar times equivariant vector is
equivariant, so the update rotates with that fragment and nothing else.

Nothing useful is lost: the relative orientation between two scattered
fragments is uniformly random noise by construction. What determines which
pieces mate is invariant interface shape, which still crosses.

SLOT ANCHORS
------------
The design document gives each slot a fixed spatial coordinate. A world-frame
coordinate does not rotate when the input fragment does, which breaks the
equivariance the backbone exists to provide. Each slot instead gets a fixed
NON-SPATIAL learned identity (plain scalars -- no xyz axis, so nothing can
rotate wrongly) which gates, by invariant scalar multiplication, an equivariant
seed pooled from the fragment's own vertices.

SCENE SCOPING
-------------
A batch concatenates independent scenes and fragment ids are unique across the
whole batch, so stage 2 must be restricted to fragments of the SAME SCENE.
Unmasked, fragments of unrelated objects would exchange information and a
scene's output would depend on what else landed in the batch.
"""
from __future__ import annotations

import tensorflow as tf

from .segment_ops import (
    at_least_float32, blockwise_softmax, segment_mean, segment_softmax, segment_sum,
)
from .vn_layers import VNInvariant, VNLeakyReLU, VNLinear, make_norm


class VirtualNodeBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, num_slots: int = 8, heads: int = 4,
                 identity_dim: int = 16, norm: str = "layer",
                 invariant_bottleneck: int = 8, **kw):
        super().__init__(**kw)
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels, self.num_slots, self.heads = channels, num_slots, heads
        self.head_channels = channels // heads
        self.scale = (self.head_channels * 3) ** -0.5
        self.token_scale = (channels // heads) ** -0.5
        self.identity_dim = identity_dim

        self.gate_map = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.seed_proj = VNLinear(channels, dtype=self.dtype)
        self.up_q = VNLinear(channels, dtype=self.dtype)
        self.up_k = VNLinear(channels, dtype=self.dtype)
        self.up_v = VNLinear(channels, dtype=self.dtype)

        self.slot_invariant = VNInvariant(invariant_bottleneck, dtype=self.dtype)
        self.token_in = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.token_q = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.token_k = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.token_v = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.token_out = tf.keras.layers.Dense(channels, dtype=self.dtype)
        self.glob_out = VNLinear(channels, dtype=self.dtype)

        self.down_q = VNLinear(channels, dtype=self.dtype)
        self.down_k = VNLinear(channels, dtype=self.dtype)
        self.down_v = VNLinear(channels, dtype=self.dtype)
        self.down_out = VNLinear(channels, dtype=self.dtype)
        self.norm = make_norm(norm, dtype=self.dtype)
        self.act = VNLeakyReLU(dtype=self.dtype)

    def build(self, input_shape):
        # Pure scalars, no xyz axis, so nothing here can rotate incorrectly.
        self.slot_identity = self.add_weight(
            name="slot_identity", shape=(self.num_slots, self.identity_dim),
            dtype=self.dtype,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1), trainable=True)
        super().build(input_shape)

    def _slot_queries(self, x, node_frag, num_fragments):
        seed = segment_mean(self.seed_proj(x), node_frag, num_fragments)   # (F, C, 3)
        gate = tf.sigmoid(self.gate_map(self.slot_identity))               # (K, C) invariant
        return tf.reshape(gate, [1, self.num_slots, self.channels, 1]) * tf.expand_dims(seed, 1)

    def _exchange(self, q, k, v, frag_scene, K):
        """Attention over INVARIANT slot tokens, restricted to same-scene slots."""
        def attend(qs, ks, vs):
            logits = tf.einsum('qhd,khd->qhk', at_least_float32(qs),
                               at_least_float32(ks)) * self.token_scale
            alpha = tf.cast(tf.nn.softmax(logits, axis=-1), vs.dtype)
            return tf.einsum('qhk,khd->qhd', alpha, vs)

        if frag_scene is None:
            return attend(q, k, v)
        slot_scene = tf.repeat(frag_scene, K)
        logits = tf.einsum('qhd,khd->qhk', at_least_float32(q),
                           at_least_float32(k)) * self.token_scale
        alpha = tf.cast(blockwise_softmax(logits, slot_scene), v.dtype)
        return tf.einsum('qhk,khd->qhd', alpha, v)

    def call(self, x, node_frag, *, num_fragments, frag_scene=None):
        n = tf.shape(x)[0]
        K, H, Ch, C = self.num_slots, self.heads, self.head_channels, self.channels

        # ---- Stage 1: vertices -> their own fragment's slots -------------
        queries = self._slot_queries(x, node_frag, num_fragments)
        q_up = tf.reshape(self.up_q(queries), [num_fragments, K, H, Ch, 3])
        k_up = tf.reshape(self.up_k(x), [-1, H, Ch, 3])
        v_up = tf.reshape(self.up_v(x), [-1, H, Ch, 3])

        q_node = tf.gather(q_up, node_frag)                                # (N,K,H,Ch,3)
        logits = tf.einsum('nkhci,nhci->nkh', at_least_float32(q_node),
                           at_least_float32(k_up)) * self.scale
        alpha = segment_softmax(logits, node_frag, num_fragments)          # (N,K,H)
        slots = segment_sum(
            tf.cast(alpha, v_up.dtype)[..., None, None] * tf.expand_dims(v_up, 1),
            node_frag, num_fragments)                                      # (F,K,H,Ch,3)

        # ---- Stage 2: cross-fragment exchange, through invariants only ----
        flat = tf.reshape(slots, [-1, C, 3])
        tokens = self.token_in(self.slot_invariant(flat))                  # (Q, C) invariant
        q = tf.reshape(self.token_q(tokens), [-1, H, C // H])
        k = tf.reshape(self.token_k(tokens), [-1, H, C // H])
        v = tf.reshape(self.token_v(tokens), [-1, H, C // H])
        context = tf.reshape(self._exchange(q, k, v, frag_scene, K), [-1, C])
        gate = tf.tanh(self.token_out(context))                            # (Q, C) invariant
        # The only equivariant quantity is the fragment's OWN slots, rescaled
        # per channel by an invariant gate -- so the update rotates with that
        # fragment and with nothing else.
        flat = flat + tf.expand_dims(gate, -1) * self.glob_out(flat)
        slots = tf.reshape(flat, [num_fragments, K, C, 3])

        # ---- Stage 3: slots -> vertices of the same fragment --------------
        dq = tf.reshape(self.down_q(x), [-1, H, Ch, 3])
        flat_slots = tf.reshape(slots, [-1, C, 3])
        dk = tf.reshape(self.down_k(flat_slots), [num_fragments, K, H, Ch, 3])
        dv = tf.reshape(self.down_v(flat_slots), [num_fragments, K, H, Ch, 3])
        dk_node, dv_node = tf.gather(dk, node_frag), tf.gather(dv, node_frag)

        down_logits = tf.einsum('nhci,nkhci->nkh', at_least_float32(dq),
                                at_least_float32(dk_node)) * self.scale
        # A plain softmax over the K slots IS the fragment-scoped one, because
        # dk_node was gathered by fragment -- no mask needed at all.
        down_alpha = tf.cast(tf.nn.softmax(down_logits, axis=1), dv_node.dtype)
        ctx = tf.reshape(tf.reduce_sum(down_alpha[..., None, None] * dv_node, axis=1), [-1, C, 3])
        return self.act(self.norm(x + self.down_out(ctx)))
