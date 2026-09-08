"""
Composite training objective, TensorFlow.

    L = w_rot L_rot + w_pos L_pos + w_node L_node + w_mid L_mid
        + w_face L_face + w_emb_v L_emb-v + w_emb_e L_emb-e

The five geometric terms follow the design document. The two interface-embedding
terms replace the document's formula, which is degenerate three times over --
see `cluster_consistency_loss`.
"""
from __future__ import annotations

import tensorflow as tf

from ..evaluation.metrics import swing_twist_error
from ..models.segment_ops import at_least_float32
from ..models.vn_layers import geodesic_rotation_loss

_HALF = (tf.float16, tf.bfloat16)


def _mean(values):
    """
    Mean that stays finite on an empty tensor.

    `reduce_mean` of an empty tensor is NaN, and a single NaN in the weighted
    total propagates into every parameter on the next step -- turning a
    momentarily empty term into a silently destroyed multi-hour run.
    """
    return tf.cond(tf.size(values) > 0,
                   lambda: tf.reduce_mean(values),
                   lambda: tf.zeros((), values.dtype))


def _reduce(values, mask=None):
    """
    Mean over `values`, restricted to `mask` when the batch has been padded.

    `mask=None` is the unpadded path. TPU batches are padded to static shapes,
    where a plain mean would divide by the padded length and silently rescale
    every loss by (padded / real).
    """
    if mask is None:
        return _mean(values)
    m = tf.cast(mask, values.dtype)
    while len(m.shape) < len(values.shape):
        m = tf.expand_dims(m, -1)
    # Sum-and-divide, not boolean_mask: the latter produces a dynamic shape,
    # which forces an XLA recompilation on every batch.
    return tf.reduce_sum(values * m) / tf.maximum(tf.reduce_sum(m), tf.constant(1.0, values.dtype))


def node_position_loss(x_pred, x_gt, mask=None):
    """L_pos = mean_v ||x_v - xhat_v||^2. Inputs (V, 3)."""
    return _reduce(tf.reduce_sum(tf.square(x_pred - x_gt), -1), mask)


def node_normal_loss(n_pred, n_gt, eps: float = 1e-8, mask=None):
    """L_node = mean_v (1 - n_v . nhat_v). Renormalised defensively."""
    n_pred = tf.math.l2_normalize(n_pred, axis=-1, epsilon=eps)
    n_gt = tf.math.l2_normalize(n_gt, axis=-1, epsilon=eps)
    return _reduce(1 - tf.reduce_sum(n_pred * n_gt, -1), mask)


def edge_midpoint_loss(m_pred, m_gt, mask=None):
    """L_mid = mean_e ||m_e - mhat_e||^2. Inputs (E, 3)."""
    return _reduce(tf.reduce_sum(tf.square(m_pred - m_gt), -1), mask)


def face_normal_loss(n1_pred, n1_gt, n2_pred, n2_gt, eps: float = 1e-8, mask=None):
    """L_face = mean_e [(1 - n1.n1hat) + (1 - n2.n2hat)]."""
    f = lambda a, b: 1 - tf.reduce_sum(  # noqa: E731
        tf.math.l2_normalize(a, -1, eps) * tf.math.l2_normalize(b, -1, eps), -1)
    return _reduce(f(n1_pred, n1_gt) + f(n2_pred, n2_gt), mask)


def cluster_consistency_loss(embeddings, cluster_id, pull_margin: float = 0.1,
                             push_margin: float = 0.5, push_weight: float = 1.0,
                             max_push_clusters: int = 512):
    """
    Discriminative interface-embedding loss on the UNIT SPHERE.

    Embeddings are L2-normalised first, then:
        pull_k = mean_{i in k} relu(||z_i - c_k|| - pull_margin)^2
        push   = mean_{a != b} relu(2*push_margin - ||c_a - c_b||)^2

    THREE DEGENERACIES, ALL OBSERVED IN TRAINING, ALL FIXED HERE

    1. The design document's `|| sum z ||^2` is minimised by embeddings that
       CANCEL rather than agree: (3,-1)+(3,-1) scores 40, (5,0)+(-5,0) scores 0.
    2. Plain within-cluster variance is zero for ANY CONSTANT embedding, so a
       real run collapsed to a single vector within two epochs, encoding
       nothing. The `push` term fixes that.
    3. But `push` on UNNORMALISED embeddings has its own escape hatch:
       separating clusters by inflating magnitude is easier than arranging them.
       A run drove the mean centroid norm to ~565, and 565^2 overflows float16,
       producing NaN losses from epoch 19. Normalising removes the escape by
       construction -- every embedding has norm 1, all distances lie in [0, 2],
       and no weight setting can make the term large. It also bounds the loss to
       O(1) so this objective cannot dominate the gradient budget (measured at
       63% of the total before the change, against rotation's 15%), and it is
       the right space for the inference-time matcher, which compares
       descriptors by distance.

    `push_margin` must be below 1.0, since normalised embeddings cannot be more
    than 2 apart. The minimum is not zero -- judge this term by whether it
    FALLS, and treat an exact 0.0000 as the collapse alarm.
    """
    mask = cluster_id >= 0
    if embeddings.shape[0] == 0:
        return tf.reduce_sum(embeddings) * 0.0

    emb = tf.boolean_mask(embeddings, mask)
    acc = tf.float32 if emb.dtype in _HALF else emb.dtype
    emb = tf.cast(emb, acc)
    n_real = tf.shape(emb)[0]

    def empty():
        # Graph-connected zero, so the embedding heads still receive a (zero)
        # gradient and every replica builds identical reduction buckets.
        return tf.reduce_sum(embeddings) * 0.0

    def compute():
        e = emb / tf.sqrt(tf.reduce_sum(emb * emb, -1, keepdims=True) + 1e-12)
        cid = tf.boolean_mask(cluster_id, mask)
        uniq, inverse = tf.unique(cid)
        num_clusters = tf.size(uniq)

        counts = tf.cast(tf.maximum(tf.math.unsorted_segment_sum(
            tf.ones_like(inverse), inverse, num_clusters), 1), acc)
        sums = tf.math.unsorted_segment_sum(e, inverse, num_clusters)
        centroids = sums / tf.expand_dims(counts, -1)

        dist = tf.sqrt(tf.reduce_sum(tf.square(e - tf.gather(centroids, inverse)), -1) + 1e-12)
        per_point = tf.square(tf.nn.relu(dist - pull_margin))
        pull = tf.reduce_mean(
            tf.math.unsorted_segment_sum(per_point, inverse, num_clusters) / counts)

        def push_term():
            picked = centroids
            k = tf.minimum(num_clusters, max_push_clusters)
            picked = tf.gather(picked, tf.random.shuffle(tf.range(num_clusters))[:k])
            # NOT a self-cdist: its diagonal is an exact zero distance where the
            # gradient is 0/0, and masking the forward does not stop the
            # backward producing NaN.
            sq = tf.reduce_sum(picked * picked, -1)
            d2 = tf.expand_dims(sq, 1) + tf.expand_dims(sq, 0) - 2.0 * tf.matmul(
                picked, picked, transpose_b=True)
            pairwise = tf.sqrt(tf.maximum(d2, 0.0) + 1e-12)
            off = tf.logical_not(tf.eye(k, dtype=tf.bool))
            return tf.reduce_mean(tf.square(tf.nn.relu(
                2 * push_margin - tf.boolean_mask(pairwise, off))))

        push = tf.cond(num_clusters > 1, push_term, lambda: tf.zeros((), acc))
        return pull + push_weight * push

    return tf.cond(n_real > 0, compute, empty)


class CompositeLoss:
    def __init__(self, w_rot=1.0, w_pos=1.0, w_node=1.0, w_mid=1.0, w_face=1.0,
                 w_emb_v=1.0, w_emb_e=1.0, emb_pull_margin=0.1, emb_push_margin=0.5,
                 symmetry_axis="z"):
        self.weights = dict(rot=w_rot, pos=w_pos, node=w_node, mid=w_mid,
                            face=w_face, emb_v=w_emb_v, emb_e=w_emb_e)
        self.emb_pull_margin = emb_pull_margin
        self.emb_push_margin = emb_push_margin
        self.symmetry_axis = symmetry_axis

    def __call__(self, outputs, targets):
        node_mask = targets.get("node_mask")
        edge_mask = targets.get("edge_mask")
        frag_mask = targets.get("frag_mask")

        rot_angles = geodesic_rotation_loss(outputs["R_pred"], targets["R_gt"])
        tilt, twist = swing_twist_error(outputs["R_pred"], targets["R_gt"],
                                        axis=self.symmetry_axis)
        tilt, twist = tf.stop_gradient(tilt), tf.stop_gradient(twist)

        l_rot = _reduce(rot_angles, frag_mask)
        l_pos = node_position_loss(outputs["x_pred"], targets["x_gt"], node_mask)
        l_node = node_normal_loss(outputs["n_pred"], targets["n_gt"], mask=node_mask)
        l_mid = edge_midpoint_loss(outputs["mid_pred"], targets["mid_gt"], edge_mask)
        l_face = face_normal_loss(outputs["n1_pred"], targets["n1_gt"],
                                  outputs["n2_pred"], targets["n2_gt"], mask=edge_mask)
        margins = dict(pull_margin=self.emb_pull_margin, push_margin=self.emb_push_margin)
        l_emb_v = cluster_consistency_loss(
            outputs["vertex_embedding"], targets["vertex_cluster_id"], **margins)
        l_emb_e = cluster_consistency_loss(
            outputs["edge_embedding"], targets["edge_cluster_id"], **margins)

        w = self.weights
        total = (w["rot"] * l_rot + w["pos"] * l_pos + w["node"] * l_node
                 + w["mid"] * l_mid + w["face"] * l_face
                 + w["emb_v"] * l_emb_v + w["emb_e"] * l_emb_e)
        return {
            "total": total, "rot": l_rot,
            # Degrees, because the thesis tables are in degrees. This is the
            # mean GEODESIC angle -- NOT the Euler-angle RMSE GARF reports.
            # They are different numbers and must not be compared directly.
            "rot_deg": _reduce(tf.stop_gradient(rot_angles), frag_mask) * (180.0 / 3.141592653589793),
            "pos": l_pos, "node": l_node, "mid": l_mid, "face": l_face,
            "emb_v": l_emb_v, "emb_e": l_emb_e,
            # Reported, never optimised. On validation this split separates
            # "has not learned the axis either" (tilt ~ 90) from "learned the
            # axis, cannot recover the azimuth" (tilt ~ 0, twist ~ 90) -- the
            # latter a structural limit of a one-shot per-fragment
            # canonicaliser, which no tuning would move.
            "tilt": _reduce(tilt, frag_mask), "twist": _reduce(twist, frag_mask),
        }
