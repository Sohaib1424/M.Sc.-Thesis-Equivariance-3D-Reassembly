"""Composite loss terms, including the three degeneracies observed in training."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from conftest import haar_rotations, make_scene
from vngat.losses.composite import (
    CompositeLoss, cluster_consistency_loss, edge_midpoint_loss,
    face_normal_loss, node_normal_loss, node_position_loss,
)
from vngat.models.vn_gat import VNGATModel
from vngat.training.bridge import (
    build_predictions, build_targets, ground_truth_rotation, to_tensors,
)


def test_geometric_terms_are_zero_at_the_optimum():
    x = tf.constant(np.random.default_rng(0).normal(size=(20, 3)))
    n = tf.math.l2_normalize(tf.constant(np.random.default_rng(1).normal(size=(20, 3))), -1)
    assert float(node_position_loss(x, x)) < 1e-12
    assert float(node_normal_loss(n, n)) < 1e-10
    assert float(edge_midpoint_loss(x, x)) < 1e-12
    assert float(face_normal_loss(n, n, n, n)) < 1e-10


def test_normal_loss_peaks_at_opposite_directions():
    n = tf.math.l2_normalize(tf.constant(np.random.default_rng(2).normal(size=(16, 3))), -1)
    assert abs(float(node_normal_loss(-n, n)) - 2.0) < 1e-8


def test_collapsed_embeddings_are_penalised():
    """
    A constant embedding drives within-cluster variance to zero, so a pull-only
    loss rates it perfect -- and a real run duly collapsed both embedding terms
    to exactly 0.0000 within two epochs. Repulsion must make collapse the WORST
    outcome, not the best.
    """
    cid = tf.constant([0, 0, 1, 1])
    good = tf.constant([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], tf.float64)
    collapsed = tf.ones((4, 2), tf.float64)
    assert float(cluster_consistency_loss(collapsed, cid)) > float(
        cluster_consistency_loss(good, cid))


def test_inflating_embedding_magnitude_does_not_help():
    """
    Second-order guard. With unnormalised embeddings, `push` could be satisfied
    by scaling everything up rather than arranging it: a real run drove mean
    centroid norm to ~565, past where squaring overflows float16, producing NaN
    losses in bursts. Normalising makes the loss exactly scale-invariant.
    """
    cid = tf.constant([0, 0, 1, 1])
    small = tf.constant([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], tf.float64)
    assert abs(float(cluster_consistency_loss(small, cid))
               - float(cluster_consistency_loss(small * 500.0, cid))) < 1e-8


def test_loss_is_bounded():
    """Bounded by construction, so this term cannot dominate the gradient budget
    -- it was 63% of the total at initialisation before normalising."""
    worst = tf.ones((64, 16), tf.float64)
    cid = tf.constant(np.arange(64) % 8)
    assert float(cluster_consistency_loss(worst, cid)) <= 1.01


def test_rejects_the_cancellation_degeneracy():
    """The design document's `|| sum z ||^2` scores opposed embeddings BETTER
    than identical ones. This loss must do the reverse."""
    identical = tf.constant([[3.0, -1.0], [3.0, -1.0]], tf.float64)
    opposed = tf.constant([[5.0, 0.0], [-5.0, 0.0]], tf.float64)
    cid = tf.constant([0, 0])
    doc_identical = float(tf.reduce_sum(tf.square(tf.reduce_sum(identical, 0))))
    doc_opposed = float(tf.reduce_sum(tf.square(tf.reduce_sum(opposed, 0))))
    assert doc_opposed < doc_identical          # the degeneracy, demonstrated
    assert float(cluster_consistency_loss(identical, cid)) < float(
        cluster_consistency_loss(opposed, cid))


def test_unshared_points_are_ignored():
    emb = tf.constant([[1.0, 1.0], [1.0, 1.0], [99.0, -99.0]], tf.float64)
    with_unshared = cluster_consistency_loss(emb, tf.constant([0, 0, -1]))
    without = cluster_consistency_loss(emb[:2], tf.constant([0, 0]))
    assert abs(float(with_unshared) - float(without)) < 1e-8


def test_empty_cluster_set_stays_connected():
    """With no shared points the loss must remain graph-connected, so the
    embedding heads still receive a (zero) gradient."""
    emb = tf.Variable(np.random.default_rng(3).normal(size=(5, 4)))
    with tf.GradientTape() as tape:
        loss = cluster_consistency_loss(emb, tf.constant([-1, -1, -1, -1, -1]))
    assert float(loss) == 0.0
    grad = tape.gradient(loss, emb)
    assert grad is not None and float(tf.reduce_max(tf.abs(grad))) == 0.0


def test_perfect_prediction_gives_zero_geometric_loss():
    """
    End-to-end consistency: diffuse by A, predict A^T, and every geometric term
    must vanish. If any convention in the chain is transposed, this fails.
    """
    scene = make_scene(seed=5)
    A = haar_rotations(scene.num_fragments, 7)
    diffused = scene.rotate_per_fragment(A)
    R = ground_truth_rotation(tf.constant(A))
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       embed_dim=4, gram_bottleneck=4, dtype="float64")
    out = model(**to_tensors(diffused, tf.float64))
    losses = CompositeLoss()(
        dict(R_pred=R, vertex_embedding=out["vertex_embedding"],
             edge_embedding=out["edge_embedding"],
             **build_predictions(diffused, R, dtype=tf.float64)),
        build_targets(scene, tf.constant(A), scene, dtype=tf.float64))
    for key in ("pos", "node", "mid", "face"):
        assert abs(float(losses[key])) < 1e-10, f"{key} = {float(losses[key])}"
    assert float(losses["rot_deg"]) < 1e-3


def test_tilt_twist_are_reported_but_not_optimised():
    scene = make_scene(seed=6)
    A = haar_rotations(scene.num_fragments, 8)
    diffused = scene.rotate_per_fragment(A)
    R = ground_truth_rotation(tf.constant(A))
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       embed_dim=4, gram_bottleneck=4, dtype="float64")
    out = model(**to_tensors(diffused, tf.float64))
    losses = CompositeLoss()(
        dict(R_pred=R, vertex_embedding=out["vertex_embedding"],
             edge_embedding=out["edge_embedding"],
             **build_predictions(diffused, R, dtype=tf.float64)),
        build_targets(scene, tf.constant(A), scene, dtype=tf.float64))
    # a perfect prediction has no residual to decompose
    assert float(losses["tilt"]) < 0.1 and float(losses["twist"]) < 0.1
    # and the total is the weighted sum of the SEVEN objective terms only
    parts = sum(float(losses[k]) for k in
                ("rot", "pos", "node", "mid", "face", "emb_v", "emb_e"))
    assert abs(float(losses["total"]) - parts) < 1e-8
