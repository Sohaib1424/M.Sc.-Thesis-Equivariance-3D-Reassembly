"""
The properties that make the thesis's central claim checkable rather than
asserted. Run in float64: a correct implementation gives ~1e-15, and float32
would give ~1e-7 for both a correct and a broken one.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from conftest import haar_rotations, make_scene
from vngat.models.gat_layer import VNGATLayer
from vngat.models.heads import InvariantEmbeddingHead, symmetric_edge_features
from vngat.models.virtual_nodes import VirtualNodeBlock
from vngat.models.vn_gat import VNGATModel
from vngat.models.vn_layers import (
    VNInvariant, VNLayerNorm, VNLeakyReLU, VNLinear,
    geodesic_rotation_loss, gram_schmidt_frame, predict_rotation,
)
from vngat.training.bridge import to_tensors

TOL = 1e-11


def _rot(x, R):
    return tf.matmul(x, R, transpose_b=True)


def test_primitive_layers_are_equivariant():
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(40, 6, 3)))
    A = tf.constant(haar_rotations(1, 3)[0])
    for layer in (VNLinear(10, dtype="float64"), VNLeakyReLU(dtype="float64"),
                  VNLayerNorm(dtype="float64")):
        _ = layer(x)
        assert float(tf.reduce_max(tf.abs(_rot(layer(x), A) - layer(_rot(x, A))))) < TOL


def test_vn_invariant_is_invariant_and_bottlenecked():
    rng = np.random.default_rng(1)
    x = tf.constant(rng.normal(size=(20, 32, 3)))
    A = tf.constant(haar_rotations(1, 4)[0])
    layer = VNInvariant(8, dtype="float64")
    _ = layer(x)
    assert float(tf.reduce_max(tf.abs(layer(x) - layer(_rot(x, A))))) < TOL
    assert layer.out_features == 64          # 8*8, not 32*32


def test_rotation_head_is_right_equivariant():
    """G(Ax) == G(x)A^T. Combined with the target R_gt = A^T, this is what lets
    a single learned canonicalisation generalise to every orientation."""
    rng = np.random.default_rng(2)
    a1 = tf.constant(rng.normal(size=(32, 3)))
    a2 = tf.constant(rng.normal(size=(32, 3)))
    A = tf.constant(haar_rotations(1, 5)[0])
    lhs = tf.matmul(predict_rotation(a1, a2), A, transpose_b=True)
    rhs = predict_rotation(_rot(a1, A), _rot(a2, A))
    assert float(tf.reduce_max(tf.abs(lhs - rhs))) < TOL


def test_transposed_head_recovers_the_target_for_every_rotation():
    """
    The decisive test. A network that has learned G(x_clean) = I must output
    exactly A^T for EVERY scattering rotation A, with no further learning. That
    is the entire payoff of equivariance -- and returning F instead of F^T makes
    the target provably unlearnable, silently.
    """
    c1 = tf.constant([[1.0, 0.0, 0.0]], tf.float64)
    c2 = tf.constant([[0.0, 1.0, 0.0]], tf.float64)
    assert float(tf.reduce_max(tf.abs(
        predict_rotation(c1, c2) - tf.eye(3, dtype=tf.float64)))) < TOL
    for A in haar_rotations(8, 6):
        A = tf.constant(A)
        R = predict_rotation(_rot(c1, A), _rot(c2, A))
        assert float(tf.reduce_max(tf.abs(R - tf.transpose(A)))) < TOL


def test_gram_schmidt_is_scale_invariant():
    """
    A fixed eps is negligible against ||a||~1 but comparable to it once the
    predicted vectors shrink -- measured, the frame drifted 2e-3 from orthogonal
    at ||a|| ~ 1e-6 before the inputs were pre-normalised. Small activations are
    exactly what an untrained network produces.
    """
    rng = np.random.default_rng(3)
    a1 = tf.constant(rng.normal(size=(64, 3)))
    a2 = tf.constant(rng.normal(size=(64, 3)))
    for s in (1e3, 1.0, 1e-3, 1e-6):
        R = gram_schmidt_frame(a1 * s, a2 * s)
        eye = tf.eye(3, dtype=tf.float64)
        assert float(tf.reduce_max(tf.abs(tf.matmul(R, R, transpose_b=True) - eye))) < 1e-12
        assert float(tf.reduce_max(tf.abs(tf.linalg.det(R) - 1))) < 1e-12


def test_gat_layer_is_equivariant(scene):
    A = tf.constant(haar_rotations(1, 7)[0])
    ei, el, ev = VNGATModel.symmetrise_edges(
        tf.constant(scene.edge_index, tf.int32),
        tf.constant(scene.edge_len), tf.constant(scene.edge_vec))
    x = tf.constant(scene.node_vec)
    layer = VNGATLayer(8, heads=2, dtype="float64")
    _ = layer(x, ei, el, ev)
    lhs = _rot(layer(x, ei, el, ev), A)
    rhs = layer(_rot(x, A), ei, el, _rot(ev, A))
    assert float(tf.reduce_max(tf.abs(lhs - rhs))) < TOL


def _rot_per_fragment(x, R, node_frag):
    return tf.einsum('nij,ncj->nci', tf.gather(R, node_frag), x)


def test_virtual_node_block_is_equivariant_per_fragment(scene):
    """
    THE property the target depends on, and the one a global-rotation check
    cannot see. Diffusion rotates every fragment INDEPENDENTLY, so fragment f's
    output must be equivariant to its own rotation and invariant to the others'.
    A block that lets slots of different fragments attend as VECTORS passes the
    global check and fails this one.
    """
    nf = tf.constant(scene.node_frag, tf.int32)
    fs = tf.constant(scene.frag_scene, tf.int32)
    x = tf.constant(np.random.default_rng(8).normal(size=(scene.num_nodes, 8, 3)))
    blk = VirtualNodeBlock(8, num_slots=4, heads=2, dtype="float64")
    run = lambda v: blk(v, nf, num_fragments=scene.num_fragments, frag_scene=fs)  # noqa: E731
    _ = run(x)
    R = tf.constant(haar_rotations(scene.num_fragments, 9))
    lhs = _rot_per_fragment(run(x), R, nf)
    rhs = run(_rot_per_fragment(x, R, nf))
    assert float(tf.reduce_max(tf.abs(lhs - rhs))) < TOL


def test_cross_fragment_information_still_flows(scene):
    """Guards the fix from being 'achieved' by cutting the connection:
    perturbing one fragment's SHAPE must still change another's output, even
    though rotating it must not."""
    nf = tf.constant(scene.node_frag, tf.int32)
    fs = tf.constant(scene.frag_scene, tf.int32)
    x = np.random.default_rng(10).normal(size=(scene.num_nodes, 8, 3))
    blk = VirtualNodeBlock(8, num_slots=4, heads=2, dtype="float64")
    run = lambda v: blk(tf.constant(v), nf, num_fragments=scene.num_fragments,  # noqa: E731
                        frag_scene=fs).numpy()
    base = run(x)
    pert = x.copy()
    pert[scene.node_frag == scene.num_fragments - 1] += 3.0
    first = scene.node_frag == 0
    assert np.abs(base[first] - run(pert)[first]).max() > 1e-6


def test_full_model_is_equivariant_per_fragment(scene):
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       embed_dim=4, gram_bottleneck=4, dtype="float64")
    base = model(**to_tensors(scene, tf.float64))
    R = haar_rotations(scene.num_fragments, 11)
    rotated = model(**to_tensors(scene.rotate_per_fragment(R), tf.float64))
    expected = tf.matmul(base["R_pred"], tf.constant(R), transpose_b=True)
    assert float(tf.reduce_max(tf.abs(rotated["R_pred"] - expected))) < TOL
    assert float(tf.reduce_max(tf.abs(
        rotated["vertex_embedding"] - base["vertex_embedding"]))) < TOL


def test_model_outputs_are_proper_rotations(scene):
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       embed_dim=4, gram_bottleneck=4, dtype="float64")
    R = model(**to_tensors(scene, tf.float64))["R_pred"]
    eye = tf.eye(3, batch_shape=[scene.num_fragments], dtype=tf.float64)
    assert float(tf.reduce_max(tf.abs(tf.matmul(R, R, transpose_b=True) - eye))) < 1e-12
    assert float(tf.reduce_max(tf.abs(tf.linalg.det(R) - 1))) < 1e-12


def test_edge_features_are_symmetric_in_endpoints(scene):
    h = tf.constant(np.random.default_rng(12).normal(size=(scene.num_nodes, 5, 3)))
    ei = tf.constant(scene.edge_index, tf.int32)
    ev = tf.constant(scene.edge_vec)
    assert float(tf.reduce_max(tf.abs(
        symmetric_edge_features(h, ei, ev)
        - symmetric_edge_features(h, tf.reverse(ei, [0]), ev)))) == 0.0


def test_symmetrise_edges_layout(scene):
    ei = tf.constant(scene.edge_index, tf.int32)
    el = tf.constant(scene.edge_len)
    ev = tf.constant(scene.edge_vec)
    di, dl, dv = VNGATModel.symmetrise_edges(ei, el, ev)
    e = scene.num_edges
    assert int(di.shape[1]) == 2 * e
    assert np.array_equal(di[:, :e].numpy(), scene.edge_index)      # forward block first
    assert np.array_equal(di[:, e:].numpy(), scene.edge_index[::-1])
    assert np.array_equal(dv[e:, 1].numpy(), scene.edge_vec[:, 2])  # normals swapped
    assert np.array_equal(dv[e:, 2].numpy(), scene.edge_vec[:, 1])


def test_geodesic_loss_matches_known_angles():
    import math
    for deg in (0.0, 30.0, 90.0, 179.0):
        t = math.radians(deg)
        R = tf.constant([[[math.cos(t), -math.sin(t), 0.0],
                          [math.sin(t), math.cos(t), 0.0],
                          [0.0, 0.0, 1.0]]], tf.float64)
        got = float(geodesic_rotation_loss(R, tf.eye(3, batch_shape=[1], dtype=tf.float64))[0])
        # ~3e-5 degree floor from the eps that keeps the gradient finite at
        # theta = 0 and pi -- four orders below anything this task measures.
        assert abs(math.degrees(got) - deg) < 1e-3
