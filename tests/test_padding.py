"""
Padding must not change any number.

XLA needs static shapes, so every batch is padded into buckets. A masking
mistake does not raise -- it silently rescales every loss by
(padded length / real length) and looks like a model that trains differently.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from conftest import haar_rotations, make_scene
from vngat.data.padding import bucket_report, choose_bucket, pad_scene_batch
from vngat.losses.composite import CompositeLoss
from vngat.models.vn_gat import VNGATModel
from vngat.training.bridge import build_predictions, build_targets, to_tensors

SMALL = dict(node_buckets=(64, 128, 256), edge_buckets=(128, 256, 512),
             frag_buckets=(8, 16, 32))


def test_choose_bucket_rounds_up():
    assert choose_bucket(1, (8, 16, 32)) == 8
    assert choose_bucket(8, (8, 16, 32)) == 8
    assert choose_bucket(9, (8, 16, 32)) == 16
    assert choose_bucket(100, (8, 16, 32)) == 128     # doubles past the ladder


def test_padding_preserves_the_real_entries():
    scene = make_scene(seed=3)
    p = pad_scene_batch(scene, **SMALL)
    g, n, e, f = p.graph, scene.num_nodes, scene.num_edges, scene.num_fragments
    assert np.array_equal(g.node_vec[:n], scene.node_vec)
    assert np.array_equal(g.edge_index[:, :e], scene.edge_index)
    assert (int(p.node_mask.sum()), int(p.edge_mask.sum()), int(p.frag_mask.sum())) == (n, e, f)


def test_padding_is_isolated_by_construction():
    """
    Padded vertices get their own fragment, that fragment its own scene, and
    padded edges point only at the pad vertex. Every segment op is scoped by
    fragment or scene, so isolation falls out of the indexing -- no masks are
    needed inside the layers.
    """
    scene = make_scene(seed=4)
    p = pad_scene_batch(scene, **SMALL)
    g, n, e, f = p.graph, scene.num_nodes, scene.num_edges, scene.num_fragments
    assert (g.node_frag[n:] == f).all()
    assert (g.frag_scene[f:] == scene.num_scenes).all()
    assert (g.edge_index[:, e:] == n).all()
    assert g.edge_index[:, :e].max() < n
    assert (g.vertex_cluster_id[n:] == -1).all()
    assert (g.edge_cluster_id[e:] == -1).all()


def _run(model, graph, rot, masks=None):
    diffused = graph.rotate_per_fragment(rot)
    out = model(**to_tensors(diffused, tf.float64))
    targets = build_targets(graph, tf.constant(rot), graph, dtype=tf.float64)
    if masks:
        targets.update(masks)
    losses = CompositeLoss()(
        dict(R_pred=out["R_pred"], vertex_embedding=out["vertex_embedding"],
             edge_embedding=out["edge_embedding"],
             **build_predictions(diffused, out["R_pred"], dtype=tf.float64)), targets)
    return losses, out


def test_padded_loss_equals_unpadded_loss():
    """
    THE test. Every reported term must be identical whether or not the batch was
    padded -- otherwise a TPU number and a GPU number are not comparable and
    neither means anything.
    """
    scene = make_scene(seed=5)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       embed_dim=4, gram_bottleneck=4, dtype="float64")
    A = haar_rotations(scene.num_fragments, 9)
    p = pad_scene_batch(scene, **SMALL)
    pad_A = np.concatenate(
        [A, np.tile(np.eye(3), (p.graph.num_fragments - scene.num_fragments, 1, 1))], 0)

    plain, out_a = _run(model, scene, A)
    padded, out_b = _run(model, p.graph, pad_A, masks=dict(
        node_mask=tf.constant(p.node_mask), edge_mask=tf.constant(p.edge_mask),
        frag_mask=tf.constant(p.frag_mask)))

    for key in ("rot", "rot_deg", "pos", "node", "mid", "face",
                "emb_v", "emb_e", "tilt", "twist"):
        a, b = float(plain[key]), float(padded[key])
        assert abs(a - b) < 1e-8, f"{key}: unpadded {a} vs padded {b}"
    assert float(tf.reduce_max(tf.abs(
        out_a["R_pred"] - out_b["R_pred"][:scene.num_fragments]))) < 1e-10


@pytest.mark.parametrize("nodes,edges", [(1, 0), (3, 1), (40, 90)])
def test_padding_handles_degenerate_scenes(nodes, edges):
    from vngat.data.graph import FragmentGraph, merge_fragments

    rng = np.random.default_rng(0)
    g = merge_fragments([FragmentGraph(
        node_vec=rng.normal(size=(nodes, 2, 3)),
        edge_index=rng.integers(0, max(nodes, 1), (2, edges)).astype(np.int64),
        edge_len=rng.random((edges, 1)), edge_vec=rng.normal(size=(edges, 3, 3)),
        centroid=np.zeros(3), vertex_cluster_id=np.full(nodes, -1),
        edge_cluster_id=np.full(edges, -1))])
    p = pad_scene_batch(g, **SMALL)
    assert p.graph.num_nodes > nodes
    assert int(p.node_mask.sum()) == nodes


def test_bucket_report_counts_compilations():
    rng = np.random.default_rng(1)
    sizes = [(int(rng.integers(4000, 90000)), int(rng.integers(12000, 270000)),
              int(rng.integers(2, 94))) for _ in range(100)]
    text = bucket_report(sizes)
    assert "distinct shape combinations" in text and "padding factor" in text
