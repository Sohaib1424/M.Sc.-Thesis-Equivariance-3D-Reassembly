"""
Padding must not change any number.

XLA needs static shapes, so every batch is padded into buckets. The risk is the
usual one for this project: a masking mistake does not raise, it just shifts
every loss by a factor of (padded length / real length) and looks like a model
that trains differently. These tests pin the equality.
"""
from __future__ import annotations

import pytest
import torch

from conftest import make_scene, random_rotation
from vngat.data.padding import (
    DEFAULT_EDGE_BUCKETS, DEFAULT_FRAG_BUCKETS, DEFAULT_NODE_BUCKETS,
    choose_bucket, masked_mean, pad_scene_batch,
)
from vngat.losses.composite import CompositeLoss
from vngat.models.vn_gat import VNGATModel
from vngat.training.bridge import build_model_inputs, build_predictions, build_targets

SMALL = dict(node_buckets=(64, 128, 256), edge_buckets=(128, 256, 512),
             frag_buckets=(8, 16, 32))


def test_choose_bucket_rounds_up():
    assert choose_bucket(1, (8, 16, 32)) == 8
    assert choose_bucket(8, (8, 16, 32)) == 8
    assert choose_bucket(9, (8, 16, 32)) == 16
    # beyond the ladder it doubles rather than failing
    assert choose_bucket(100, (8, 16, 32)) == 128


def test_masked_mean_matches_indexing():
    values = torch.randn(50)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:17] = True
    assert abs(float(masked_mean(values, mask)) - float(values[:17].mean())) < 1e-6


def test_masked_mean_needs_no_host_sync():
    """The sum-and-divide form keeps the graph static; `values[mask]` would need
    the mask on the host to size its output, forcing a sync every step."""
    values = torch.randn(20, requires_grad=True)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[:5] = True
    out = masked_mean(values, mask)
    out.backward()
    # gradient only where the mask is set, and equal to 1/n there
    assert torch.allclose(values.grad[:5], torch.full((5,), 0.2), atol=1e-6)
    assert float(values.grad[5:].abs().max()) == 0.0


def test_padding_preserves_the_real_entries():
    scene = make_scene(seed=3)
    padded = pad_scene_batch(scene, **SMALL)
    g, n, e, f = padded.graph, scene.num_nodes, scene.num_edges, scene.num_fragments

    assert g.num_nodes >= n + 1 and g.num_edges >= e and g.num_fragments >= f + 1
    assert torch.equal(g.node_vec[:n], scene.node_vec)
    assert torch.equal(g.edge_index[:, :e], scene.edge_index)
    assert torch.equal(g.edge_vec[:e], scene.edge_vec)
    assert torch.equal(g.node_frag[:n], scene.node_frag)
    assert int(padded.node_mask.sum()) == n
    assert int(padded.edge_mask.sum()) == e
    assert int(padded.frag_mask.sum()) == f


def test_padding_is_isolated_by_construction():
    """
    Padded vertices belong to a dedicated pad fragment, which belongs to a
    dedicated pad scene, and padded edges point only at the pad vertex. Every
    segment op in the model is scoped by fragment or scene, so the isolation
    falls out of the indexing rather than needing masks inside the layers.
    """
    scene = make_scene(seed=4)
    padded = pad_scene_batch(scene, **SMALL)
    g, n, e, f = padded.graph, scene.num_nodes, scene.num_edges, scene.num_fragments

    assert (g.node_frag[n:] == f).all(), "padded vertices must share one pad fragment"
    assert (g.frag_scene[f:] == scene.num_scenes).all(), "pad fragment needs its own scene"
    if g.num_edges > e:
        assert (g.edge_index[:, e:] == n).all(), "padded edges must point at the pad vertex"
    # real edges never touch padding
    assert int(g.edge_index[:, :e].max()) < n
    # -1 keeps padding out of the interface-embedding terms automatically
    assert (g.vertex_cluster_id[n:] == -1).all()
    assert (g.edge_cluster_id[e:] == -1).all()


def _run(model, graph, rot, masks=None):
    out = model(**build_model_inputs(graph.rotate_per_fragment(rot)))
    diffused = graph.rotate_per_fragment(rot)
    merged = dict(R_pred=out["R_pred"], vertex_embedding=out["vertex_embedding"],
                  edge_embedding=out["edge_embedding"],
                  **build_predictions(diffused, out["R_pred"]))
    targets = build_targets(graph, rot, graph)
    if masks:
        targets.update(masks)
    return CompositeLoss()(merged, targets), out


def test_padded_loss_equals_unpadded_loss():
    """
    THE test. Every reported term must be identical whether or not the batch was
    padded -- otherwise TPU and GPU runs are not comparable and neither number
    means anything.
    """
    torch.manual_seed(0)
    scene = make_scene(seed=5)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=4, gram_bottleneck=4).double().eval()
    scene.node_vec = scene.node_vec.double()
    scene.edge_len = scene.edge_len.double()
    scene.edge_vec = scene.edge_vec.double()
    scene.frag_centroid = scene.frag_centroid.double()
    rot = random_rotation(scene.num_fragments, dtype=torch.float64)

    padded = pad_scene_batch(scene, **SMALL)
    pad_rot = torch.cat([
        rot, torch.eye(3, dtype=torch.float64).expand(
            padded.graph.num_fragments - scene.num_fragments, 3, 3)], 0)

    with torch.no_grad():
        plain, _ = _run(model, scene, rot)
        pad, _ = _run(model, padded.graph, pad_rot, masks=dict(
            node_mask=padded.node_mask, edge_mask=padded.edge_mask,
            frag_mask=padded.frag_mask))

    for key in ("rot", "rot_deg", "pos", "node", "mid", "face", "tilt", "twist"):
        a, b = float(plain[key]), float(pad[key])
        assert abs(a - b) < 1e-8, f"{key}: unpadded {a} vs padded {b}"


def test_padded_embedding_terms_match():
    """Cluster ids of -1 exclude padding, so these need no explicit mask."""
    torch.manual_seed(1)
    scene = make_scene(seed=6)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=4, gram_bottleneck=4).double().eval()
    for name in ("node_vec", "edge_len", "edge_vec", "frag_centroid"):
        setattr(scene, name, getattr(scene, name).double())
    rot = random_rotation(scene.num_fragments, dtype=torch.float64)
    padded = pad_scene_batch(scene, **SMALL)
    pad_rot = torch.cat([
        rot, torch.eye(3, dtype=torch.float64).expand(
            padded.graph.num_fragments - scene.num_fragments, 3, 3)], 0)

    with torch.no_grad():
        plain, _ = _run(model, scene, rot)
        pad, _ = _run(model, padded.graph, pad_rot, masks=dict(
            node_mask=padded.node_mask, edge_mask=padded.edge_mask,
            frag_mask=padded.frag_mask))
    for key in ("emb_v", "emb_e"):
        assert abs(float(plain[key]) - float(pad[key])) < 1e-8, key


def test_padding_does_not_change_predicted_rotations():
    """The real fragments' predictions must be untouched by the padding."""
    torch.manual_seed(2)
    scene = make_scene(seed=7)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=4, gram_bottleneck=4).double().eval()
    for name in ("node_vec", "edge_len", "edge_vec", "frag_centroid"):
        setattr(scene, name, getattr(scene, name).double())
    padded = pad_scene_batch(scene, **SMALL)

    with torch.no_grad():
        a = model(**build_model_inputs(scene))["R_pred"]
        b = model(**build_model_inputs(padded.graph))["R_pred"]
    assert torch.allclose(a, b[:scene.num_fragments], atol=1e-9)


@pytest.mark.parametrize("nodes,edges,frags", [(1, 0, 1), (3, 1, 1), (40, 90, 6)])
def test_padding_handles_degenerate_scenes(nodes, edges, frags):
    from vngat.data.graph import FragmentGraph, merge_fragments

    g = merge_fragments([FragmentGraph(
        node_vec=torch.randn(nodes, 2, 3),
        edge_index=torch.randint(0, max(nodes, 1), (2, edges)),
        edge_len=torch.rand(edges, 1), edge_vec=torch.randn(edges, 3, 3),
        centroid=torch.zeros(3),
        vertex_cluster_id=torch.full((nodes,), -1),
        edge_cluster_id=torch.full((edges,), -1))])
    padded = pad_scene_batch(g, **SMALL)
    assert padded.graph.num_nodes > nodes
    assert int(padded.node_mask.sum()) == nodes
