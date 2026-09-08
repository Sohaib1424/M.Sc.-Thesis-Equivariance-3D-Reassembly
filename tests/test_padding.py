"""
Padding must not change any number.

XLA needs static shapes, so every batch is padded into buckets. The risk is the
usual one for this project: a masking mistake does not raise, it silently
rescales every loss by (padded length / real length) and looks like a model
that trains differently. These tests pin the equality, so a TPU number and a
GPU number remain comparable.
"""
from __future__ import annotations

import pytest
import torch

from conftest import make_scene, random_rotation
from vngat.data.padding import choose_bucket, masked_mean, pad_scene_batch
from vngat.losses.composite import CompositeLoss
from vngat.models.vn_gat import VNGATModel
from vngat.training.bridge import build_model_inputs, build_predictions, build_targets

SMALL = dict(node_buckets=(64, 128, 256), edge_buckets=(128, 256, 512),
             frag_buckets=(8, 16, 32))


def test_choose_bucket_rounds_up():
    assert choose_bucket(1, (8, 16, 32)) == 8
    assert choose_bucket(8, (8, 16, 32)) == 8
    assert choose_bucket(9, (8, 16, 32)) == 16
    assert choose_bucket(100, (8, 16, 32)) == 128     # doubles past the ladder


def test_masked_mean_matches_indexing():
    values = torch.randn(50)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:17] = True
    assert abs(float(masked_mean(values, mask)) - float(values[:17].mean())) < 1e-6


def test_masked_mean_gradient_reaches_only_real_entries():
    values = torch.randn(20, requires_grad=True)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[:5] = True
    masked_mean(values, mask).backward()
    assert torch.allclose(values.grad[:5], torch.full((5,), 0.2), atol=1e-6)
    assert float(values.grad[5:].abs().max()) == 0.0


def test_padding_preserves_the_real_entries():
    scene = make_scene(seed=3)
    p = pad_scene_batch(scene, **SMALL)
    g, n, e, f = p.graph, scene.num_nodes, scene.num_edges, scene.num_fragments
    assert torch.equal(g.node_vec[:n], scene.node_vec)
    assert torch.equal(g.edge_index[:, :e], scene.edge_index)
    assert torch.equal(g.edge_vec[:e], scene.edge_vec)
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
    if g.num_edges > e:
        assert (g.edge_index[:, e:] == n).all()
    assert int(g.edge_index[:, :e].max()) < n
    assert (g.vertex_cluster_id[n:] == -1).all()
    assert (g.edge_cluster_id[e:] == -1).all()


def _double(scene):
    for name in ("node_vec", "edge_len", "edge_vec", "frag_centroid"):
        setattr(scene, name, getattr(scene, name).double())
    return scene


def _run(model, graph, rot, masks=None):
    diffused = graph.rotate_per_fragment(rot)
    out = model(**build_model_inputs(diffused))
    merged = dict(R_pred=out["R_pred"], vertex_embedding=out["vertex_embedding"],
                  edge_embedding=out["edge_embedding"],
                  **build_predictions(diffused, out["R_pred"]))
    targets = build_targets(graph, rot, graph)
    if masks:
        targets.update(masks)
    return CompositeLoss()(merged, targets)


def test_padded_loss_equals_unpadded_loss():
    """
    THE test. Every reported term must be identical whether or not the batch
    was padded -- otherwise TPU and GPU runs are not comparable and neither
    number means anything.
    """
    torch.manual_seed(0)
    scene = _double(make_scene(seed=5))
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=4, gram_bottleneck=4).double().eval()
    rot = random_rotation(scene.num_fragments, dtype=torch.float64)
    p = pad_scene_batch(scene, **SMALL)
    pad_rot = torch.cat([rot, torch.eye(3, dtype=torch.float64).expand(
        p.graph.num_fragments - scene.num_fragments, 3, 3)], 0)

    with torch.no_grad():
        plain = _run(model, scene, rot)
        padded = _run(model, p.graph, pad_rot, masks=dict(
            node_mask=p.node_mask, edge_mask=p.edge_mask, frag_mask=p.frag_mask))

    for key in ("rot", "rot_deg", "pos", "node", "mid", "face",
                "emb_v", "emb_e", "tilt", "twist"):
        a, b = float(plain[key]), float(padded[key])
        assert abs(a - b) < 1e-8, f"{key}: unpadded {a} vs padded {b}"


def test_padding_does_not_change_predicted_rotations():
    torch.manual_seed(2)
    scene = _double(make_scene(seed=7))
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=4, gram_bottleneck=4).double().eval()
    p = pad_scene_batch(scene, **SMALL)
    with torch.no_grad():
        a = model(**build_model_inputs(scene))["R_pred"]
        b = model(**build_model_inputs(p.graph))["R_pred"]
    assert torch.allclose(a, b[:scene.num_fragments], atol=1e-9)


@pytest.mark.parametrize("nodes,edges", [(1, 0), (3, 1), (40, 90)])
def test_padding_handles_degenerate_scenes(nodes, edges):
    from vngat.data.graph import FragmentGraph, merge_fragments

    g = merge_fragments([FragmentGraph(
        node_vec=torch.randn(nodes, 2, 3),
        edge_index=torch.randint(0, max(nodes, 1), (2, edges)),
        edge_len=torch.rand(edges, 1), edge_vec=torch.randn(edges, 3, 3),
        centroid=torch.zeros(3),
        vertex_cluster_id=torch.full((nodes,), -1),
        edge_cluster_id=torch.full((edges,), -1))])
    p = pad_scene_batch(g, **SMALL)
    assert p.graph.num_nodes > nodes
    assert int(p.node_mask.sum()) == nodes
