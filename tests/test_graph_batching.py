"""Batching, splitting and rotation of scene graphs."""
from __future__ import annotations

import torch

from conftest import make_scene, random_rotation
from vngat.data.dataset import split_batch_by_scene
from vngat.data.graph import collate_scenes


def _dummy_batch(batch):
    return {
        "target": batch, "input": None,
        "rot": random_rotation(batch.num_fragments),
        "scene_dirs": [f"s{i}" for i in range(batch.num_scenes)],
        "categories": [f"c{i}" for i in range(batch.num_scenes)],
        "num_scenes": batch.num_scenes,
    }


def test_collate_offsets_are_consistent(batch):
    assert int(batch.edge_index.max()) < batch.num_nodes
    assert int(batch.node_frag.max()) == batch.num_fragments - 1
    assert batch.frag_scene.shape[0] == batch.num_fragments
    assert batch.frag_centroid.shape == (batch.num_fragments, 3)
    assert batch.frag_unit.shape == (batch.num_fragments,)
    assert batch.frag_log_scale.shape == (batch.num_fragments, 1)


def test_no_edge_crosses_a_fragment_or_scene(batch):
    frag_src = batch.node_frag[batch.edge_index[0]]
    frag_dst = batch.node_frag[batch.edge_index[1]]
    assert torch.equal(frag_src, frag_dst)
    assert torch.equal(batch.frag_scene[frag_src], batch.frag_scene[frag_dst])
    assert torch.equal(batch.edge_frag, frag_src)


def test_cluster_id_spaces_are_disjoint_across_scenes(batch):
    scene_of_node = batch.frag_scene[batch.node_frag]
    seen = []
    for s in range(batch.num_scenes):
        ids = batch.vertex_cluster_id[(scene_of_node == s) & (batch.vertex_cluster_id >= 0)]
        seen.append(set(ids.tolist()))
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            assert not (a & b), "cluster ids leaked between scenes"


def test_unshared_marker_survives_collation():
    scenes = [make_scene(seed=i) for i in range(3)]
    before = sum(int((s.vertex_cluster_id == -1).sum()) for s in scenes)
    batch = collate_scenes(scenes)
    assert int((batch.vertex_cluster_id == -1).sum()) == before


def test_split_round_trips_exactly(batch):
    scenes = [make_scene(((11, 23), (7, 15), (19, 31)), seed=1),
              make_scene(((5, 9), (13, 27)), seed=2),
              make_scene(((9, 17), (6, 11), (8, 14), (12, 22)), seed=3)]
    combined = collate_scenes(scenes)
    parts = split_batch_by_scene(_dummy_batch(combined))
    assert len(parts) == len(scenes)
    assert [p["categories"] for p in parts] == [["c0"], ["c1"], ["c2"]]
    for part, original in zip(parts, scenes):
        g = part["target"]
        assert torch.equal(g.node_vec, original.node_vec)
        assert torch.equal(g.edge_index, original.edge_index)
        assert torch.equal(g.edge_attr, original.edge_attr)
        assert torch.equal(g.node_frag, original.node_frag)
        assert torch.equal(g.frag_centroid, original.frag_centroid)
        assert torch.equal(g.frag_unit, original.frag_unit)
        assert torch.equal(g.frag_log_scale, original.frag_log_scale)
        assert g.num_fragments == original.num_fragments
        assert g.num_scenes == 1


def test_rotate_per_fragment_is_a_rigid_motion(scene):
    R = random_rotation(scene.num_fragments)
    rotated = scene.rotate_per_fragment(R)

    manual = torch.einsum('nij,ncj->nci', R[scene.node_frag], scene.node_vec)
    assert torch.allclose(rotated.node_vec, manual, atol=1e-5)
    assert torch.equal(rotated.edge_index, scene.edge_index)
    manual_edges = torch.einsum('eij,ecj->eci', R[scene.edge_frag], scene.edge_attr)
    assert torch.allclose(rotated.edge_attr, manual_edges, atol=1e-5)
    assert torch.equal(rotated.frag_unit, scene.frag_unit)
    assert torch.equal(rotated.frag_log_scale, scene.frag_log_scale)

    def lengths(g):
        p = g.node_vec[:, 0]
        return (p[g.edge_index[0]] - p[g.edge_index[1]]).norm(dim=-1)

    assert torch.allclose(lengths(scene), lengths(rotated), atol=1e-4)
    n_before = scene.node_vec[:, 1].norm(dim=-1)
    n_after = rotated.node_vec[:, 1].norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-5)


def test_single_scene_collation_is_a_passthrough():
    scene = make_scene(seed=9)
    assert collate_scenes([scene]) is scene


def test_relative_position_channel_is_source_minus_destination(scene):
    p = scene.node_vec[:, 0]
    delta = scene.edge_attr[:, 2]
    assert torch.allclose(delta, p[scene.edge_index[0]] - p[scene.edge_index[1]], atol=1e-6)


def test_scene_normalisation_puts_the_largest_fragment_on_the_unit_sphere():
    scene = make_scene(((10, 21), (7, 15), (13, 29)), seed=4)
    radius = torch.stack([scene.node_vec[scene.node_frag == f, 0].norm(dim=-1).max()
                          for f in range(scene.num_fragments)])
    assert abs(float(radius.max()) - 1.0) < 1e-5
    assert torch.allclose(scene.frag_unit, scene.frag_unit[:1].expand_as(scene.frag_unit))
    # world units are recoverable, and the scale feature is the WORLD radius
    world = scene.world_pos()
    world_radius = torch.stack([world[scene.node_frag == f].norm(dim=-1).max()
                                for f in range(scene.num_fragments)])
    assert torch.allclose(scene.frag_log_scale.squeeze(-1), world_radius.log(), atol=1e-5)


def test_fragment_normalisation_puts_every_fragment_on_the_unit_sphere():
    scene = make_scene(((10, 21), (7, 15), (13, 29)), seed=4, normalize_mode="fragment")
    for f in range(scene.num_fragments):
        r = scene.node_vec[scene.node_frag == f, 0].norm(dim=-1).max()
        assert abs(float(r) - 1.0) < 1e-5


def test_no_normalisation_keeps_world_units():
    a = make_scene(seed=5, normalize_mode="none")
    b = make_scene(seed=5, normalize_mode="scene")
    assert torch.equal(a.frag_unit, torch.ones_like(a.frag_unit))
    assert torch.allclose(a.node_vec[:, 0], b.world_pos(), atol=1e-5)
    assert torch.equal(a.frag_log_scale, b.frag_log_scale)


def test_split_scenes_keeps_tensors_on_their_device(batch):
    from vngat.data.graph import split_scenes
    for part in split_scenes(batch):
        assert part.frag_scene.device == batch.node_frag.device
        assert int(part.frag_scene.max()) == 0
