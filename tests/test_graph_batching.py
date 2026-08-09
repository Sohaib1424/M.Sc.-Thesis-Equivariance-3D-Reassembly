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
        "trans": torch.randn(batch.num_fragments, 3),
        "scene_dirs": [f"s{i}" for i in range(batch.num_scenes)],
        "num_scenes": batch.num_scenes,
    }


def test_collate_offsets_are_consistent(batch):
    assert int(batch.edge_index.max()) < batch.num_nodes
    assert int(batch.node_frag.max()) == batch.num_fragments - 1
    assert batch.frag_scene.shape[0] == batch.num_fragments
    assert batch.frag_centroid.shape == (batch.num_fragments, 3)


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
    for part, original in zip(parts, scenes):
        g = part["target"]
        assert torch.equal(g.node_vec, original.node_vec)
        assert torch.equal(g.edge_index, original.edge_index)
        assert torch.equal(g.edge_len, original.edge_len)
        assert torch.equal(g.edge_vec, original.edge_vec)
        assert torch.equal(g.node_frag, original.node_frag)
        assert torch.equal(g.frag_centroid, original.frag_centroid)
        assert g.num_fragments == original.num_fragments
        assert g.num_scenes == 1


def test_rotate_per_fragment_is_a_rigid_motion(scene):
    R = random_rotation(scene.num_fragments)
    rotated = scene.rotate_per_fragment(R)

    manual = torch.einsum('nij,ncj->nci', R[scene.node_frag], scene.node_vec)
    assert torch.allclose(rotated.node_vec, manual, atol=1e-5)
    assert torch.equal(rotated.edge_index, scene.edge_index)
    assert torch.equal(rotated.edge_len, scene.edge_len)

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
