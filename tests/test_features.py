"""
Edge features on real meshes: the canonical face-normal order, the relative
position channel, and what rotation does to both.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh

from conftest import random_rotation
from vngat.data.features import get_features
from vngat.data.graph import merge_fragments


def _fragment(seed: int = 0) -> trimesh.Trimesh:
    """A closed, irregular mesh: an icosphere with jittered vertices."""
    rng = np.random.default_rng(seed)
    sphere = trimesh.creation.icosphere(subdivisions=2)
    vertices = sphere.vertices * (1.0 + 0.15 * rng.standard_normal((len(sphere.vertices), 1)))
    return trimesh.Trimesh(vertices + rng.standard_normal(3), sphere.faces, process=False)


def _open_fragment() -> trimesh.Trimesh:
    """Half an icosphere, so it has boundary edges with a single face."""
    sphere = _fragment(3)
    keep = sphere.triangles_center[:, 2] > sphere.centroid[2]
    return trimesh.Trimesh(sphere.vertices, sphere.faces[keep], process=True)


def _pairs(graph):
    """(u, v) -> (n1, n2) for every stored edge."""
    out = {}
    ei = graph.edge_index.t().tolist()
    for k, (u, v) in enumerate(ei):
        out[(u, v)] = (graph.edge_attr[k, 0], graph.edge_attr[k, 1])
    return out


def test_stored_edges_run_low_to_high_and_carry_source_minus_destination():
    g = get_features(_fragment())
    assert bool((g.edge_index[0] < g.edge_index[1]).all())
    p = g.node_vec[:, 0]
    assert torch.allclose(g.edge_attr[:, 2], p[g.edge_index[0]] - p[g.edge_index[1]], atol=1e-6)


def test_normal_order_follows_the_triple_product_rule():
    """(n1 x n2) . (x_v - x_u) >= 0 for every interior edge."""
    g = get_features(_fragment(1))
    n1, n2 = g.edge_attr[:, 0].double(), g.edge_attr[:, 1].double()
    d = -g.edge_attr[:, 2].double()                          # x_v - x_u
    s = (torch.cross(n1, n2, dim=-1) * d).sum(-1)
    assert float(s.min()) > -1e-6


def test_normal_order_does_not_depend_on_face_order():
    """
    The previous builder filled the slots in FACE-ARRAY order, which the
    decompression step reorders lexicographically: measured through the real
    decompression code it agreed with the geometric rule on 50.1% of edges -- a
    coin flip. Shuffling the faces must leave every edge's (n1, n2) exactly
    where it was.
    """
    mesh = _fragment(2)
    rng = np.random.default_rng(0)
    shuffled = trimesh.Trimesh(mesh.vertices, mesh.faces[rng.permutation(len(mesh.faces))],
                               process=False)
    a, b = _pairs(get_features(mesh)), _pairs(get_features(shuffled))
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key][0], b[key][0], atol=1e-6)
        assert torch.allclose(a[key][1], b[key][1], atol=1e-6)


def test_rotating_the_mesh_rotates_the_slots_without_swapping_them():
    """
    Recomputing features on a rotated mesh must equal rotating the features --
    which is what lets the training loop derive the diffused view by rotation
    alone (`rotate_per_fragment`) instead of re-running feature extraction.
    """
    mesh = _fragment(4)
    R = random_rotation(1, dtype=torch.float64)[0].numpy()
    rotated = mesh.copy()
    rotated.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    a, b = get_features(mesh), get_features(rotated)
    assert torch.equal(a.edge_index, b.edge_index)
    Rt = torch.as_tensor(R, dtype=torch.float32)
    assert torch.allclose(b.edge_attr, a.edge_attr @ Rt.T, atol=1e-5)
    assert torch.allclose(b.node_vec, a.node_vec @ Rt.T, atol=1e-5)
    assert abs(a.radius - b.radius) < 1e-9


def test_boundary_edges_repeat_their_single_normal():
    g = get_features(_open_fragment())
    mesh = _open_fragment()
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary = torch.as_tensor(counts == 1)
    assert bool(boundary.any())
    assert torch.equal(g.edge_attr[boundary, 0], g.edge_attr[boundary, 1])


def test_reverse_copy_is_what_the_rule_gives_for_the_reversed_edge():
    """The model builds v -> u as (n2, n1, -delta); that must be exactly what
    the canonical rule would have produced had the edge been stored v -> u."""
    from vngat.models.vn_gat import VNGATModel

    g = get_features(_fragment(5))
    ei, ea = VNGATModel.symmetrise_edges(g.edge_index, g.edge_attr)
    E = g.num_edges
    n1, n2, delta = ea[E:, 0].double(), ea[E:, 1].double(), ea[E:, 2].double()
    s = (torch.cross(n1, n2, dim=-1) * (-delta)).sum(-1)     # d = x_dst - x_src
    assert float(s.min()) > -1e-6


def test_positions_stay_in_world_units_until_the_scene_is_merged():
    mesh = _fragment(6)
    g = get_features(mesh)
    centred = np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(0)
    assert np.allclose(g.node_vec[:, 0].numpy(), centred, atol=1e-5)
    assert abs(g.radius - np.linalg.norm(centred, axis=1).max()) < 1e-6
    scene = merge_fragments([g, get_features(_fragment(7))])
    assert torch.allclose(scene.world_pos()[: g.num_nodes], g.node_vec[:, 0], atol=1e-5)
    # normals are never scaled
    assert torch.allclose(scene.node_vec[: g.num_nodes, 1], g.node_vec[:, 1])
    assert torch.allclose(scene.edge_attr[: g.num_edges, :2], g.edge_attr[:, :2])


def test_degenerate_triangles_do_not_reach_the_tensors():
    mesh = _fragment(8)
    faces = np.vstack([mesh.faces, [[0, 0, 1]]])             # a zero-area face
    broken = trimesh.Trimesh(mesh.vertices, faces, process=False)
    g = get_features(broken)
    assert torch.isfinite(g.node_vec).all() and torch.isfinite(g.edge_attr).all()
