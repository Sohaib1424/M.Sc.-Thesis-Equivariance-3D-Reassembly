"""
Topology, repair and the array primitives.

The theme is *ordering contracts*. A set-level bug here is loud; an
ordering bug is silent, because every array still has the right length and
every value is still plausible. Positional alignment between edges, faces and
their attributes is what actually breaks.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.arrays import (
    compact_indices,
    dense_group_ids,
    first_index_per_group,
    pair_key,
    pair_unkey,
    triple_key,
    unique_sorted,
)
from reassembly.mesh.repair import resolve_duplicated_faces
from reassembly.mesh.topology import (
    count_unique_edges,
    edge_topology,
    face_adjacency,
    face_normals,
    unique_edges,
    vertex_normals,
)


# ---------------------------------------------------------------- arrays --

def test_unique_sorted_matches_numpy():
    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 50, 5000):
        keys = rng.integers(0, 20, n).astype(np.int64)
        assert np.array_equal(unique_sorted(keys), np.unique(keys))


def test_pair_key_round_trip():
    rng = np.random.default_rng(1)
    n = 5000
    a = rng.integers(0, n, 1000).astype(np.int64)
    b = rng.integers(0, n, 1000).astype(np.int64)
    ra, rb = pair_unkey(pair_key(a, b, n), n)
    assert np.array_equal(a, ra) and np.array_equal(b, rb)


def test_triple_key_is_injective_and_lexicographic():
    rows = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.int64)
    keys = triple_key(rows, 4)
    assert len(np.unique(keys)) == len(rows)
    assert np.array_equal(keys, np.sort(keys)), "keys must order lexicographically"


def test_dense_group_ids_matches_numpy_inverse():
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 30, 500).astype(np.int64)
    gid, n_groups = dense_group_ids(keys)
    expected = np.unique(keys, return_inverse=True)[1].ravel()
    assert n_groups == len(np.unique(keys))
    assert np.array_equal(gid, expected)


def test_first_index_per_group_takes_the_lowest():
    gid = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    assert first_index_per_group(gid, 3).tolist() == [0, 1, 4]
    mask = np.array([False, True, True, True, True])
    assert first_index_per_group(gid, 3, mask).tolist()[:2] == [2, 1]


def test_compact_indices_preserves_triangles():
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    faces = np.array([[0, 2, 4]])
    kept, new_faces = compact_indices(faces, 5)
    assert kept.tolist() == [0, 2, 4]
    assert np.allclose(vertices[faces], vertices[kept][new_faces])


# ----------------------------------------------------------------- edges --

def test_unique_edges_matches_trimesh_as_a_set(simple_meshes):
    for name, mesh in simple_meshes.items():
        mine = unique_edges(np.asarray(mesh.faces), len(mesh.vertices))
        theirs = np.sort(np.asarray(mesh.edges_unique), axis=1)
        assert set(map(tuple, mine.tolist())) == set(map(tuple, theirs.tolist())), name
        assert count_unique_edges(np.asarray(mesh.faces), len(mesh.vertices)) == len(mine)


def test_unique_edges_is_lexicographically_ordered(simple_meshes):
    """The canonical ordering contract. Do not relax this."""
    for name, mesh in simple_meshes.items():
        edges = unique_edges(np.asarray(mesh.faces), len(mesh.vertices))
        assert (edges[:, 0] <= edges[:, 1]).all(), name
        keys = pair_key(edges[:, 0], edges[:, 1], len(mesh.vertices))
        assert np.array_equal(keys, np.sort(keys)), name


def test_edge_topology_agrees_with_unique_edges(simple_meshes):
    """One ordering, two entry points -- they must not drift apart."""
    for name, mesh in simple_meshes.items():
        faces = np.asarray(mesh.faces)
        topo = edge_topology(faces, len(mesh.vertices))
        assert np.array_equal(topo.edges, unique_edges(faces, len(mesh.vertices))), name


def test_edge_topology_incidence_is_consistent(simple_meshes):
    """Each half-edge maps to an edge that its own face is recorded as touching."""
    for name, mesh in simple_meshes.items():
        faces = np.asarray(mesh.faces)
        topo = edge_topology(faces, len(mesh.vertices))
        owns = (topo.faces_per_edge[topo.edge_of_half] == topo.face_of_half[:, None])
        assert owns.any(axis=1).all(), name


def test_closed_mesh_has_two_faces_per_edge(simple_meshes):
    topo = edge_topology(np.asarray(simple_meshes["icosphere"].faces))
    assert (topo.faces_per_edge >= 0).all()


def test_open_patch_has_boundary_edges(open_patch):
    topo = edge_topology(np.asarray(open_patch.faces), len(open_patch.vertices))
    assert (topo.faces_per_edge[:, 1] == -1).sum() > 0


# ------------------------------------------------------------- adjacency --

def test_face_adjacency_matches_shared_vertex_definition(simple_meshes):
    """
    The claim the fast path rests on: for triangles, "shares >= 2 vertices" and
    "shares an edge" are the same relation. Checked against the brute-force
    shared-vertex count.
    """
    for name, mesh in simple_meshes.items():
        faces = np.asarray(mesh.faces)
        if len(faces) > 600:
            continue
        src, dst = face_adjacency(faces, len(mesh.vertices))
        fast = set(zip(src.tolist(), dst.tolist()))
        brute = {
            (i, j)
            for i in range(len(faces))
            for j in range(len(faces))
            if i != j and len(set(faces[i]) & set(faces[j])) >= 2
        }
        assert fast == brute, name


def test_face_adjacency_is_symmetric(simple_meshes):
    src, dst = face_adjacency(np.asarray(simple_meshes["torus"].faces))
    assert set(zip(src.tolist(), dst.tolist())) == set(zip(dst.tolist(), src.tolist()))


def test_duplicate_faces_are_not_double_counted():
    """Duplicate faces share three edges and would otherwise appear three times."""
    faces = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 3]])
    src, dst = face_adjacency(faces, 4, has_duplicate_faces=True)
    pairs = list(zip(src.tolist(), dst.tolist()))
    assert len(pairs) == len(set(pairs))


def test_face_adjacency_empty():
    src, dst = face_adjacency(np.zeros((0, 3), dtype=np.int64), 0)
    assert len(src) == 0 and len(dst) == 0


# ----------------------------------------------------------------- normals --

def test_face_normals_match_trimesh(simple_meshes):
    for name, mesh in simple_meshes.items():
        mine = face_normals(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        assert np.allclose(mine, np.asarray(mesh.face_normals), atol=1e-9), name


def test_face_normals_are_unit_length(simple_meshes):
    mesh = simple_meshes["torus"]
    mine = face_normals(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    assert np.allclose(np.linalg.norm(mine, axis=1), 1.0)


def test_normals_never_nan_on_degenerate_input():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    faces = np.array([[0, 1, 2]])
    normals, bad = face_normals(vertices, faces, return_degenerate=True)
    assert bad == 1
    assert np.isfinite(normals).all()
    assert np.allclose(normals, 0.0)


def test_vertex_normals_are_finite_and_unit(simple_meshes):
    mesh = simple_meshes["icosphere"]
    normals = vertex_normals(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    assert np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


# ------------------------------------------------------------------ repair --

def _cases():
    rng = np.random.default_rng(0)
    faces = np.asarray(trimesh.creation.icosphere(subdivisions=2).faces).astype(np.int64)
    return {
        "clean": faces,
        "exact duplicates": np.vstack([faces, faces[:10]]),
        "flipped duplicates": np.vstack([faces, faces[:10][:, ::-1]]),
        "triple duplicates": np.vstack([faces, faces[:5], faces[:5], faces[:5][:, ::-1]]),
        "degenerate rows": np.vstack([faces, [[1, 1, 2], [3, 3, 3]]]),
        "random": rng.integers(0, 12, (200, 3)).astype(np.int64),
        "empty": np.zeros((0, 3), dtype=np.int64),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_resolve_returns_a_subset_in_group_order(name):
    faces = _cases()[name]
    kept_faces, kept = resolve_duplicated_faces(faces.copy())
    assert len(kept) <= len(faces)
    assert np.array_equal(kept_faces, faces[kept])
    if len(kept) > 1:
        # ordering contract: lexicographic by sorted vertex triple
        canonical = np.sort(faces[kept], axis=1)
        keys = triple_key(canonical, int(faces.max()) + 1)
        assert np.array_equal(keys, np.sort(keys)), name


def test_resolve_leaves_a_clean_mesh_untouched():
    faces = np.asarray(trimesh.creation.icosphere(subdivisions=2).faces).astype(np.int64)
    kept_faces, kept = resolve_duplicated_faces(faces.copy())
    assert len(kept) == len(faces)
    assert set(kept.tolist()) == set(range(len(faces)))


def test_resolve_cancels_a_face_against_its_own_flip():
    faces = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    kept_faces, kept = resolve_duplicated_faces(faces)
    assert len(kept) == 0, "a face and its reverse should cancel"


def test_resolve_keeps_the_majority_winding():
    faces = np.array([[0, 1, 2], [0, 1, 2], [2, 1, 0]], dtype=np.int64)
    kept_faces, kept = resolve_duplicated_faces(faces)
    assert len(kept) == 1
    assert kept_faces[0].tolist() == [0, 1, 2]
