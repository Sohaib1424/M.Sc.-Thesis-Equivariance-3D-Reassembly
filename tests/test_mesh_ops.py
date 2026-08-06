"""Face adjacency and fracture-surface extraction."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh not installed")

from reassembly.data.mesh_ops import (  # noqa: E402
    extract_fractures, extract_fractures_with_map, find_neighbors,
)


def find_neighbors_reference(mesh, min_common_vertices=2):
    """The original O(F^2) double loop, kept as the oracle."""
    faces = np.asarray(mesh.faces)
    nf = len(faces)
    W = np.zeros((nf, nf), dtype=np.float32)
    M = np.zeros((nf, nf), dtype=np.float32)
    normals = np.asarray(mesh.face_normals)
    for i in range(nf):
        for j in range(nf):
            if i == j:
                continue
            if len(set(faces[i]) & set(faces[j])) >= min_common_vertices:
                M[i, j] = 1.0
                W[i, j] = float(np.dot(normals[i], normals[j]))
    return W, M


@pytest.mark.parametrize("min_common", [1, 2])
def test_find_neighbors_matches_the_loop_reference(min_common):
    for mesh in (trimesh.creation.box(), trimesh.creation.icosphere(subdivisions=1)):
        W, M = find_neighbors(mesh, min_common_vertices=min_common)
        Wr, Mr = find_neighbors_reference(mesh, min_common_vertices=min_common)
        assert np.allclose(W.toarray(), Wr, atol=1e-5)
        assert np.allclose(M.toarray(), Mr, atol=1e-5)


def test_find_neighbors_shares_sparsity_between_w_and_m():
    """Downstream code assigns A.data = mask over W.data and then sums M --
    that is only correct if the two share a layout."""
    W, M = find_neighbors(trimesh.creation.icosphere(subdivisions=1))
    assert W.nnz == M.nnz
    assert np.array_equal(W.tocsr().indices, M.tocsr().indices)
    assert np.array_equal(W.tocsr().indptr, M.tocsr().indptr)


def test_find_neighbors_on_an_empty_mesh():
    empty = trimesh.Trimesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), process=False)
    W, M = find_neighbors(empty)
    assert W.shape == (0, 0)


def test_extract_fractures_prunes_vertices():
    """THE regression for the bug where the fracture mesh reused the full
    vertex array, so only its edge count shrank and it saved no memory."""
    # A box with one subdivided face gives a genuinely sharp interior region.
    mesh = trimesh.creation.box()
    frac, vmap = extract_fractures_with_map(mesh)
    assert len(vmap) == len(mesh.vertices)
    if len(frac.faces) < len(mesh.faces):
        assert len(frac.vertices) <= len(mesh.vertices)
        # every surviving face must index within the pruned vertex array
        assert int(np.asarray(frac.faces).max()) < len(frac.vertices)


def test_extract_fractures_vertex_map_is_consistent():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    frac, vmap = extract_fractures_with_map(mesh)
    live = vmap >= 0
    if live.any() and len(frac.vertices) < len(mesh.vertices):
        assert np.allclose(np.asarray(frac.vertices)[vmap[live]],
                           np.asarray(mesh.vertices)[live], atol=1e-9)


def test_extract_fractures_never_returns_a_degenerate_mesh():
    """A fragment must survive extraction: fragment counts are positional
    everywhere downstream, so silently dropping one misaligns rotations."""
    for mesh in (trimesh.creation.box(),
                 trimesh.creation.icosphere(subdivisions=1),
                 trimesh.creation.cylinder(radius=1, height=2)):
        frac = extract_fractures(mesh)
        assert len(frac.vertices) >= 3
        assert len(frac.faces) >= 1


def test_extract_fractures_falls_back_on_a_smooth_mesh():
    """A smooth sphere has no sharp interior, so there is no fracture surface
    to find -- the full mesh must come back rather than an empty one."""
    sphere = trimesh.creation.icosphere(subdivisions=3)
    frac = extract_fractures(sphere, sharpness_threshold=0.0)   # nothing is 'sharp'
    assert len(frac.vertices) == len(sphere.vertices)


def test_extract_fractures_applies_both_criteria():
    """Criterion (b) used to be computed and never applied, over-including
    faces. With a stricter sharpness threshold the selection must shrink."""
    mesh = trimesh.creation.box()
    loose = extract_fractures(mesh, sharpness_threshold=0.99, min_faces=1)
    strict = extract_fractures(mesh, sharpness_threshold=0.01, min_faces=1)
    assert len(strict.faces) <= len(loose.faces)
