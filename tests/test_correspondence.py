"""Cross-fragment correspondence detection and label transfer."""
import numpy as np
import pytest

from reassembly.data.correspondence import (
    UnionFind, find_shared_points, transfer_vertex_clusters,
)


def test_exact_match_finds_touching_points():
    shared = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    a = np.concatenate([shared, np.array([[5.0, 5, 5]])])
    b = np.concatenate([shared, np.array([[-5.0, -5, -5]])])
    clusters = find_shared_points([a, b])
    assert len(clusters) == 3
    for c in clusters:
        assert len({f for f, _ in c}) == 2


def test_sub_tolerance_noise_still_matches():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(20, 3))
    a = shared
    b = shared + rng.normal(0, 1e-9, shared.shape)
    assert len(find_shared_points([a, b], tol=1e-5)) == 20


def test_nearest_neighbour_fallback_handles_larger_noise():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(15, 3)) * 3.0
    b = shared + rng.normal(0, 1e-4, shared.shape)
    # too far apart to round-match at 6 decimals, close enough for the KD-tree
    assert len(find_shared_points([shared, b], tol=1e-3, exact_decimals=6)) == 15


def test_far_apart_fragments_share_nothing():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(30, 3))
    b = rng.normal(size=(30, 3)) + 100.0
    assert find_shared_points([a, b], tol=1e-5) == []


def test_three_way_shared_point_forms_one_cluster():
    p = np.array([[1.0, 2.0, 3.0]])
    clusters = find_shared_points([p, p, p])
    assert len(clusters) == 1
    assert {f for f, _ in clusters[0]} == {0, 1, 2}


def test_single_fragment_yields_no_clusters():
    rng = np.random.default_rng(3)
    assert find_shared_points([rng.normal(size=(10, 3))]) == []


def test_duplicate_points_within_one_fragment_are_not_a_cluster():
    """Two coincident points of the SAME fragment must not be reported --
    the loss needs cross-fragment agreement, not self-agreement."""
    p = np.array([[0.0, 0, 0], [0.0, 0, 0]])
    assert find_shared_points([p]) == []


def test_union_find_merges_chains():
    uf = UnionFind(5)
    uf.union(0, 1); uf.union(1, 2); uf.union(3, 4)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


def test_transfer_through_a_pruning_map():
    ids = np.array([-1, 7, 7, 3, -1])
    vmap = np.array([-1, 0, 0, 1, -1])          # verts 1,2 -> 0 ; vert 3 -> 1
    out = transfer_vertex_clusters(ids, vmap, 2)
    assert out.tolist() == [7, 3]


def test_transfer_handles_everything_dropped():
    out = transfer_vertex_clusters(np.array([1, 2, 3]), np.array([-1, -1, -1]), 0)
    assert len(out) == 0
