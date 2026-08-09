"""Cross-fragment correspondence."""
from __future__ import annotations

import numpy as np

from vngat.data.correspondence import cluster_shared_points


def ids_to_clusters(id_lists):
    out = {}
    for frag, arr in enumerate(id_lists):
        for local, cid in enumerate(np.asarray(arr)):
            if cid >= 0:
                out.setdefault(int(cid), []).append((frag, local))
    return sorted(tuple(sorted(v)) for v in out.values())


def test_exact_shared_points_are_clustered():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(15, 3))
    a = np.vstack([rng.normal(size=(50, 3)), shared])
    b = np.vstack([rng.normal(size=(40, 3)), shared])
    clusters = ids_to_clusters(cluster_shared_points([a, b]))
    assert len(clusters) == 15
    assert all(len(c) == 2 for c in clusters)


def test_sub_tolerance_noise_still_matches():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(12, 3))
    a = np.vstack([rng.normal(size=(30, 3)), shared])
    b = np.vstack([rng.normal(size=(30, 3)), shared + 1e-7])
    assert len(ids_to_clusters(cluster_shared_points([a, b]))) == 12


def test_disjoint_fragments_produce_no_clusters():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(30, 3))
    b = rng.normal(size=(30, 3)) + 100.0
    ids = cluster_shared_points([a, b])
    assert all(int((np.asarray(x) >= 0).sum()) == 0 for x in ids)


def test_three_way_shared_points_form_one_cluster_each():
    rng = np.random.default_rng(3)
    shared = rng.normal(size=(6, 3))
    frags = [np.vstack([rng.normal(size=(20, 3)), shared]) for _ in range(3)]
    clusters = ids_to_clusters(cluster_shared_points(frags))
    assert len(clusters) == 6
    assert all(len(c) == 3 for c in clusters)


def test_points_shared_within_one_fragment_are_not_clusters():
    """Duplicated coordinates inside a single fragment are not correspondences."""
    p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    ids = cluster_shared_points([p, np.array([[9.0, 9.0, 9.0]])])
    assert int((np.asarray(ids[0]) >= 0).sum()) == 0


def test_empty_fragments_are_handled():
    rng = np.random.default_rng(4)
    shared = rng.normal(size=(8, 3))
    frags = [np.vstack([rng.normal(size=(10, 3)), shared]),
             np.zeros((0, 3)),
             np.vstack([rng.normal(size=(10, 3)), shared])]
    ids = cluster_shared_points(frags)
    assert len(ids) == 3 and len(ids[1]) == 0
    assert len(ids_to_clusters(ids)) == 8


def test_all_empty_input():
    assert cluster_shared_points([]) == []
    ids = cluster_shared_points([np.zeros((0, 3)), np.zeros((0, 3))])
    assert all(len(x) == 0 for x in ids)
