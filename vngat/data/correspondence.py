"""
Cross-fragment correspondence: which vertices / edges of *different* fragments
occupied the same physical location before the object broke.

This is the ground truth for the interface-embedding-consistency losses.

Algorithm (unchanged in semantics from the original implementation):
  1. Exact match -- coordinates rounded to `exact_decimals` and grouped.
  2. Nearest-neighbour fallback within `tol` for anything still unmatched.
  3. Union-find merges match chains; only clusters spanning >= 2 distinct
     fragments survive. Everything else is labelled -1.

What changed is the *implementation*, not the result: the original ran three
Python-level loops over every point in the scene (roughly 17k vertices +
47k edge midpoints for a typical Breaking Bad object), which made this one of
the two dominant costs in the data path. The version below is fully
vectorised apart from a union-find loop over actual matched pairs, which is
tiny. `tests/test_correspondence.py` asserts the two produce identical
clusters on exact-match, sub-tolerance-noise, NN-fallback, disjoint,
three-way-shared, and empty-fragment cases.

IMPORTANT: correspondence must be computed on fragments in a COMMON,
untransformed frame -- the meshes as loaded, before per-fragment
centralisation or diffusion. Both of those move fragments independently and
destroy the "same 3D location" property. Rigid diffusion does not change
vertex/face ordering, so ids computed here stay valid for the diffused view.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _union_find_roots(num_groups: int, pairs: np.ndarray) -> np.ndarray:
    """Roots after unioning `pairs` (M, 2) of group ids. Loop is over matched
    pairs only -- typically a few thousand interface points, not every point."""
    parent = np.arange(num_groups, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(g) for g in range(num_groups)], dtype=np.int64)


def cluster_shared_points(
    points_list: Sequence[np.ndarray],
    tol: float = 1e-5,
    exact_decimals: int = 6,
) -> List[np.ndarray]:
    """
    points_list[i]: (N_i, 3) points of fragment i, all in the same frame.

    Returns a list of (N_i,) int64 arrays of cluster ids: a non-negative id
    shared with at least one other fragment, or -1 for "not shared".
    """
    sizes = [int(np.asarray(p).reshape(-1, 3).shape[0]) for p in points_list]
    if not sizes:
        return []
    total = int(sum(sizes))
    if total == 0:
        return [np.full(s, -1, dtype=np.int64) for s in sizes]

    coords = np.concatenate(
        [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in points_list], axis=0
    )
    frag_ids = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)

    # ---- Stage 1: exact match via lexicographic grouping of rounded coords --
    rounded = np.round(coords, decimals=exact_decimals)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    sorted_rounded = rounded[order]
    is_new_group = np.ones(total, dtype=bool)
    if total > 1:
        is_new_group[1:] = np.any(sorted_rounded[1:] != sorted_rounded[:-1], axis=1)
    group_id = np.empty(total, dtype=np.int64)
    group_id[order] = np.cumsum(is_new_group) - 1
    num_groups = int(group_id.max()) + 1

    # A group is already "matched" if it touches more than one fragment.
    gf_pairs = np.unique(np.stack([group_id, frag_ids], axis=1), axis=0)
    group_frag_count = np.bincount(gf_pairs[:, 0], minlength=num_groups)
    matched = group_frag_count[group_id] > 1

    # ---- Stage 2: nearest-neighbour fallback for the rest -------------------
    link_pairs = np.zeros((0, 2), dtype=np.int64)
    unmatched = np.where(~matched)[0]
    if tol > 0 and len(unmatched) > 1:
        k = min(8, len(unmatched))
        tree = cKDTree(coords[unmatched])
        # distance_upper_bound lets the tree abandon a query as soon as
        # nothing within tol remains, which is the common case here (most
        # points are exterior surface with no counterpart at all).
        dist, idx = tree.query(coords[unmatched], k=k, distance_upper_bound=tol)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        valid = np.isfinite(dist)
        idx = np.where(valid, idx, 0)
        cand = unmatched[idx]
        ok = valid & (cand != unmatched[:, None]) & (frag_ids[cand] != frag_ids[unmatched][:, None])
        first = np.argmax(ok, axis=1)          # results are distance-sorted
        has = ok.any(axis=1)
        if has.any():
            link_pairs = np.stack(
                [group_id[unmatched[has]], group_id[cand[has, first[has]]]], axis=1
            )

    roots = _union_find_roots(num_groups, link_pairs)
    root_of_point = roots[group_id]

    # ---- Keep only roots spanning >= 2 fragments, then densify ids ----------
    rf_pairs = np.unique(np.stack([root_of_point, frag_ids], axis=1), axis=0)
    root_frag_count = np.bincount(rf_pairs[:, 0], minlength=num_groups)
    shared = root_frag_count[root_of_point] > 1

    cluster_id = np.full(total, -1, dtype=np.int64)
    if shared.any():
        _, dense = np.unique(root_of_point[shared], return_inverse=True)
        cluster_id[shared] = dense

    return list(np.split(cluster_id, np.cumsum(sizes)[:-1]))


def compute_scene_correspondence(
    fragment_meshes: list,
    tol: float = 1e-5,
    exact_decimals: int = 6,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Per-fragment vertex and edge cluster ids for one scene.

    Returns (vertex_cluster_ids, edge_cluster_ids):
      vertex_cluster_ids[i]: (V_i,)  aligned with mesh.vertices
      edge_cluster_ids[i]:   (E_i,)  aligned with mesh.edges_unique

    Vertex-cluster ids and edge-cluster ids are INDEPENDENT numbering schemes;
    a vertex cluster 3 and an edge cluster 3 are unrelated.

    Must be called on whichever mesh variant the model will actually consume
    (full fragment or pruned fracture surface), because `extract_fractures`
    reindexes vertices and rebuilds `edges_unique` -- ids derived from one
    variant do not address the other.
    """
    vertex_points = [np.asarray(m.vertices, dtype=np.float64) for m in fragment_meshes]
    vertex_cluster_ids = cluster_shared_points(vertex_points, tol=tol, exact_decimals=exact_decimals)

    edge_midpoints = []
    for m in fragment_meshes:
        eu = m.edges_unique
        if len(eu) == 0:
            edge_midpoints.append(np.zeros((0, 3), dtype=np.float64))
        else:
            v = np.asarray(m.vertices, dtype=np.float64)
            edge_midpoints.append((v[eu[:, 0]] + v[eu[:, 1]]) / 2.0)
    edge_cluster_ids = cluster_shared_points(edge_midpoints, tol=tol, exact_decimals=exact_decimals)

    return vertex_cluster_ids, edge_cluster_ids
