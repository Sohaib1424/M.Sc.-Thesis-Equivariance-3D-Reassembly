"""
Cross-fragment correspondence: which vertices / edge midpoints from *different*
fragments occupied the same physical location before the object broke.

This is the ground-truth supervision for the interface-embedding-consistency
losses, and later the thing the classical translation solver anchors on.

WHERE IT MUST BE COMPUTED
-------------------------
On fragments expressed in a COMMON, UNTRANSFORMED frame -- i.e. straight off
``load_scene`` (optionally after decimation, which keeps every fragment on the
same shared grid), and *before* per-fragment centralization or SE(3) diffusion.
Both of those move fragments independently and destroy the "same 3D location"
property this relies on. Diffusion does not change vertex/face count or
ordering, so ids computed here stay valid for the diffused copies.

ALGORITHM
---------
1. Exact match: coordinates rounded to ``exact_decimals`` are grouped by a
   lexicographic sort. Any group spanning more than one fragment is resolved
   here. This is the common case on undecimated Breaking Bad meshes, where the
   two sides of a cut share bit-identical coordinates.
2. Nearest-neighbour fallback: whatever is left is matched through a k-d tree
   (k = 8 candidates, nearest first), accepted only if the nearest candidate
   *from another fragment* is within ``tol``. This is what carries the method
   through decimated meshes and any float noise.

Clusters are merged with union-find; only clusters spanning >= 2 distinct
fragments are returned. Everything else is labelled -1 ("not shared"), and the
losses skip those entries entirely.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


class UnionFind:
    """Union-find with path compression and union by size."""

    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def find_shared_points(
    points_list: Sequence[np.ndarray],
    tol: float = 1e-5,
    exact_decimals: int = 6,
    knn: int = 8,
) -> List[List[Tuple[int, int]]]:
    """Cluster points from different fragments that share a physical location.

    ``points_list[i]`` is an ``(N_i, 3)`` array for fragment ``i``, all in one
    frame. Returns clusters as lists of ``(fragment_index, local_point_index)``,
    keeping only clusters that span 2+ fragments.
    """
    frag_ids, local_ids, coords = [], [], []
    for fi, pts in enumerate(points_list):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        frag_ids.append(np.full(len(pts), fi, dtype=np.int64))
        local_ids.append(np.arange(len(pts), dtype=np.int64))
        coords.append(pts)

    if not coords:
        return []
    frag_ids = np.concatenate(frag_ids)
    local_ids = np.concatenate(local_ids)
    coords = np.concatenate(coords, axis=0)
    n = len(coords)
    if n == 0:
        return []

    uf = UnionFind(n)
    matched = np.zeros(n, dtype=bool)

    # ---- Stage 1: exact match on rounded coordinates ----
    rounded = np.round(coords, decimals=exact_decimals)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    srt = rounded[order]
    new_group = np.ones(n, dtype=bool)
    if n > 1:
        new_group[1:] = np.any(srt[1:] != srt[:-1], axis=1)
    group_id = np.cumsum(new_group) - 1

    # Group boundaries in the sorted order; only groups with >1 member can
    # possibly span two fragments, so the rest are skipped without a dict.
    boundaries = np.flatnonzero(new_group)
    ends = np.append(boundaries[1:], n)
    for start, stop in zip(boundaries, ends):
        if stop - start < 2:
            continue
        members = order[start:stop]
        if len({int(frag_ids[m]) for m in members}) < 2:
            continue
        base = int(members[0])
        for m in members[1:]:
            uf.union(base, int(m))
        matched[members] = True

    # ---- Stage 2: nearest-neighbour fallback ----
    unmatched = np.flatnonzero(~matched)
    if len(unmatched) > 1 and tol > 0:
        tree = cKDTree(coords[unmatched])
        k = min(knn, len(unmatched))
        dists, idxs = tree.query(coords[unmatched], k=k)
        if k == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        for i, gi in enumerate(unmatched):
            gi_frag = frag_ids[gi]
            for rank in range(k):
                cand = unmatched[idxs[i, rank]]
                if cand == gi:
                    continue
                if dists[i, rank] > tol:
                    break                       # sorted by distance; nothing closer left
                if frag_ids[cand] != gi_frag:
                    uf.union(int(gi), int(cand))
                    break

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    clusters = defaultdict(list)
    for i, r in enumerate(roots):
        clusters[int(r)].append((int(frag_ids[i]), int(local_ids[i])))
    return [m for m in clusters.values() if len({f for f, _ in m}) > 1]


def _cluster_ids_from_clusters(
    clusters: List[List[Tuple[int, int]]],
    sizes: Sequence[int],
) -> List[np.ndarray]:
    ids = [np.full(n, -1, dtype=np.int64) for n in sizes]
    for cid, members in enumerate(clusters):
        for frag_idx, local_idx in members:
            ids[frag_idx][local_idx] = cid
    return ids


def compute_scene_correspondence(
    fragment_meshes: Sequence,
    tol: float = 1e-5,
    exact_decimals: int = 6,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Per-fragment vertex and edge cluster-id arrays for one scene.

    Returns ``(vertex_cluster_ids, edge_cluster_ids)``:
      * ``vertex_cluster_ids[i]``: ``(V_i,)``, aligned with ``mesh.vertices``
      * ``edge_cluster_ids[i]``:   ``(E_i,)``, aligned with ``mesh.edges_unique``
        (i.e. the *undoubled*, forward-only edge ordering that feature
        construction starts from before it builds the bidirectional graph)

    Vertex-cluster ids and edge-cluster ids are independent numbering schemes;
    vertex cluster 3 and edge cluster 3 have nothing to do with each other.
    """
    vertex_points = [np.asarray(m.vertices) for m in fragment_meshes]
    vertex_clusters = find_shared_points(vertex_points, tol=tol, exact_decimals=exact_decimals)
    vertex_ids = _cluster_ids_from_clusters(vertex_clusters, [len(p) for p in vertex_points])

    edge_midpoints = []
    for m in fragment_meshes:
        eu = np.asarray(m.edges_unique)
        if len(eu) == 0:
            edge_midpoints.append(np.zeros((0, 3)))
        else:
            v = np.asarray(m.vertices)
            edge_midpoints.append((v[eu[:, 0]] + v[eu[:, 1]]) / 2.0)
    edge_clusters = find_shared_points(edge_midpoints, tol=tol, exact_decimals=exact_decimals)
    edge_ids = _cluster_ids_from_clusters(edge_clusters, [len(p) for p in edge_midpoints])

    return vertex_ids, edge_ids


def transfer_vertex_clusters(
    vertex_cluster_ids: np.ndarray,
    vertex_map: np.ndarray,
    num_new_vertices: int,
) -> np.ndarray:
    """Carry vertex cluster ids through a vertex-pruning/remapping step.

    ``vertex_map[i]`` is the new index of old vertex ``i``, or -1 if it was
    dropped (this is exactly what ``extract_fractures_with_map`` and
    ``decimate_arrays`` return). Where several old vertices collapse into one
    new vertex, the first non-negative id wins -- they were coincident, so if
    they disagreed the disagreement was already meaningless.
    """
    out = np.full(num_new_vertices, -1, dtype=np.int64)
    live = vertex_map >= 0
    if not np.any(live):
        return out
    src_ids = np.asarray(vertex_cluster_ids, dtype=np.int64)[live]
    dst = np.asarray(vertex_map, dtype=np.int64)[live]
    keep = src_ids >= 0
    if np.any(keep):
        # Reverse order so that the *first* source wins after the later writes.
        out[dst[keep][::-1]] = src_ids[keep][::-1]
    return out


def derive_edge_clusters_for_scene(
    fragment_meshes: Sequence,
    vertex_cluster_ids: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Derive edge cluster ids from *vertex* cluster ids, for a whole scene.

    An edge is shared across fragments exactly when both its endpoints are, so
    the (sorted) pair of endpoint vertex-cluster ids is a canonical key for the
    edge's physical location. Two advantages over clustering edge midpoints
    spatially a second time:

    * the vertex-level and edge-level supervision are guaranteed to agree,
      rather than being two independent clusterings that can disagree at the
      margins;
    * it needs no coordinates at all, which is what makes it usable on the
      *pruned fracture-surface mesh*, where ``edges_unique`` is rebuilt from a
      face subset and therefore has an entirely different ordering from the
      full mesh. Transferring vertex ids through ``vertex_map`` and then
      re-deriving edge ids here is exactly the missing step that previously
      made fracture-mesh edge supervision unusable.

    Clusters that end up spanning fewer than two distinct fragments are
    dropped (relabelled -1) and the rest renumbered contiguously, matching the
    contract of :func:`compute_scene_correspondence`.
    """
    key_to_id: dict = {}
    per_fragment_keys: List[np.ndarray] = []

    for mesh, vc in zip(fragment_meshes, vertex_cluster_ids):
        edges = np.asarray(mesh.edges_unique, dtype=np.int64).reshape(-1, 2)
        if len(edges) == 0:
            per_fragment_keys.append(np.empty((0,), dtype=np.int64))
            continue
        vc = np.asarray(vc, dtype=np.int64)
        a, b = vc[edges[:, 0]], vc[edges[:, 1]]
        shared = (a >= 0) & (b >= 0)
        lo, hi = np.minimum(a, b), np.maximum(a, b)

        ids = np.full(len(edges), -1, dtype=np.int64)
        for i in np.flatnonzero(shared):
            key = (int(lo[i]), int(hi[i]))
            cid = key_to_id.get(key)
            if cid is None:
                cid = key_to_id[key] = len(key_to_id)
            ids[i] = cid
        per_fragment_keys.append(ids)

    # Drop clusters that do not actually span two fragments.
    n_clusters = len(key_to_id)
    if n_clusters == 0:
        return per_fragment_keys

    fragments_per_cluster = [set() for _ in range(n_clusters)]
    for fi, ids in enumerate(per_fragment_keys):
        for cid in np.unique(ids[ids >= 0]):
            fragments_per_cluster[int(cid)].add(fi)

    remap = np.full(n_clusters, -1, dtype=np.int64)
    next_id = 0
    for cid, frags in enumerate(fragments_per_cluster):
        if len(frags) >= 2:
            remap[cid] = next_id
            next_id += 1

    out = []
    for ids in per_fragment_keys:
        new = np.full(len(ids), -1, dtype=np.int64)
        live = ids >= 0
        if np.any(live):
            new[live] = remap[ids[live]]
        out.append(new)
    return out
