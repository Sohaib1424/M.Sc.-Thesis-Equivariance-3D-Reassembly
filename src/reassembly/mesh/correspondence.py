"""
Cross-fragment correspondence.

Finds vertices and edge midpoints from *different* fragments that occupy the
same physical location -- the pairs that faced each other before the object
broke. Used both as supervision for interface-embedding losses and, at
inference, as the constraint set for a translation solver.

Correspondence must be computed on fragments in a **common, untransformed
frame**: the meshes as loaded, before per-fragment centralisation and before
any SE(3) perturbation. Both of those move fragments independently and
destroy the "same 3D location" property this relies on. A rigid transform
applied uniformly afterwards is fine, and does not change vertex ordering, so
cluster ids computed here stay valid for the perturbed copies.

Changes from the original implementation, all behaviour-preserving:

* the exact-match pass groups with a void view over rounded coordinates
  instead of a Python ``defaultdict`` keyed by group id, which was the
  dominant cost on scenes with many vertices;
* the nearest-neighbour fallback uses ``cKDTree.query_pairs``, one call
  returning every within-tolerance pair, instead of a k=8 query followed by a
  Python loop over every unmatched point;
* union-find carries union-by-size in addition to path compression, so
  pathological chains cannot degrade it.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


class UnionFind:
    """Union-find over ``n`` elements, with path halving and union by size."""

    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def union_many(self, a: np.ndarray, b: np.ndarray) -> None:
        for x, y in zip(a.tolist(), b.tolist()):
            self.union(x, y)

    def labels(self) -> np.ndarray:
        """Root of every element, as an array."""
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


def find_shared_points(
    points_list: Sequence[np.ndarray],
    tol: float = 1e-5,
    exact_decimals: int = 6,
) -> List[List[Tuple[int, int]]]:
    """
    Cluster points from different fragments that share a physical location.

    ``points_list[i]`` is the ``(N_i, 3)`` point set of fragment ``i``, all in
    one frame. Returns clusters as lists of ``(fragment_index, local_index)``;
    only clusters touching two or more fragments are returned.

    Two passes: exact match on coordinates rounded to ``exact_decimals``,
    then a ``tol``-radius pair query over whatever is left. The second pass
    matters because the two sides of an interface can come from independent
    remeshing and differ in the last bits.
    """
    sizes = np.array([len(np.asarray(p)) for p in points_list], dtype=np.int64)
    if sizes.size == 0 or sizes.sum() == 0:
        return []

    coords = np.concatenate(
        [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in points_list], axis=0
    )
    frag_ids = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    local_ids = np.concatenate([np.arange(s, dtype=np.int64) for s in sizes])
    n = coords.shape[0]

    uf = UnionFind(n)
    matched = np.zeros(n, dtype=bool)

    # ---- pass 1: exact match on rounded coordinates -----------------------
    rounded = np.round(coords, decimals=exact_decimals)
    view = np.ascontiguousarray(rounded).view(
        np.dtype((np.void, rounded.dtype.itemsize * 3))
    ).ravel()
    _, group = np.unique(view, return_inverse=True)
    group = group.ravel()

    order = np.argsort(group, kind="stable")
    gs = group[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    if n > 1:
        np.not_equal(gs[1:], gs[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, n))

    for start, count in zip(starts[counts > 1].tolist(), counts[counts > 1].tolist()):
        members = order[start:start + count]
        if np.unique(frag_ids[members]).size > 1:
            base = int(members[0])
            for m in members[1:].tolist():
                uf.union(base, m)
            matched[members] = True

    # ---- pass 2: tolerance match on the remainder -------------------------
    rest = np.flatnonzero(~matched)
    if rest.size > 1 and tol > 0:
        pairs = cKDTree(coords[rest]).query_pairs(r=tol, output_type="ndarray")
        if pairs.size:
            a, b = rest[pairs[:, 0]], rest[pairs[:, 1]]
            cross = frag_ids[a] != frag_ids[b]
            uf.union_many(a[cross], b[cross])

    # ---- collect ----------------------------------------------------------
    roots = uf.labels()
    order = np.argsort(roots, kind="stable")
    rs = roots[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    if n > 1:
        np.not_equal(rs[1:], rs[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, n))

    clusters: List[List[Tuple[int, int]]] = []
    for start, count in zip(starts.tolist(), counts.tolist()):
        if count < 2:
            continue
        members = order[start:start + count]
        if np.unique(frag_ids[members]).size < 2:
            continue
        clusters.append(
            [(int(frag_ids[m]), int(local_ids[m])) for m in members.tolist()]
        )
    return clusters


def _cluster_ids(clusters, sizes) -> List[np.ndarray]:
    out = [np.full(int(s), -1, dtype=np.int64) for s in sizes]
    for cid, members in enumerate(clusters):
        for frag_idx, local_idx in members:
            out[frag_idx][local_idx] = cid
    return out


def compute_scene_correspondence(
    fragment_meshes: Sequence,
    tol: float = 1e-5,
    exact_decimals: int = 6,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Per-fragment vertex and edge cluster ids for one scene.

    Returns ``(vertex_cluster_ids, edge_cluster_ids)``. Each is a list with
    one ``int64`` array per fragment: a non-negative cluster id where the
    vertex/edge is shared with another fragment, ``-1`` where it is not.
    Edge arrays are aligned with :func:`reassembly.mesh.topology.unique_edges`
    ordering, matching ``get_features``' pre-doubling edge order.

    Vertex ids and edge ids are separate numbering schemes -- vertex cluster 3
    and edge cluster 3 have nothing to do with each other.
    """
    from .topology import unique_edges

    vertex_points = [np.asarray(m.vertices) for m in fragment_meshes]
    vertex_sizes = [len(p) for p in vertex_points]
    vertex_clusters = find_shared_points(vertex_points, tol=tol, exact_decimals=exact_decimals)

    edge_midpoints, edge_sizes = [], []
    for mesh in fragment_meshes:
        vertices = np.asarray(mesh.vertices)
        edges = unique_edges(np.asarray(mesh.faces), vertices.shape[0])
        edge_sizes.append(len(edges))
        if len(edges) == 0:
            edge_midpoints.append(np.zeros((0, 3)))
        else:
            edge_midpoints.append((vertices[edges[:, 0]] + vertices[edges[:, 1]]) * 0.5)
    edge_clusters = find_shared_points(edge_midpoints, tol=tol, exact_decimals=exact_decimals)

    return _cluster_ids(vertex_clusters, vertex_sizes), _cluster_ids(edge_clusters, edge_sizes)
