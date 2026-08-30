"""
Mesh topology from raw ``(vertices, faces)`` arrays.

These replace the trimesh cached properties (``edges_unique``,
``face_adjacency``, ``face_normals``) used across the original code. Two
reasons, both of which cost real time earlier in this project:

* trimesh's cached properties hash their inputs on every access, which is
  wasted work when a mesh is built, queried once, and thrown away -- exactly
  the access pattern of a dataset loader.
* ``mesh.face_normals`` silently yields NaN on zero-area triangles. Here the
  degenerate faces are repaired *and counted*, so a bad input shows up in a
  tally instead of quietly propagating into features.

Everything takes plain arrays so it can be reused on igl output without
constructing a Trimesh first.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..arrays import group_by_key, pair_key, pair_unkey, unique_sorted


def half_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The 3F undirected half-edges as ``(lo, hi)`` vertex-index arrays.

    Ordering is ``[all (0,1); all (1,2); all (2,0)]``, so half-edge ``k``
    belongs to face ``k % F``.
    """
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    e = np.concatenate([e01, e12, e20], axis=0)
    return np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])


def _nv(faces: np.ndarray, num_vertices: int | None) -> int:
    if num_vertices is not None:
        return int(num_vertices)
    return int(faces.max()) + 1 if faces.size else 1


def unique_edges(faces: np.ndarray, num_vertices: int | None = None) -> np.ndarray:
    """
    ``(E, 2)`` unique undirected edges, ``lo < hi``, in lexicographic order.

    **This is the canonical edge ordering for the whole package.** Anything
    that stores a per-edge array -- edge features, edge cluster ids, edge
    embeddings -- is aligned by position, so every producer and every consumer
    must agree on the order.

    ``trimesh.edges_unique`` returns the same *set* in a different order. It
    is therefore never used anywhere in this package, and mixing the two
    would misalign edge attributes without raising anything. That failure has
    already happened once on this project. If a mesh's edges are needed, they
    come from here or from :func:`edge_topology`, which returns this same
    ordering.
    """
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    n = _nv(faces, num_vertices)
    lo, hi = half_edges(faces)
    key = unique_sorted(pair_key(lo, hi, n))
    a, b = pair_unkey(key, n)
    return np.stack([a, b], axis=1)


def count_unique_edges(faces: np.ndarray, num_vertices: int | None = None) -> int:
    """Number of unique undirected edges, without materialising them."""
    if faces.size == 0:
        return 0
    n = _nv(faces, num_vertices)
    lo, hi = half_edges(faces)
    return int(unique_sorted(pair_key(lo, hi, n)).shape[0])


class EdgeTopology(NamedTuple):
    """Everything :func:`reassembly.data.features.get_features` needs, in one pass."""
    edges: np.ndarray          # (E, 2) unique undirected edges, lo < hi
    edge_of_half: np.ndarray   # (3F,) unique-edge index of each half-edge
    face_of_half: np.ndarray   # (3F,) owning face of each half-edge
    faces_per_edge: np.ndarray # (E, 2) up to two incident faces, -1 when absent


def edge_topology(faces: np.ndarray, num_vertices: int | None = None) -> EdgeTopology:
    """
    Unique edges plus edge->face incidence, computed together.

    ``faces_per_edge`` holds the (at most two) faces touching each edge.
    Boundary edges get ``-1`` in the second slot; non-manifold edges keep the
    two lowest-numbered incident faces, matching what the original
    ``get_features`` did by taking the first two entries of the group.
    """
    if faces.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return EdgeTopology(np.zeros((0, 2), np.int64), z, z, np.zeros((0, 2), np.int64))

    nf = faces.shape[0]
    n = _nv(faces, num_vertices)
    lo, hi = half_edges(faces)
    keys = pair_key(lo, hi, n)
    face_of_half = np.tile(np.arange(nf, dtype=np.int64), 3)

    # stable: the incident faces of an edge come out in ascending face order
    order, starts, sizes = group_by_key(keys, stable=True)
    n_edges = starts.shape[0]

    edge_of_half = np.empty(keys.shape[0], dtype=np.int64)
    edge_of_half[order] = np.repeat(np.arange(n_edges, dtype=np.int64), sizes)

    a, b = pair_unkey(keys[order[starts]], n)
    edges = np.stack([a, b], axis=1)

    owners = face_of_half[order]
    faces_per_edge = np.full((n_edges, 2), -1, dtype=np.int64)
    faces_per_edge[:, 0] = owners[starts]
    has_second = sizes >= 2
    faces_per_edge[has_second, 1] = owners[starts[has_second] + 1]

    return EdgeTopology(edges, edge_of_half, face_of_half, faces_per_edge)


def face_adjacency(faces: np.ndarray, num_vertices: int | None = None,
                   has_duplicate_faces: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    All ordered pairs ``(i, j)``, ``i != j``, of faces sharing >= 2 vertices.

    This reproduces the original ``find_neighbors(min_common_vertices=2)``
    exactly, but without forming the ``(F, F)`` sparse product ``V^T V``.

    The equivalence that makes it possible: for *triangles*, "shares two
    vertices" and "shares an edge" are the same condition, because any two
    vertices of a triangle span one of its three edges. So the pairs can be
    read off by grouping the 3F half-edges, which costs one sort of 3F keys
    instead of ``sum_v deg(v)**2`` sparse-product work.

    Two duplicate faces share all three edges and would therefore be emitted
    three times. Passing ``has_duplicate_faces=False`` (the normal case for
    meshes that have been through :mod:`reassembly.mesh.repair`) skips the
    deduplication pass, which is the single most expensive step here.
    """
    nf = faces.shape[0]
    if nf == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z

    n = _nv(faces, num_vertices)
    lo, hi = half_edges(faces)
    keys = pair_key(lo, hi, n)
    owner = np.tile(np.arange(nf, dtype=np.int64), 3)

    order, starts, sizes = group_by_key(keys, stable=False)
    owner_sorted = owner[order]

    src_parts, dst_parts = [], []
    for k in unique_sorted(sizes[sizes >= 2]):
        k = int(k)
        sel = starts[sizes == k]
        block = owner_sorted[sel[:, None] + np.arange(k)[None, :]]   # (G, k)
        a = np.repeat(block, k, axis=1).ravel()
        b = np.tile(block, (1, k)).ravel()
        keep = a != b
        src_parts.append(a[keep])
        dst_parts.append(b[keep])

    if not src_parts:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)

    if has_duplicate_faces is None:
        has_duplicate_faces = _any_duplicate_faces(faces, n)
    if has_duplicate_faces:
        key = unique_sorted(pair_key(src, dst, nf))
        return pair_unkey(key, nf)
    return src, dst


def _any_duplicate_faces(faces: np.ndarray, num_vertices: int) -> bool:
    """Cheap check: does any triple of vertex indices appear on two faces?"""
    from ..arrays import can_triple_key, triple_key

    srt = np.sort(faces, axis=1)
    if can_triple_key(num_vertices):
        keys = triple_key(srt, num_vertices)
    else:                                                      # pragma: no cover
        keys = np.ascontiguousarray(srt).view(
            np.dtype((np.void, srt.dtype.itemsize * 3))
        ).ravel()
        return len(np.unique(keys)) != srt.shape[0]
    return unique_sorted(keys).shape[0] != srt.shape[0]


def face_normals(vertices: np.ndarray, faces: np.ndarray,
                 return_degenerate: bool = False):
    """
    Unit face normals, with zero-area faces set to the zero vector.

    trimesh returns NaN for these. NaN normals propagated silently through an
    entire earlier run of this project, so the degenerate count is surfaced
    rather than swallowed: pass ``return_degenerate=True`` to get
    ``(normals, n_degenerate)``.
    """
    if faces.size == 0:
        out = np.zeros((0, 3), dtype=np.float64)
        return (out, 0) if return_degenerate else out

    tri = vertices[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(nrm, axis=1)
    bad = ~(length > 0) | ~np.isfinite(length)
    nrm /= np.where(bad, 1.0, length)[:, None]
    nrm[bad] = 0.0
    return (nrm, int(bad.sum())) if return_degenerate else nrm


def vertex_normals(vertices: np.ndarray, faces: np.ndarray,
                   face_norms: np.ndarray | None = None) -> np.ndarray:
    """
    Area-weighted vertex normals, with isolated/degenerate vertices zeroed.

    Matches trimesh's default vertex-normal convention closely enough for
    feature extraction while staying NaN-free by construction.
    """
    if faces.size == 0:
        return np.zeros((vertices.shape[0], 3), dtype=np.float64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # area-weighted
    if face_norms is not None:
        pass  # cross already carries the area weight; face_norms is unit-length
    out = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    np.add.at(out, faces[:, 0], cross)
    np.add.at(out, faces[:, 1], cross)
    np.add.at(out, faces[:, 2], cross)
    length = np.linalg.norm(out, axis=1)
    bad = ~(length > 0) | ~np.isfinite(length)
    out /= np.where(bad, 1.0, length)[:, None]
    out[bad] = 0.0
    return out
