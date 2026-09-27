"""
Low-level array primitives.

Everything in `reassembly.mesh` is built on these three ideas:

1. Encode small integer tuples as a single ``int64`` key, so grouping and
   deduplication become a 1-D sort instead of a multi-column ``lexsort`` or a
   structured-void view. On the meshes in this project that is roughly 2-5x
   faster.

2. Never call ``np.unique``. Measured on NumPy 2.4, ``np.unique`` on a 245k
   ``int64`` array takes ~50 ms, while ``sort`` + neighbour-compare takes
   ~2 ms -- a 24x difference, because ``np.unique`` computes masks and index
   arrays that are almost always discarded. ``unique_sorted`` below is the
   drop-in replacement and is never slower on any NumPy version.

3. Ask for stability only when it is actually needed. ``kind="stable"`` on
   ``int64`` falls back to mergesort and costs ~4x an introsort. Grouping for
   adjacency does not need it; picking the lowest original index within a
   group does.
"""
from __future__ import annotations

import numpy as np

INT64_MAX = np.iinfo(np.int64).max

# (max_vertex_index + 1) below which a 3-tuple still fits in int64:
# nv**3 must stay under 2**63 - 1.
_TRIPLE_KEY_LIMIT = 2_000_000


def pair_key(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Encode (a, b) with 0 <= a, b < n as a single int64. Requires n**2 < 2**63."""
    return a.astype(np.int64, copy=False) * np.int64(n) + b.astype(np.int64, copy=False)


def pair_unkey(key: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pair_key`."""
    n64 = np.int64(n)
    return key // n64, key % n64


def triple_key(rows: np.ndarray, n: int) -> np.ndarray:
    """Encode (a, b, c) with 0 <= a, b, c < n as a single int64."""
    n64 = np.int64(n)
    r = rows.astype(np.int64, copy=False)
    return (r[:, 0] * n64 + r[:, 1]) * n64 + r[:, 2]


def can_triple_key(n: int) -> bool:
    return n < _TRIPLE_KEY_LIMIT


def unique_sorted(keys: np.ndarray) -> np.ndarray:
    """Sorted unique values of a 1-D array. Faster ``np.unique`` for this use."""
    n = keys.shape[0]
    if n == 0:
        return keys
    ks = np.sort(keys)
    if n == 1:
        return ks
    keep = np.empty(n, dtype=bool)
    keep[0] = True
    np.not_equal(ks[1:], ks[:-1], out=keep[1:])
    return ks[keep]


def group_by_key(keys: np.ndarray, stable: bool = False):
    """
    Group equal values of a 1-D int array.

    Returns ``(order, starts, sizes)`` where ``keys[order]`` is sorted,
    ``starts`` indexes the first element of each group inside ``order``, and
    ``sizes[g]`` is that group's length. With ``stable=True`` the original
    indices stay ascending inside each group.
    """
    n = keys.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    order = np.argsort(keys, kind="stable" if stable else None)
    ks = keys[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    if n > 1:
        np.not_equal(ks[1:], ks[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    return order, starts, sizes


def dense_group_ids(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Map each element of ``keys`` to a dense group id in ``[0, n_groups)``."""
    n = keys.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    order, starts, _ = group_by_key(keys, stable=True)
    ks = keys[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    if n > 1:
        np.not_equal(ks[1:], ks[:-1], out=new[1:])
    gid = np.empty(n, dtype=np.int64)
    gid[order] = np.cumsum(new) - 1
    return gid, len(starts)


def first_index_per_group(gid: np.ndarray, n_groups: int, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Lowest original index in each group, restricted to ``mask`` if given.
    Groups with no member are filled with ``INT64_MAX``.
    """
    out = np.full(n_groups, INT64_MAX, dtype=np.int64)
    sel = np.arange(gid.shape[0], dtype=np.int64) if mask is None else np.flatnonzero(mask)
    if sel.size == 0:
        return out
    g = gid[sel]
    order = np.argsort(g, kind="stable")   # stability -> ascending index per group
    gs, ss = g[order], sel[order]
    new = np.empty(gs.shape[0], dtype=bool)
    new[0] = True
    if gs.shape[0] > 1:
        np.not_equal(gs[1:], gs[:-1], out=new[1:])
    out[gs[new]] = ss[new]
    return out


def compact_indices(faces: np.ndarray, num_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop unreferenced vertices.

    Returns ``(kept, new_faces)`` where ``kept`` are the surviving vertex
    indices in ascending order and ``new_faces`` re-indexes into them.
    Equivalent to ``trimesh.remove_unreferenced_vertices`` but without
    rebuilding a mesh.
    """
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64), faces.reshape(0, 3).astype(np.int64)
    used = np.zeros(num_vertices, dtype=bool)
    used[faces.ravel()] = True
    kept = np.flatnonzero(used)
    remap = np.full(num_vertices, -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)
    return kept, remap[faces]
