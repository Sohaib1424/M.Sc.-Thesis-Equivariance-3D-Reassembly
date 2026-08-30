"""
Mesh repair: the pure-Python stand-in for ``igl::resolve_duplicated_faces``.

libigl's pip wheels (v2.6.x) do not expose ``resolve_duplicated_faces``, so
the loader needs its own. The original implementation looped over every face
in Python; this one is fully vectorised and returns the identical face
selection, verified face-for-face against it in ``tests/test_repair.py``.
"""
from __future__ import annotations

import numpy as np

from ..arrays import (
    INT64_MAX,
    can_triple_key,
    dense_group_ids,
    first_index_per_group,
    triple_key,
)


def _cyclically_consistent(faces: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    """True where ``faces[i]`` is a cyclic rotation (not a reflection) of ``canonical[i]``."""
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    c0, c1, c2 = canonical[:, 0], canonical[:, 1], canonical[:, 2]
    return (
        ((f0 == c0) & (f1 == c1) & (f2 == c2))
        | ((f0 == c1) & (f1 == c2) & (f2 == c0))
        | ((f0 == c2) & (f1 == c0) & (f2 == c1))
    )


def resolve_duplicated_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve duplicate faces by winding balance, following ``igl``'s rule.

    Faces are grouped by their vertex *set*. Within a group, let ``c`` be the
    number of faces wound consistently with the canonical (sorted) ordering
    minus the number wound against it:

    * a group of one keeps that face;
    * ``c == +1`` keeps the lowest-indexed consistently-wound face;
    * ``c == -1`` keeps the lowest-indexed oppositely-wound face;
    * ``c == 0`` keeps nothing (the duplicates cancel);
    * otherwise the lowest-indexed face in the group is kept.

    Returns ``(kept_faces, kept_indices)``.

    **Output row order is part of the contract.** Faces come back ordered by
    their group, i.e. lexicographically by sorted vertex triple -- not in
    input order. That is what the original implementation produced, because
    it appended while looping over ``np.unique``'s output, and downstream
    artifacts index faces by position: a stored per-face mask lines up with a
    mesh only if both were built under the same ordering. Do not "tidy" this
    by sorting the indices.

    One simplification over the original: it built ``uF`` via
    ``np.unique(..., axis=0)`` and then looked up ``uF[IC]`` for each face's
    canonical ordering. But ``uF[IC]`` is by definition each face's own sorted
    row, so the canonical array is just ``np.sort(faces, axis=1)`` and ``uF``
    never needs to exist.
    """
    faces = np.asarray(faces)
    n = faces.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=faces.dtype), np.empty((0,), dtype=np.int64)

    canonical = np.sort(faces, axis=1)
    n_vertices = int(canonical.max()) + 1

    if can_triple_key(n_vertices):
        keys = triple_key(canonical, n_vertices)
    else:                                                       # pragma: no cover
        view = np.ascontiguousarray(canonical).view(
            np.dtype((np.void, canonical.dtype.itemsize * 3))
        ).ravel()
        keys = np.unique(view, return_inverse=True)[1].astype(np.int64)

    gid, n_groups = dense_group_ids(keys)

    # Common case after igl's own cleanup: no duplicates. Still has to be
    # emitted in group order, so the fast path reuses the grouping already
    # computed rather than returning arange(n).
    if n_groups == n:
        kept = np.empty(n, dtype=np.int64)
        kept[gid] = np.arange(n, dtype=np.int64)
        return faces[kept], kept

    consistent = _cyclically_consistent(faces, canonical)
    group_size = np.bincount(gid, minlength=n_groups)
    balance = np.bincount(gid, weights=np.where(consistent, 1.0, -1.0), minlength=n_groups)

    first_any = first_index_per_group(gid, n_groups)
    first_pos = first_index_per_group(gid, n_groups, consistent)
    first_neg = first_index_per_group(gid, n_groups, ~consistent)

    keep = np.full(n_groups, -1, dtype=np.int64)
    single = group_size == 1
    keep[single] = first_any[single]

    multi = ~single
    take_pos = multi & (balance == 1)
    take_neg = multi & (balance == -1)
    take_any = multi & (balance != 1) & (balance != -1) & (balance != 0)
    keep[take_pos] = first_pos[take_pos]
    keep[take_neg] = first_neg[take_neg]
    keep[take_any] = first_any[take_any]

    if np.any(keep[take_pos] == INT64_MAX) or np.any(keep[take_neg] == INT64_MAX):
        raise AssertionError(
            "resolve_duplicated_faces: a group's winding balance implies a face "
            "orientation that is not present -- the input face array is inconsistent"
        )

    kept = keep[keep >= 0].astype(np.int64)   # group order, deliberately unsorted
    return faces[kept], kept
