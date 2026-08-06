"""
Correspondence discovery at inference (design doc S1.7).

After the network has oriented every fragment, matching interface points are
found by mutual nearest-neighbour search in the INVARIANT embedding space, then
filtered geometrically. Those matches are the constraints the translation
solver consumes.

Mutual (rather than one-sided) nearest neighbours: a pair is accepted only if
each point is the other's closest embedding. That single condition removes most
of the "popular point" failures where one distinctive vertex is the nearest
neighbour of half the other fragment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class MatchSet:
    """Accepted correspondences across a whole scene.

    All arrays share length M (the number of matches).
    ``frag_a`` / ``frag_b``   fragment indices
    ``point_a`` / ``point_b`` coordinates in each fragment's ORIENTED, still
                              un-translated frame (i.e. after applying R_pred,
                              before the solver)
    ``normal_a`` / ``normal_b`` matching surface normals, same frame
    ``score``                 embedding distance (lower is better)
    """
    frag_a: np.ndarray
    frag_b: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    normal_a: np.ndarray
    normal_b: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return len(self.frag_a)

    def filter(self, mask: np.ndarray) -> "MatchSet":
        return MatchSet(*(getattr(self, f)[mask] for f in
                          ("frag_a", "frag_b", "point_a", "point_b",
                           "normal_a", "normal_b", "score")))


def mutual_nearest_neighbors(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    max_distance: Optional[float] = None,
    ratio_threshold: Optional[float] = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual nearest neighbours between two embedding sets.

    ``ratio_threshold`` applies Lowe's ratio test (best / second-best): a match
    is kept only if the best neighbour is clearly better than the runner-up.
    On a fracture surface, large regions can be nearly featureless and produce
    many almost-equally-good candidates; the ratio test is what discards those
    rather than committing to an arbitrary one.

    Returns ``(idx_a, idx_b, distance)``.
    """
    if len(emb_a) == 0 or len(emb_b) == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0)

    tree_b = cKDTree(emb_b)
    tree_a = cKDTree(emb_a)

    k = 2 if (ratio_threshold is not None and len(emb_b) > 1) else 1
    d_ab, i_ab = tree_b.query(emb_a, k=k)
    if k == 1:
        d_ab, i_ab = d_ab[:, None], i_ab[:, None]

    best_b = i_ab[:, 0]
    best_d = d_ab[:, 0]

    _d_ba, i_ba = tree_a.query(emb_b, k=1)
    mutual = i_ba[best_b] == np.arange(len(emb_a))

    if max_distance is not None:
        mutual &= best_d <= max_distance
    if k == 2 and ratio_threshold is not None:
        second = np.maximum(d_ab[:, 1], 1e-12)
        mutual &= (best_d / second) <= ratio_threshold

    idx_a = np.flatnonzero(mutual)
    return idx_a, best_b[idx_a], best_d[idx_a]


def match_scene(
    embeddings: Sequence[np.ndarray],
    points: Sequence[np.ndarray],
    normals: Sequence[np.ndarray],
    max_distance: Optional[float] = None,
    auto_distance_quantile: Optional[float] = 0.10,
    auto_distance_scale: float = 4.0,
    ratio_threshold: Optional[float] = 0.9,
    normal_opposition: float = -0.3,
    min_matches_per_pair: int = 8,
    max_matches_per_pair: Optional[int] = 512,
) -> MatchSet:
    """Match every fragment pair and pool the accepted correspondences.

    ``normal_opposition``: keep a match only if ``n_a . n_b <= threshold``.
    Two surfaces that were in contact before the break must face each other,
    so their outward normals should be roughly antiparallel. This is the
    design document's "normal cancellation" idea used where it actually has
    force -- as a MATCH FILTER.

    (It cannot be a term in the translation objective, which is what S1.8
    proposes: normals are translation-invariant, so ``E_normal`` has exactly
    zero gradient with respect to every translation and contributes nothing to
    that minimization. Verified in ``validation/v05_translation_solver.py``.
    Applied here instead, the same geometric insight does real work.)

    ``max_matches_per_pair`` caps how many matches a single fragment pair can
    contribute, so one large interface cannot dominate the least-squares system
    and drown out the constraints from smaller ones.

    TWO FILTERS THAT ARE NOT OPTIONAL IN PRACTICE
    ---------------------------------------------
    Mutual nearest-neighbour matching between two point sets is essentially
    never empty. For generic embeddings, *some* pair is always mutually
    closest, so two fragments that never touched still produce matches. Left
    unchecked, that makes every scene look fully connected and quietly invents
    constraints between fragments that share no interface -- which is worse
    than reporting the fragment as unconstrained, because the resulting
    translation looks plausible and is wrong.

    ``auto_distance_quantile`` / ``auto_distance_scale`` set an
    embedding-distance ceiling from the scene's own candidate distribution.
    The training objective drives true twins towards *identical* embeddings, so
    genuine matches occupy the very bottom of that distribution: the low
    quantile estimates the true-match scale, and the ceiling is a multiple of
    it. Anchoring on the bottom of the distribution rather than cutting at a
    fixed quantile matters -- a fixed quantile discards a fixed FRACTION of
    candidates regardless of how many are genuine, so on a scene where most
    matches are correct it throws away correct ones and can disconnect the
    correspondence graph.

    Pass ``max_distance`` to override with an absolute threshold once you have
    calibrated one on real embeddings (``scripts/evaluate.py --match-report``
    prints the distributions needed to do that).

    ``min_matches_per_pair`` drops fragment pairs contributing only a handful
    of survivors. A genuine fracture interface yields many mutually-consistent
    matches; a spurious pairing yields a few scattered ones.
    """
    n_frags = len(embeddings)

    # Pass 1: raw mutual-NN candidates, with the ratio and normal filters.
    candidates = []
    for a in range(n_frags):
        for b in range(a + 1, n_frags):
            ia, ib, dist = mutual_nearest_neighbors(
                embeddings[a], embeddings[b],
                max_distance=max_distance, ratio_threshold=ratio_threshold,
            )
            if len(ia) == 0:
                continue
            cos = (normals[a][ia] * normals[b][ib]).sum(axis=1)
            keep = cos <= normal_opposition
            if not np.any(keep):
                continue
            candidates.append((a, b, ia[keep], ib[keep], dist[keep]))

    # Scene-level embedding-distance ceiling.
    if max_distance is None and auto_distance_quantile is not None and candidates:
        all_dist = np.concatenate([c[4] for c in candidates])
        true_scale = float(np.quantile(all_dist, auto_distance_quantile))
        max_distance = max(true_scale * auto_distance_scale, 1e-12)

    # Pass 2: apply the ceiling and the per-pair support requirement.
    fa, fb, pa, pb, na, nb, sc = [], [], [], [], [], [], []
    for a, b, ia, ib, dist in candidates:
        if max_distance is not None:
            keep = dist <= max_distance
            ia, ib, dist = ia[keep], ib[keep], dist[keep]
        if len(ia) < min_matches_per_pair:
            continue

        if max_matches_per_pair is not None and len(ia) > max_matches_per_pair:
            best = np.argsort(dist)[:max_matches_per_pair]
            ia, ib, dist = ia[best], ib[best], dist[best]

        fa.append(np.full(len(ia), a, dtype=np.int64))
        fb.append(np.full(len(ia), b, dtype=np.int64))
        pa.append(points[a][ia])
        pb.append(points[b][ib])
        na.append(normals[a][ia])
        nb.append(normals[b][ib])
        sc.append(dist)

    if not fa:
        z = np.empty(0, dtype=np.int64)
        e3 = np.empty((0, 3))
        return MatchSet(z, z, e3, e3, e3, e3, np.empty(0))

    return MatchSet(
        np.concatenate(fa), np.concatenate(fb),
        np.concatenate(pa), np.concatenate(pb),
        np.concatenate(na), np.concatenate(nb),
        np.concatenate(sc),
    )


def correspondence_components(
    frag_a: np.ndarray, frag_b: np.ndarray, num_fragments: int
) -> np.ndarray:
    """Connected components of the fragment correspondence graph.

    Worth checking before solving: translations are determined only up to one
    free global offset PER COMPONENT. A fragment nothing matched against is not
    estimated badly -- it is genuinely unconstrained, and reporting that is far
    more useful than silently returning zero for it.
    """
    parent = np.arange(num_fragments)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(frag_a.tolist(), frag_b.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(num_fragments)], dtype=np.int64)
