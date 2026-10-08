"""
Stage two, rotations: each fragment's rotation from the embedding matches, by
geometry alone.

The network answers "how is this fragment turned?" twice: once with its
rotation head, and once, implicitly, with the per-vertex embedding that the
translation solver already matches across fragments. On unseen shapes only the
second answer works. Measured on a trained run (W10, epoch 334, 1023
validation scenes, ``probe_val.py --procrustes``): between fragments that
touch, the head's relative rotation was 98.7 deg off at the median -- near
chance -- while the rotation that lines up the two fragments' *matched* points
was 0.1 deg off, and chaining those from the anchor took the validation error
from 102.0 to 14.8 deg (median 0.3). The embedding generalises; the head does
not. This module is the second answer, made explicit.

Pipeline, for one scene
-----------------------
1. **Match.** Mutual nearest neighbours in embedding space between
   fracture-surface vertices of different fragments -- the same matches
   :func:`~reassembly.assembly.translation.assemble` uses.
2. **Fit.** For every pair of fragments with at least ``min_matches``
   matches, the rotation that carries one fragment's matched points onto the
   other's (Kabsch), with RANSAC against the wrong matches. The points are the
   network's *input* coordinates, so the fit is a relative rotation between
   the two input frames and does not involve the rotation head at all.
3. **Chain.** Outward from a root fragment whose rotation is known -- under the
   benchmark protocol the anchor, set to its true pose -- along a maximum
   spanning tree whose edge weights are the RANSAC inlier counts. A fragment
   the tree does not reach keeps the rotation it was given (the head's).

The inlier count is a safe edge weight. On the run above, fragment pairs that
really touch had a median of 38 inliers (10th percentile 9) and pairs that do
not a median of 3 (90th percentile 5); every edge the tree chose was a real
contact. ``min_matches`` is the floor for both a fit and an edge.

Each fit keeps its whole rigid motion -- the offset beside the rotation -- and
the matches that agree with it. The rotation chain needs only the rotation;
:mod:`reassembly.assembly.placement` places the fragments from the rest.

Why the fit can be exact here -- and what that means for comparisons
--------------------------------------------------------------------
A Breaking Bad object's fragments are cut from one mesh, so the two sides of a
break share the *same* vertices: a correct match is the same physical point on
both fragments, and Kabsch on correct matches is exact. Pipelines that sample
points independently on each fragment (GARF and most published tables) never
see coincident points, so a number from this route is not directly comparable
with theirs without saying so.

Units: :func:`pairwise_rotations` expects every fragment's points in one scale,
the scene's largest fragment's radius (``tau`` is in those units), which is what
:func:`~reassembly.assembly.scoring.score_batch` passes.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .translation import mutual_nearest_neighbours, subsample_per_fragment

MIN_MATCHES = 6
"""Matches a pair needs to be fitted, and RANSAC inliers a fit needs to be an edge."""
INLIER_DISTANCE = 0.05
"""A match agrees with a fitted motion within this distance, in largest-fragment radii."""
RANSAC_ITERATIONS = 256
"""Hypotheses per pair: an all-correct triple is drawn with probability over 0.99
when as few as 27% of the matches are right, ``1 - (1 - 0.27**3)**256``."""


class PairRotation(NamedTuple):
    """
    One fitted pair of fragments (local indices, ``i < j``): the motion
    ``q = rotation @ p + shift`` carries a point ``p`` of fragment i's input
    frame onto its partner ``q`` in fragment j's, in the units the fit was
    given. ``source`` and ``target`` index the matches that agree with it --
    ``inliers`` of them -- into the points :func:`pairwise_rotations` was
    given. The last three are ``None`` for a pair built by hand with only a
    rotation, which the chain needs and nothing else does.
    """
    i: int
    j: int
    rotation: Tensor     # (3, 3): carries fragment i's input frame onto fragment j's
    matches: int
    inliers: int
    shift: Optional[Tensor] = None     # (3,)
    source: Optional[Tensor] = None    # (inliers,) points of fragment i
    target: Optional[Tensor] = None    # (inliers,) their partners on fragment j


def kabsch(source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """
    ``(R, t)`` minimising ``sum ||R p + t - q||^2`` over the points (the
    second-to-last axis), batched over any leading axes.

    Always a proper rotation: when the best orthogonal fit is a reflection --
    coplanar or degenerate points -- the last singular direction is flipped.
    """
    centre_s, centre_t = source.mean(-2), target.mean(-2)
    covariance = ((source - centre_s.unsqueeze(-2)).transpose(-1, -2)
                  @ (target - centre_t.unsqueeze(-2)))
    U, _, Vt = torch.linalg.svd(covariance)
    V, Ut = Vt.transpose(-1, -2), U.transpose(-1, -2)
    sign = torch.sign(torch.linalg.det(V @ Ut))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    ones = torch.ones_like(sign)
    rotation = V @ torch.diag_embed(torch.stack([ones, ones, sign], dim=-1)) @ Ut
    shift = centre_t - (rotation @ centre_s.unsqueeze(-1)).squeeze(-1)
    return rotation, shift


def ransac_motion(source: Tensor, target: Tensor, *, iterations: int = RANSAC_ITERATIONS,
                  tau: float = INLIER_DISTANCE,
                  generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    ``(rotation, shift, inliers)``: the rigid motion carrying ``source`` onto
    ``target`` that the most matches agree with, refitted on those matches,
    and the ``(M,)`` mask of the ones that agree (within ``tau`` once moved).
    Three matches fix a rigid motion, so each hypothesis is fitted on three
    drawn without replacement.
    """
    count = source.shape[0]
    if count < 3:
        raise ValueError(f"a rotation needs at least three matches, got {count}")
    weights = torch.ones(iterations, count, device=source.device)
    draws = torch.multinomial(weights, 3, replacement=False, generator=generator)
    rotations, shifts = kabsch(source[draws], target[draws])          # (K, 3, 3), (K, 3)
    moved = torch.einsum("kij,mj->kmi", rotations, source) + shifts[:, None, :]
    agree = (moved - target[None]).norm(dim=-1) < tau                 # (K, M)
    best = int(agree.sum(dim=1).argmax())
    rotation, shift, inliers = rotations[best], shifts[best], agree[best]
    for _ in range(2):                         # refit on the agreeing matches
        if int(inliers.sum()) < 3:
            break
        rotation, shift = kabsch(source[inliers], target[inliers])
        inliers = (source @ rotation.transpose(-1, -2) + shift - target).norm(dim=-1) < tau
    return rotation, shift, inliers


def ransac_rotation(source: Tensor, target: Tensor, *, iterations: int = RANSAC_ITERATIONS,
                    tau: float = INLIER_DISTANCE,
                    generator: Optional[torch.Generator] = None) -> Tuple[Tensor, int]:
    """
    The rotation of :func:`ransac_motion` and how many matches agree with it:
    the same draws, so the same rotation.
    """
    rotation, _, inliers = ransac_motion(source, target, iterations=iterations, tau=tau,
                                         generator=generator)
    return rotation, int(inliers.sum())


def pairwise_rotations(points: Tensor, point_fragment: Tensor, embeddings: Tensor,
                       num_fragments: int, *, candidates: Optional[Tensor] = None,
                       max_points: int = 2048, min_matches: int = MIN_MATCHES,
                       tau: float = INLIER_DISTANCE, iterations: int = RANSAC_ITERATIONS,
                       generator: Optional[torch.Generator] = None) -> List[PairRotation]:
    """
    Every pair of fragments of ONE scene with at least ``min_matches`` matches,
    fitted. ``points`` are the fragments' input coordinates in one common scale
    (module docstring); ``point_fragment`` is ``0..F-1``; ``candidates`` limits
    the matching to the fracture surface, as in the translation solver. Each
    pair carries its whole motion and the indices of its inliers.
    """
    keep = subsample_per_fragment(point_fragment, max_points, candidates)
    source, target = mutual_nearest_neighbours(embeddings[keep], point_fragment[keep])
    source, target = keep[source], keep[target]
    if source.numel() == 0:
        return []
    a, b = point_fragment[source].long(), point_fragment[target].long()
    swap = a > b                               # order every match low -> high fragment
    first = torch.where(swap, target, source)
    second = torch.where(swap, source, target)
    pair = torch.minimum(a, b) * num_fragments + torch.maximum(a, b)
    points = points.double()
    fitted: List[PairRotation] = []
    for key in torch.unique(pair).tolist():
        chosen = pair == key
        count = int(chosen.sum())
        if count < max(min_matches, 3):
            continue
        i, j = divmod(int(key), num_fragments)
        on_i, on_j = first[chosen], second[chosen]
        rotation, shift, agree = ransac_motion(points[on_i], points[on_j],
                                               iterations=iterations, tau=tau,
                                               generator=generator)
        fitted.append(PairRotation(i, j, rotation, count, int(agree.sum()), shift,
                                   on_i[agree], on_j[agree]))
    return fitted


def chain_rotations(pairs: Sequence[PairRotation], rotations: Tensor, root: int,
                    min_inliers: int = MIN_MATCHES) -> Tuple[Tensor, Tensor]:
    """
    ``rotations`` ``(F, 3, 3)`` with every fragment reachable from ``root``
    replaced by the fitted rotations composed along the way, and the ``(F,)``
    mask of the fragments so placed (``root`` included). ``rotations[root]``
    is trusted; the others are only fallbacks.

    The tree grows from the root by the largest inlier count across its
    boundary (Prim's algorithm for a maximum spanning tree), so each fragment
    is placed through the best-supported path there is.

    Composition: a fitted ``R_ij`` carries i's input frame onto j's, and a
    fragment's rotation carries its input frame into the assembled one, so a
    child ``j`` of a placed ``i`` gets ``rotations[i] @ R_ij^T`` and a child
    ``i`` of a placed ``j`` gets ``rotations[j] @ R_ij``.
    """
    count = rotations.shape[0]
    chained = rotations.clone()
    reached = torch.zeros(count, dtype=torch.bool, device=rotations.device)
    if not 0 <= root < count:
        return chained, reached
    reached[root] = True
    edges = [p for p in pairs if p.inliers >= min_inliers]
    placed = {root: rotations[root].double()}
    while True:
        best = None
        for p in edges:
            if (p.i in placed) != (p.j in placed) and (best is None or p.inliers > best.inliers):
                best = p
        if best is None:
            break
        fitted = best.rotation.double()
        if best.i in placed:
            new, pose = best.j, placed[best.i] @ fitted.transpose(-1, -2)
        else:
            new, pose = best.i, placed[best.j] @ fitted
        placed[new] = pose
        chained[new] = pose.to(chained.dtype)
        reached[new] = True
    return chained, reached


def match_rotations(points: Tensor, point_fragment: Tensor, embeddings: Tensor,
                    rotations: Tensor, root: int, *, candidates: Optional[Tensor] = None,
                    max_points: int = 2048, min_matches: int = MIN_MATCHES,
                    tau: float = INLIER_DISTANCE, iterations: int = RANSAC_ITERATIONS,
                    generator: Optional[torch.Generator] = None
                    ) -> Tuple[Tensor, Tensor, List[PairRotation]]:
    """Fit and chain, for ONE scene: ``(rotations, reached, pairs)``."""
    pairs = pairwise_rotations(points, point_fragment, embeddings, rotations.shape[0],
                               candidates=candidates, max_points=max_points,
                               min_matches=min_matches, tau=tau, iterations=iterations,
                               generator=generator)
    chained, reached = chain_rotations(pairs, rotations, root, min_matches)
    return chained, reached, pairs
