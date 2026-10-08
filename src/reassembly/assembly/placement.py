"""
Stage two, placement: where each fragment goes, from the matches its rotation
was fitted on rather than from every match.

The global solve, and how it fails
----------------------------------
:func:`~reassembly.assembly.translation.assemble` places every fragment at
once, by least squares over every embedding match. That is right when nearly
every match is right. In a scene of many pieces many are not -- matches
between pieces that do not touch -- and the solve answers each one by pulling
its two pieces together. Huber caps a wrong match's pull; it does not remove
it, and a few hundred of them outweigh the right ones. The assembly contracts
onto its centre.

Measured on W10's prediction dumps with matched rotations: the placed Spoon and
Ring spread over 0.10 and 0.21 of their true spread (RMS distance from the
centre; vase, wine bottle, bowl and plate 0.86-0.97). With the same rotations
and the true correspondences instead, the well-turned pieces landed 0.0065 and
0.005 world units from their places (median). The rotations were right; the
placement was not -- which is how a piece can show a small position loss, a
measure of its rotation alone (``nn/losses.py``), and still sit in the wrong
place.

What this module does instead
-----------------------------
The matched route has already sorted the matches: every fitted pair keeps its
RANSAC inliers -- matches that agree with one rigid motion -- and the chain
says which fits it believes. So, for ONE scene:

1. **Verify.** A fitted pair is kept when the chain reached both fragments, it
   has at least ``min_inliers`` inliers, and its relative rotation agrees with
   the chained rotations within ``agreement_deg``. The chain's own edges agree
   by construction; another pair that agrees closes a loop and is kept, and
   one that does not -- a wrong fit, or a pair that does not touch -- is
   dropped with all its matches.
2. **Place the reached fragments** by least squares over the kept pairs'
   inliers alone, with the root held at its own position -- under the
   benchmark protocol the anchor, at its true position -- rather than the
   zero-mean gauge, and Huber reweighting as in the global solve.
3. **Place the rest.** The fragments the chain did not reach keep the head's
   rotation; they are placed from every match that touches them, with the
   reached fragments held where step 2 put them, so a wrong rotation can
   misplace its own fragment and no other. A fragment with no match at all
   goes to the mean position of the reached ones, where the global solve's
   gauge would have put it.

Replayed through :func:`~reassembly.assembly.scoring.score_batch` on those
dumps' meshes and head rotations, with matches made exact at every break and
then a quarter, or half, of them made wrong (pairs of break points on two
random pieces): the share of well-turned pieces placed (part Chamfer < 0.01)
went from 5% (Spoon) and 12% / 0% (Ring) under the global solve to 100%, and
part accuracy from 0.04 to 0.94 / 0.90 (Spoon) and from 0.09 / 0.00 to
0.77 / 0.73 (Ring); what is left is pieces whose rotation is wrong. On the
five scenes of 3-7 pieces the checked placement was never worse. Wrong matches
from a trained network need not look like random ones, so the number that
counts is the re-run evaluation.

Units are those of the points given: world units from
:func:`~reassembly.assembly.scoring.score_batch`, where the Huber scale is
defined, as for the global solve.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from ..nn.losses import geodesic_angle
from .rotation import MIN_MATCHES, PairRotation
from .translation import (Matches, mutual_nearest_neighbours, normal_compatibility,
                          resolve_collisions, solve_translations, subsample_per_fragment)

AGREEMENT_DEG = 5.0
"""A fitted pair whose relative rotation is further than this from the chained
rotations' is not trusted for placement. Fits on correct matches are a tenth of
a degree off (``rotation.py``), so this leaves room for drift along the chain
while a fit to the wrong motion -- a random rotation -- lands within it with a
probability near 1e-5."""


class Placement(NamedTuple):
    """One scene placed."""
    translations: Tensor   # (F, 3): relative to the root's, which is 0
    matches: Matches       # every embedding match -- what the global solve would use
    links: Matches         # the verified pairs' inliers the reached fragments were placed by


def agreement_deg(pair: PairRotation, rotations: Tensor) -> float:
    """
    Degrees between a pair's fitted relative rotation and the one the chained
    ``rotations`` imply. A fit ``R_ij`` carries i's input frame onto j's, so
    the chain agrees with it when ``rotations[i] == rotations[j] @ R_ij``.
    """
    implied = rotations[pair.j].double() @ pair.rotation.double()
    return math.degrees(float(geodesic_angle(rotations[pair.i].double(), implied)))


def verified_pairs(pairs: Sequence[PairRotation], rotations: Tensor, reached: Tensor, *,
                   min_inliers: int = MIN_MATCHES,
                   agreement: float = AGREEMENT_DEG) -> List[PairRotation]:
    """
    The fitted pairs a placement can rest on: both fragments reached by the
    chain, at least ``min_inliers`` inliers, and a relative rotation within
    ``agreement`` degrees of the chained ``rotations``. A pair that carries no
    inlier indices (built by hand, with a rotation only) is never kept.
    """
    kept = []
    for pair in pairs:
        if pair.source is None or pair.target is None:
            continue
        if not (bool(reached[pair.i]) and bool(reached[pair.j])):
            continue
        if pair.inliers < min_inliers or pair.source.numel() < min_inliers:
            continue
        if agreement_deg(pair, rotations) <= agreement:
            kept.append(pair)
    return kept


def solve_with_held(points: Tensor, point_fragment: Tensor, matches: Matches,
                    free: Tensor, held: Tensor, *, iterations: int = 5, huber: float = 0.05,
                    ridge: float = 1e-6, prior: Optional[Tensor] = None) -> Tensor:
    """
    ``(F, 3)`` translations: the ``free`` fragments solved for, every other
    fragment held at its row of ``held``.

    The same energy as :func:`~reassembly.assembly.translation.solve_translations`
    -- matched points should coincide, ``sum w |(p + t_a) - (q + t_b)|^2`` --
    and the same Huber IRLS, but the gauge comes from the held fragments
    instead of a zero mean. Matches between two held fragments change nothing
    and are skipped; a free fragment no match reaches settles at ``prior``
    (default: the mean of the held fragments), which a ``ridge`` too small to
    move a matched fragment pulls it to. ``points`` are already rotated and
    fragment-centred; float64 throughout, as there.
    """
    device = points.device
    dtype = torch.float64
    out = held.to(dtype).clone()
    free = free.to(device=device, dtype=torch.bool)
    index = torch.nonzero(free, as_tuple=False).flatten()
    count = index.numel()
    if count == 0:
        return out.to(points.dtype)
    if prior is None:
        anchored = ~free
        prior = (out[anchored].mean(dim=0) if bool(anchored.any())
                 else torch.zeros(3, dtype=dtype, device=device))
    prior = prior.to(device=device, dtype=dtype)
    position = torch.full(free.shape, -1, dtype=torch.long, device=device)
    position[index] = torch.arange(count, device=device)

    a = point_fragment[matches.source].long()
    b = point_fragment[matches.target].long()
    use = free[a] | free[b]
    a, b = a[use], b[use]
    wanted = (points[matches.target[use]].to(dtype)
              - points[matches.source[use]].to(dtype))      # t_a - t_b should equal q - p
    base = matches.weight[use].to(dtype)
    free_a, free_b = free[a], free[b]
    both = free_a & free_b
    ia, ib = position[a], position[b]
    eye = torch.eye(count, dtype=dtype, device=device)
    weight = base.clone()
    for _ in range(max(int(iterations), 1)):
        system = ridge * eye
        system = system.index_put((ia[free_a], ia[free_a]), weight[free_a], accumulate=True)
        system = system.index_put((ib[free_b], ib[free_b]), weight[free_b], accumulate=True)
        system = system.index_put((ia[both], ib[both]), -weight[both], accumulate=True)
        system = system.index_put((ib[both], ia[both]), -weight[both], accumulate=True)
        # A held end enters the right-hand side at its fixed position.
        held_b = torch.where(free_b[:, None], torch.zeros_like(wanted), out[b])
        held_a = torch.where(free_a[:, None], torch.zeros_like(wanted), out[a])
        rhs = (ridge * prior).expand(count, 3).clone()
        rhs.index_add_(0, ia[free_a], (weight[:, None] * (wanted + held_b))[free_a])
        rhs.index_add_(0, ib[free_b], (weight[:, None] * (held_a - wanted))[free_b])
        out[index] = torch.linalg.solve(system, rhs)
        residual = (out[a] - out[b] - wanted).norm(dim=-1)
        weight = base * torch.where(residual <= huber, torch.ones_like(residual),
                                    huber / residual.clamp_min(1e-12))
    return out.to(points.dtype)


def _links(pairs: Sequence[PairRotation], like: Tensor) -> Matches:
    """The kept pairs' inliers as matches of weight 1: RANSAC has vouched for them."""
    if not pairs:
        empty = torch.zeros(0, dtype=torch.long, device=like.device)
        return Matches(empty, empty, like.new_zeros(0))
    source = torch.cat([p.source for p in pairs]).to(like.device).long()
    target = torch.cat([p.target for p in pairs]).to(like.device).long()
    return Matches(source, target, like.new_ones(source.numel()))


def place(points: Tensor, normals: Tensor, point_fragment: Tensor, embeddings: Tensor,
          num_fragments: int, pairs: Sequence[PairRotation], rotations: Tensor,
          reached: Tensor, root: int, *, candidates: Optional[Tensor] = None,
          max_points: int = 2048, iterations: int = 5, huber: float = 0.05,
          min_inliers: int = MIN_MATCHES, agreement: float = AGREEMENT_DEG,
          collision_radii: Optional[Tensor] = None) -> Placement:
    """
    Verify, place the reached fragments, then the rest -- for ONE scene
    (module docstring).

    ``points`` and ``normals`` are rotated by ``rotations`` and
    fragment-centred, in world units, exactly as
    :func:`~reassembly.assembly.translation.assemble` takes them;
    ``point_fragment`` is ``0..F-1``. ``pairs`` are
    :func:`~reassembly.assembly.rotation.pairwise_rotations`' fits on the same
    points' indices, and ``reached`` and ``root`` the chain's. The
    translations come back relative to the root's.
    """
    device = points.device
    dtype = points.dtype
    reached = reached.to(device=device, dtype=torch.bool)
    translations = torch.zeros(num_fragments, 3, dtype=dtype, device=device)

    # Every embedding match, as the global solve finds and weights them: the
    # count is reported either way, and step 3 places the unreached from them.
    keep = subsample_per_fragment(point_fragment, max_points, candidates)
    source, target = mutual_nearest_neighbours(embeddings[keep], point_fragment[keep])
    source, target = keep[source], keep[target]
    weight = (normal_compatibility(normals[source], normals[target]) if source.numel()
              else points.new_zeros(0))
    matches = Matches(source, target, weight)

    if not 0 <= root < num_fragments:
        # No chain to rest on (no fragment at all): the global solve's answer.
        translations = solve_translations(points, point_fragment, matches, num_fragments,
                                          iterations=iterations, huber=huber)
        return Placement(translations, matches, _links([], points))

    kept = verified_pairs(pairs, rotations, reached, min_inliers=min_inliers,
                          agreement=agreement)
    links = _links(kept, points)
    # 2. The reached fragments, from the verified inliers, the root held at 0.
    free = reached.clone()
    free[root] = False
    translations = solve_with_held(points, point_fragment, links, free, translations,
                                   iterations=iterations, huber=huber,
                                   prior=torch.zeros(3, dtype=torch.float64, device=device))
    # 3. The rest, from every match that touches them, the reached held.
    free = ~reached
    free[root] = False
    if bool(free.any()):
        translations = solve_with_held(points, point_fragment, matches, free, translations,
                                       iterations=iterations, huber=huber,
                                       prior=translations[reached].double().mean(dim=0))
    if collision_radii is not None:
        translations = resolve_collisions(translations, collision_radii)
        translations = translations - translations[root]
    return Placement(translations, matches, links)
