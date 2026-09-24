"""
Stage two: translations from the predicted rotations, by geometry alone.

The network predicts orientation only. Once every fragment is correctly
oriented, placing it is a well-posed least-squares problem rather than
something to learn -- the bet behind splitting reassembly in two instead of
regressing SE(3) jointly the way GARF does. This module is that second stage,
and it is what turns a rotation model into an assembly that Breaking Bad's
translation RMSE, Chamfer distance and part accuracy can be computed on.

Pipeline
--------
1. **Match.** Mutual nearest neighbours in the network's *invariant* per-vertex
   embedding, between fracture-surface vertices of *different* fragments.
   Invariant is the point: the two sides of an interface are matched while the
   fragments are still apart, because the embedding does not depend on pose.
2. **Weight.** A matched pair on a real interface has opposed normals, so
   ``(1 - cos) / 2`` of the two (rotated) normals weights each match.
3. **Solve.** With rotations fixed, making matched points coincide,

       E(t) = sum_m w_m || (p_m + t_a(m)) - (q_m + t_b(m)) ||^2 ,

   is linear least squares in the translations: its normal equations are a
   weighted graph Laplacian over the fragments. The gauge (adding one constant
   to every translation changes nothing) is fixed by pinning their mean to
   zero, and wrong matches are handled by iteratively reweighted least squares
   with a Huber weight, so a handful of them cannot drag the assembly.

A normal term does **not** appear in ``E``, and that is deliberate: normals do
not change under translation, so any energy in them has zero gradient with
respect to ``t``. What they genuinely measure is whether a *match* is
plausible, which is where they are used.

Everything is in **world units**: the Huber scale here and the benchmark's
0.01 part-accuracy threshold are both defined there, while the network works
in per-scene normalised coordinates. :mod:`reassembly.assembly.scoring` does
the conversion with the divisor and centroids the batch carries.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor


class Matches(NamedTuple):
    """Correspondences between points of different fragments."""
    source: Tensor       # (M,) index into the point array
    target: Tensor       # (M,)
    weight: Tensor       # (M,) in [0, 1]


def subsample_per_fragment(point_fragment: Tensor, cap: int,
                           candidates: Optional[Tensor] = None) -> Tensor:
    """
    Indices of at most ``cap`` points per fragment, evenly strided -- so the
    same scene always gives the same subset (a random draw would make two
    evaluations of one checkpoint disagree for no reason).

    ``candidates`` (a boolean mask) restricts the choice, e.g. to the fracture
    surface; a fragment with no candidate at all falls back to every one of its
    points, because a fragment without matches is a fragment the solver cannot
    place.
    """
    device = point_fragment.device
    if point_fragment.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    keep = []
    for fragment in torch.unique(point_fragment):
        own = point_fragment == fragment
        pool = own & candidates if candidates is not None else own
        if candidates is not None and not bool(pool.any()):
            pool = own
        index = torch.nonzero(pool, as_tuple=False).flatten()
        if cap > 0 and index.numel() > cap:
            index = index[torch.linspace(0, index.numel() - 1, cap,
                                         device=device).round().long()]
        keep.append(index)
    return torch.cat(keep)


def mutual_nearest_neighbours(embeddings: Tensor, point_fragment: Tensor,
                              chunk: int = 2048) -> tuple:
    """
    Mutual nearest neighbours in embedding space, between points of DIFFERENT
    fragments. Each pair is returned once. Chunked over rows, so the full
    pairwise matrix is never held.
    """
    device = embeddings.device
    n = embeddings.shape[0]
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if n < 2:
        return empty, empty
    embeddings = embeddings.float()
    best = torch.empty(n, dtype=torch.long, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        distance = torch.cdist(embeddings[start:stop], embeddings)
        same = point_fragment[start:stop, None] == point_fragment[None, :]
        best[start:stop] = distance.masked_fill(same, float("inf")).argmin(dim=1)
    index = torch.arange(n, device=device)
    mutual = (best[best] == index) & (point_fragment[best] != point_fragment)
    source, target = index[mutual], best[mutual]
    once = source < target
    return source[once], target[once]


def normal_compatibility(normal_a: Tensor, normal_b: Tensor) -> Tensor:
    """1 for exactly opposed normals (a plausible interface), 0 for aligned."""
    cosine = torch.nn.functional.cosine_similarity(normal_a, normal_b, dim=-1)
    return ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)


def solve_translations(points: Tensor, point_fragment: Tensor, matches: Matches,
                       num_fragments: int, iterations: int = 5,
                       huber: float = 0.05, ridge: float = 1e-6) -> Tensor:
    """
    ``(F, 3)`` translations, zero-mean, by IRLS on the weighted Laplacian.

    ``points`` are already rotated and fragment-centred. In float64 throughout:
    the normal equations of a sparsely connected assembly are ill-conditioned,
    and float32 loses the small fragments first.
    """
    device = points.device
    if matches.source.numel() == 0 or num_fragments < 2:
        return torch.zeros(num_fragments, 3, dtype=points.dtype, device=device)
    dtype = torch.float64
    a = point_fragment[matches.source].long()
    b = point_fragment[matches.target].long()
    p = points[matches.source].to(dtype)
    q = points[matches.target].to(dtype)
    base = matches.weight.to(dtype)
    wanted = q - p                         # t_a - t_b should equal q - p
    weight = base.clone()
    eye = torch.eye(num_fragments, dtype=dtype, device=device)
    translations = torch.zeros(num_fragments, 3, dtype=dtype, device=device)
    for _ in range(max(int(iterations), 1)):
        laplacian = torch.zeros(num_fragments, num_fragments, dtype=dtype, device=device)
        laplacian.index_put_((a, a), weight, accumulate=True)
        laplacian.index_put_((b, b), weight, accumulate=True)
        laplacian.index_put_((a, b), -weight, accumulate=True)
        laplacian.index_put_((b, a), -weight, accumulate=True)
        rhs = torch.zeros(num_fragments, 3, dtype=dtype, device=device)
        rhs.index_add_(0, a, weight[:, None] * wanted)
        rhs.index_add_(0, b, -weight[:, None] * wanted)
        # ridge + 1/F: the second term adds (1/F) 1 1^T, which pins sum(t) = 0
        # and makes a connected Laplacian invertible; the ridge covers
        # fragments no match reaches.
        system = laplacian + ridge * eye + 1.0 / num_fragments
        translations = torch.linalg.solve(system, rhs)
        translations = translations - translations.mean(dim=0, keepdim=True)
        residual = ((p + translations[a]) - (q + translations[b])).norm(dim=-1)
        weight = base * torch.where(residual <= huber, torch.ones_like(residual),
                                    huber / residual.clamp_min(1e-12))
    return translations.to(points.dtype)


def resolve_collisions(translations: Tensor, radii: Tensor, iterations: int = 20,
                       step: float = 0.5) -> Tensor:
    """
    Optional: push apart fragments whose bounding spheres overlap.

    A cheap stand-in for a volumetric penetration term. Off by default, because
    it trades interface alignment for visual plausibility -- the wrong trade
    when the numbers are what is being reported.
    """
    t = translations.clone()
    for _ in range(iterations):
        difference = t[:, None, :] - t[None, :, :]
        distance = difference.norm(dim=-1)
        overlap = (radii[:, None] + radii[None, :] - distance).clamp_min(0.0)
        overlap.fill_diagonal_(0.0)
        if float(overlap.max()) < 1e-6:
            break
        direction = difference / distance.clamp_min(1e-9).unsqueeze(-1)
        t = t + step * 0.5 * (direction * overlap.unsqueeze(-1)).sum(dim=1)
    return t - t.mean(dim=0, keepdim=True)


def assemble(points: Tensor, normals: Tensor, point_fragment: Tensor,
             embeddings: Tensor, num_fragments: int, *,
             candidates: Optional[Tensor] = None, max_points: int = 2048,
             iterations: int = 5, huber: float = 0.05,
             collision_radii: Optional[Tensor] = None) -> tuple:
    """
    Match, weight, solve -- for ONE scene.

    ``points`` and ``normals`` must already be rotated by the predicted
    rotations (and fragment-centred, in world units); ``point_fragment`` is
    ``0..F-1``. Returns ``(translations, matches)``: the match count is worth
    reporting, since an assembly resting on a handful of matches is not one
    to trust.
    """
    keep = subsample_per_fragment(point_fragment, max_points, candidates)
    source, target = mutual_nearest_neighbours(embeddings[keep], point_fragment[keep])
    source, target = keep[source], keep[target]
    weight = (normal_compatibility(normals[source], normals[target]) if source.numel()
              else points.new_zeros(0))
    matches = Matches(source, target, weight)
    translations = solve_translations(points, point_fragment, matches, num_fragments,
                                      iterations=iterations, huber=huber)
    if collision_radii is not None:
        translations = resolve_collisions(translations, collision_radii)
    return translations, matches
