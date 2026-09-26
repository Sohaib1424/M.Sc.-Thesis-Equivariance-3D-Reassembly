"""
Stage 2: the classical translation solver.

The network predicts orientation only. Once every fragment is correctly
oriented, recovering the translations is a well-posed geometric optimisation
rather than something that needs to be learned -- which is the whole bet
behind splitting the problem in two instead of regressing SE(3) jointly the
way GARF does.

Pipeline
--------
1. Mutual-nearest-neighbour matching in the INVARIANT embedding space between
   interface points of different fragments (the embeddings are invariant, so
   two sides of the same interface can be matched while still misaligned).
2. Solve for the per-fragment translations that make matched points coincide.
3. Optionally push apart fragments that ended up interpenetrating.

A CORRECTION TO THE DESIGN DOCUMENT'S ENERGY
--------------------------------------------
The document minimises `E_pos + l1 * E_normal + l2 * E_collision` jointly over
the translations, with

    E_normal = sum_i || n_i^(A) + n_i^(B) ||^2 .

E_normal does not depend on the translations at all. Normals are invariant
under translation, so its gradient with respect to every t is identically zero
-- including it in the objective changes nothing, and reporting it as part of
a "solved" energy is misleading. What that term genuinely measures is whether
a candidate CORRESPONDENCE is plausible: matching fracture surfaces should
face each other, i.e. n_A ~ -n_B. So it is used here as a per-match WEIGHT
rather than as an energy over t, which is where the information in it actually
belongs.

With that, `E_pos` alone is a weighted linear least squares in t:

    E(t) = sum_m w_m || (p_m + t_{a_m}) - (q_m + t_{b_m}) ||^2

whose normal equations form a weighted graph Laplacian, solved in closed form.
Rotations are already fixed, so no alternation is required. The gauge freedom
(adding a constant to every t) is fixed by pinning the mean to zero. Wrong
matches are handled by iteratively reweighted least squares with a Huber
weight, which is what stops a handful of bad correspondences from dragging the
whole assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Matches:
    """Mutual-NN correspondences between points of different fragments."""

    src_idx: torch.Tensor     # (M,) index into the point array
    dst_idx: torch.Tensor     # (M,)
    weight: torch.Tensor      # (M,) confidence in [0, 1]


def mutual_nearest_neighbours(
    embeddings: torch.Tensor,
    point_frag: torch.Tensor,
    max_points_per_fragment: int = 4096,
    chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mutual nearest neighbours in embedding space, restricted to pairs from
    DIFFERENT fragments.

    Distances are computed in chunks: the full pairwise matrix over every
    interface point of a scene would be tens of thousands squared.
    `max_points_per_fragment` subsamples very dense fragments -- this affects
    only the inference-time solver, never the data the network is trained on.
    """
    device = embeddings.device
    keep = _subsample_per_fragment(point_frag, max_points_per_fragment)
    emb = embeddings[keep]
    frag = point_frag[keep]
    n = emb.shape[0]
    if n < 2:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    best_idx = torch.zeros(n, dtype=torch.long, device=device)
    best_val = torch.full((n,), float("inf"), device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dist = torch.cdist(emb[start:stop], emb)                     # (c, n)
        same = frag[start:stop, None] == frag[None, :]
        dist = dist.masked_fill(same, float("inf"))
        val, idx = dist.min(dim=1)
        best_val[start:stop] = val
        best_idx[start:stop] = idx

    arange = torch.arange(n, device=device)
    mutual = best_idx[best_idx] == arange
    src_local = arange[mutual]
    dst_local = best_idx[mutual]
    ordered = src_local < dst_local                                   # keep each pair once
    src_local, dst_local = src_local[ordered], dst_local[ordered]
    return keep[src_local], keep[dst_local]


def _subsample_per_fragment(point_frag: torch.Tensor, cap: int) -> torch.Tensor:
    if cap <= 0:
        return torch.arange(point_frag.shape[0], device=point_frag.device)
    keep = []
    for frag_id in torch.unique(point_frag):
        idx = torch.nonzero(point_frag == frag_id, as_tuple=False).flatten()
        if idx.numel() > cap:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]
        keep.append(idx)
    return torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long, device=point_frag.device)


def normal_compatibility(n_src: torch.Tensor, n_dst: torch.Tensor) -> torch.Tensor:
    """
    Per-match weight from the normal-cancellation criterion: 1 when the two
    normals are exactly opposed (a plausible interface), 0 when they are
    aligned. This is where `E_normal` is actually used -- see the module
    docstring for why it cannot serve as an energy over the translations.
    """
    cos = torch.nn.functional.cosine_similarity(n_src, n_dst, dim=-1)
    return ((1.0 - cos) * 0.5).clamp(0.0, 1.0)


def solve_translations(
    points: torch.Tensor,
    point_frag: torch.Tensor,
    matches: Matches,
    num_fragments: int,
    irls_iterations: int = 5,
    huber_delta: float = 0.05,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted least-squares translations, refined by IRLS.

    points:     (P, 3) already-rotated, fragment-centralised points
    point_frag: (P,) fragment id per point
    Returns:    (F, 3) translations, gauge-fixed to zero mean.
    """
    device = points.device
    dtype = torch.float64                       # the normal equations are ill-conditioned in fp32
    if matches.src_idx.numel() == 0:
        return torch.zeros(num_fragments, 3, device=device, dtype=points.dtype)

    a = point_frag[matches.src_idx].long()
    b = point_frag[matches.dst_idx].long()
    p = points[matches.src_idx].to(dtype)
    q = points[matches.dst_idx].to(dtype)
    base_w = matches.weight.to(dtype)
    residual_target = q - p                     # want t_a - t_b = q - p

    weights = base_w.clone()
    translations = torch.zeros(num_fragments, 3, device=device, dtype=dtype)

    for _ in range(max(1, irls_iterations)):
        # Weighted graph Laplacian: L t = rhs, with a ridge term and a
        # zero-mean constraint to remove the global translation gauge.
        L = torch.zeros(num_fragments, num_fragments, device=device, dtype=dtype)
        rhs = torch.zeros(num_fragments, 3, device=device, dtype=dtype)
        L.index_put_((a, a), weights, accumulate=True)
        L.index_put_((b, b), weights, accumulate=True)
        L.index_put_((a, b), -weights, accumulate=True)
        L.index_put_((b, a), -weights, accumulate=True)
        rhs.index_add_(0, a, weights[:, None] * residual_target)
        rhs.index_add_(0, b, -weights[:, None] * residual_target)

        L = L + ridge * torch.eye(num_fragments, device=device, dtype=dtype)
        L = L + (1.0 / num_fragments)           # pins sum(t) = 0, making L invertible
        translations = torch.linalg.solve(L, rhs)
        translations = translations - translations.mean(dim=0, keepdim=True)

        residual = (p + translations[a]) - (q + translations[b])
        norm = residual.norm(dim=-1)
        huber = torch.where(
            norm <= huber_delta,
            torch.ones_like(norm),
            huber_delta / norm.clamp_min(1e-12),
        )
        weights = base_w * huber

    return translations.to(points.dtype)


def resolve_collisions(
    translations: torch.Tensor,
    centroids: torch.Tensor,
    radii: torch.Tensor,
    iterations: int = 20,
    step: float = 0.5,
) -> torch.Tensor:
    """
    Optional collision relief: a few iterations of pairwise push-apart on
    bounding spheres.

    A deliberately cheap stand-in for the design document's volumetric
    `E_collision`. True volume-intersection is non-convex and would need a
    mesh-level penetration depth per pair per step; on Breaking Bad fragments
    the residual overlap after the least-squares solve is small, so a sphere
    approximation removes visible interpenetration for a negligible cost. Off
    by default -- it trades a small amount of interface-alignment accuracy for
    visual plausibility, which is the wrong trade when reporting metrics.
    """
    t = translations.clone()
    for _ in range(iterations):
        centres = centroids + t
        diff = centres[:, None, :] - centres[None, :, :]
        dist = diff.norm(dim=-1)
        min_dist = radii[:, None] + radii[None, :]
        overlap = (min_dist - dist).clamp_min(0.0)
        overlap.fill_diagonal_(0.0)
        if float(overlap.max()) < 1e-6:
            break
        direction = diff / dist.clamp_min(1e-9).unsqueeze(-1)
        push = (direction * overlap.unsqueeze(-1)).sum(dim=1)
        t = t + step * 0.5 * push
    return t - t.mean(dim=0, keepdim=True)


def assemble(
    points: torch.Tensor,
    normals: torch.Tensor,
    point_frag: torch.Tensor,
    embeddings: torch.Tensor,
    num_fragments: int,
    max_points_per_fragment: int = 4096,
    irls_iterations: int = 5,
    collision: bool = False,
    centroids: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Matches]:
    """
    End-to-end stage 2: match, weight, solve.

    `points`/`normals` must already be ROTATED by the network's predicted
    rotations (and still fragment-centralised). Returns the (F, 3)
    translations and the matches used, so the caller can inspect match count
    and quality -- both are strong predictors of whether an assembly is
    trustworthy.
    """
    src, dst = mutual_nearest_neighbours(
        embeddings, point_frag, max_points_per_fragment=max_points_per_fragment
    )
    weight = normal_compatibility(normals[src], normals[dst]) if src.numel() else torch.zeros(0, device=points.device)
    matches = Matches(src_idx=src, dst_idx=dst, weight=weight)

    translations = solve_translations(
        points, point_frag, matches, num_fragments, irls_iterations=irls_iterations
    )
    if collision and centroids is not None and radii is not None:
        translations = resolve_collisions(translations, centroids, radii)
    return translations, matches
