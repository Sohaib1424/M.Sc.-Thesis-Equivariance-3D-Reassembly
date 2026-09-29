"""
The largest fragment as the anchor: the frame predictions are scored in.

Why
---
`R_gt` is each fragment's rotation back into the frame its object happens to
be stored in. For a shape the model has never seen, that frame cannot be read
off the input: most Everyday objects are round, so how far the stored object
is turned about its axis is arbitrary. A model asked for `R_gt` itself can
lower that loss on training shapes only by remembering them -- which is what
Thesis 1 and My Thesis Work showed on the official split: validation at chance
on every rotation-dependent term while training fell.

What IS determined by the input is how the fragments sit relative to one
another. So the benchmark fixes one fragment at its true pose and scores the
others relative to it (GARF supplementary C.3 and Table VI; PuzzleFusion++).

The operation
-------------
Under this package's convention (`vngat.training.bridge`) a prediction acts on
column vectors, `x_pred = R_pred x_diffused`, so a global rotation `G` of the
whole predicted assembly multiplies every prediction ON THE LEFT. One rotation
per scene undoes it::

    C = R_gt[a] R_pred[a]^T          aligned_i = C R_pred_i       aligned_a = R_gt[a]

If every prediction in a scene is off by the same `G`, `C = G^T` and every
aligned prediction is exact: the one thing no input determines is removed and
nothing else is. Because the geodesic is bi-invariant, the aligned error IS the
error of the rotation relative to the anchor::

    angle(aligned_i, R_gt_i) = angle(R_pred_a^T R_pred_i, R_gt_a^T R_gt_i)

Which fragment
--------------
The largest by radius -- `frag_log_scale` is the log world radius, and the
largest radius is what `normalize_mode='scene'` already divides each scene by,
so "the largest" means the same thing everywhere. Rotation-invariant and known
at inference, as an anchor must be. Ties go to the first fragment. The anchor's
aligned error is zero by construction, so it is left out of every average,
where it would be a free perfect score (a third of the median scene).

The model is untouched: its symmetry -- equivariant to its own fragment's pose,
invariant to the others' (`test_rotating_one_fragment_leaves_the_others_alone`)
-- is the same under either target. Only what the prediction is compared
against changes.
"""
from __future__ import annotations

from typing import Tuple

import torch


def _float(x: torch.Tensor) -> torch.Tensor:
    """At least float32: the correction is a product of two rotations, and a
    half-precision one would put ~1e-3 of error into every aligned fragment."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def anchor_fragments(frag_log_scale: torch.Tensor, frag_scene: torch.Tensor,
                     num_scenes: int) -> torch.Tensor:
    """
    `(S,)` global index of each scene's anchor -- its largest fragment.

    Ties go to the lowest index. A scene with no fragments gets `F` (one past
    the end), which `align_to_anchor` reads as "no anchor". Float scatters
    only, as elsewhere in this package; exact for any batch below 2**24
    fragments.
    """
    scale = _float(frag_log_scale.reshape(-1))
    count = scale.shape[0]
    scene = frag_scene.long()
    largest = scale.new_full((num_scenes,), float("-inf")).scatter_reduce(
        0, scene, scale, reduce="amax", include_self=True)
    index = torch.arange(count, device=scale.device).to(scale.dtype)
    candidate = torch.where(scale == largest[scene], index,
                            torch.full_like(index, float(count)))
    first = scale.new_full((num_scenes,), float(count)).scatter_reduce(
        0, scene, candidate, reduce="amin", include_self=True)
    return first.round().long()


def align_to_anchor(R_pred: torch.Tensor, R_gt: torch.Tensor, frag_scene: torch.Tensor,
                    anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    `(aligned, scored)`: every prediction rotated so its scene's anchor is
    exact, and an `(F,)` mask that is False on the anchors.

    Differentiable in `R_pred`, including through the anchor's own prediction:
    under the anchor target the anchor is what its scene is scored against,
    so it is trained by every other fragment in the scene.
    """
    R = _float(R_pred)
    G = _float(R_gt).to(R.dtype)
    count = R.shape[0]
    scored = torch.ones(count, dtype=torch.bool, device=R.device)
    if count == 0:
        return R, scored
    present = anchor < count
    safe = torch.where(present, anchor, torch.zeros_like(anchor))
    # Outside autocast: under AMP it would run these products in half
    # precision whatever their inputs are.
    with torch.autocast(device_type=R.device.type, enabled=False):
        correction = torch.matmul(G[safe], R[safe].transpose(-1, -2))
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        correction = torch.where(present[:, None, None], correction, eye)
        aligned = torch.matmul(correction[frag_scene.long()], R)
    scored[safe[present]] = False
    return aligned, scored


def anchor_alignment(R_pred: torch.Tensor, R_gt: torch.Tensor,
                     graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """`align_to_anchor` for a `SceneBatch` (its `frag_log_scale`, `frag_scene`)."""
    anchor = anchor_fragments(graph.frag_log_scale, graph.frag_scene, graph.num_scenes)
    return align_to_anchor(R_pred, R_gt, graph.frag_scene, anchor)


def scored_fragments(graph) -> int:
    """Fragments that are not an anchor: one fewer per scene that has any."""
    scenes = int(torch.unique(graph.frag_scene).numel()) if graph.num_fragments else 0
    return int(graph.num_fragments) - scenes
