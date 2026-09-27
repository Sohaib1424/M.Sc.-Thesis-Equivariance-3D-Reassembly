"""
The largest fragment as the anchor: the frame every loss and metric is read in.

Why there is an anchor at all
-----------------------------
The label for fragment *i* is ``R_i``: the rotation back into the frame its
object happens to be stored in. For an object the model has never seen, that
frame cannot be read off the input. Most Everyday objects are round, so how far
the stored object is turned about its axis is arbitrary, and nothing in the
fragments says what it was. A model asked for ``R_i`` itself can lower that loss
on training shapes only by remembering them -- which is what the official-split
runs showed: validation at chance on every rotation-dependent term while the
training error fell, and the only term that improved on validation was the one
that ignores orientation.

What *is* determined by the input is how the fragments sit relative to one
another. So the benchmark fixes one fragment at its true pose and asks for the
others relative to it. GARF (supplementary C.3, Table VI) and PuzzleFusion++ use
the largest fragment; GARF's training fixes its anchors "with identity
rotations and zero translations".

The operation
-------------
One rotation per scene, ``C = R_a R_hat_a^T``, applied on the left of every
prediction in that scene::

    aligned_i = C R_hat_i          so    aligned_a = R_a   exactly

``C`` is the rotation that puts the anchor where it belongs, applied to the
whole predicted assembly. If every prediction in a scene is off by the same
global rotation, ``R_hat_i = G R_i``, then ``C = G^T`` and every aligned
prediction is exact: the one thing no input determines is removed and nothing
else is. The geodesic error of an aligned prediction is the error of the
*relative* rotation to the anchor (the geodesic is bi-invariant)::

    angle(aligned_i, R_i) = angle(R_hat_a^T R_hat_i, R_a^T R_i)

Left, not right: under this project's convention (``nn/losses.py``) a
prediction maps the perturbed fragment into the assembled frame, and a global
rotation of the assembled result multiplies it on the left.

Which fragment
--------------
The largest by radius -- the fragment whose radius already sets each scene's
normalisation divisor (``data/transforms.normalize_fragments``), so "the
largest" means the same thing everywhere in the pipeline. The radius is
rotation-invariant and known at inference, as an anchor must be. Ties go to the
first fragment. GARF found a random anchor as good as the largest (6.09 vs
6.10 deg), so the rule is a convention rather than a lever.

The anchor's own error is zero by construction, so it is left out of every
average -- counting it would hand each scene one free perfect fragment, which in
a three-fragment scene (the median) is a third of the score.

How the model is affected
-------------------------
Not at all. The network, its cross-fragment rule (equivariant to its own
fragment's pose, invariant to the others') and both stages are unchanged; this
module only decides which rotation the loss and the metrics compare against.
A model trained on the absolute target is scored under this protocol with no
retraining (``--evaluate``), and ``--rotation_target anchor`` trains on it.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def anchor_fragments(log_scale: Tensor, fragment_scene: Tensor,
                     num_scenes: int) -> Tensor:
    """
    ``(S,)`` global index of each scene's anchor -- its largest fragment.

    ``log_scale`` is the batch's ``(F, 1)`` or ``(F,)`` log-radius. Ties go to
    the lowest index. A scene with no fragments gets ``F`` (one past the end),
    which :func:`align_to_anchor` treats as "no anchor".
    """
    scale = log_scale.reshape(-1)
    count = scale.shape[0]
    scene = fragment_scene.long()
    largest = scale.new_full((num_scenes,), float("-inf")).scatter_reduce(
        0, scene, scale, reduce="amax", include_self=True)
    # The first index reaching the maximum, reduced in the scale's own float
    # dtype -- the same float scatter `nn/segment.py` already relies on, on
    # every device -- rather than an integer one. Exact: a batch holds far
    # fewer than 2**24 fragments.
    index = torch.arange(count, device=scale.device).to(scale.dtype)
    candidate = torch.where(scale == largest[scene], index,
                            torch.full_like(index, float(count)))
    first = scale.new_full((num_scenes,), float(count)).scatter_reduce(
        0, scene, candidate, reduce="amin", include_self=True)
    return first.round().long()


def align_to_anchor(rotation: Tensor, target: Tensor, fragment_scene: Tensor,
                    anchor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    ``(aligned, scored)``: every prediction rotated so its scene's anchor is
    exact, and a ``(F,)`` mask that is False on the anchors.

    Differentiable in ``rotation``, including through the anchor's own
    prediction: under the relative target the anchor is the reference every
    other fragment in its scene is scored against, so its prediction is
    trained by all of them.
    """
    count = rotation.shape[0]
    present = anchor < count
    safe = torch.where(present, anchor, torch.zeros_like(anchor))
    if count == 0:
        return rotation, torch.zeros(0, dtype=torch.bool, device=rotation.device)
    correction = torch.matmul(target[safe], rotation[safe].transpose(-1, -2))
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    correction = torch.where(present[:, None, None], correction, eye)
    aligned = torch.matmul(correction[fragment_scene.long()], rotation)
    scored = torch.ones(count, dtype=torch.bool, device=rotation.device)
    scored[safe[present]] = False
    return aligned, scored


def anchor_alignment(batch, rotation: Tensor) -> Tuple[Tensor, Tensor]:
    """:func:`align_to_anchor` for a :class:`~reassembly.data.features.Batch`."""
    anchor = anchor_fragments(batch.log_scale, batch.fragment_scene, batch.num_scenes)
    return align_to_anchor(rotation, batch.target_rotation.to(rotation.dtype),
                           batch.fragment_scene, anchor)


def scored_count(batch) -> int:
    """Fragments that are not an anchor: one fewer per scene that has any."""
    ptr = batch.fragment_ptr
    scenes_with_fragments = int((ptr[1:] > ptr[:-1]).sum()) if ptr.numel() > 1 else 0
    return int(batch.num_fragments) - scenes_with_fragments
