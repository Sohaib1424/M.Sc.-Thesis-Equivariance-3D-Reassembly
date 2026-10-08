"""
The training loss, the four geometric scores, and the reference values every
one of them is checked against.

Since v7 the network is trained on one term: the contrastive embedding term
(:func:`correspondence_loss`). The four geometric terms -- rotation, position,
normal, face -- are *scores*: they measure the rotations stage two fits from
the embedding matches (``assembly/rotation.py``), are computed without a
gradient, and never enter the total. :class:`ReassemblyLoss` returns both.

A geometric term that is silently wrong still reads like a number. The defence
is to know what each reads at chance, when the rotation is random, and to
compare: a term far from its chance value is measuring something other than
what its name says. ``tests/test_losses.py`` asserts these by Monte Carlo.

    L_rotation (geodesic)      126.48 deg = pi/2 + 2/pi rad
    Euler RMSE, random guess    83.25 deg
    Euler RMSE, always identity 83.18 deg  <- the same, and that is the point
    L_normal  (cosine)           1.0
    L_face    (two normals)      2.0
    L_position                   depends on fragment scale -- see the docstring
    perfect fit                  0 (geodesic exactly; the rest to the 1e-8 floor)

The rotation convention
-----------------------
``diffuse_fragments`` applies ``v_pert = v_gt Q^T + t``, so after centring
``v_pert = v_gt Q^T``. The rotation that undoes it satisfies
``v_pert R^T = v_gt``, giving

    R_gt = Q^T

which is ``matrices[i][:3, :3].T``. Every function here takes ``(..., 3, 3)``
rotation matrices in that convention: **the rotation that maps the perturbed
fragment back to its assembled pose**, applied to row vectors as ``v @ R.T``.
Getting this backwards costs nothing at initialisation -- chance is chance in
either direction -- and shows up only as a model that will not converge.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .segment import segment_mean, segment_sum
from .vn import EPS, safe_norm


def geodesic_angle(a: Tensor, b: Tensor) -> Tensor:
    """
    Angle in radians between two rotations, via ``atan2``.

    For ``E = a^T b`` the rotation angle satisfies

        cos(theta) = (tr(E) - 1) / 2        sin(theta) = ||axis(E)|| / 2

    so ``theta = atan2(||axis||, tr(E) - 1)``, correct over the full ``[0, pi]``.

    **Not** ``arccos((tr - 1) / 2)``, which is the obvious formula and the wrong
    one. ``arccos`` has an infinite derivative at both ends of its range, and
    one of those ends is ``theta = 0`` -- exactly where a converging model
    lives. Its gradient blows up as the fit improves, drowning out every other
    term in the loss, and clamping the argument to dodge the NaN converts the
    problem into a hard error floor instead. ``atan2`` is smooth at both ends:
    its derivative at ``theta = 0`` is 1/2, not infinity.
    """
    relative = torch.matmul(a.transpose(-1, -2), b)
    axis = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    # safe_norm, not linalg.norm: at a perfect prediction the axis is exactly
    # zero and the plain norm's gradient there is NaN. The floor is 1e-20
    # rather than the module default 1e-8 because this norm is not divided by:
    # d(atan2)/d||axis|| -> 1/2 as the axis vanishes, exactly cancelling the
    # 1/||axis|| in the norm's own derivative, so the composite gradient is
    # bounded by 1/2 regardless. A larger floor would only add a spurious
    # 5e-9 rad at a perfect fit and the same deficit at 180 degrees.
    return torch.atan2(safe_norm(axis, dim=-1, eps=1e-20), trace - 1.0)


def rotation_loss(predicted: Tensor, target: Tensor,
                  keep: Optional[Tensor] = None) -> Tensor:
    """
    Mean geodesic angle in radians. Chance is ``pi/2 + 2/pi`` = 126.48 deg.

    ``keep`` (``(F,)`` bool) restricts the mean to some fragments -- under the
    anchor target, every fragment but the anchors (``nn/anchor.py``).
    """
    return _fragment_mean(geodesic_angle(predicted, target), keep)


def _fragment_mean(values: Tensor, keep: Optional[Tensor]) -> Tensor:
    """
    Mean of per-fragment ``values`` over the fragments ``keep`` marks.

    ``torch.where`` rather than indexing, so the shape does not depend on the
    data and a left-out fragment passes back an exactly zero gradient. With no
    fragment kept the mean is 0, not NaN: a batch of single-fragment scenes has
    nothing to score under the anchor target, and it is weighted by zero
    fragments in the step anyway.
    """
    if keep is None:
        return values.mean()
    if keep.shape != values.shape:
        raise ValueError(f"keep has shape {tuple(keep.shape)}, values "
                         f"{tuple(values.shape)}: one flag per fragment")
    kept = torch.where(keep, values, torch.zeros_like(values))
    return kept.sum() / keep.sum().clamp(min=1).to(values.dtype)


def euler_rmse(predicted: Tensor, target: Tensor) -> Tensor:
    """
    RMSE over the XYZ Euler angles of the **residual** rotation, in degrees --
    GARF's table metric.

    Reported alongside the geodesic angle and never instead of it: the two are
    different numbers for the same error. A 30 deg error about one axis is
    30 deg geodesic but ``sqrt((30^2 + 0 + 0)/3)`` = 17.3 deg Euler RMSE, so
    quoting the smaller one without saying which is a 40% free improvement.

    THE RESIDUAL, NOT A DIFFERENCE OF ANGLES
    ----------------------------------------
    This used to compute ``euler(predicted) - euler(target)`` componentwise,
    which is not a metric on SO(3): Euler angles are chart coordinates, and
    subtracting two charts weights the same physical error differently
    depending on where in the chart the pair happens to sit. Taking the angles
    of ``predicted^T @ target`` -- the rotation still needed after the
    prediction -- measures the residual itself, and is what GARF reports.

    The difference is not cosmetic. Measured over 400k Haar-uniform pairs:

        convention                   random      identity     gap
        euler(pred) - euler(target)   86.23        83.18      3.06
        euler(pred^T @ target)        83.25        83.18      0.07

    Under the old convention, a model that collapsed to the identity scored
    **three degrees better than guessing** while the geodesic angle said it had
    learned nothing -- and that artefact was documented here, in the reference
    table and in the training banner as though it were a property of the
    metric. It is a property of the chart. Under the residual convention the
    gap is 0.07 deg, i.e. Monte-Carlo noise: collapsing to the identity buys
    exactly nothing, which is the correct behaviour and one fewer trap to warn
    about.

    The geodesic angle remains the primary number; this one exists for
    comparability with published tables.
    """
    residual = torch.matmul(predicted.transpose(-1, -2), target)
    # Wrap each angle into (-pi, pi] before squaring: 359 deg and 1 deg differ
    # by 2 deg, not 358. Still needed -- `_euler_xyz` returns atan2 output in
    # (-pi, pi], but the wrap costs nothing and documents the intent.
    angles = _euler_xyz(residual)
    wrapped = torch.atan2(torch.sin(angles), torch.cos(angles))
    return torch.sqrt(torch.mean(wrapped ** 2, dim=-1)).mean() * (180.0 / torch.pi)


def _euler_xyz(R: Tensor) -> Tensor:
    """Intrinsic XYZ Euler angles from a rotation matrix."""
    sy = torch.clamp(-R[..., 2, 0], -1.0, 1.0)
    pitch = torch.asin(sy)
    # Gimbal lock: |R[2,0]| -> 1 leaves roll and yaw individually undetermined.
    singular = torch.abs(R[..., 2, 0]) > 1.0 - 1e-7
    roll = torch.where(singular,
                       torch.atan2(-R[..., 1, 2], R[..., 1, 1]),
                       torch.atan2(R[..., 2, 1], R[..., 2, 2]))
    yaw = torch.where(singular, torch.zeros_like(pitch),
                      torch.atan2(R[..., 1, 0], R[..., 0, 0]))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _cosine_items(predicted: Tensor, target: Tensor) -> Tensor:
    """``1 - cos`` per item, averaged over any axes between the first and the last."""
    a = predicted / safe_norm(predicted, dim=-1, keepdim=True)
    b = target / safe_norm(target, dim=-1, keepdim=True)
    per_item = 1.0 - torch.sum(a * b, dim=-1)
    if per_item.dim() > 1:
        per_item = per_item.mean(dim=tuple(range(1, per_item.dim())))
    return per_item


def cosine_per_fragment(predicted: Tensor, target: Tensor, batch: Tensor,
                        num_segments: Optional[int] = None) -> Tensor:
    """:func:`cosine_loss` for each segment (fragment) on its own: ``(F,)``."""
    per_item = _cosine_items(predicted, target)
    return segment_mean(per_item.unsqueeze(-1), batch, num_segments).squeeze(-1)


def cosine_loss(predicted: Tensor, target: Tensor,
                batch: Optional[Tensor] = None,
                num_segments: Optional[int] = None,
                keep: Optional[Tensor] = None) -> Tensor:
    """
    ``1 - cos`` between two sets of directions. Chance is 1.0.

    Both arguments are normalised first, so an input that is not unit length
    (a vertex normal averaged over degenerate faces, say) cannot inflate the
    term. With ``batch``, the mean is taken per segment and then over segments,
    so a 83,039-vertex fragment does not outweigh a 4-vertex one; ``keep``
    then selects which segments (fragments) count.
    """
    if batch is None:
        if keep is not None:
            raise ValueError("keep selects fragments, so it needs batch")
        return _cosine_items(predicted, target).mean()
    return _fragment_mean(cosine_per_fragment(predicted, target, batch, num_segments), keep)


def position_per_fragment(predicted: Tensor, target: Tensor, batch: Tensor,
                          num_segments: Optional[int] = None) -> Tensor:
    """:func:`position_loss` for each segment (fragment) on its own: ``(F,)``."""
    distance = safe_norm(predicted - target, dim=-1)
    return segment_mean(distance.unsqueeze(-1), batch, num_segments).squeeze(-1)


def position_loss(predicted: Tensor, target: Tensor,
                  batch: Optional[Tensor] = None,
                  num_segments: Optional[int] = None,
                  keep: Optional[Tensor] = None) -> Tensor:
    """
    Mean distance between corresponding vertices, both centred.

    There is no universal chance value: it scales with the fragment. For points
    on a unit sphere and a uniformly random rotation the expectation is 4/3,
    which is the number to compare against under per-fragment normalisation,
    but the default here is per-*scene* normalisation, where a small fragment
    sits well inside the unit ball and reads lower.
    """
    if batch is None:
        if keep is not None:
            raise ValueError("keep selects fragments, so it needs batch")
        return safe_norm(predicted - target, dim=-1).mean()
    return _fragment_mean(position_per_fragment(predicted, target, batch, num_segments), keep)


def embedding_consistency_loss(
    embeddings: Tensor,
    cluster: Tensor,
    num_clusters: int,
) -> Tensor:
    """
    Variance of the embeddings within each set of coincident vertices.

    Two vertices that touch in the assembled object should embed to the same
    point, so the loss is the scatter of each coincidence cluster about its own
    centroid::

        L = mean_c  mean_{i in c}  ||z_i - mean(z_c)||^2

    **Not** ``||sum_i z_i||^2``, which the design document specifies and which
    is minimised by embeddings that *cancel* rather than agree: two opposite
    vectors score 0, two identical ones score ``4||z||^2``. It rewards exactly
    the wrong configuration, and it does so while looking like a perfectly
    reasonable consistency penalty.

    ``embeddings`` must be **invariant** features. The vertices in a cluster
    belong to different fragments, and each fragment arrives under its own
    independent perturbation, so equivariant features from two fragments are
    expressed in two unrelated frames and comparing them directly would
    penalise the perturbation rather than the geometry. This is the same
    constraint that shapes the cross-fragment layer, applied to the loss.

    .. warning::
       This term is **minimised by a constant embedding**, and that is not
       hypothetical -- it is what the first real training run did. Measured on
       synthetic scenes: the loss fell 0.0109 -> 0.0010 over 180 steps while the
       spread of the embeddings fell 0.166 -> 0.021 and their norm stayed at
       ~2.0. The head was not learning to agree, it was converging on one
       vector, and the loss reported success the whole way down.

       It survives only as a *component* of :func:`correspondence_loss`, which
       adds the repulsion that makes the trivial solution expensive. Do not use
       it alone.
    """
    if embeddings.numel() == 0 or num_clusters == 0:
        return embeddings.new_zeros(())
    centroid = segment_mean(embeddings, cluster, num_clusters)
    deviation = embeddings - centroid[cluster]
    per_vertex = torch.sum(deviation * deviation, dim=-1)
    return segment_mean(per_vertex.unsqueeze(-1), cluster, num_clusters).mean()


def correspondence_loss(
    embeddings: Tensor,
    cluster: Tensor,
    num_clusters: int,
    temperature: float = 0.1,
    max_anchors: int = 1024,
    generator: Optional[torch.Generator] = None,
    return_accuracy: bool = False,
):
    """
    InfoNCE over coincidence clusters: coincident vertices close, others far.

    Why this and not the pure agreement term
    ----------------------------------------
    :func:`embedding_consistency_loss` asks only that coincident vertices agree,
    and the cheapest way to agree is for *everything* to agree. That is a real
    minimum, not a corner case, and the model finds it within a few hundred
    steps. A collapsed embedding is worse than an untrained one for what this
    head exists to do: stage two matches interface points by mutual nearest
    neighbours, and when every embedding is the same vector every distance ties.

    Repulsion is what removes the trivial solution. Written as InfoNCE rather
    than as a variance floor because the negatives here are exactly the
    confusion set at matching time -- the *other* fracture vertices of the same
    scene -- so the training objective and the downstream use are the same
    question::

        L = -mean_i  log[ sum_{p in pos(i)} exp(s_ip / T)
                          / sum_{k != i}    exp(s_ik / T) ]

    with ``s`` the cosine similarity. Collapse scores ``log(A - 1) - log|pos|``:
    identical embeddings make positives and negatives indistinguishable, so
    there is nothing for a collapsed head to gain here, unlike under the pure
    agreement term.

    COLLAPSE IS NOT THE WORST VALUE, and reading the curve depends on knowing
    that. This docstring used to say it was. Measured on 260 clustered vertices
    with 32-dimensional embeddings::

        random (untrained)       7.29
        collapsed (identical)    5.56      = log(259) - log(1)
        perfect clusters         0.05

    Random embeddings score *above* collapse: at temperature 0.1 the spread of
    their similarities inflates the log-sum-exp in the denominator by about
    ``var(s / T) / 2``. So the value can fall early in training by shrinking
    that spread -- towards collapse -- as well as by learning to cluster, and
    the curve alone cannot tell the two apart. ``match@1`` can: it stays near
    zero for a collapsed head.

    Anchors are subsampled to ``max_anchors`` because the similarity matrix is
    quadratic in them and a scene can label thousands of vertices. 1024 anchors
    is a 4 MB matrix; the whole set would be gigabytes on the largest scenes,
    which is the same trap the cross-attention layer had.

    With ``return_accuracy`` it also reports the fraction of anchors whose
    nearest neighbour is a true coincidence partner. That is the number worth
    watching: it is exactly what stage two does at inference, it runs from
    chance (``|pos| / (A - 1)``, near zero) to 1.0, and unlike the loss it does
    not need a reference value computed per batch to be read.
    """
    empty = embeddings.new_zeros(())
    if embeddings.numel() == 0 or num_clusters == 0:
        return (empty, empty) if return_accuracy else empty

    # Only clustered vertices participate; -1 marks "no coincidence partner".
    anchors = torch.nonzero(cluster >= 0, as_tuple=False).flatten()
    if anchors.numel() < 2:
        return (empty, empty) if return_accuracy else empty
    if anchors.numel() > max_anchors:
        pick = torch.randperm(anchors.numel(), device=embeddings.device,
                              generator=generator)[:max_anchors]
        anchors = anchors[pick]

    z = torch.nn.functional.normalize(embeddings[anchors], dim=-1)
    labels = cluster[anchors]
    similarity = (z @ z.t()) / temperature

    identity = torch.eye(len(anchors), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    # An anchor whose cluster-mates all fell outside the subsample has no
    # positive, so its log-ratio is undefined. Dropped rather than given a zero,
    # which would quietly reward whatever the subsample happened to exclude.
    keep = positive.any(dim=1)
    if not bool(keep.any()):
        return (empty, empty) if return_accuracy else empty

    similarity = similarity.masked_fill(identity, float("-inf"))
    denominator = torch.logsumexp(similarity, dim=1)
    numerator = torch.logsumexp(
        similarity.masked_fill(~positive, float("-inf")), dim=1
    )
    loss = (denominator - numerator)[keep].mean()
    if not return_accuracy:
        return loss

    with torch.no_grad():
        nearest = similarity.argmax(dim=1)
        hit = positive[torch.arange(len(anchors), device=z.device), nearest]
        accuracy = hit[keep].to(loss.dtype).mean()
    return loss, accuracy


SCORE_TERMS = ("rotation", "position", "normal", "face")
"""The four geometric terms: scores of a rotation, never trained on (since v7)."""


class ReassemblyLoss(torch.nn.Module):
    """
    The training loss -- the contrastive embedding term, alone -- and the four
    geometric terms beside it as scores.

    Until v7 the four were weighted into the total, and trained the rotation
    head this network no longer has. They now score the rotations stage two
    fits from the embedding matches (``assembly/rotation.py``): computed under
    ``no_grad``, reported, never part of the total, so nothing they say can
    reach a gradient. Their chance values are unchanged -- 126.48 deg, 1.0 and
    2.0 -- because a fragment the matching cannot reach is left at a rotation
    that is random relative to its true one.

    ``embedding`` scales the one term that is trained; it is 1.0 and there is
    nothing to balance it against.
    """

    def __init__(self, embedding: float = 1.0, temperature: float = 0.1,
                 max_anchors: int = 1024) -> None:
        super().__init__()
        self.weights = {"embedding": embedding}
        self.temperature = temperature
        self.max_anchors = max_anchors

    def forward(
        self,
        predicted_rotation: Optional[Tensor] = None,
        target_rotation: Optional[Tensor] = None,
        *,
        vertices: Optional[Tensor] = None,
        target_vertices: Optional[Tensor] = None,
        vertex_batch: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        target_normals: Optional[Tensor] = None,
        face_normals: Optional[Tensor] = None,
        target_face_normals: Optional[Tensor] = None,
        edge_batch: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        cluster: Optional[Tensor] = None,
        num_clusters: int = 0,
        keep: Optional[Tensor] = None,
        return_fragments: bool = False,
    ):
        """
        ``(total, report)``: the trained total -- the embedding term times its
        weight, or a constant 0 with no gradient when the batch has no
        coincidence cluster to train on -- and every term as a float, the
        scores included.

        The scores need ``predicted_rotation`` and ``target_rotation``, and
        each of the other three its own pair of inputs; a term whose inputs
        are not given is not reported. ``keep`` (``(F,)`` bool) is which
        fragments they average over -- under the anchor protocol all but each
        scene's anchor, whose rotation is its true pose by construction
        (``nn/anchor.py``). The embedding term is unaffected: it is invariant,
        so there is no frame for an anchor to fix.

        ``return_fragments`` adds a third value: each score per fragment,
        ``(F,)`` tensors (every fragment; ``keep`` is not applied), for
        breakdowns by category. Per fragment needs the segment index of the
        term (``vertex_batch``, ``edge_batch``); without it the term is left
        out of the per-fragment values.
        """
        terms, fragments = {}, {}
        with torch.no_grad():
            if predicted_rotation is not None and target_rotation is not None:
                n = predicted_rotation.shape[0]
                fragments["rotation"] = geodesic_angle(predicted_rotation, target_rotation)
                terms["rotation"] = _fragment_mean(fragments["rotation"], keep)
                if vertices is not None and target_vertices is not None:
                    if vertex_batch is None:
                        terms["position"] = position_loss(vertices, target_vertices, keep=keep)
                    else:
                        fragments["position"] = position_per_fragment(
                            vertices, target_vertices, vertex_batch, n)
                        terms["position"] = _fragment_mean(fragments["position"], keep)
                if normals is not None and target_normals is not None:
                    if vertex_batch is None:
                        terms["normal"] = cosine_loss(normals, target_normals, keep=keep)
                    else:
                        fragments["normal"] = cosine_per_fragment(
                            normals, target_normals, vertex_batch, n)
                        terms["normal"] = _fragment_mean(fragments["normal"], keep)
                if face_normals is not None and target_face_normals is not None:
                    # Edges carry three channels -- (n1, n2, relative position) --
                    # but only the two normals are directions to be matched.
                    # Sliced here rather than left to the caller: passing the
                    # whole `edge_attr` is the obvious thing to do, and it would
                    # silently normalise a vector whose *length* is its content
                    # and move chance from 2.0 to 3.0.
                    predicted_faces = face_normals[..., :2, :]
                    target_faces = target_face_normals[..., :2, :]
                    # Both adjacent normals, so chance is 2.0 not 1.0.
                    if edge_batch is None:
                        terms["face"] = 2.0 * cosine_loss(predicted_faces, target_faces,
                                                          keep=keep)
                    else:
                        fragments["face"] = 2.0 * cosine_per_fragment(
                            predicted_faces, target_faces, edge_batch, n)
                        terms["face"] = _fragment_mean(fragments["face"], keep)

        match_top1 = None
        total = None
        if embeddings is not None and cluster is not None:
            embedding, match_top1 = correspondence_loss(
                embeddings, cluster, num_clusters,
                temperature=self.temperature, max_anchors=self.max_anchors,
                return_accuracy=True,
            )
            terms["embedding"] = embedding
            total = self.weights["embedding"] * embedding
        if total is None:
            # Nothing to train on: a constant, so `requires_grad` says so.
            like = next((t for t in terms.values()), None)
            total = (like.new_zeros(()) if like is not None
                     else torch.zeros((), device=getattr(embeddings, "device", None)))
        report = {k: float(v.detach()) for k, v in terms.items()}
        report["total"] = float(total.detach())
        if "rotation" in report:
            report["rotation_degrees"] = report["rotation"] * 180.0 / 3.141592653589793
        if match_top1 is not None:
            # The embedding loss needs a per-batch reference to interpret; this
            # does not. It is stage two's own operation -- is a vertex's true
            # coincidence partner its nearest neighbour -- so it reads straight
            # from ~0 at chance to 1.0, and a collapsed embedding cannot fake it.
            report["match@1"] = float(match_top1.detach())
        if return_fragments:
            return total, report, fragments
        return total, report
