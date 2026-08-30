"""
Losses, and the reference values every one of them is checked against.

A geometric loss that is silently wrong still descends. The defence is to know
what each term reads at initialisation, when the prediction is a random
rotation, and to compare: a term that starts far from its chance value is
measuring something other than what its name says.
``tests/test_losses.py`` asserts these by Monte Carlo.

    L_rotation (geodesic)      126.48 deg = pi/2 + 2/pi rad
    Euler RMSE, random guess    86.29 deg
    Euler RMSE, always identity 83.14 deg  <- BETTER than guessing; see below
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

from typing import Optional, Tuple

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


def rotation_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean geodesic angle in radians. Chance is ``pi/2 + 2/pi`` = 126.48 deg."""
    return geodesic_angle(predicted, target).mean()


def euler_rmse(predicted: Tensor, target: Tensor) -> Tensor:
    """
    RMSE over XYZ Euler angles, in degrees -- GARF's table metric.

    Reported alongside the geodesic angle and never instead of it: the two are
    different numbers for the same error. A 30 deg error about one axis is
    30 deg geodesic but ``sqrt((30^2 + 0 + 0)/3)`` = 17.3 deg Euler RMSE, so
    quoting the smaller one without saying which is a 40% free improvement.

    "Chance" also has two values here, and the gap is a trap. Guessing randomly
    scores 86.29 deg; always predicting the **identity** scores 83.14 deg --
    three degrees *better*. Collapsing towards the identity is the cheapest
    early way to reduce a rotation loss, so a model can appear to beat chance on
    this metric having learned nothing. The geodesic angle reads 126.48 deg for
    both and does not pay for the collapse, which is why it is the primary
    number and this one is only for comparability.
    """
    difference = _euler_xyz(predicted) - _euler_xyz(target)
    # Wrap each angle into (-pi, pi] before squaring: 359 deg and 1 deg differ
    # by 2 deg, not 358.
    wrapped = torch.atan2(torch.sin(difference), torch.cos(difference))
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


def cosine_loss(predicted: Tensor, target: Tensor,
                batch: Optional[Tensor] = None,
                num_segments: Optional[int] = None) -> Tensor:
    """
    ``1 - cos`` between two sets of directions. Chance is 1.0.

    Both arguments are normalised first, so an input that is not unit length
    (a vertex normal averaged over degenerate faces, say) cannot inflate the
    term. With ``batch``, the mean is taken per segment and then over segments,
    so a 83,039-vertex fragment does not outweigh a 4-vertex one.
    """
    a = predicted / safe_norm(predicted, dim=-1, keepdim=True)
    b = target / safe_norm(target, dim=-1, keepdim=True)
    per_item = 1.0 - torch.sum(a * b, dim=-1)
    if per_item.dim() > 1:
        per_item = per_item.mean(dim=tuple(range(1, per_item.dim())))
    if batch is None:
        return per_item.mean()
    return segment_mean(per_item.unsqueeze(-1), batch, num_segments).mean()


def position_loss(predicted: Tensor, target: Tensor,
                  batch: Optional[Tensor] = None,
                  num_segments: Optional[int] = None) -> Tensor:
    """
    Mean distance between corresponding vertices, both centred.

    There is no universal chance value: it scales with the fragment. For points
    on a unit sphere and a uniformly random rotation the expectation is 4/3,
    which is the number to compare against under per-fragment normalisation,
    but the default here is per-*scene* normalisation, where a small fragment
    sits well inside the unit ball and reads lower.
    """
    distance = safe_norm(predicted - target, dim=-1)
    if batch is None:
        return distance.mean()
    return segment_mean(distance.unsqueeze(-1), batch, num_segments).mean()


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

    with ``s`` the cosine similarity. Collapse now scores its *worst* value,
    ``log(A - 1) - log|pos|``: identical embeddings make positives and negatives
    indistinguishable, which is exactly what the term measures.

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


class ReassemblyLoss(torch.nn.Module):
    """
    The composite objective, with every weight at 1.0 and untuned.

    Deliberately untuned: the terms have very different natural scales
    (radians, a cosine in [0, 2], a distance in normalised units) and choosing
    weights before seeing how each one moves is guesswork dressed as a
    decision. The per-term values are returned alongside the total so the first
    training run *measures* the relative scales instead of assuming them.
    """

    def __init__(self, rotation: float = 1.0, position: float = 1.0,
                 normal: float = 1.0, face: float = 1.0,
                 embedding: float = 1.0, temperature: float = 0.1,
                 max_anchors: int = 1024) -> None:
        super().__init__()
        self.weights = {
            "rotation": rotation, "position": position, "normal": normal,
            "face": face, "embedding": embedding,
        }
        self.temperature = temperature
        self.max_anchors = max_anchors

    def forward(
        self,
        predicted_rotation: Tensor,
        target_rotation: Tensor,
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
    ) -> Tuple[Tensor, dict]:
        n = predicted_rotation.shape[0]
        terms = {"rotation": rotation_loss(predicted_rotation, target_rotation)}

        if vertices is not None and target_vertices is not None:
            terms["position"] = position_loss(vertices, target_vertices, vertex_batch, n)
        if normals is not None and target_normals is not None:
            terms["normal"] = cosine_loss(normals, target_normals, vertex_batch, n)
        if face_normals is not None and target_face_normals is not None:
            # Edges carry three channels -- (n1, n2, relative position) -- but
            # only the two normals are directions to be matched. Sliced here
            # rather than left to the caller: passing the whole `edge_attr` is
            # the obvious thing to do, and it would silently normalise a vector
            # whose *length* is its content and move chance from 2.0 to 3.0.
            predicted_faces = face_normals[..., :2, :]
            target_faces = target_face_normals[..., :2, :]
            # Both adjacent normals, so chance is 2.0 not 1.0.
            terms["face"] = 2.0 * cosine_loss(
                predicted_faces, target_faces, edge_batch, n
            )
        match_top1 = None
        if embeddings is not None and cluster is not None:
            terms["embedding"], match_top1 = correspondence_loss(
                embeddings, cluster, num_clusters,
                temperature=self.temperature, max_anchors=self.max_anchors,
                return_accuracy=True,
            )

        total = sum(self.weights[k] * v for k, v in terms.items())
        report = {k: float(v.detach()) for k, v in terms.items()}
        report["total"] = float(total.detach())
        report["rotation_degrees"] = report["rotation"] * 180.0 / 3.141592653589793
        if match_top1 is not None:
            # The embedding loss needs a per-batch reference to interpret; this
            # does not. It is stage two's own operation -- is a vertex's true
            # coincidence partner its nearest neighbour -- so it reads straight
            # from ~0 at chance to 1.0, and a collapsed embedding cannot fake it.
            report["match@1"] = float(match_top1.detach())
        return total, report
