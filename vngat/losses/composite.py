"""
Composite training objective:

    L = w_rot L_rot + w_pos L_pos + w_node L_node + w_mid L_mid
        + w_face L_face + w_emb_v L_emb-v + w_emb_e L_emb-e

The five geometric terms follow the design document directly. The two
interface-embedding terms replace the document's formula, which is degenerate
(see `cluster_consistency_loss`).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evaluation.metrics import swing_twist_error
from ..models.vn_layers import geodesic_rotation_loss


def _mean(values: torch.Tensor) -> torch.Tensor:
    """
    Mean that stays finite (and graph-connected) on an empty tensor.

    `torch.empty(0).mean()` is NaN, and a single NaN in the weighted total
    propagates into every parameter on the next optimiser step -- turning a
    momentarily empty term into a silently destroyed 12-hour run. Returning a
    connected zero keeps the parameter reachable for DDP's reduction, exactly
    as `cluster_consistency_loss` does.
    """
    if values.numel() == 0:
        return values.sum() * 0.0
    return values.mean()


def _reduce(values: torch.Tensor, mask=None) -> torch.Tensor:
    """
    Mean over `values`, restricted to `mask` when the batch has been padded.

    `mask=None` is the unpadded path and behaves exactly as `_mean`, so this
    file stays a drop-in match for the GPU branch. Only the XLA trainer passes
    masks; everything else is unaffected.
    """
    if mask is None:
        return _mean(values)
    from ..data.padding import masked_mean

    return masked_mean(values, mask)


def node_position_loss(x_pred, x_gt, mask=None) -> torch.Tensor:
    """L_pos = mean_v ||x_v - xhat_v||^2. Inputs (V, 3)."""
    return _reduce(((x_pred - x_gt) ** 2).sum(dim=-1), mask)


def node_normal_loss(n_pred, n_gt, eps: float = 1e-8, mask=None) -> torch.Tensor:
    """L_node = mean_v (1 - n_v . nhat_v). Inputs (V, 3), renormalised defensively."""
    n_pred = F.normalize(n_pred, dim=-1, eps=eps)
    n_gt = F.normalize(n_gt, dim=-1, eps=eps)
    return _reduce(1 - (n_pred * n_gt).sum(dim=-1), mask)


def edge_midpoint_loss(m_pred, m_gt, mask=None) -> torch.Tensor:
    """L_mid = mean_e ||m_e - mhat_e||^2. Inputs (E, 3)."""
    return _reduce(((m_pred - m_gt) ** 2).sum(dim=-1), mask)


def face_normal_loss(
    n1_pred: torch.Tensor, n1_gt: torch.Tensor,
    n2_pred: torch.Tensor, n2_gt: torch.Tensor,
    eps: float = 1e-8,
    mask=None,
) -> torch.Tensor:
    """L_face = mean_e [(1 - n1.n1hat) + (1 - n2.n2hat)]. Inputs (E, 3) each."""
    n1_pred = F.normalize(n1_pred, dim=-1, eps=eps)
    n1_gt = F.normalize(n1_gt, dim=-1, eps=eps)
    n2_pred = F.normalize(n2_pred, dim=-1, eps=eps)
    n2_gt = F.normalize(n2_gt, dim=-1, eps=eps)
    term1 = 1 - (n1_pred * n1_gt).sum(dim=-1)
    term2 = 1 - (n2_pred * n2_gt).sum(dim=-1)
    return _reduce(term1 + term2, mask)


def cluster_consistency_loss(
    embeddings: torch.Tensor,
    cluster_id: torch.Tensor,
    pull_margin: float = 0.1,
    push_margin: float = 0.5,
    push_weight: float = 1.0,
    max_push_clusters: int = 512,
) -> torch.Tensor:
    """
    Discriminative interface-embedding loss on the UNIT SPHERE.

    Embeddings are L2-normalised first, then:

        pull_k = mean_{i in k} relu(||z_i - c_k|| - pull_margin)^2
        push   = mean_{a != b} relu(2*push_margin - ||c_a - c_b||)^2
        L      = mean_k pull_k + push_weight * push

    THREE DEGENERACIES, ALL OBSERVED IN TRAINING, ALL FIXED HERE
    ------------------------------------------------------------
    1. The design document's `|| sum_{(f,v) in C(x)} z ||^2` is minimised by
       embeddings that CANCEL rather than agree.

    2. Plain within-cluster variance is zero for ANY CONSTANT embedding, so the
       model collapsed to a single vector within two epochs, encoding nothing.
       The `push` term fixes that.

    3. But `push` on UNNORMALISED embeddings has an escape hatch of its own:
       separating clusters by inflating their magnitude is easier than
       arranging them, and a weak norm penalty does not stop it. A real run
       drove mean centroid norm to ~565 while emb_v still looked healthy at
       0.56. Squaring anything that large overflows float16 (max 65504) to
       inf, and the first inf/inf produces NaN -- which is exactly what
       happened, in bursts, from epoch 19 onward.

       Normalising removes the escape hatch by construction: every embedding
       has norm 1, all distances lie in [0, 2], and no setting of the weights
       can make the term large. It also bounds the loss to O(1), so the
       embedding objective can no longer dominate the gradient budget --
       measured at 63% of the total at initialisation before this change,
       against rotation's 15%, which is why the rotation error sat at chance
       while the embedding terms fell.

       As a bonus it is the right space for the inference-time matcher, which
       compares descriptors by distance.

    Because distances are bounded by 2, `push_margin` must be below 1.0; the
    default 0.5 asks for a separation of 1.0 between centroids. The minimum is
    not zero and need not be -- judge this term by whether it FALLS, and treat
    an exact 0.0000 as the collapse alarm.

    Points marked -1 ("shared with nothing") are excluded from every term.

    DDP: returns a graph-connected zero when a batch contains no shared points,
    so the embedding heads always receive a gradient and every rank builds
    identical reduction buckets.
    """
    mask = cluster_id >= 0
    if embeddings.numel() == 0 or not bool(mask.any()):
        return embeddings.sum() * 0.0

    # float32 whenever the input is half. NOT just precision: autocast promotes
    # `pow` and `sum` to float32, so a buffer allocated from a Half embedding
    # dtype would meet a Float source and `index_add_` would reject the pair.
    emb = embeddings[mask]
    acc = torch.float32 if emb.dtype in (torch.float16, torch.bfloat16) else emb.dtype
    emb = emb.to(acc)

    # `sqrt(sum + eps)` rather than `.norm()`: the latter has an undefined
    # gradient at the zero vector, which an untrained head can produce.
    emb = emb / torch.sqrt(emb.pow(2).sum(-1, keepdim=True) + 1e-12)

    cid = cluster_id[mask]
    _, inverse = torch.unique(cid, return_inverse=True)
    num_clusters = int(inverse.max().item()) + 1
    D = emb.shape[-1]

    counts = torch.bincount(inverse, minlength=num_clusters).clamp_min(1).to(acc)
    sums = torch.zeros(num_clusters, D, device=emb.device, dtype=acc)
    sums.index_add_(0, inverse, emb)
    centroids = sums / counts.unsqueeze(-1)

    # -- pull: members toward their own centroid, averaged WITHIN a cluster
    #    first so a large cluster cannot dominate by member count alone.
    dist = torch.sqrt((emb - centroids[inverse]).pow(2).sum(-1) + 1e-12)
    per_point = torch.relu(dist - pull_margin).pow(2).to(acc)
    pull_sum = torch.zeros(num_clusters, device=emb.device, dtype=acc)
    pull_sum.index_add_(0, inverse, per_point)
    pull = (pull_sum / counts).mean()

    # -- push: centroids apart. Subsampled above `max_push_clusters`: this is
    #    the only O(C^2) term and a dense scene yields thousands of clusters,
    #    so a random subset each step is an unbiased estimate.
    push = pull.new_zeros(())
    if num_clusters > 1:
        picked = centroids
        if num_clusters > max_push_clusters:
            sel = torch.randperm(num_clusters, device=emb.device)[:max_push_clusters]
            picked = centroids[sel]
        # NOT `torch.cdist(picked, picked)`. Its diagonal is an exact zero
        # distance, where the gradient is 0/0 -- cdist against itself is a
        # documented NaN-gradient trap, and masking the diagonal out of the
        # FORWARD does not stop the backward from producing it.
        sq = picked.pow(2).sum(-1)
        d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (picked @ picked.t())
        pairwise = torch.sqrt(d2.clamp_min(0) + 1e-12)
        n = picked.shape[0]
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=picked.device)
        push = torch.relu(2 * push_margin - pairwise[off_diagonal]).pow(2).mean()

    return pull + push_weight * push


class CompositeLoss(nn.Module):
    def __init__(
        self,
        w_rot: float = 1.0,
        w_pos: float = 1.0,
        w_node: float = 1.0,
        w_mid: float = 1.0,
        w_face: float = 1.0,
        w_emb_v: float = 1.0,
        w_emb_e: float = 1.0,
        emb_pull_margin: float = 0.1,
        emb_push_margin: float = 0.5,
        symmetry_axis: str = "z",
    ):
        super().__init__()
        self.weights = dict(
            rot=w_rot, pos=w_pos, node=w_node, mid=w_mid,
            face=w_face, emb_v=w_emb_v, emb_e=w_emb_e,
        )
        self.emb_pull_margin = emb_pull_margin
        self.emb_push_margin = emb_push_margin
        self.symmetry_axis = symmetry_axis

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rot_angles = geodesic_rotation_loss(outputs["R_pred"], targets["R_gt"])
        with torch.no_grad():
            tilt, twist = swing_twist_error(
                outputs["R_pred"], targets["R_gt"], axis=self.symmetry_axis)
        # Masks are present only on the padded (XLA) path; None everywhere else.
        node_mask = targets.get("node_mask")
        edge_mask = targets.get("edge_mask")
        frag_mask = targets.get("frag_mask")

        l_rot = _reduce(rot_angles, frag_mask)
        l_pos = node_position_loss(outputs["x_pred"], targets["x_gt"], node_mask)
        l_node = node_normal_loss(outputs["n_pred"], targets["n_gt"], mask=node_mask)
        l_mid = edge_midpoint_loss(outputs["mid_pred"], targets["mid_gt"], edge_mask)
        l_face = face_normal_loss(
            outputs["n1_pred"], targets["n1_gt"], outputs["n2_pred"], targets["n2_gt"],
            mask=edge_mask,
        )
        margins = dict(pull_margin=self.emb_pull_margin, push_margin=self.emb_push_margin)
        l_emb_v = cluster_consistency_loss(
            outputs["vertex_embedding"], targets["vertex_cluster_id"], **margins)
        l_emb_e = cluster_consistency_loss(
            outputs["edge_embedding"], targets["edge_cluster_id"], **margins)

        w = self.weights
        total = (
            w["rot"] * l_rot + w["pos"] * l_pos + w["node"] * l_node
            + w["mid"] * l_mid + w["face"] * l_face
            + w["emb_v"] * l_emb_v + w["emb_e"] * l_emb_e
        )
        return {
            "total": total,
            "rot": l_rot,
            # Reported alongside the radian loss because degrees are what the
            # thesis's comparison tables are in. This is the mean GEODESIC
            # angle -- not the Euler-angle RMSE GARF reports. They are
            # different numbers and must not be compared directly; see
            # `vngat/evaluation/metrics.py`, which computes both.
            "rot_deg": _reduce(rot_angles.detach(), frag_mask) * (180.0 / torch.pi),
            # Reported, never optimised. Splits the residual into TILT off the
            # object's symmetry axis and TWIST about it. On validation this is
            # the measurement that separates "has not learned the axis either"
            # (tilt ~ 90) from "learned the axis, cannot recover the azimuth"
            # (tilt ~ 0, twist ~ 90) -- the latter being a structural limit of a
            # one-shot per-fragment canonicaliser on surfaces of revolution,
            # which no amount of tuning or compute would move.
            "tilt": _reduce(tilt.detach(), frag_mask),
            "twist": _reduce(twist.detach(), frag_mask),
            "pos": l_pos,
            "node": l_node,
            "mid": l_mid,
            "face": l_face,
            "emb_v": l_emb_v,
            "emb_e": l_emb_e,
        }
