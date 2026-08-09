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


def node_position_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """L_pos = mean_v ||x_v - xhat_v||^2. Inputs (V, 3)."""
    return _mean(((x_pred - x_gt) ** 2).sum(dim=-1))


def node_normal_loss(n_pred: torch.Tensor, n_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L_node = mean_v (1 - n_v . nhat_v). Inputs (V, 3), renormalised defensively."""
    n_pred = F.normalize(n_pred, dim=-1, eps=eps)
    n_gt = F.normalize(n_gt, dim=-1, eps=eps)
    return _mean(1 - (n_pred * n_gt).sum(dim=-1))


def edge_midpoint_loss(m_pred: torch.Tensor, m_gt: torch.Tensor) -> torch.Tensor:
    """L_mid = mean_e ||m_e - mhat_e||^2. Inputs (E, 3)."""
    return _mean(((m_pred - m_gt) ** 2).sum(dim=-1))


def face_normal_loss(
    n1_pred: torch.Tensor, n1_gt: torch.Tensor,
    n2_pred: torch.Tensor, n2_gt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_face = mean_e [(1 - n1.n1hat) + (1 - n2.n2hat)]. Inputs (E, 3) each."""
    n1_pred = F.normalize(n1_pred, dim=-1, eps=eps)
    n1_gt = F.normalize(n1_gt, dim=-1, eps=eps)
    n2_pred = F.normalize(n2_pred, dim=-1, eps=eps)
    n2_gt = F.normalize(n2_gt, dim=-1, eps=eps)
    term1 = 1 - (n1_pred * n1_gt).sum(dim=-1)
    term2 = 1 - (n2_pred * n2_gt).sum(dim=-1)
    return _mean(term1 + term2)


def cluster_consistency_loss(
    embeddings: torch.Tensor,
    cluster_id: torch.Tensor,
    pull_margin: float = 0.5,
    push_margin: float = 1.5,
    push_weight: float = 1.0,
    reg_weight: float = 1e-3,
    max_push_clusters: int = 512,
) -> torch.Tensor:
    """
    Discriminative interface-embedding loss (De Brabandere et al., 2017):

        pull_k = mean_{i in k} relu(||z_i - c_k|| - pull_margin)^2
        push   = mean_{a != b} relu(2*push_margin - ||c_a - c_b||)^2
        reg    = mean_k ||c_k||
        L      = mean_k pull_k + push_weight * push + reg_weight * reg

    TWO DEGENERACIES, BOTH OBSERVED, BOTH FIXED HERE
    ------------------------------------------------
    1. The design document's `|| sum_{(f,v) in C(x)} z ||^2` is minimised by
       embeddings that CANCEL rather than agree: (3, -1) with (3, -1) scores
       40, while (5, 0) with (-5, 0) scores 0. It rewards disagreement.

    2. Replacing it with plain within-cluster variance fixes that but leaves a
       worse one: variance is zero for ANY CONSTANT embedding. Emitting the
       same vector everywhere is a global minimum that encodes no geometry.
       This is not hypothetical -- it is what the first training run did,
       driving both embedding terms from 0.0030/0.0257 to exactly 0.0000
       within two epochs while every other term stayed at chance.

    The `push` term is what removes (2): collapsing all centroids together
    makes it maximal. The hinges matter too -- `pull_margin` stops the pull
    term demanding infinite precision once a cluster is tight enough, and
    `2*push_margin` stops the push term from separating clusters that are
    already far apart, so gradient goes to the pairs that are actually
    confusable.

    Note the minimum is NOT zero: `reg` is only zero when centroids sit at the
    origin, which `push` opposes. Judge this term by whether it FALLS, not by
    whether it reaches zero -- and treat an exact 0.0000 as the collapse alarm.

    Why the embeddings must be good, not merely present: at inference the
    translation solver finds correspondences by mutual nearest neighbours in
    this space. Collapsed embeddings make every point equidistant from every
    other, so the matching -- and with it the entire assembly stage -- is noise.

    Points marked -1 ("shared with nothing") are excluded from every term. They
    are not currently pushed AWAY from interface clusters, so a non-interface
    point can still land near one; that is a known limitation, not an oversight.

    DDP: returns a graph-connected zero when a batch contains no shared points
    at all, so the embedding heads always receive a gradient and every rank
    builds identical reduction buckets.
    """
    mask = cluster_id >= 0
    if embeddings.numel() == 0 or not bool(mask.any()):
        return embeddings.sum() * 0.0

    # Everything below runs in float32 when the input is half precision.
    #
    # NOT just a precision preference -- without it this function CRASHES under
    # AMP. Autocast promotes both `pow` and `sum` to float32 (they are on
    # torch's float32 cast list), so a per-point term comes back Float while a
    # buffer allocated from `emb.dtype` is Half, and `index_add_` rejects the
    # pair with "self (Half) and source (Float) must have the same scalar
    # type". Deriving every buffer from one accumulation dtype removes the
    # whole class of mismatch rather than patching the one call that raised.
    emb = embeddings[mask]
    acc = torch.float32 if emb.dtype in (torch.float16, torch.bfloat16) else emb.dtype
    emb = emb.to(acc)

    cid = cluster_id[mask]
    _, inverse = torch.unique(cid, return_inverse=True)
    num_clusters = int(inverse.max().item()) + 1
    D = emb.shape[-1]

    # Integer counts: exact, and never a float accumulation to get wrong.
    counts = torch.bincount(inverse, minlength=num_clusters).clamp_min(1).to(acc)

    sums = torch.zeros(num_clusters, D, device=emb.device, dtype=acc)
    sums.index_add_(0, inverse, emb)
    centroids = sums / counts.unsqueeze(-1)

    # -- pull: members toward their own centroid, averaged WITHIN a cluster
    #    first so a large cluster cannot dominate by member count alone.
    dist = (emb - centroids[inverse]).norm(dim=-1)
    per_point = torch.relu(dist - pull_margin).pow(2).to(acc)
    pull_sum = torch.zeros(num_clusters, device=emb.device, dtype=acc)
    pull_sum.index_add_(0, inverse, per_point)
    pull = (pull_sum / counts).mean()

    # -- push: centroids apart. Subsampled above `max_push_clusters` because
    #    this is the only O(C^2) term and a dense scene can produce thousands
    #    of clusters; a random subset each step is an unbiased estimate.
    push = pull.new_zeros(())
    if num_clusters > 1:
        picked = centroids
        if num_clusters > max_push_clusters:
            sel = torch.randperm(num_clusters, device=emb.device)[:max_push_clusters]
            picked = centroids[sel]
        n = picked.shape[0]
        pairwise = torch.cdist(picked, picked)
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=picked.device)
        push = torch.relu(2 * push_margin - pairwise[off_diagonal]).pow(2).mean()

    reg = centroids.norm(dim=-1).mean()
    return pull + push_weight * push + reg_weight * reg


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
        emb_pull_margin: float = 0.5,
        emb_push_margin: float = 1.5,
    ):
        super().__init__()
        self.weights = dict(
            rot=w_rot, pos=w_pos, node=w_node, mid=w_mid,
            face=w_face, emb_v=w_emb_v, emb_e=w_emb_e,
        )
        self.emb_pull_margin = emb_pull_margin
        self.emb_push_margin = emb_push_margin

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rot_angles = geodesic_rotation_loss(outputs["R_pred"], targets["R_gt"])
        l_rot = _mean(rot_angles)
        l_pos = node_position_loss(outputs["x_pred"], targets["x_gt"])
        l_node = node_normal_loss(outputs["n_pred"], targets["n_gt"])
        l_mid = edge_midpoint_loss(outputs["mid_pred"], targets["mid_gt"])
        l_face = face_normal_loss(
            outputs["n1_pred"], targets["n1_gt"], outputs["n2_pred"], targets["n2_gt"]
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
            "rot_deg": _mean(rot_angles.detach()) * (180.0 / torch.pi),
            "pos": l_pos,
            "node": l_node,
            "mid": l_mid,
            "face": l_face,
            "emb_v": l_emb_v,
            "emb_e": l_emb_e,
        }
