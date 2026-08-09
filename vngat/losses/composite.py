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


def cluster_consistency_loss(embeddings: torch.Tensor, cluster_id: torch.Tensor) -> torch.Tensor:
    """
    Interface-embedding consistency: mean squared distance of each cluster
    member to its own cluster centroid, averaged within a cluster first (so a
    large cluster does not dominate purely by member count) and then across
    clusters. Zero if and only if every member of every cluster has an
    identical embedding.

    WHY NOT THE DESIGN DOCUMENT'S FORMULA. The document writes
    `sum_x || sum_{(f,v) in C_V(x)} z_v^(f) ||^2`, which is minimised by
    embeddings that CANCEL rather than agree: two identical embeddings (3, -1)
    and (3, -1) score 40, while two maximally opposed ones (5, 0) and (-5, 0)
    score 0. It rewards exactly the wrong thing.

    DDP NOTE. When a batch happens to contain no shared vertices/edges at all,
    this returns `embeddings.sum() * 0` rather than a fresh constant zero.
    Numerically identical, but it keeps the embedding heads' parameters
    connected to the graph so they receive a (zero) gradient. Without it those
    parameters intermittently receive no gradient at all, and
    DistributedDataParallel raises
    "Expected to have finished reduction in the prior iteration" and dies.
    The usual workaround, `find_unused_parameters=True`, costs a full graph
    traversal every single step; keeping the graph connected costs nothing and
    also guarantees every rank builds the identical reduction buckets, which
    is what stops ranks from deadlocking against each other.
    """
    mask = cluster_id >= 0
    if embeddings.numel() == 0 or not bool(mask.any()):
        return embeddings.sum() * 0.0

    # Everything below runs in float32 when the input is half precision.
    #
    # NOT just a precision preference -- without it this function CRASHES under
    # AMP. Autocast promotes both `pow` and `sum` to float32 (they are on
    # torch's float32 cast list), so `per_point` comes back Float while a buffer
    # allocated from `emb.dtype` is Half, and `index_add_` rejects the pair with
    # "self (Half) and source (Float) must have the same scalar type". Deriving
    # every buffer from one accumulation dtype removes the whole class of
    # mismatch rather than patching the one call that happened to raise.
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

    per_point = ((emb - centroids[inverse]) ** 2).sum(dim=-1).to(acc)

    cluster_sums = torch.zeros(num_clusters, device=emb.device, dtype=acc)
    cluster_sums.index_add_(0, inverse, per_point)
    return (cluster_sums / counts).mean()


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
    ):
        super().__init__()
        self.weights = dict(
            rot=w_rot, pos=w_pos, node=w_node, mid=w_mid,
            face=w_face, emb_v=w_emb_v, emb_e=w_emb_e,
        )

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rot_angles = geodesic_rotation_loss(outputs["R_pred"], targets["R_gt"])
        l_rot = _mean(rot_angles)
        l_pos = node_position_loss(outputs["x_pred"], targets["x_gt"])
        l_node = node_normal_loss(outputs["n_pred"], targets["n_gt"])
        l_mid = edge_midpoint_loss(outputs["mid_pred"], targets["mid_gt"])
        l_face = face_normal_loss(
            outputs["n1_pred"], targets["n1_gt"], outputs["n2_pred"], targets["n2_gt"]
        )
        l_emb_v = cluster_consistency_loss(outputs["vertex_embedding"], targets["vertex_cluster_id"])
        l_emb_e = cluster_consistency_loss(outputs["edge_embedding"], targets["edge_cluster_id"])

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
