"""
The composite training objective.

    L = w_rot L_rot + w_pos L_pos + w_node L_node + w_mid L_mid
        + w_face L_face + w_embv L_emb-v + w_embe L_emb-e

The four local-geometry terms are NOT separate network outputs. They are the
input geometry rotated by the network's predicted rotation:
``x_pred = R_pred[frag(v)] @ x_diffused_v``, and likewise for normals,
midpoints and face normals. That gives a spatially distributed gradient on top
of the single global rotation term, at the cost of one matmul.

(An earlier draft read "position" and "normal" off fixed channel indices of the
learned hidden features. That is meaningless the moment any channel-mixing
layer runs -- nothing keeps hidden channel 0 semantically "position" once
training starts moving weights.)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vn_layers import geodesic_rotation_angle, rotation_loss

LOSS_KEYS = ("rot", "pos", "node", "mid", "face", "embv", "embe")


def _zero_like_connected(t: torch.Tensor) -> torch.Tensor:
    """A zero that still carries a gradient path back to ``t``.

    THIS IS THE REAL FIX FOR THE DDP CRASH.

    ``cluster_consistency_loss`` returns 0 whenever a batch happens to contain
    no cross-fragment shared vertices/edges. Returning a fresh constant
    (``embeddings.new_tensor(0.0)``) detaches the embedding heads from the
    graph entirely, so their parameters receive no gradient that step. Plain
    single-GPU training tolerates that silently; DistributedDataParallel does
    not, and raises

        RuntimeError: Expected to have finished reduction ...
        Parameter indices which did not receive grad for rank 0: 89 ... 96

    (8 indices = the 2 embedding heads' 4 parameters each -- they are the
    last-registered submodules.)

    The usual workaround is ``find_unused_parameters=True``, which makes DDP
    traverse the autograd graph every single iteration to discover which
    parameters participated. That is a permanent per-step cost imposed to
    tolerate an occasional data condition. Multiplying by zero instead keeps
    the parameters in the graph with a gradient of exactly zero -- numerically
    identical, no per-step overhead, and it lets ``static_graph=True`` stay on.
    """
    return t.sum() * 0.0


def node_position_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    """``mean_v ||x_v - x_hat_v||^2``. Inputs ``(V, 3)``."""
    return ((x_pred - x_gt) ** 2).sum(dim=-1).mean()


def node_normal_loss(n_pred: torch.Tensor, n_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``mean_v (1 - n_v . n_hat_v)``, both re-normalized defensively."""
    n_pred = F.normalize(n_pred, dim=-1, eps=eps)
    n_gt = F.normalize(n_gt, dim=-1, eps=eps)
    return (1 - (n_pred * n_gt).sum(dim=-1)).mean()


def edge_midpoint_loss(m_pred: torch.Tensor, m_gt: torch.Tensor) -> torch.Tensor:
    """``mean_e ||m_e - m_hat_e||^2``. Inputs ``(E, 3)``."""
    return ((m_pred - m_gt) ** 2).sum(dim=-1).mean()


def face_normal_loss(n1_pred, n1_gt, n2_pred, n2_gt, eps: float = 1e-8) -> torch.Tensor:
    """``mean_e [(1 - n1.n1_hat) + (1 - n2.n2_hat)]``."""
    n1_pred, n1_gt = F.normalize(n1_pred, dim=-1, eps=eps), F.normalize(n1_gt, dim=-1, eps=eps)
    n2_pred, n2_gt = F.normalize(n2_pred, dim=-1, eps=eps), F.normalize(n2_gt, dim=-1, eps=eps)
    return ((1 - (n1_pred * n1_gt).sum(-1)) + (1 - (n2_pred * n2_gt).sum(-1))).mean()


def cluster_consistency_loss(embeddings: torch.Tensor, cluster_id: torch.Tensor) -> torch.Tensor:
    """Interface-embedding consistency: mean squared distance to the cluster
    centroid, averaged within a cluster first and then across clusters.

    Zero if and only if every member of every cluster has an identical
    embedding.

    WHY NOT THE DESIGN DOCUMENT'S FORMULA. The document writes
    ``sum_x || sum_{(f,v) in C(x)} z_v ||^2``, i.e. sum the embeddings of a
    cluster and penalize the norm of that sum. That has a degenerate global
    minimizer: it is minimized by embeddings that CANCEL, not by embeddings
    that agree. Two identical embeddings (3, -1) and (3, -1) score 40; two
    maximally opposed ones (5, 0) and (-5, 0) score 0. Optimizing it drives the
    interface representation towards mutual cancellation -- the opposite of the
    stated intent, and it would have looked like healthy loss curves the whole
    time. Averaging within a cluster before averaging across clusters also
    stops one large interface from dominating purely by member count.

    Entries with ``cluster_id == -1`` (not shared with any other fragment) are
    excluded; there is nothing for them to be consistent with.
    """
    mask = cluster_id >= 0
    if not bool(mask.any()):
        return _zero_like_connected(embeddings)

    emb = embeddings[mask]
    cid = cluster_id[mask]
    _unique, inverse = torch.unique(cid, return_inverse=True)
    num_clusters = int(inverse.max().item()) + 1
    D = emb.shape[-1]

    sums = torch.zeros(num_clusters, D, device=emb.device, dtype=emb.dtype)
    counts = torch.zeros(num_clusters, device=emb.device, dtype=emb.dtype)
    sums.index_add_(0, inverse, emb)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=emb.dtype))
    centroids = sums / counts.clamp_min(1).unsqueeze(-1)

    per_point = ((emb - centroids[inverse]) ** 2).sum(dim=-1)

    cluster_sums = torch.zeros(num_clusters, device=emb.device, dtype=emb.dtype)
    cluster_sums.index_add_(0, inverse, per_point)
    return (cluster_sums / counts.clamp_min(1)).mean()


class CompositeLoss(nn.Module):
    """All seven terms, plus reported metrics.

    ``auto_balance`` enables homoscedastic uncertainty weighting (Kendall,
    Gal & Cipolla, CVPR 2018): each term gets a learned log-variance ``s_i``
    and contributes ``exp(-s_i) * L_i + s_i``. Worth having here because the
    terms live on genuinely different scales -- squared distances in mesh
    units, cosine distances in [0, 2], a chordal rotation term in [0, 8] --
    and all seven weights are currently 1.0 and untuned. This gives a
    defensible starting point without a manual sweep. It is OFF by default so
    the baseline stays a fixed, reproducible objective.
    """

    def __init__(
        self,
        w_rot: float = 1.0,
        w_pos: float = 1.0,
        w_node: float = 1.0,
        w_mid: float = 1.0,
        w_face: float = 1.0,
        w_embv: float = 1.0,
        w_embe: float = 1.0,
        rot_loss: str = "chordal",
        auto_balance: bool = False,
    ):
        super().__init__()
        self.rot_loss = rot_loss
        self.auto_balance = auto_balance
        weights = dict(rot=w_rot, pos=w_pos, node=w_node, mid=w_mid,
                       face=w_face, embv=w_embv, embe=w_embe)
        self.register_buffer(
            "weight_vector",
            torch.tensor([weights[k] for k in LOSS_KEYS], dtype=torch.float32),
            persistent=False,
        )
        self.weights = weights
        if auto_balance:
            self.log_vars = nn.Parameter(torch.zeros(len(LOSS_KEYS)))

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        terms = {
            "rot": rotation_loss(outputs["R_pred"], targets["R_gt"], kind=self.rot_loss).mean(),
            "pos": node_position_loss(outputs["x_pred"], targets["x_gt"]),
            "node": node_normal_loss(outputs["n_pred"], targets["n_gt"]),
            "mid": edge_midpoint_loss(outputs["mid_pred"], targets["mid_gt"]),
            "face": face_normal_loss(outputs["n1_pred"], targets["n1_gt"],
                                     outputs["n2_pred"], targets["n2_gt"]),
            "embv": cluster_consistency_loss(
                outputs["vertex_embedding"], targets["vertex_cluster_id"]
            ),
            "embe": cluster_consistency_loss(
                outputs["edge_embedding"], targets["edge_cluster_id"]
            ),
        }

        if self.auto_balance:
            total = sum(
                torch.exp(-self.log_vars[i]) * terms[k] + self.log_vars[i]
                for i, k in enumerate(LOSS_KEYS)
            )
        else:
            total = sum(self.weights[k] * terms[k] for k in LOSS_KEYS)

        out = {"total": total, **terms}
        # Always report the geodesic angle in degrees, whatever is optimized:
        # this is the quantity GARF's RMSE(R) is expressed in, so it is the
        # number the thesis comparison ultimately turns on.
        with torch.no_grad():
            angle = geodesic_rotation_angle(outputs["R_pred"], targets["R_gt"])
            out["rot_deg"] = torch.rad2deg(angle).mean()
        return out
