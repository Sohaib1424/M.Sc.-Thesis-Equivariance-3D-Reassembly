"""Composite loss terms, including the degeneracy fix and the DDP zero."""
from __future__ import annotations

import torch

from conftest import random_rotation
from vngat.losses.composite import (
    CompositeLoss, cluster_consistency_loss, edge_midpoint_loss,
    face_normal_loss, node_normal_loss, node_position_loss,
)


def test_geometric_terms_are_zero_at_the_optimum():
    x = torch.randn(20, 3)
    n = torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)
    assert float(node_position_loss(x, x)) < 1e-8
    assert float(node_normal_loss(n, n)) < 1e-6
    assert float(edge_midpoint_loss(x, x)) < 1e-8
    assert float(face_normal_loss(n, n, n, n)) < 1e-6


def test_normal_losses_peak_at_opposite_directions():
    n = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    assert abs(float(node_normal_loss(-n, n)) - 2.0) < 1e-5


def test_cluster_consistency_is_zero_only_when_members_agree():
    emb = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0], [5.0, 5.0]])
    cid = torch.tensor([0, 0, 1, 1])
    assert float(cluster_consistency_loss(emb, cid)) < 1e-8

    emb2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert float(cluster_consistency_loss(emb2, torch.tensor([0, 0]))) > 0.5


def test_cluster_consistency_rejects_the_cancellation_degeneracy():
    """
    The design document's `|| sum z ||^2` scores opposed embeddings BETTER
    than identical ones. This loss must do the reverse.
    """
    identical = torch.tensor([[3.0, -1.0], [3.0, -1.0]])
    opposed = torch.tensor([[5.0, 0.0], [-5.0, 0.0]])
    cid = torch.tensor([0, 0])

    doc_identical = float(identical.sum(0).pow(2).sum())
    doc_opposed = float(opposed.sum(0).pow(2).sum())
    assert doc_opposed < doc_identical            # the degeneracy, demonstrated

    assert float(cluster_consistency_loss(identical, cid)) < float(
        cluster_consistency_loss(opposed, cid))


def test_cluster_consistency_ignores_unshared_points():
    emb = torch.tensor([[1.0, 1.0], [1.0, 1.0], [99.0, -99.0]])
    cid = torch.tensor([0, 0, -1])
    assert float(cluster_consistency_loss(emb, cid)) < 1e-8


def test_cluster_consistency_keeps_parameters_connected_when_empty():
    """
    With no shared points the loss must still be graph-connected, so every
    parameter gets a (zero) gradient. Otherwise DDP raises
    "Expected to have finished reduction in the prior iteration".
    """
    emb = torch.randn(5, 4, requires_grad=True)
    loss = cluster_consistency_loss(emb, torch.full((5,), -1))
    assert float(loss) == 0.0
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert emb.grad is not None and torch.equal(emb.grad, torch.zeros_like(emb))


def test_cluster_consistency_averages_within_clusters_first():
    """A 100-member cluster must not dominate a 2-member one by size alone."""
    big = torch.zeros(100, 2)
    small = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    emb = torch.cat([big, small])
    cid = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(2, dtype=torch.long)])
    value = float(cluster_consistency_loss(emb, cid))
    assert abs(value - 0.5) < 1e-5     # mean of (0.0, 1.0)


def test_composite_loss_reports_every_component():
    loss_fn = CompositeLoss()
    F, V, E = 3, 40, 60
    R_gt = random_rotation(F)
    outputs = {
        "R_pred": R_gt.clone(),
        "x_pred": torch.randn(V, 3), "n_pred": torch.randn(V, 3),
        "mid_pred": torch.randn(E, 3), "n1_pred": torch.randn(E, 3),
        "n2_pred": torch.randn(E, 3),
        "vertex_embedding": torch.randn(V, 8), "edge_embedding": torch.randn(E, 8),
    }
    targets = {
        "R_gt": R_gt,
        "x_gt": torch.randn(V, 3), "n_gt": torch.randn(V, 3),
        "mid_gt": torch.randn(E, 3), "n1_gt": torch.randn(E, 3),
        "n2_gt": torch.randn(E, 3),
        "vertex_cluster_id": torch.randint(-1, 4, (V,)),
        "edge_cluster_id": torch.randint(-1, 4, (E,)),
    }
    out = loss_fn(outputs, targets)
    for key in ("total", "rot", "rot_deg", "pos", "node", "mid", "face", "emb_v", "emb_e"):
        assert key in out and torch.isfinite(out[key]).all()
    assert float(out["rot_deg"]) < 0.5          # exact rotation -> ~0 degrees


def test_zero_weight_removes_a_term():
    loss_fn = CompositeLoss(w_pos=0.0)
    assert loss_fn.weights["pos"] == 0.0
