"""Composite loss terms, including the degeneracy fix and the DDP zero."""
from __future__ import annotations

import pytest
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


def test_cluster_consistency_prefers_tight_clusters():
    tight = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    sloppy = torch.tensor([[1.0, 0.4], [1.0, -0.4], [-1.0, 0.4], [-1.0, -0.4]])
    cid = torch.tensor([0, 0, 1, 1])
    assert float(cluster_consistency_loss(tight, cid)) < float(
        cluster_consistency_loss(sloppy, cid))


def test_collapsed_embeddings_are_penalised():
    """A constant embedding drives within-cluster variance to zero, so a
    pull-only loss rates it perfect -- and a real run duly collapsed both
    embedding terms to 0.0000 within two epochs."""
    cid = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    collapsed = torch.ones(4, 2)
    assert float(cluster_consistency_loss(collapsed, cid)) > float(
        cluster_consistency_loss(good, cid))


def test_inflating_embedding_magnitude_does_not_help():
    """
    THE second-order regression guard. With unnormalised embeddings, `push`
    could be satisfied by scaling everything up rather than arranging it, and a
    real run drove mean centroid norm to ~565 -- past the point where squaring
    overflows float16, which produced NaN losses in bursts from epoch 19.
    Normalising makes the loss exactly scale-invariant.
    """
    cid = torch.tensor([0, 0, 1, 1])
    small = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    huge = small * 500.0
    assert abs(float(cluster_consistency_loss(small, cid))
               - float(cluster_consistency_loss(huge, cid))) < 1e-6


def test_loss_is_bounded():
    """Bounded by construction, so this term cannot dominate the gradient
    budget -- it was 63% of the total at initialisation before normalising."""
    torch.manual_seed(0)
    worst = torch.ones(64, 16)                      # everything identical
    cid = torch.arange(64) % 8
    assert float(cluster_consistency_loss(worst, cid)) <= 1.01


def test_disabling_repulsion_reproduces_the_collapse_degeneracy():
    cid = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    collapsed = torch.ones(4, 2)
    assert float(cluster_consistency_loss(collapsed, cid, push_margin=0.0)) <= float(
        cluster_consistency_loss(good, cid, push_margin=0.0)) + 1e-3


def test_cluster_consistency_rejects_the_cancellation_degeneracy():
    """The design document's `|| sum z ||^2` scores opposed embeddings BETTER
    than identical ones. This loss must do the reverse."""
    identical = torch.tensor([[3.0, -1.0], [3.0, -1.0]])
    opposed = torch.tensor([[5.0, 0.0], [-5.0, 0.0]])
    cid = torch.tensor([0, 0])
    assert float(opposed.sum(0).pow(2).sum()) < float(identical.sum(0).pow(2).sum())
    assert float(cluster_consistency_loss(identical, cid)) < float(
        cluster_consistency_loss(opposed, cid))


def test_cluster_consistency_ignores_unshared_points():
    emb = torch.tensor([[1.0, 1.0], [1.0, 1.0], [99.0, -99.0]])
    with_unshared = cluster_consistency_loss(emb, torch.tensor([0, 0, -1]))
    without = cluster_consistency_loss(emb[:2], torch.tensor([0, 0]))
    assert abs(float(with_unshared) - float(without)) < 1e-6


def test_cluster_consistency_averages_within_clusters_first():
    """A 100-member cluster must not dominate a 2-member one by size alone."""
    geometry = [[1.0, 0.0], [-1.0, 0.0]]
    small = torch.tensor([geometry[0]] * 2 + [geometry[1]] * 2)
    big = torch.tensor([geometry[0]] * 100 + [geometry[1]] * 2)
    assert abs(float(cluster_consistency_loss(small, torch.tensor([0, 0, 1, 1])))
               - float(cluster_consistency_loss(big, torch.tensor([0] * 100 + [1, 1])))) < 1e-6


def test_cluster_consistency_keeps_parameters_connected_when_empty():
    """
    With no shared points the loss must still be graph-connected, so every
    parameter gets a (zero) gradient. Otherwise DDP raises
    "Expected to have finished reduction in the prior iteration".
    """
    emb = torch.randn(5, 4, requires_grad=True)
    loss = cluster_consistency_loss(emb, torch.full((5,), -1))
    assert float(loss.detach()) == 0.0
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert emb.grad is not None and torch.equal(emb.grad, torch.zeros_like(emb))


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


def test_cluster_consistency_survives_mixed_precision_inputs():
    """
    Half-precision embeddings must not crash or return NaN.

    Regression guard for a crash that only appeared under AMP: autocast
    promotes `pow` and `sum` to float32, so the per-point term came back Float
    while the accumulator had been allocated from the (Half) embedding dtype,
    and `index_add_` refused the pair.
    """
    emb = torch.randn(40, 8, dtype=torch.float16)
    cid = torch.randint(-1, 5, (40,))
    loss = cluster_consistency_loss(emb, cid)
    assert torch.isfinite(loss).all()
    assert loss.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
def test_full_loss_runs_under_autocast():
    """
    The whole composite loss inside `torch.autocast`, which is how the trainer
    calls it. None of the other tests exercise this path -- and the smoke
    config sets amp: false, so it does not either.
    """
    from conftest import random_rotation

    device = torch.device("cuda")
    loss_fn = CompositeLoss()
    F_, V, E = 3, 60, 90
    R_gt = random_rotation(F_).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = {
            "R_pred": R_gt.clone(),
            "x_pred": torch.randn(V, 3, device=device),
            "n_pred": torch.randn(V, 3, device=device),
            "mid_pred": torch.randn(E, 3, device=device),
            "n1_pred": torch.randn(E, 3, device=device),
            "n2_pred": torch.randn(E, 3, device=device),
            # half, exactly as the embedding heads produce under autocast
            "vertex_embedding": torch.randn(V, 8, device=device).half(),
            "edge_embedding": torch.randn(E, 8, device=device).half(),
        }
        targets = {
            "R_gt": R_gt,
            "x_gt": torch.randn(V, 3, device=device),
            "n_gt": torch.randn(V, 3, device=device),
            "mid_gt": torch.randn(E, 3, device=device),
            "n1_gt": torch.randn(E, 3, device=device),
            "n2_gt": torch.randn(E, 3, device=device),
            "vertex_cluster_id": torch.randint(-1, 4, (V,), device=device),
            "edge_cluster_id": torch.randint(-1, 4, (E,), device=device),
        }
        out = loss_fn(outputs, targets)
    for key in ("total", "rot", "rot_deg", "pos", "node", "mid", "face", "emb_v", "emb_e"):
        assert torch.isfinite(out[key]).all(), f"{key} not finite under autocast"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
def test_model_forward_and_backward_under_autocast(scene):
    """Full forward + backward under AMP -- the exact trainer configuration."""
    from vngat.models.vn_gat import VNGATModel
    from vngat.training.bridge import build_model_inputs, build_predictions, build_targets
    from conftest import random_rotation

    device = torch.device("cuda")
    graph = scene.to(device)
    model = VNGATModel(hidden_channels=16, num_layers=2, num_vn_slots=3,
                       heads=2, embed_dim=8).to(device)
    A = random_rotation(graph.num_fragments).to(device)
    diffused = graph.rotate_per_fragment(A)
    loss_fn = CompositeLoss()

    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**build_model_inputs(diffused))
        merged = dict(R_pred=out["R_pred"],
                      vertex_embedding=out["vertex_embedding"],
                      edge_embedding=out["edge_embedding"],
                      **build_predictions(diffused, out["R_pred"]))
        losses = loss_fn(merged, build_targets(graph, A, diffused))
    losses["total"].backward()

    assert torch.isfinite(losses["total"]).all()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters received no gradient (breaks DDP): {missing}"
