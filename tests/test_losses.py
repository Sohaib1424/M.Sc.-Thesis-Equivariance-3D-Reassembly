"""Loss terms: correctness, non-degeneracy, and the DDP gradient-path fix."""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from reassembly.training.losses import (  # noqa: E402
    CompositeLoss, cluster_consistency_loss, edge_midpoint_loss,
    face_normal_loss, node_normal_loss, node_position_loss,
)

pytestmark = pytest.mark.torch


def test_position_loss_is_zero_at_the_optimum_and_known_otherwise():
    x = torch.randn(20, 3)
    assert float(node_position_loss(x, x)) == pytest.approx(0.0, abs=1e-12)
    shifted = x + torch.tensor([1.0, 0.0, 0.0])
    assert float(node_position_loss(shifted, x)) == pytest.approx(1.0, rel=1e-6)


def test_normal_loss_is_zero_when_aligned_and_two_when_opposed():
    n = torch.nn.functional.normalize(torch.randn(30, 3), dim=-1)
    assert float(node_normal_loss(n, n)) == pytest.approx(0.0, abs=1e-6)
    assert float(node_normal_loss(-n, n)) == pytest.approx(2.0, abs=1e-6)


def test_normal_loss_ignores_magnitude():
    n = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    assert float(node_normal_loss(n * 7.0, n)) == pytest.approx(0.0, abs=1e-6)


def test_midpoint_loss_matches_position_loss_semantics():
    m = torch.randn(15, 3)
    assert float(edge_midpoint_loss(m, m)) == pytest.approx(0.0, abs=1e-12)


def test_face_normal_loss_sums_both_slots():
    n = torch.nn.functional.normalize(torch.randn(12, 3), dim=-1)
    assert float(face_normal_loss(n, n, n, n)) == pytest.approx(0.0, abs=1e-6)
    assert float(face_normal_loss(-n, n, -n, n)) == pytest.approx(4.0, abs=1e-6)
    assert float(face_normal_loss(n, n, -n, n)) == pytest.approx(2.0, abs=1e-6)


def test_cluster_loss_is_zero_iff_members_agree():
    emb = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, -1.0], [5.0, -1.0]])
    cid = torch.tensor([0, 0, 1, 1])
    assert float(cluster_consistency_loss(emb, cid)) == pytest.approx(0.0, abs=1e-9)


def test_cluster_loss_is_not_degenerate_under_cancellation():
    """The design document's literal formula, ``||sum_c z||^2``, is minimized by
    embeddings that CANCEL rather than agree: (3,-1)+(3,-1) scores 40 while
    (5,0)+(-5,0) scores 0. The corrected centroid form must order these the
    other way round."""
    cid = torch.tensor([0, 0])
    identical = torch.tensor([[3.0, -1.0], [3.0, -1.0]])
    opposed = torch.tensor([[5.0, 0.0], [-5.0, 0.0]])

    literal = lambda e: float(e.sum(0).pow(2).sum())
    assert literal(identical) > literal(opposed)          # the bug, reproduced

    assert float(cluster_consistency_loss(identical, cid)) < \
        float(cluster_consistency_loss(opposed, cid))     # corrected ordering
    assert float(cluster_consistency_loss(identical, cid)) == pytest.approx(0.0, abs=1e-9)


def test_cluster_loss_excludes_unlabelled_entries():
    emb = torch.tensor([[0.0, 0.0], [0.0, 0.0], [99.0, 99.0]])
    cid = torch.tensor([0, 0, -1])
    assert float(cluster_consistency_loss(emb, cid)) == pytest.approx(0.0, abs=1e-9)


def test_cluster_loss_averages_within_before_across():
    """A large cluster must not dominate purely by member count."""
    big = torch.zeros(100, 2)
    big[:, 0] = torch.linspace(-1, 1, 100)
    emb = torch.cat([big, torch.tensor([[0.0, 0.0], [0.0, 10.0]])])
    cid = torch.cat([torch.zeros(100, dtype=torch.long), torch.tensor([1, 1])])
    value = float(cluster_consistency_loss(emb, cid))
    # cluster 1's per-member squared distance to its centroid is 25 each
    assert value > 10.0


def test_cluster_loss_returns_a_gradient_connected_zero():
    """THE DDP fix. With no shared entries the loss is zero -- but it must
    still be attached to the graph, or the embedding heads receive no gradient
    and DistributedDataParallel raises 'Expected to have finished reduction'."""
    emb = torch.randn(10, 4, requires_grad=True)
    cid = torch.full((10,), -1, dtype=torch.long)
    loss = cluster_consistency_loss(emb, cid)

    assert float(loss) == 0.0
    assert loss.requires_grad, "detached zero -- DDP would fail on this batch"
    loss.backward()
    assert emb.grad is not None
    assert torch.allclose(emb.grad, torch.zeros_like(emb.grad))


def _make_io(shared=True, requires_grad=False):
    torch.manual_seed(0)
    F, V, E, D = 3, 12, 9, 5
    R_gt = torch.linalg.qr(torch.randn(F, 3, 3))[0]
    R_gt = R_gt * torch.sign(torch.det(R_gt)).view(F, 1, 1)
    x = torch.randn(V, 3)
    n = torch.nn.functional.normalize(torch.randn(V, 3), dim=-1)
    m = torch.randn(E, 3)
    n1 = torch.nn.functional.normalize(torch.randn(E, 3), dim=-1)
    n2 = torch.nn.functional.normalize(torch.randn(E, 3), dim=-1)
    vemb = torch.randn(V, D, requires_grad=requires_grad)
    eemb = torch.randn(E, D, requires_grad=requires_grad)
    vcid = torch.tensor([0, 0, 1, 1] + [-1] * (V - 4)) if shared else torch.full((V,), -1)
    ecid = torch.tensor([0, 0] + [-1] * (E - 2)) if shared else torch.full((E,), -1)
    outputs = dict(R_pred=R_gt.clone(), x_pred=x, n_pred=n, mid_pred=m,
                   n1_pred=n1, n2_pred=n2, vertex_embedding=vemb, edge_embedding=eemb)
    targets = dict(R_gt=R_gt, x_gt=x, n_gt=n, mid_gt=m, n1_gt=n1, n2_gt=n2,
                   vertex_cluster_id=vcid.long(), edge_cluster_id=ecid.long())
    return outputs, targets


def test_composite_loss_is_zero_for_a_perfect_prediction():
    outputs, targets = _make_io(shared=False)
    losses = CompositeLoss()(outputs, targets)
    for key in ("rot", "pos", "node", "mid", "face"):
        assert float(losses[key]) == pytest.approx(0.0, abs=1e-5), key
    assert float(losses["rot_deg"]) == pytest.approx(0.0, abs=1e-2)


def test_composite_loss_reports_every_component():
    outputs, targets = _make_io()
    losses = CompositeLoss()(outputs, targets)
    for key in ("total", "rot", "pos", "node", "mid", "face", "embv", "embe", "rot_deg"):
        assert key in losses


def test_composite_loss_weights_are_applied():
    outputs, targets = _make_io()
    outputs["x_pred"] = outputs["x_pred"] + 1.0
    base = float(CompositeLoss(w_pos=1.0)(outputs, targets)["total"])
    doubled = float(CompositeLoss(w_pos=2.0)(outputs, targets)["total"])
    single = float(CompositeLoss()(outputs, targets)["pos"])
    assert doubled == pytest.approx(base + single, rel=1e-5)


def test_composite_loss_keeps_gradients_when_nothing_is_shared():
    outputs, targets = _make_io(shared=False, requires_grad=True)
    CompositeLoss()(outputs, targets)["total"].backward()
    assert outputs["vertex_embedding"].grad is not None
    assert outputs["edge_embedding"].grad is not None


def test_auto_balance_learns_weights():
    outputs, targets = _make_io()
    loss_fn = CompositeLoss(auto_balance=True)
    assert any(p.requires_grad for p in loss_fn.parameters())
    loss_fn(outputs, targets)["total"].backward()
    assert loss_fn.log_vars.grad is not None


def test_rot_deg_is_reported_in_degrees():
    outputs, targets = _make_io()
    import numpy as np
    th = np.deg2rad(30.0)
    pert = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                         [np.sin(th), np.cos(th), 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)
    outputs["R_pred"] = pert @ targets["R_gt"]
    assert float(CompositeLoss()(outputs, targets)["rot_deg"]) == pytest.approx(30.0, abs=0.2)
