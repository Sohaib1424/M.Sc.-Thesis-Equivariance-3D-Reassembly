"""Full-model tests: shapes, valid rotations, equivariance law, DDP-safety."""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from reassembly.models.vn_gat import VNGATLayer  # noqa: E402
from reassembly.models.vn_gat_model import VNGATModel  # noqa: E402

pytestmark = pytest.mark.torch


def random_rotation_t(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def rot(t, Q):
    return torch.einsum("ij,...j->...i", Q, t)


def synthetic_batch(frag_sizes=(6, 8, 7, 9), scene_of=(0, 0, 1, 1), seed=0, degree=4):
    """A batch shaped exactly like collation produces: no cross-fragment edges,
    bidirectional with an explicit forward mask."""
    g = torch.Generator().manual_seed(seed)
    fragment_id = torch.cat(
        [torch.full((n,), i) for i, n in enumerate(frag_sizes)]
    ).long()
    N = sum(frag_sizes)

    src, dst, base = [], [], 0
    for n in frag_sizes:
        e = max(n * degree // 2, 1)
        s = torch.randint(0, n, (e,), generator=g) + base
        d = torch.randint(0, n, (e,), generator=g) + base
        keep = s != d
        src.append(s[keep]); dst.append(d[keep])
        base += n
    src, dst = torch.cat(src), torch.cat(dst)
    E = src.numel()

    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    forward_mask = torch.cat([torch.ones(E, dtype=torch.bool),
                              torch.zeros(E, dtype=torch.bool)])
    return dict(
        x=torch.randn(N, 2, 3, generator=g),
        edge_index=edge_index,
        edge_scalar=torch.rand(2 * E, 1, generator=g),
        edge_vec=torch.randn(2 * E, 3, 3, generator=g),
        fragment_id=fragment_id,
        num_fragments=len(frag_sizes),
        fragment_scene_id=torch.tensor(scene_of, dtype=torch.long),
        forward_edge_mask=forward_mask,
    )


def make_model(**kw):
    defaults = dict(hidden_channels=8, num_layers=2, num_vn_slots=4,
                    heads=2, embed_dim=6)
    defaults.update(kw)
    return VNGATModel(**defaults).eval()


def test_output_shapes():
    torch.manual_seed(0)
    batch = synthetic_batch()
    out = make_model()(**batch)
    N = batch["x"].shape[0]
    E_fwd = int(batch["forward_edge_mask"].sum())
    assert out["R_pred"].shape == (batch["num_fragments"], 3, 3)
    assert out["node_features"].shape == (N, 8, 3)
    assert out["vertex_embedding"].shape == (N, 6)
    assert out["edge_embedding"].shape == (E_fwd, 6)


def test_predicted_rotation_is_a_proper_rotation():
    torch.manual_seed(1)
    R = make_model()(**synthetic_batch())["R_pred"]
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)
    assert torch.allclose(torch.det(R), torch.ones(R.shape[0]), atol=1e-4)


def test_rotation_head_matches_target_law():
    """THE end-to-end regression for the rotation-convention bug.

    Rotating the whole input scene by Q must give ``R_pred @ Q^T`` -- the law
    the supervision target obeys -- and NOT ``Q @ R_pred``, which is what the
    equivariant frame alone would give.
    """
    torch.manual_seed(2)
    model = make_model()
    batch = synthetic_batch()
    Q = random_rotation_t(5)

    with torch.no_grad():
        base = model(**batch)
        rotated_batch = dict(batch)
        rotated_batch["x"] = rot(batch["x"], Q)
        rotated_batch["edge_vec"] = rot(batch["edge_vec"], Q)
        rotated = model(**rotated_batch)

    required = base["R_pred"] @ Q.T
    wrong = Q @ base["R_pred"]
    assert torch.allclose(rotated["R_pred"], required, atol=1e-3), \
        (rotated["R_pred"] - required).abs().max()
    assert not torch.allclose(rotated["R_pred"], wrong, atol=1e-2)


def test_backbone_features_co_rotate():
    torch.manual_seed(3)
    model = make_model()
    batch = synthetic_batch()
    Q = random_rotation_t(6)
    with torch.no_grad():
        base = model(**batch)
        rotated_batch = dict(batch)
        rotated_batch["x"] = rot(batch["x"], Q)
        rotated_batch["edge_vec"] = rot(batch["edge_vec"], Q)
        rotated = model(**rotated_batch)
    assert torch.allclose(rotated["node_features"], rot(base["node_features"], Q), atol=1e-3)


def test_embeddings_are_invariant():
    torch.manual_seed(4)
    model = make_model()
    batch = synthetic_batch()
    Q = random_rotation_t(7)
    with torch.no_grad():
        base = model(**batch)
        rotated_batch = dict(batch)
        rotated_batch["x"] = rot(batch["x"], Q)
        rotated_batch["edge_vec"] = rot(batch["edge_vec"], Q)
        rotated = model(**rotated_batch)
    assert torch.allclose(rotated["vertex_embedding"], base["vertex_embedding"], atol=1e-3)
    assert torch.allclose(rotated["edge_embedding"], base["edge_embedding"], atol=1e-3)


def test_no_cross_scene_leakage_at_model_level():
    torch.manual_seed(5)
    model = make_model()
    batch = synthetic_batch(frag_sizes=(6, 8, 7, 9), scene_of=(0, 0, 1, 1))
    scene_a = batch["fragment_id"] < 2

    with torch.no_grad():
        base = model(**batch)
        perturbed = dict(batch)
        x2 = batch["x"].clone()
        x2[~scene_a] = torch.randn_like(x2[~scene_a]) * 50
        perturbed["x"] = x2
        out2 = model(**perturbed)

    assert torch.allclose(base["R_pred"][:2], out2["R_pred"][:2], atol=1e-5), \
        "scene B's input changed scene A's predicted rotations"


def test_every_parameter_receives_a_gradient():
    """DDP with find_unused_parameters=False refuses to run otherwise. The
    embedding heads are the ones at risk, since their loss can be structurally
    zero on a batch with no shared interface."""
    torch.manual_seed(6)
    model = make_model().train()
    out = model(**synthetic_batch())
    loss = (out["R_pred"].square().sum()
            + out["vertex_embedding"].square().sum() * 0.0     # the zero case
            + out["edge_embedding"].square().sum() * 0.0)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient (DDP would fail): {missing}"


def test_gradient_checkpointing_gives_the_same_forward():
    torch.manual_seed(7)
    plain = make_model(gradient_checkpointing=False).train()
    ckpt = make_model(gradient_checkpointing=True).train()
    ckpt.load_state_dict(plain.state_dict())
    batch = synthetic_batch()
    a = plain(**batch)["R_pred"]
    b = ckpt(**batch)["R_pred"]
    assert torch.allclose(a, b, atol=1e-5)


def test_gradient_checkpointing_gives_the_same_gradients():
    torch.manual_seed(8)
    plain = make_model(gradient_checkpointing=False).train()
    ckpt = make_model(gradient_checkpointing=True).train()
    ckpt.load_state_dict(plain.state_dict())
    batch = synthetic_batch()

    for model in (plain, ckpt):
        model.zero_grad()
        model(**batch)["R_pred"].square().sum().backward()

    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert n1 == n2
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), n1


def test_head_dim_default_is_standard_multihead_splitting():
    """The memory fix: heads split a fixed width rather than each getting the
    full width. Message tensors are `heads` times smaller as a result."""
    layer = VNGATLayer(16, 16, heads=4)
    assert layer.head_dim == 4
    wide = VNGATLayer(16, 16, heads=4, head_dim=16)
    assert wide.head_dim == 16


def test_hidden_channels_must_divide_by_heads():
    with pytest.raises(ValueError):
        VNGATModel(hidden_channels=10, heads=4)


def test_angular_variant_runs_and_stays_equivariant():
    torch.manual_seed(9)
    model = make_model(angular=True)
    batch = synthetic_batch()
    Q = random_rotation_t(11)
    with torch.no_grad():
        base = model(**batch)
        rotated_batch = dict(batch)
        rotated_batch["x"] = rot(batch["x"], Q)
        rotated_batch["edge_vec"] = rot(batch["edge_vec"], Q)
        rotated = model(**rotated_batch)
    assert torch.allclose(rotated["R_pred"], base["R_pred"] @ Q.T, atol=1e-3)


def test_vn_gat_layer_alone_is_equivariant():
    torch.manual_seed(10)
    batch = synthetic_batch()
    layer = VNGATLayer(8, 8, heads=2).eval()
    x = torch.randn(batch["x"].shape[0], 8, 3)
    Q = random_rotation_t(12)
    base = layer(x, batch["edge_index"], batch["edge_scalar"], batch["edge_vec"])
    rotated = layer(rot(x, Q), batch["edge_index"], batch["edge_scalar"],
                    rot(batch["edge_vec"], Q))
    assert torch.allclose(rotated, rot(base, Q), atol=1e-4)
