"""Vector Neuron primitives: equivariance, and the rotation head's transform law."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from reassembly.models.vn_layers import (  # noqa: E402
    VNBatchNorm, VNInvariant, VNLayerNorm, VNLeakyReLU, VNLinear,
    chordal_rotation_loss, geodesic_rotation_angle, gram_schmidt_frame,
    rotation_6d_to_matrix, rotation_loss,
)

pytestmark = pytest.mark.torch


def random_rotation_t(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def rot(t, Q):
    """Apply Q to the trailing xyz axis of a (..., C, 3) tensor."""
    return torch.einsum("ij,...j->...i", Q, t)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vnlinear_is_equivariant(seed):
    torch.manual_seed(seed)
    layer = VNLinear(5, 7)
    x = torch.randn(11, 5, 3)
    Q = random_rotation_t(seed)
    assert torch.allclose(layer(rot(x, Q)), rot(layer(x), Q), atol=1e-5)


@pytest.mark.parametrize("share", [False, True])
def test_vnleakyrelu_is_equivariant(share):
    torch.manual_seed(0)
    layer = VNLeakyReLU(6, share_nonlinearity=share)
    x = torch.randn(13, 6, 3)
    Q = random_rotation_t(3)
    assert torch.allclose(layer(rot(x, Q)), rot(layer(x), Q), atol=1e-5)


def test_vnlayernorm_is_equivariant():
    torch.manual_seed(0)
    layer = VNLayerNorm(6)
    x = torch.randn(20, 6, 3)
    Q = random_rotation_t(4)
    assert torch.allclose(layer(rot(x, Q)), rot(layer(x), Q), atol=1e-5)


def test_vnlayernorm_is_per_sample():
    """Unlike batch norm, one node's output must not depend on the others --
    that is why it is the default here."""
    torch.manual_seed(0)
    layer = VNLayerNorm(4).eval()
    x = torch.randn(10, 4, 3)
    alone = layer(x[:1])
    together = layer(x)[:1]
    assert torch.allclose(alone, together, atol=1e-6)


def test_vnbatchnorm_is_equivariant():
    torch.manual_seed(0)
    layer = VNBatchNorm(6).eval()
    x = torch.randn(20, 6, 3)
    Q = random_rotation_t(5)
    assert torch.allclose(layer(rot(x, Q)), rot(layer(x), Q), atol=1e-5)


def test_vninvariant_output_does_not_change_under_rotation():
    layer = VNInvariant(5)
    x = torch.randn(9, 5, 3)
    Q = random_rotation_t(6)
    assert torch.allclose(layer(rot(x, Q)), layer(x), atol=1e-5)


def test_gram_schmidt_produces_a_proper_rotation():
    torch.manual_seed(0)
    a1, a2 = torch.randn(50, 3), torch.randn(50, 3)
    R = gram_schmidt_frame(a1, a2)
    eye = torch.eye(3).expand(50, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(50), atol=1e-5)


def test_gram_schmidt_frame_is_left_equivariant():
    torch.manual_seed(0)
    a1, a2 = torch.randn(20, 3), torch.randn(20, 3)
    Q = random_rotation_t(7)
    lhs = gram_schmidt_frame(a1 @ Q.T, a2 @ Q.T)
    rhs = Q @ gram_schmidt_frame(a1, a2)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_rotation_head_obeys_the_target_transform_law():
    """THE regression test for the rotation-convention bug.

    The supervision target is ``R_gt = R_diffuse^T``. Rotating the scattered
    input by Q makes the effective diffusion ``Q R_diffuse``, so the target
    transforms as ``R_gt -> R_gt Q^T``. The Gram-Schmidt FRAME transforms as
    ``F -> Q F``, which is a different law -- returning it directly made the
    model structurally unable to match its own supervision. The head must
    return ``F^T``.
    """
    torch.manual_seed(0)
    a1, a2 = torch.randn(20, 3), torch.randn(20, 3)
    Q = random_rotation_t(8)

    R = rotation_6d_to_matrix(a1, a2)
    R_rotated_input = rotation_6d_to_matrix(a1 @ Q.T, a2 @ Q.T)

    required = R @ Q.T                       # the law R_gt obeys
    wrong = Q @ R                            # the law the old head obeyed
    assert torch.allclose(R_rotated_input, required, atol=1e-5)
    assert not torch.allclose(R_rotated_input, wrong, atol=1e-3)


def test_rotation_head_still_returns_a_proper_rotation():
    torch.manual_seed(1)
    R = rotation_6d_to_matrix(torch.randn(30, 3), torch.randn(30, 3))
    assert torch.allclose(torch.det(R), torch.ones(30), atol=1e-5)


def test_geodesic_angle_matches_known_rotations():
    for deg in (0.0, 15.0, 90.0, 179.0):
        th = np.deg2rad(deg)
        R = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                          [np.sin(th), np.cos(th), 0.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)[None]
        angle = geodesic_rotation_angle(R, torch.eye(3)[None])
        assert abs(float(torch.rad2deg(angle)) - deg) < 1e-3


def test_chordal_loss_is_zero_at_the_optimum_and_increases_with_angle():
    R = rotation_6d_to_matrix(torch.randn(5, 3), torch.randn(5, 3))
    assert float(chordal_rotation_loss(R, R).max()) < 1e-8

    prev = -1.0
    for deg in (1.0, 10.0, 45.0, 120.0):
        th = np.deg2rad(deg)
        pert = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                             [np.sin(th), np.cos(th), 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
        value = float(chordal_rotation_loss(pert @ R, R).mean())
        assert value > prev
        prev = value


def test_chordal_gradient_vanishes_at_the_optimum_and_geodesic_does_not():
    """Why chordal is the default.

    NOT because the geodesic gradient explodes -- with the atan2 formulation it
    does not. The explosion in the original code came from the
    arccos-of-trace parameterisation, and it is gone.

    The real reason is the shape of the gradient near the optimum. Chordal
    behaves like an L2 loss: the gradient decays smoothly to zero as the
    prediction converges. The geodesic angle behaves like an L1 loss: bounded
    (a constant 1/sqrt(2)) but non-vanishing, with a kink at zero, so the
    update size never shrinks however close you get.
    """
    R_gt = rotation_6d_to_matrix(torch.randn(8, 3), torch.randn(8, 3)).detach()

    def grad_norm(kind, deg):
        th = np.deg2rad(deg)
        pert = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                             [np.sin(th), np.cos(th), 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
        R = (pert @ R_gt).clone().requires_grad_(True)
        rotation_loss(R, R_gt, kind=kind).sum().backward()
        return float(R.grad.norm())

    chordal_far, chordal_near = grad_norm("chordal", 30.0), grad_norm("chordal", 0.05)
    geodesic_far, geodesic_near = grad_norm("geodesic", 30.0), grad_norm("geodesic", 0.05)

    # chordal: vanishes as the prediction converges
    assert chordal_near < chordal_far / 100

    # geodesic: bounded -- this is the regression against the old arccos form,
    # where the same measurement gave ~500 at 0.1 degrees
    assert geodesic_near < 10.0, "geodesic gradient should not explode any more"
    assert geodesic_near > chordal_near, "geodesic does not vanish at the optimum"


def test_geodesic_angle_is_exactly_zero_for_identical_rotations():
    """The eps clamp in the old arccos form put a hard 0.028-degree floor on
    the reported angle, so RMSE(R) could never read zero -- and the perfect
    prediction test failed on it."""
    R = rotation_6d_to_matrix(torch.randn(16, 3), torch.randn(16, 3))
    angle = geodesic_rotation_angle(R, R)
    assert float(angle.abs().max()) == 0.0


def test_geodesic_angle_is_accurate_across_the_whole_range():
    for deg in (1e-3, 0.01, 0.1, 1.0, 45.0, 90.0, 150.0, 179.0, 179.99):
        th = np.deg2rad(deg)
        R = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                          [np.sin(th), np.cos(th), 0.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)[None]
        got = float(torch.rad2deg(geodesic_rotation_angle(R, torch.eye(3)[None])))
        assert abs(got - deg) < 1e-3, f"{deg} -> {got}"


def test_rotation_loss_rejects_unknown_kinds():
    R = torch.eye(3)[None]
    with pytest.raises(ValueError):
        rotation_loss(R, R, kind="nonsense")
