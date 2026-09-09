"""
The rotation-convention regression test.

If any of these fail, the model is being asked to learn something it provably
cannot, and training will plateau near chance with no error message. See the
long argument in `vngat/models/vn_layers.predict_rotation`.
"""
from __future__ import annotations

import pytest
import torch

from conftest import random_rotation
from vngat.models.vn_layers import (
    geodesic_rotation_loss, gram_schmidt_frame, predict_rotation,
)


def test_gram_schmidt_produces_proper_rotations():
    a1, a2 = torch.randn(64, 3), torch.randn(64, 3)
    R = gram_schmidt_frame(a1, a2)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(64), atol=1e-5)


def test_frame_is_left_equivariant():
    """F(A x) == A F(x): the property every VN primitive has."""
    a1, a2 = torch.randn(32, 3), torch.randn(32, 3)
    A = random_rotation(1)[0]
    lhs = gram_schmidt_frame(a1 @ A.T, a2 @ A.T)
    rhs = A @ gram_schmidt_frame(a1, a2)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_predicted_rotation_is_right_equivariant():
    """G(A x) == G(x) A^T -- what makes the target learnable."""
    a1, a2 = torch.randn(32, 3), torch.randn(32, 3)
    A = random_rotation(1)[0]
    lhs = predict_rotation(a1 @ A.T, a2 @ A.T)
    rhs = predict_rotation(a1, a2) @ A.T
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_transposed_head_recovers_the_target_for_every_rotation():
    """
    The decisive test. A network that has learned G(x_clean) = I on canonical
    geometry must output exactly A^T for EVERY scattering rotation A -- with no
    further learning. That is the entire payoff of equivariance.
    """
    clean = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a1_clean, a2_clean = clean[0], clean[1]
    assert torch.allclose(predict_rotation(a1_clean, a2_clean), torch.eye(3), atol=1e-6)

    for _ in range(8):
        A = random_rotation(1)[0]
        R_pred = predict_rotation(a1_clean @ A.T, a2_clean @ A.T)
        assert torch.allclose(R_pred, A.T, atol=1e-5), "transposed head failed to recover A^T"
        assert geodesic_rotation_loss(R_pred, A.T) < 1e-3


def test_untransposed_head_is_inconsistent_across_rotations():
    """
    The bug, made explicit: WITHOUT the transpose, the value the head would
    have to output for the same clean geometry depends on A, so no single
    function of the clean input can satisfy it.
    """
    clean = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    required = []
    for _ in range(2):
        A = random_rotation(1)[0]
        # If the head returned F directly, then F(A x) = A F(x) = A^T would
        # force F(x_clean) = A^T A^T = (A^2)^T.
        required.append((A @ A).T)
    difference = (required[0] - required[1]).abs().max()
    assert difference > 0.1, "expected two different A to demand different F(x_clean)"


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 90.0, 179.0])
def test_geodesic_loss_matches_known_angles(angle_deg):
    theta = torch.deg2rad(torch.tensor(angle_deg))
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
    measured = geodesic_rotation_loss(R, torch.eye(3).unsqueeze(0))
    assert torch.allclose(torch.rad2deg(measured), torch.tensor([angle_deg]), atol=0.1)


def test_gram_schmidt_is_scale_invariant():
    """
    A fixed epsilon is negligible against ||a|| ~ 1 but comparable to it once
    the predicted vectors shrink. Before the inputs were pre-normalised the
    frame drifted 2e-9 from orthogonal at ||a|| ~ 1e-3 and 2e-3 at ||a|| ~ 1e-6,
    worst for pairs whose channels are nearly parallel -- exactly the
    high-`head_cos` regime real training runs sat in. Small activations are also
    what an untrained network produces, so this is not a corner case.
    """
    torch.manual_seed(0)
    a1 = torch.randn(64, 3, dtype=torch.float64)
    a2 = torch.randn(64, 3, dtype=torch.float64)
    for scale in (1e3, 1.0, 1e-3, 1e-6, 1e-9):
        R = gram_schmidt_frame(a1 * scale, a2 * scale)
        eye = torch.eye(3, dtype=torch.float64).expand_as(R)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-11), f"scale {scale}"
        assert torch.allclose(torch.linalg.det(R), torch.ones(64, dtype=torch.float64),
                              atol=1e-11), f"scale {scale}"


def test_pre_normalisation_does_not_change_the_frame():
    """The whole point is that it is mathematically identical: the frame depends
    only on the directions of a1 and a2, so scaling either input must leave the
    result untouched."""
    torch.manual_seed(1)
    a1 = torch.randn(32, 3, dtype=torch.float64)
    a2 = torch.randn(32, 3, dtype=torch.float64)
    base = gram_schmidt_frame(a1, a2)
    assert torch.allclose(base, gram_schmidt_frame(a1 * 7.0, a2), atol=1e-12)
    assert torch.allclose(base, gram_schmidt_frame(a1, a2 * 0.01), atol=1e-12)


def test_scale_invariance_holds_for_nearly_parallel_channels():
    """The regime the fix targets: |cos| ~ 0.9 between the two channels."""
    torch.manual_seed(2)
    a1 = torch.randn(64, 3, dtype=torch.float64)
    a1 = a1 / a1.norm(dim=-1, keepdim=True)
    a2 = a1 + 0.2 * torch.randn(64, 3, dtype=torch.float64)     # |cos| ~ 0.9
    for scale in (1.0, 1e-4, 1e-7):
        R = gram_schmidt_frame(a1 * scale, a2 * scale)
        eye = torch.eye(3, dtype=torch.float64).expand_as(R)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-11), f"scale {scale}"
