"""Evaluation metrics, including the Euler-vs-geodesic distinction."""
from __future__ import annotations

import math

import torch

from conftest import random_rotation
from vngat.evaluation.metrics import (
    chamfer_distance, euler_rmse, evaluate_scene, geodesic_angle,
    matrix_to_euler_xyz, part_accuracy, translation_rmse,
)


def _rot_z(deg):
    t = math.radians(deg)
    return torch.tensor([[math.cos(t), -math.sin(t), 0.0],
                         [math.sin(t), math.cos(t), 0.0],
                         [0.0, 0.0, 1.0]]).unsqueeze(0)


def test_perfect_prediction_has_zero_error():
    R = random_rotation(6)
    assert float(geodesic_angle(R, R).max()) < 1e-2
    assert float(euler_rmse(R, R).max()) < 1e-2


def test_geodesic_matches_a_known_angle():
    assert abs(float(geodesic_angle(_rot_z(30), torch.eye(3).unsqueeze(0))) - 30.0) < 0.05


def test_euler_rmse_is_smaller_than_the_geodesic_angle():
    """
    Documented, deliberate, and the reason both are reported: a single-axis
    30-degree error is 30 degrees geodesic but sqrt((30^2+0+0)/3) ~ 17.3
    degrees as an Euler RMSE. Comparing one against a table of the other
    misstates results.
    """
    geo = float(geodesic_angle(_rot_z(30), torch.eye(3).unsqueeze(0)))
    euler = float(euler_rmse(_rot_z(30), torch.eye(3).unsqueeze(0)))
    assert abs(euler - 30.0 / math.sqrt(3)) < 0.1
    assert euler < geo


def test_euler_conversion_round_trips_for_small_angles():
    R = _rot_z(20)
    angles = matrix_to_euler_xyz(R)
    assert abs(float(angles[0, 2]) - 20.0) < 0.05


def test_euler_rmse_wraps_around_360():
    assert float(euler_rmse(_rot_z(359.9), torch.eye(3).unsqueeze(0))) < 0.5


def test_gimbal_lock_does_not_produce_garbage():
    R = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).unsqueeze(0)
    angles = matrix_to_euler_xyz(R)
    assert torch.isfinite(angles).all()
    assert float(angles.abs().max()) <= 180.0 + 1e-3


def test_translation_rmse():
    t = torch.zeros(3, 3)
    t2 = torch.ones(3, 3)
    assert abs(float(translation_rmse(t, t2).mean()) - 1.0) < 1e-6


def test_chamfer_is_zero_for_identical_clouds():
    p = torch.randn(64, 3)
    assert float(chamfer_distance(p, p)) < 1e-8


def test_part_accuracy_threshold():
    chamfer = torch.tensor([0.001, 0.02, 0.005, float("nan")])
    assert abs(float(part_accuracy(chamfer, 0.01)) - 2 / 3) < 1e-6


def test_evaluate_scene_omits_translation_metrics_when_unavailable():
    R = random_rotation(4)
    metrics = evaluate_scene(R, R)
    assert "rmse_R_euler_deg" in metrics and "geodesic_deg" in metrics
    assert "rmse_T" not in metrics and "part_accuracy" not in metrics
