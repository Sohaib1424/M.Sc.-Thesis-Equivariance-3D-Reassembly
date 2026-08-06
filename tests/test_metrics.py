"""GARF-comparable metrics."""
import numpy as np
import pytest

from reassembly.evaluation.metrics import (
    aggregate, chamfer_distance, evaluate_assembly, rmse_rotation_euler,
    rmse_rotation_geodesic, rmse_translation, rotation_error_euler_deg,
    rotation_error_geodesic_deg, sample_surface,
)
from conftest import random_rotation


def test_identical_assembly_scores_perfectly(rng):
    R = np.stack([random_rotation(rng) for _ in range(4)])
    t = rng.normal(size=(4, 3))
    verts = [rng.normal(size=(120, 3)) for _ in range(4)]
    m = evaluate_assembly(R, R, t, t, verts, points_per_part=300)
    assert m.rmse_rotation_euler_deg == pytest.approx(0.0, abs=1e-6)
    assert m.rmse_rotation_geodesic_deg == pytest.approx(0.0, abs=1e-4)
    assert m.rmse_translation == pytest.approx(0.0, abs=1e-12)
    assert m.part_accuracy == 1.0
    assert m.chamfer_distance == pytest.approx(0.0, abs=1e-12)


def test_euler_error_wraps_around():
    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    err = rotation_error_euler_deg(rz(np.deg2rad(1))[None], rz(np.deg2rad(-1))[None])
    assert err.max() == pytest.approx(2.0, abs=1e-6)   # not 358


def test_euler_and_geodesic_are_different_conventions(rng):
    """They are not interchangeable; a comparison table must state which."""
    R_gt = np.stack([random_rotation(rng) for _ in range(200)])
    R_pred = np.stack([random_rotation(rng) for _ in range(200)])
    eu = rmse_rotation_euler(R_pred, R_gt)
    ge = rmse_rotation_geodesic(R_pred, R_gt)
    assert not np.isclose(eu, ge, rtol=0.1)
    assert 0.4 < eu / ge < 0.9


def test_geodesic_matches_a_known_angle(rng):
    axis = np.array([0.0, 0.0, 1.0])
    for deg in (5.0, 30.0, 90.0, 179.0):
        th = np.deg2rad(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        err = rotation_error_geodesic_deg(R[None], np.eye(3)[None])
        assert err[0] == pytest.approx(deg, abs=1e-4)


def test_rotation_error_grows_monotonically(rng):
    R_gt = np.stack([random_rotation(rng) for _ in range(30)])
    prev = -1.0
    for deg in (1, 5, 20, 60):
        th = np.deg2rad(deg)
        K = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        pert = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        err = rmse_rotation_geodesic(np.einsum("ij,njk->nik", pert, R_gt), R_gt)
        assert err > prev
        prev = err


def test_chamfer_is_zero_for_identical_clouds(rng):
    a = rng.normal(size=(200, 3))
    assert chamfer_distance(a, a) == pytest.approx(0.0, abs=1e-12)


def test_chamfer_grows_with_displacement(rng):
    a = rng.normal(size=(200, 3))
    prev = -1.0
    for shift in (0.0, 0.1, 0.5, 2.0):
        d = chamfer_distance(a, a + shift)
        assert d > prev
        prev = d


def test_chamfer_squared_flag_is_consistent():
    """squared=True averages SQUARED nearest distances -- the convention this
    literature uses, and the reason a 0.01 PA threshold corresponds to a 0.1
    gap in shape units rather than 0.01."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 0.0, 0.0]])
    plain = chamfer_distance(a, b, squared=False, bidirectional=False)
    squared = chamfer_distance(a, b, squared=True, bidirectional=False)
    assert plain == pytest.approx(3.0)
    assert squared == pytest.approx(9.0)
    assert squared == pytest.approx(plain ** 2)


def test_chamfer_bidirectional_sums_both_directions():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 0.0, 0.0]])
    one = chamfer_distance(a, b, squared=True, bidirectional=False)
    both = chamfer_distance(a, b, squared=True, bidirectional=True)
    assert both == pytest.approx(2 * one)


def test_translation_rmse(rng):
    t = rng.normal(size=(6, 3))
    assert rmse_translation(t, t) == pytest.approx(0.0)
    assert rmse_translation(t + 1.0, t) == pytest.approx(np.sqrt(3.0), rel=1e-9)


def test_part_accuracy_responds_to_the_threshold(rng):
    R = np.stack([np.eye(3)] * 3)
    verts = [rng.normal(size=(80, 3)) for _ in range(3)]
    t_gt = np.zeros((3, 3))
    t_pred = np.array([[0.0, 0, 0], [0.3, 0, 0], [3.0, 0, 0]])
    loose = evaluate_assembly(R, R, t_pred, t_gt, verts, pa_threshold=1.0, points_per_part=200)
    tight = evaluate_assembly(R, R, t_pred, t_gt, verts, pa_threshold=1e-4, points_per_part=200)
    assert loose.part_accuracy > tight.part_accuracy
    assert tight.part_accuracy == pytest.approx(1 / 3)


def test_surface_sampling_is_area_weighted(rng):
    """A big triangle should receive far more samples than a tiny one."""
    V = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 0.01], [0.01, 0, 0.01]])
    F = np.array([[0, 1, 2], [0, 3, 4]])
    pts = sample_surface(V, F, 4000, rng)
    on_big = (pts[:, 2] < 1e-6).sum()
    assert on_big / len(pts) > 0.95


def test_aggregate_averages_over_scenes(rng):
    R = np.stack([random_rotation(rng) for _ in range(2)])
    t = rng.normal(size=(2, 3))
    verts = [rng.normal(size=(50, 3)) for _ in range(2)]
    ms = [evaluate_assembly(R, R, t, t, verts, points_per_part=100) for _ in range(3)]
    agg = aggregate(ms)
    assert agg["num_scenes"] == 3
    assert agg["PA"] == pytest.approx(1.0)
