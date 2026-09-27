"""
The reported-only diagnostics.

Each of these exists to split one ambiguous number into two unambiguous ones,
so the thing to test is the *separation*, not just that the arithmetic runs.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from reassembly.evaluation.metrics import (
    chamfer_distance,
    format_group_table,
    group_means,
    head_collinearity,
    matrix_to_quaternion,
    part_accuracy,
    swing_twist_error,
)

DTYPE = torch.float64


def _about(axis: str, degrees: float) -> "torch.Tensor":
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    table = {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }
    return torch.tensor(table[axis], dtype=DTYPE).unsqueeze(0)


def _identity(n: int = 1):
    return torch.eye(3, dtype=DTYPE).expand(n, 3, 3)


def _haar(n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=generator, dtype=DTYPE)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


# --------------------------------------------------------------- quaternion --

def test_quaternion_round_trips_for_random_rotations():
    q = matrix_to_quaternion(_haar(2000, 0))
    assert torch.allclose(q.norm(dim=-1), torch.ones(2000, dtype=DTYPE), atol=1e-12)


def test_quaternion_is_accurate_near_180_degrees():
    """
    Where the textbook ``w = sqrt(1 + tr)/2`` loses its precision -- and where a
    model at chance lives, so the naive version degrades worst exactly where
    the diagnostic is read.
    """
    for degrees in (179.0, 179.9, 179.999, 180.0):
        q = matrix_to_quaternion(_about("x", degrees))
        angle = 2 * math.degrees(math.atan2(float(q[0, 1:].norm()), abs(float(q[0, 0]))))
        assert angle == pytest.approx(degrees, abs=1e-6)


# --------------------------------------------------------------- swing/twist --

@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 179.0])
def test_rotation_about_the_axis_is_all_twist(degrees):
    tilt, twist = swing_twist_error(_identity(), _about("z", degrees), axis="z")
    assert float(tilt) < 1e-9
    assert float(twist) == pytest.approx(degrees, abs=1e-6)


@pytest.mark.parametrize("degrees", [20.0, 60.0, 90.0])
def test_rotation_off_the_axis_is_all_tilt(degrees):
    tilt, twist = swing_twist_error(_identity(), _about("x", degrees), axis="z")
    assert float(tilt) == pytest.approx(degrees, abs=1e-6)
    assert float(twist) < 1e-9


def test_tilt_has_no_arccos_floor():
    """
    The regime the metric exists to detect is ``tilt ~ 0``. An ``arccos`` with
    the usual clamp bottoms out around 0.026 deg there, which is
    indistinguishable from a genuinely small tilt -- so the metric would be
    blind in exactly the case it was added for.
    """
    tilt, _twist = swing_twist_error(_identity(), _identity())
    assert float(tilt) < 1e-12


def test_the_two_failure_regimes_are_separated():
    """
    THE point of the metric. Both of these read ~90 deg of mean geodesic error
    and they call for opposite responses.
    """
    n = 4000
    uniform = _haar(n, 1)
    tilt_uniform, _ = swing_twist_error(_identity(n), uniform, axis="z")

    generator = torch.Generator().manual_seed(2)
    phi = (torch.rand(n, generator=generator, dtype=DTYPE) * 2 - 1) * math.pi
    c, s = torch.cos(phi), torch.sin(phi)
    zero, one = torch.zeros(n, dtype=DTYPE), torch.ones(n, dtype=DTYPE)
    azimuth = torch.stack([
        torch.stack([c, -s, zero], -1),
        torch.stack([s, c, zero], -1),
        torch.stack([zero, zero, one], -1),
    ], dim=-2)
    tilt_azimuth, twist_azimuth = swing_twist_error(_identity(n), azimuth, axis="z")

    assert float(tilt_uniform.mean()) > 80          # axis not recovered
    assert float(tilt_azimuth.mean()) < 1e-9        # axis recovered
    assert 80 < float(twist_azimuth.mean()) < 100   # azimuth not


def test_swing_and_twist_account_for_the_whole_residual():
    """A decomposition that dropped part of the error would look fine and
    under-report it; the two angles must bound the geodesic angle."""
    from reassembly.nn.losses import geodesic_angle

    predicted, target = _haar(500, 3), _haar(500, 4)
    tilt, twist = swing_twist_error(predicted, target, axis="z")
    geodesic = torch.rad2deg(geodesic_angle(predicted, target))
    # The swing-twist factorisation composes, so neither part can exceed the
    # whole by more than the composition allows, and their sum cannot be less.
    assert bool((tilt + twist + 1e-9 >= geodesic).all())


def test_a_perfect_prediction_has_no_residual_to_decompose():
    target = _haar(200, 5)
    tilt, twist = swing_twist_error(target, target)
    assert float(tilt.max()) < 1e-7 and float(twist.max()) < 1e-7


def test_an_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="axis must be one of"):
        swing_twist_error(_identity(), _identity(), axis="w")


# --------------------------------------------------------------------- head --

def test_collinearity_is_one_for_parallel_channels_and_zero_for_orthogonal():
    parallel = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    orthogonal = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    assert float(head_collinearity(parallel)) == pytest.approx(1.0)
    assert float(head_collinearity(orthogonal)) == pytest.approx(0.0, abs=1e-12)


def test_collinearity_ignores_sign_and_magnitude():
    """Gram-Schmidt cares about the angle between the channels, not their
    lengths or which way round they point."""
    a = torch.tensor([[[1.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]])
    assert float(head_collinearity(a)) == pytest.approx(1.0)


def test_collinearity_survives_a_zero_channel():
    """An untrained head can emit an exactly zero vector; a plain normalise
    would return NaN and take the whole epoch's diagnostic with it."""
    a = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    assert torch.isfinite(head_collinearity(a))


def test_collinearity_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(F, 2, 3\)"):
        head_collinearity(torch.zeros(4, 3, 3))


# ----------------------------------------------------------------- assembly --

def test_chamfer_is_exactly_zero_for_identical_clouds():
    """
    ``cdist`` expands the squared distance and cancels catastrophically for
    near-coincident points, putting a floor under every reported distance that
    grows with the coordinate magnitude. Differencing directly does not.
    """
    points = torch.randn(300, 3, dtype=DTYPE)
    assert float(chamfer_distance(points, points)) == 0.0
    assert float(chamfer_distance(points * 100.0, points * 100.0)) == 0.0


def test_chamfer_is_symmetric_and_grows_with_separation():
    a = torch.randn(200, 3, dtype=DTYPE)
    b = a + 0.5
    assert float(chamfer_distance(a, b)) == pytest.approx(float(chamfer_distance(b, a)))
    assert float(chamfer_distance(a, a + 1.0)) > float(chamfer_distance(a, b))


def test_chamfer_chunking_does_not_change_the_answer():
    a, b = torch.randn(500, 3, dtype=DTYPE), torch.randn(400, 3, dtype=DTYPE)
    assert float(chamfer_distance(a, b, chunk=64)) == pytest.approx(
        float(chamfer_distance(a, b, chunk=4096)), rel=1e-12)


def test_chamfer_on_an_empty_cloud_is_nan_not_zero():
    """Zero would read as a perfect match. A part that could not be scored is
    missing data, and the two must not be confused."""
    assert math.isnan(float(chamfer_distance(torch.zeros(0, 3, dtype=DTYPE),
                                             torch.randn(5, 3, dtype=DTYPE))))


def test_part_accuracy_excludes_unscorable_parts():
    """
    Counting a NaN as a miss quietly converts a pipeline defect into a
    worse-looking model, which is the direction of error that never gets
    investigated.
    """
    chamfer = torch.tensor([0.001, 0.02, 0.005, float("nan")])
    assert float(part_accuracy(chamfer, 0.01)) == pytest.approx(2 / 3)


def test_part_accuracy_with_nothing_scorable_is_nan():
    assert math.isnan(float(part_accuracy(torch.tensor([float("nan")]))))


# ----------------------------------------------------------------- grouping --

def test_group_means_are_sorted_worst_first_and_carry_counts():
    means = group_means([10.0, 20.0, 1.0], ["a", "a", "b"])
    assert list(means) == ["a", "b"]
    assert means["a"] == (15.0, 2)
    assert means["b"] == (1.0, 1)


def test_group_means_skip_nan_without_dropping_the_group():
    means = group_means([float("nan"), 4.0], ["a", "a"])
    assert means["a"] == (4.0, 1)


def test_group_means_reject_a_length_mismatch():
    """A silent zip() would truncate to the shorter list and mislabel every
    value after the first missing one."""
    with pytest.raises(ValueError, match="same length"):
        group_means([1.0, 2.0], ["a"])


def test_the_table_renders_and_truncates():
    means = group_means([3.0, 2.0, 1.0], ["a", "b", "c"])
    text = format_group_table(means, "error by category", top=2)
    assert "a" in text and "... 1 more" in text
    assert format_group_table({}, "empty") == "  empty: nothing to report"
