"""
Tests for centring and normalisation.

The property that decides the default is the last one: per-scene normalisation
keeps two mating surfaces the same size, per-fragment normalisation does not.
Everything else here guards the invertibility the translation solver depends on.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.data.transforms import (
    denormalize_fragments,
    normalize_fragments,
    random_rotations,
)


def _scene():
    """A big body and a small chip -- deliberately very different sizes."""
    body = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    chip = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    chip.apply_translation([5.0, 0.0, 0.0])
    return [body, chip]


def test_scene_mode_uses_one_divisor_for_every_fragment():
    norm = normalize_fragments(_scene(), mode="scene")
    assert np.allclose(norm.divisor, norm.divisor[0])
    assert norm.divisor[0] == pytest.approx(norm.radius.max())


def test_fragment_mode_puts_every_fragment_on_the_unit_sphere():
    norm = normalize_fragments(_scene(), mode="fragment")
    for v in norm.vertices:
        assert np.linalg.norm(v, axis=1).max() == pytest.approx(1.0, abs=1e-9)


def test_scene_mode_preserves_relative_size_and_fragment_mode_destroys_it():
    """
    The reason `scene` is the default. Two mating surfaces are the same size in
    world units; per-fragment normalisation rescales them by different factors,
    so at the cross-fragment attention they no longer look like they fit.
    """
    fragments = _scene()
    world_ratio = 1.0 / 0.1                       # body radius / chip radius

    scene = normalize_fragments(fragments, mode="scene")
    scene_ratio = (np.linalg.norm(scene.vertices[0], axis=1).max()
                   / np.linalg.norm(scene.vertices[1], axis=1).max())
    assert scene_ratio == pytest.approx(world_ratio, rel=1e-6)

    per_frag = normalize_fragments(fragments, mode="fragment")
    frag_ratio = (np.linalg.norm(per_frag.vertices[0], axis=1).max()
                  / np.linalg.norm(per_frag.vertices[1], axis=1).max())
    assert frag_ratio == pytest.approx(1.0, abs=1e-9)
    assert abs(frag_ratio - world_ratio) > 8.0, "per-fragment should flatten the ratio"


@pytest.mark.parametrize("mode", ["scene", "fragment"])
def test_round_trip_recovers_world_coordinates(mode):
    """
    Stage two solves translation in the original frame, so normalisation has to
    be invertible or the centroids it is trying to recover are gone.
    """
    fragments = _scene()
    norm = normalize_fragments(fragments, mode=mode)
    back = denormalize_fragments(norm)
    for original, recovered in zip(fragments, back):
        assert np.allclose(np.asarray(original.vertices), recovered, atol=1e-9)


@pytest.mark.parametrize("mode", ["scene", "fragment"])
@pytest.mark.parametrize("seed", [0, 1])
def test_normalisation_is_rotation_equivariant(mode, seed):
    """
    Centroids rotate with the fragment and radii do not change, so normalising a
    rotated scene must give the rotated normalisation of the original. If it did
    not, the backbone's equivariance would be broken before the first layer.
    """
    fragments = _scene()
    R = random_rotations(1, np.random.default_rng(seed))[0]

    plain = normalize_fragments(fragments, mode=mode)
    rotated_inputs = []
    for f in fragments:
        g = f.copy()
        g.vertices = np.asarray(f.vertices) @ R.T
        rotated_inputs.append(g)
    rotated = normalize_fragments(rotated_inputs, mode=mode)

    assert np.allclose(rotated.divisor, plain.divisor, atol=1e-12)
    assert np.allclose(rotated.radius, plain.radius, atol=1e-12)
    for a, b in zip(rotated.vertices, plain.vertices):
        assert np.allclose(a, b @ R.T, atol=1e-12)


def test_rescale_false_only_centres():
    fragments = _scene()
    norm = normalize_fragments(fragments, rescale=False)
    assert np.allclose(norm.divisor, 1.0)
    for v, f in zip(norm.vertices, fragments):
        assert np.allclose(v.mean(axis=0), 0.0, atol=1e-9)
        assert np.linalg.norm(v, axis=1).max() == pytest.approx(
            np.linalg.norm(np.asarray(f.vertices) - np.asarray(f.vertices).mean(0),
                           axis=1).max())


def test_radius_is_reported_in_world_units_for_the_scale_feature():
    """
    `radius` is what feeds the invariant scale feature, so it must stay in world
    units regardless of which divisor was applied -- otherwise "how big is this
    fragment" would depend on the normalisation mode.
    """
    fragments = _scene()
    a = normalize_fragments(fragments, mode="scene").radius
    b = normalize_fragments(fragments, mode="fragment").radius
    assert np.allclose(a, b)
    assert a[0] == pytest.approx(1.0, rel=1e-6)
    assert a[1] == pytest.approx(0.1, rel=1e-6)


def test_empty_and_degenerate_inputs():
    empty = normalize_fragments([])
    assert empty.vertices == [] and empty.radius.size == 0

    point = trimesh.Trimesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), process=False)
    norm = normalize_fragments([point])
    assert np.isfinite(norm.vertices[0]).all(), "a zero-radius fragment must not divide by 0"
