"""
An empty fracture mask is always a bug.

Every fragment of a broken object tore away from at least one neighbour, so it
has fracture surface by definition. The full-dataset run nevertheless reports
``frac_vertices`` min = 0, which means the dihedral heuristic returns nothing
for some fragments.

These tests pin down *why* (so the fix is aimed at the real mechanism, not at a
guessed threshold) and then pin the fix itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.mesh.fracture import (
    DEFAULT_SHARP_THRESHOLD,
    fracture_face_mask,
)
from reassembly.mesh.topology import face_adjacency, face_normals


def _shard(tilt: float):
    """A small, gently curved fragment -- the shape that defeats the heuristic."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, tilt], [0.5, 0.5, -0.02]],
                 dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return V, F


def test_the_mechanism_is_an_absolute_angle_threshold():
    """
    Documents the cause. The heuristic asks whether adjacent faces differ by
    more than ~26 degrees (|cos| < 0.9). Roughness is relative to triangle size,
    so a shard whose facets differ by less than that has *no* sharp adjacency
    and the mask is empty -- not because it lacks fracture surface, but because
    the test is absolute where the property is relative.
    """
    V, F = _shard(0.1)
    src, dst = face_adjacency(F, len(V))
    normals = face_normals(V, F)
    cosine = np.abs(np.einsum("ij,ij->i", normals[src], normals[dst]))

    assert (cosine >= DEFAULT_SHARP_THRESHOLD).all(), \
        "expected every adjacency to look smooth at the default threshold"
    assert not fracture_face_mask(V, F).face_mask.any(), \
        "and therefore an empty mask -- the bug being fixed"


@pytest.mark.parametrize("tilt", [0.2, 0.1, 0.05, 0.01, 0.001])
def test_fallback_never_returns_an_empty_mask(tilt):
    """The fix: min_faces >= 1 guarantees a non-empty labelling."""
    V, F = _shard(tilt)
    assert not fracture_face_mask(V, F).face_mask.any()
    assert fracture_face_mask(V, F, min_faces=1).face_mask.any()


def test_fallback_handles_a_perfectly_flat_sheet():
    """
    Every dihedral identical (cos == 1) is the worst case: a quantile-based
    relaxation must still mark something, which is why the threshold is nudged
    with nextafter rather than used as-is.
    """
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    assert not fracture_face_mask(V, F).face_mask.any()
    assert fracture_face_mask(V, F, min_faces=1).face_mask.all()


def test_fallback_leaves_working_fragments_bit_identical(fractured_solid):
    """
    The property that makes this safe to switch on globally. A fragment that
    already produced a mask must be untouched -- otherwise enabling the fallback
    would silently invalidate the numbers from the completed dataset run.
    """
    for fragment in fractured_solid:
        V = np.asarray(fragment.vertices)
        F = np.asarray(fragment.faces)
        plain = fracture_face_mask(V, F).face_mask
        assert plain.any(), "fixture should label something at the default threshold"
        relaxed = fracture_face_mask(V, F, min_faces=1).face_mask
        assert np.array_equal(plain, relaxed)


def test_fallback_is_rotation_invariant(fractured_solid):
    """
    The relaxation is driven by quantiles of |cos| between face normals, all of
    which are rotation-invariant. If it were not invariant, two views of one
    fragment would get different fracture surfaces.
    """
    rng = np.random.default_rng(0)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0

    V, F = _shard(0.05)
    plain = fracture_face_mask(V, F, min_faces=1).face_mask
    rotated = fracture_face_mask(V @ R.T, F, min_faces=1).face_mask
    assert np.array_equal(plain, rotated)


def test_lowering_the_global_threshold_is_the_worse_fix(fractured_solid):
    """
    Why the fallback is per-fragment rather than a looser global threshold:
    loosening globally also relabels fragments that were already correct,
    sweeping in the smooth exterior. Measured on the fixture, whose interface
    is genuinely rough and whose far side is genuinely flat.
    """
    fragment = fractured_solid[0]
    V = np.asarray(fragment.vertices)
    F = np.asarray(fragment.faces)

    tight = fracture_face_mask(V, F, 0.9).face_mask
    loose = fracture_face_mask(V, F, 0.999).face_mask
    assert loose.sum() > tight.sum(), "a looser threshold should label more"

    # ...whereas the fallback changes nothing at all here.
    fallback = fracture_face_mask(V, F, 0.9, min_faces=1).face_mask
    assert np.array_equal(fallback, tight)
