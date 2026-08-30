"""
Fracture-surface extraction.

The tests that matter here are the ones that fail loudly on a plausible wrong
answer. Fracture extraction can silently return an empty mask, or the whole
mesh, and either looks like "it ran". So every test asserts a *bound*, not
just absence of an exception.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.arrays import compact_indices
from reassembly.mesh.fracture import (
    boundary_edges,
    extract_fracture_surface,
    fracture_agreement,
    fracture_face_mask,
    fracture_face_mask_from_vertices,
    fracture_vertex_masks,
)


# --------------------------------------------------------------- dihedral --

def test_mask_is_a_strict_subset_and_not_degenerate(simple_meshes):
    """Neither empty-everywhere nor everything: both are silent failure modes."""
    picked = 0
    for name, mesh in simple_meshes.items():
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        mask = fracture_face_mask(vertices, faces).face_mask
        assert mask.shape == (len(faces),), name
        assert mask.sum() <= len(faces), name
        picked += int(mask.sum() > 0)
    assert picked > 0, "no mesh produced any fracture faces at all"


def test_smooth_closed_sphere_has_no_fracture(simple_meshes):
    """An intact smooth object has no break, so the mask must be empty."""
    mesh = simple_meshes["icosphere"]
    mask = fracture_face_mask(np.asarray(mesh.vertices), np.asarray(mesh.faces)).face_mask
    assert mask.sum() == 0


def test_flat_caps_are_found(simple_meshes):
    """A cylinder's flat end caps meet the wall at a sharp angle and qualify."""
    mesh = simple_meshes["cylinder"]
    mask = fracture_face_mask(np.asarray(mesh.vertices), np.asarray(mesh.faces)).face_mask
    assert mask.sum() > 0


def test_threshold_is_monotone(simple_meshes):
    """Raising the threshold makes more pairs count as sharp, never fewer."""
    mesh = simple_meshes["cylinder"]
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    counts = [
        int(fracture_face_mask(vertices, faces, t).sharp_neighbours.sum())
        for t in (0.5, 0.8, 0.9, 0.99)
    ]
    assert counts == sorted(counts), counts


def test_empty_and_single_face_meshes():
    empty = fracture_face_mask(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert empty.face_mask.shape == (0,)

    one = fracture_face_mask(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    )
    assert one.face_mask.shape == (1,)
    assert one.total_neighbours.tolist() == [0]


def test_open_patch_does_not_crash(open_patch):
    mask = fracture_face_mask(
        np.asarray(open_patch.vertices), np.asarray(open_patch.faces)
    ).face_mask
    assert mask.shape == (len(open_patch.faces),)


def test_degenerate_faces_are_counted_not_hidden():
    """A zero-area triangle must be reported, not silently turned into NaN."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])          # first is collinear
    result = fracture_face_mask(vertices, faces)
    assert result.degenerate_faces == 1


def test_extraction_is_deterministic(simple_meshes):
    mesh = simple_meshes["cylinder"]
    a = extract_fracture_surface(mesh)
    b = extract_fracture_surface(mesh)
    assert np.array_equal(np.asarray(a.faces), np.asarray(b.faces))
    assert np.allclose(np.asarray(a.vertices), np.asarray(b.vertices))


def test_compact_preserves_geometry(simple_meshes):
    """compact=True must renumber, not move: the triangles are the same points."""
    mesh = simple_meshes["cylinder"]
    loose = extract_fracture_surface(mesh, compact=False)
    tight = extract_fracture_surface(mesh, compact=True)
    assert len(tight.vertices) <= len(loose.vertices)
    assert len(tight.faces) == len(loose.faces)
    a = np.asarray(loose.vertices)[np.asarray(loose.faces)]
    b = np.asarray(tight.vertices)[np.asarray(tight.faces)]
    assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


# ---------------------------------------------------------- coincidence --

def test_coincidence_finds_the_shared_interface(fractured_solid):
    """
    The two halves share their cut surface, so both must report a substantial
    set of shared vertices -- and the same number on each side.
    """
    lower, upper = fractured_solid
    masks = fracture_vertex_masks([lower, upper], tol=1e-9)
    assert len(masks) == 2
    assert masks[0].sum() > 0 and masks[1].sum() > 0
    assert masks[0].sum() == masks[1].sum()


def test_coincidence_lifts_to_a_real_surface(fractured_solid):
    """Shared vertices must form faces, not just a rim -- require='all'."""
    lower, upper = fractured_solid
    masks = fracture_vertex_masks([lower, upper], tol=1e-9)
    faces = fracture_face_mask_from_vertices(np.asarray(lower.faces), masks[0])
    assert faces.sum() > 0, "shared vertices did not span any complete face"
    assert faces.sum() < len(lower.faces), "everything was labelled fracture"


def test_separated_fragments_share_nothing(simple_meshes):
    a = simple_meshes["box"]
    b = a.copy()
    b.apply_translation([100.0, 100.0, 100.0])
    masks = fracture_vertex_masks([a, b])
    assert masks[0].sum() == 0 and masks[1].sum() == 0


def test_touching_boxes_share_exactly_the_contact_face(simple_meshes):
    a = simple_meshes["box"]
    b = a.copy()
    b.apply_translation([1.0, 0.0, 0.0])
    masks = fracture_vertex_masks([a, b])
    assert masks[0].sum() == 4 and masks[1].sum() == 4


def test_coincidence_tolerates_small_perturbation(simple_meshes):
    """Two sides of an interface can differ in the last bits after remeshing."""
    rng = np.random.default_rng(0)
    a = simple_meshes["box"]
    b = a.copy()
    b.apply_translation([1.0, 0.0, 0.0])
    noisy = np.asarray(b.vertices) + rng.normal(scale=1e-8, size=(len(b.vertices), 3))
    masks = fracture_vertex_masks([np.asarray(a.vertices), noisy], tol=1e-6)
    assert masks[0].sum() == 4 and masks[1].sum() == 4


def test_single_fragment_shares_nothing_with_itself(simple_meshes):
    masks = fracture_vertex_masks([simple_meshes["box"]])
    assert masks[0].sum() == 0


# ------------------------------------------------------------- agreement --

def test_agreement_perfect_and_disjoint():
    truth = np.array([True, True, False, False])
    assert fracture_agreement(truth, truth)["f1"] == pytest.approx(1.0)
    assert fracture_agreement(~truth, truth)["f1"] == 0.0


def test_agreement_partial():
    predicted = np.array([True, True, True, False])
    truth = np.array([True, True, False, False])
    score = fracture_agreement(predicted, truth)
    assert score["tp"] == 2 and score["fp"] == 1 and score["fn"] == 0
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == pytest.approx(1.0)
    assert score["iou"] == pytest.approx(2 / 3)


# -------------------------------------------------------------- boundary --

def test_boundary_of_a_closed_mesh_is_empty(simple_meshes):
    mesh = simple_meshes["icosphere"]
    assert len(boundary_edges(np.asarray(mesh.faces), len(mesh.vertices))) == 0


def test_boundary_of_an_open_patch_is_a_closed_loop(open_patch):
    """Every rim vertex has exactly two rim neighbours, so the rim is a cycle."""
    edges = boundary_edges(np.asarray(open_patch.faces), len(open_patch.vertices))
    assert len(edges) > 0
    degree = np.bincount(edges.ravel())
    assert set(np.unique(degree[degree > 0]).tolist()) == {2}
