"""
Tests for the canonical edge face-normal ordering.

The failure this guards against is silent: a swapped ``(n1, n2)`` still has the
right shape and the right magnitudes, so nothing raises and the loss still
descends -- the model just receives a different input than it did last epoch.
Every test here asserts a property that would still "look fine" if violated.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.mesh.orientation import (
    canonical_edge_normals,
    directed_edge_normals,
)
from reassembly.mesh.repair import resolve_duplicated_faces
from reassembly.mesh.topology import edge_topology, face_normals


def _rotation(seed: int) -> np.ndarray:
    """A uniformly random proper rotation (det = +1)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


@pytest.fixture(scope="module")
def mesh():
    return trimesh.creation.icosphere(subdivisions=3)


def _canon(V, F):
    return canonical_edge_normals(np.asarray(V, float), np.asarray(F))


def _by_edge(canon):
    """Map each undirected edge to its ordered normal pair, for comparison
    across runs whose edge arrays may be ordered differently."""
    return {tuple(e): (tuple(np.round(a, 9)), tuple(np.round(b, 9)))
            for e, a, b in zip(canon.edges, canon.n1, canon.n2)}


def test_determinism_under_face_permutation(mesh):
    """
    The point of the module. Shuffling the face array must not change which
    normal lands in which slot -- otherwise resolve_duplicated_faces, which
    reorders faces lexicographically, silently changes the model's input.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    base = _by_edge(_canon(V, F))

    rng = np.random.default_rng(0)
    for _ in range(3):
        perm = rng.permutation(len(F))
        shuffled = _by_edge(_canon(V, F[perm]))
        assert shuffled == base


def test_determinism_after_resolve_duplicated_faces(mesh):
    """The specific reordering this project actually performs."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    before = _by_edge(_canon(V, F))
    resolved, _ = resolve_duplicated_faces(F)
    after = _by_edge(_canon(V, resolved))
    assert after == before


def test_naive_face_index_order_is_NOT_stable(mesh):
    """
    Demonstrates the bug being fixed: taking the normals in `faces_per_edge`
    order -- the obvious implementation -- does change under a face permutation.
    If this test ever fails, the hazard has gone away and this module could be
    simplified; until then it documents why the module exists.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)

    def naive(faces):
        topo = edge_topology(faces, len(V))
        norms = face_normals(V, faces)
        second = np.where(topo.faces_per_edge[:, 1] < 0,
                          topo.faces_per_edge[:, 0], topo.faces_per_edge[:, 1])
        return {tuple(e): (tuple(np.round(norms[a], 9)), tuple(np.round(norms[b], 9)))
                for e, a, b in zip(topo.edges, topo.faces_per_edge[:, 0], second)}

    rng = np.random.default_rng(1)
    base = naive(F)
    assert any(naive(F[rng.permutation(len(F))]) != base for _ in range(5)), \
        "expected the naive ordering to be unstable under face permutation"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equivariance_under_rotation(mesh, seed):
    """
    Rotating the fragment must rotate the normals *without* exchanging slots.
    A swap here would pass any global-magnitude check and still feed the network
    inconsistent features across epochs, since the loader rotates every sample.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    R = _rotation(seed)

    plain = _canon(V, F)
    rotated = _canon(V @ R.T, F)

    # Same edge set, same order (edges depend only on connectivity).
    assert np.array_equal(plain.edges, rotated.edges)
    assert np.allclose(rotated.n1, plain.n1 @ R.T, atol=1e-12)
    assert np.allclose(rotated.n2, plain.n2 @ R.T, atol=1e-12)


def test_reflection_does_flip_the_order(mesh):
    """
    Documents the assumption rather than hiding it: the rule depends on det(R),
    so an improper transform exchanges the slots. Breaking Bad applies only
    proper rotations. If a mirroring augmentation is ever added, this test is
    the thing that will fail, and it should.

    Two sign flips compose here, which is what makes this easy to get wrong.
    A normal is a *pseudovector*: mirroring the vertices also reverses face
    winding, so ``n -> -M n``, not ``M n``. The triple product then picks up
    ``(-1)(-1)det(M) = det(M) = -1`` overall, so the slots do exchange -- but
    the vector in the exchanged slot is the *negated* mirror of the original.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    mirror = np.diag([1.0, 1.0, -1.0])          # det = -1

    plain = _canon(V, F)
    flipped = _canon(V @ mirror.T, F)

    # Normals transform as pseudovectors: n -> -M n.
    assert np.allclose(flipped.n1, -(plain.n2 @ mirror.T), atol=1e-12), \
        "reflection should exchange the slots (and negate the pseudovector)"
    assert np.allclose(flipped.n2, -(plain.n1 @ mirror.T), atol=1e-12)

    # And the contrast with a proper rotation, which does NOT exchange them.
    R = _rotation(7)
    rotated = _canon(V @ R.T, F)
    assert np.allclose(rotated.n1, plain.n1 @ R.T, atol=1e-12)


def test_boundary_edges_duplicate_the_single_normal():
    """One incident face -> the same normal in both slots, no swap logic."""
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    canon = _canon(V, F)
    assert canon.boundary.all()
    assert np.allclose(canon.n1, canon.n2)
    assert not canon.ambiguous.any(), "boundary edges are not 'ambiguous'"


def test_coplanar_faces_are_reported_ambiguous():
    """
    Two coplanar triangles sharing an edge: the normals are parallel, so the
    rule cannot decide -- and does not need to, because both slots hold the
    same vector. The flag lets a caller count these rather than trust them.
    """
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    F = np.array([[0, 1, 2], [1, 3, 2]])
    canon = _canon(V, F)
    shared = [i for i, e in enumerate(canon.edges)
              if tuple(e) == (1, 2) or tuple(e) == (2, 1)]
    assert shared, "expected a shared edge between the two triangles"
    i = shared[0]
    assert canon.ambiguous[i]
    # Ambiguous, but harmless: the two slots agree, so the swap is a no-op.
    assert np.allclose(canon.n1[i], canon.n2[i], atol=1e-12)


def test_ambiguity_implies_a_vanishing_discontinuity(mesh):
    """
    The safety property behind the whole rule: wherever the order is close to
    ambiguous, the two normals are close to identical, so a flip is close to a
    no-op. Formally ||n1 - n2|| -> 0 as the ambiguity measure -> 0.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    canon = _canon(V, F)
    interior = ~canon.boundary
    sin_angle = np.linalg.norm(np.cross(canon.n1[interior], canon.n2[interior]), axis=1)
    gap = np.linalg.norm(canon.n1[interior] - canon.n2[interior], axis=1)
    # For unit normals, ||n1 - n2|| <= 2 sin(theta/2) and sin(theta) >= gap/2 * ...
    # the bound worth asserting is simply that they vanish together.
    near_parallel = sin_angle < 1e-3
    if near_parallel.any():
        assert gap[near_parallel].max() < 1e-2


def test_directed_copies_mirror_the_slots(mesh):
    """
    The reverse copy of an edge must carry the swapped pair -- and must do so
    because the rule already implies it, not because a caller remembered to.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    canon = _canon(V, F)
    edge_index, n1, n2 = directed_edge_normals(canon)

    e = len(canon.edges)
    assert edge_index.shape == (2, 2 * e)
    # forward half
    assert np.array_equal(edge_index[:, :e], canon.edges.T)
    # reverse half is the same edges, endpoints exchanged
    assert np.array_equal(edge_index[:, e:], canon.edges[:, ::-1].T)
    # slots mirrored
    assert np.allclose(n1[e:], canon.n2)
    assert np.allclose(n2[e:], canon.n1)


def test_reverse_direction_reproduces_the_mirror_from_the_rule_itself(mesh):
    """
    Independent confirmation: recomputing the rule with the edge direction
    negated must give the swap that `directed_edge_normals` hands out. This is
    what makes the mirroring a property rather than a convention.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    canon = _canon(V, F)

    d = V[canon.edges[:, 1]] - V[canon.edges[:, 0]]
    s_forward = np.einsum("ij,ij->i", np.cross(canon.n1, canon.n2), d)
    s_reverse = np.einsum("ij,ij->i", np.cross(canon.n1, canon.n2), -d)

    interior = ~canon.boundary & ~canon.ambiguous
    assert (s_forward[interior] > 0).all(), "forward copies should be positively oriented"
    assert (s_reverse[interior] < 0).all(), "reversing must flip the sign, hence the slots"
