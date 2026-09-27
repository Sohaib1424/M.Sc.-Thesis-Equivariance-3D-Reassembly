"""
Canonical ordering of an edge's two adjacent face normals.

Why this module exists
----------------------
With the edge midpoint and length dropped from the feature set, an edge's two
adjacent face normals ``(n1, n2)`` carry nearly all of its signal. Which normal
lands in which slot is therefore a real input, not a bookkeeping detail -- and
the obvious source of that order is wrong.

``edge_topology`` fills ``faces_per_edge`` with the two lowest-numbered incident
faces. Face *numbers* come from the face array, and
``repair.resolve_duplicated_faces`` reorders faces lexicographically by sorted
vertex triple. So the slot assignment depends on an ordering that has no
geometric meaning and that a mesh-cleaning step is free to change. Two runs
over the same fragment can disagree, and nothing raises: the model simply sees
``(n1, n2)`` where it saw ``(n2, n1)``.

The rule
--------
For an edge ``(u, v)`` with ``u < v``, let ``d = x_v - x_u``. Order the normals
by the sign of the scalar triple product::

    s = (n_a x n_b) . d        keep (n_a, n_b) if s > 0, else swap

This is exactly the "which normal is on the left of the edge" test, and it is
**invariant under proper rotation**: for ``R`` in SO(3),

    (R n_a x R n_b) . (R d) = det(R) (n_a x n_b) . d = (n_a x n_b) . d

so a rotated fragment produces ``(R n1, R n2)`` -- the same slots, rotated --
which is precisely what an equivariant layer needs. Note ``det(R)`` appears:
under a *reflection* the order does flip. That is correct behaviour, not a bug,
because a reflection genuinely exchanges the two sides of the edge. Breaking Bad
applies only proper rotations, and there is a test asserting the flip so the
assumption stays visible.

The degenerate case is benign
-----------------------------
For a well-formed manifold edge both faces contain the edge, so ``d`` is
perpendicular to both normals and therefore parallel to ``n_a x n_b``. Hence

    |s| = ||n_a x n_b|| ||d|| = sin(angle between normals) ||n_a|| ||n_b|| ||d||

The rule is ambiguous exactly when ``s -> 0``, which happens only when the
normals are parallel -- and then the two slots hold the same vector, so the
swap changes nothing. **The ordering is ill-defined precisely where it does not
matter**, and the size of the discontinuity goes to zero with the ambiguity.
Antiparallel normals (a zero-volume sliver) are the one case where a flip is
visible; those are counted and reported rather than silently ordered.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .topology import edge_topology, face_normals

# sin(angle) below which the two normals count as parallel and the order is
# reported as ambiguous. 1e-6 rad is far below any real dihedral variation.
DEFAULT_PARALLEL_TOL = 1e-6


class EdgeNormals(NamedTuple):
    """Canonically ordered adjacent face normals, one row per unique edge."""
    edges: np.ndarray        # (E, 2) unique undirected edges, lo < hi
    n1: np.ndarray           # (E, 3) first slot
    n2: np.ndarray           # (E, 3) second slot
    boundary: np.ndarray     # (E,) bool -- only one incident face; n1 == n2
    ambiguous: np.ndarray    # (E,) bool -- normals parallel, order arbitrary


def canonical_edge_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    topology=None,
    face_norms: np.ndarray | None = None,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
) -> EdgeNormals:
    """
    Order each edge's two adjacent face normals by the triple-product rule.

    Independent of face-array order and equivariant under SO(3). Boundary edges
    (one incident face) get that face's normal in both slots, matching the
    convention the feature builder expects.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if faces.size == 0:
        z3 = np.zeros((0, 3), np.float64)
        z = np.zeros(0, bool)
        return EdgeNormals(np.zeros((0, 2), np.int64), z3, z3, z, z)

    topo = topology if topology is not None else edge_topology(faces, vertices.shape[0])
    normals = face_normals(vertices, faces) if face_norms is None else np.asarray(face_norms)

    first = topo.faces_per_edge[:, 0]
    second = topo.faces_per_edge[:, 1]
    boundary = second < 0
    # Boundary edges reuse the first face, so the arithmetic below stays
    # branch-free; the triple product is then zero and no swap happens.
    second_safe = np.where(boundary, first, second)

    n_a = normals[first]
    n_b = normals[second_safe]

    d = vertices[topo.edges[:, 1]] - vertices[topo.edges[:, 0]]
    cross = np.cross(n_a, n_b)
    s = np.einsum("ij,ij->i", cross, d)

    swap = s < 0.0
    n1 = np.where(swap[:, None], n_b, n_a)
    n2 = np.where(swap[:, None], n_a, n_b)

    sin_angle = np.linalg.norm(cross, axis=1)
    ambiguous = (sin_angle <= parallel_tol) & ~boundary

    return EdgeNormals(topo.edges, n1, n2, boundary, ambiguous)


def directed_edge_normals(canon: EdgeNormals):
    """
    Both directed copies of every undirected edge, with the slots mirrored on
    the reverse copy.

    Reversing an edge negates ``d`` and therefore flips the sign of the triple
    product, so the canonical rule *itself* swaps the slots on the reverse
    copy -- the mirroring is not an extra convention bolted on, it is what the
    rule already does. Returning it explicitly means message passing on the
    directed graph never has to remember that.

    Returns ``(edge_index, n1, n2)`` where ``edge_index`` is ``(2, 2E)`` in COO
    form: forward copies first, then reverse copies in the same order.
    """
    edges = canon.edges
    forward = edges.T
    reverse = edges[:, ::-1].T
    edge_index = np.concatenate([forward, reverse], axis=1)
    n1 = np.concatenate([canon.n1, canon.n2], axis=0)
    n2 = np.concatenate([canon.n2, canon.n1], axis=0)
    return edge_index, n1, n2
