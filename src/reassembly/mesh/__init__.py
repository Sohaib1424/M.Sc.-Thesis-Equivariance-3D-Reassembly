"""
Mesh geometry, on raw ``(V, F)`` numpy arrays.

Nothing here constructs a ``trimesh.Trimesh`` on the hot path — building one
runs ``merge_vertices()`` by default, which is ~67x the cost of the work
itself when ``igl`` has already cleaned the fragment.

``topology``
    Half-edges, face adjacency, unique edges, degeneracy-safe normals.
    ``unique_edges`` defines the canonical edge ordering for the whole
    package; do not mix it with ``trimesh.edges_unique``, which returns the
    same set in a different order.
``repair``
    ``resolve_duplicated_faces`` — vectorised port of the igl routine.
``fracture``
    ★ Fracture-surface extraction. Two independent labellings:
    ``method="dihedral"`` (computable on one fragment in isolation, so
    available at inference) and ``method="coincidence"`` (ground truth from
    the assembled frame). ``fracture_agreement`` scores one against the
    other.
``orientation``
    Canonical, SO(3)-equivariant ordering of an edge's two adjacent face
    normals. Face-array order is not geometric and ``repair`` reorders it, so
    the naive slot assignment is unstable across runs.
``patches``
    Reduces a fracture surface to a bounded token set, so inter-fragment
    attention stops scaling with mesh resolution. Three modes:
    ``"patch"`` pools vertices into centroids, ``"sample"`` selects real
    vertices by farthest-point sampling (the deterministic, rotation-invariant
    analogue of GARF's Poisson-disk sampling), ``"vertex"`` is the unreduced
    baseline. ``allocate_budget`` splits one fixed per-scene budget across
    fragments and ``scene_tokens`` does the whole split-and-build in one
    call, weighted by fracture-surface area.
``correspondence``
    Cross-fragment coincidence matching.
``shell``
    Ray-cast visibility. UNUSED — nothing in the package imports it; kept
    in case a thesis figure wants it.
"""
from __future__ import annotations

__all__ = ["correspondence", "fracture", "orientation", "patches",
           "repair", "shell", "topology"]
