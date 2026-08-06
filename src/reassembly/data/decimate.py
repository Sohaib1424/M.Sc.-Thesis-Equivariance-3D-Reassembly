"""
Vertex-budget control via shared-grid voxel-cluster decimation.

WHY THIS EXISTS
---------------
Peak GPU memory in this pipeline is close to linear in the number of *mesh
vertices per batch*, and Breaking Bad fragment resolution varies by well over
an order of magnitude between shapes. That combination is what made training
crash intermittently rather than immediately: most batches fit in 16 GB, and
then one draw happens to pull a dense shape and the run dies. Capping batch
size to survive the worst case wastes the GPU on every other batch.

Decimating each scene to a *vertex budget* makes peak memory a function of a
number you choose rather than a number the dataset chooses for you. It is also
squarely on-thesis: reducing the input resolution needed to reach a given
accuracy is a compute reduction, and the accuracy-vs-budget curve is directly
measurable (see ``scripts/benchmark_compute.py``).

WHY VOXEL CLUSTERING AND NOT QUADRIC DECIMATION
-----------------------------------------------
Quadric edge collapse preserves shape better, but in trimesh it requires an
optional backend (``fast_simplification`` / ``open3d``) that is not guaranteed
present on Kaggle and would add an install step to a project whose whole point
is being cheap to run. Voxel clustering is ~40 lines of numpy, has no
dependencies, is O(V), and is deterministic.

THE SHARED-GRID DETAIL (this one matters)
-----------------------------------------
All fragments of a scene are clustered on ONE grid, anchored at the scene's
bounding-box corner, *before* any centralization or SE(3) diffusion. Two
fragments' coincident interface vertices therefore fall in the same cell and
collapse to nearby representatives, so cross-fragment correspondence survives
decimation. Decimating fragments independently on their own local grids would
shear the two sides of every interface apart by up to a voxel and quietly
destroy the supervision signal for the embedding losses.

Representatives are the *mean* of a cell's members (better shape preservation
than snapping to cell centres). That means the two sides of an interface land
within roughly a voxel of each other rather than exactly on top of each other,
so the correspondence tolerance must scale with the voxel size --
:func:`suggested_correspondence_tol` returns the right value and
``BreakingBadDataset`` applies it automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DecimationInfo:
    """What actually happened, for logging and for the compute benchmark."""
    applied: bool
    voxel_size: float
    vertices_before: int
    vertices_after: int
    faces_before: int
    faces_after: int
    fragments_kept_original: int = 0   # fragments that would have degenerated
    target_met: bool = True            # did the search actually reach the target?

    @property
    def vertex_ratio(self) -> float:
        return self.vertices_after / max(self.vertices_before, 1)

    @property
    def reduced(self) -> bool:
        """True only if the vertex count actually went down.

        Distinct from ``applied``: a search can run to completion, report a
        voxel size, and still return the scene untouched because every
        fragment hit the per-fragment floor. Callers that scale a tolerance by
        ``voxel_size`` must check THIS, not ``applied`` -- see the note on
        ``decimate_scene``.
        """
        return self.applied and self.vertices_after < self.vertices_before


def voxel_cluster_ids(
    vertices: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Integer cell coordinate per vertex on a grid anchored at ``origin``."""
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    return np.floor((vertices - origin) / voxel_size).astype(np.int64)


def decimate_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-cluster one mesh given as raw arrays.

    Returns ``(new_vertices, new_faces, vertex_map)`` where ``vertex_map[i]``
    is the index in ``new_vertices`` that old vertex ``i`` collapsed into.
    ``vertex_map`` is what lets any per-vertex label survive decimation.

    Faces that become degenerate (two or three corners collapsing into the
    same representative) are dropped, as are duplicate faces created by the
    collapse; unreferenced representatives are then pruned, so the output has
    no isolated vertices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if len(vertices) == 0:
        return vertices.copy(), faces.copy(), np.empty((0,), dtype=np.int64)

    cells = voxel_cluster_ids(vertices, origin, voxel_size)
    # Unique cells, in a deterministic (lexicographic) order.
    _uniq, vertex_map = np.unique(cells, axis=0, return_inverse=True)
    vertex_map = np.asarray(vertex_map).reshape(-1)
    n_new = int(vertex_map.max()) + 1 if len(vertex_map) else 0

    # Representative = centroid of the cell's members.
    sums = np.zeros((n_new, 3), dtype=np.float64)
    np.add.at(sums, vertex_map, vertices)
    counts = np.bincount(vertex_map, minlength=n_new).astype(np.float64)
    new_vertices = sums / counts[:, None]

    if len(faces) == 0:
        return new_vertices, faces.reshape(0, 3).copy(), vertex_map

    new_faces = vertex_map[faces]

    # Drop collapsed (degenerate) triangles.
    non_degenerate = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[non_degenerate]

    # Drop duplicate triangles (same corner set, any winding). Keep the first
    # occurrence so the output is deterministic.
    if len(new_faces):
        keyed = np.sort(new_faces, axis=1)
        _, first_idx = np.unique(keyed, axis=0, return_index=True)
        new_faces = new_faces[np.sort(first_idx)]

    # Prune representatives no surviving face references, and reindex.
    if len(new_faces):
        referenced = np.zeros(n_new, dtype=bool)
        referenced[new_faces.reshape(-1)] = True
        remap = np.full(n_new, -1, dtype=np.int64)
        remap[referenced] = np.arange(int(referenced.sum()), dtype=np.int64)
        new_vertices = new_vertices[referenced]
        new_faces = remap[new_faces]
        vertex_map = remap[vertex_map]      # -1 for old vertices that vanished
    else:
        new_vertices = new_vertices[:0]
        vertex_map = np.full_like(vertex_map, -1)

    return new_vertices, new_faces, vertex_map


def scene_origin(meshes: Sequence) -> np.ndarray:
    """Lower corner of the axis-aligned bounding box of the whole scene."""
    mins = [np.asarray(m.vertices).min(axis=0) for m in meshes if len(m.vertices)]
    if not mins:
        return np.zeros(3)
    return np.min(np.stack(mins, axis=0), axis=0)


def scene_diagonal(meshes: Sequence) -> float:
    """Bounding-box diagonal of the whole scene (its characteristic length)."""
    mins, maxs = [], []
    for m in meshes:
        v = np.asarray(m.vertices)
        if len(v):
            mins.append(v.min(axis=0))
            maxs.append(v.max(axis=0))
    if not mins:
        return 1.0
    lo = np.min(np.stack(mins), axis=0)
    hi = np.max(np.stack(maxs), axis=0)
    return float(max(np.linalg.norm(hi - lo), 1e-9))


def suggested_correspondence_tol(
    voxel_size: float,
    base_tol: float = 1e-5,
    factor: float = 1.0,
) -> float:
    """Correspondence tolerance that stays meaningful after decimation.

    On undecimated meshes, interface twins are bit-identical and 1e-5 is
    plenty. Once cell means replace exact coordinates, twins separate by up to
    about one voxel, so the same tolerance would find nothing at all and the
    embedding losses would silently degenerate to "no clusters, zero loss".

    ``factor=1.0`` was chosen by measuring the worst-case twin separation on
    an oblique, asymmetric synthetic interface across voxel sizes: 0.75 was
    already too tight at aggressive decimation. The failure modes are not
    symmetric -- a tolerance that is slightly too loose adds a few spurious
    pairs between genuinely nearby surfaces, while one that is too tight
    deletes the supervision signal entirely and does so silently.
    """
    if voxel_size <= 0:
        return base_tol
    return max(base_tol, factor * voxel_size)


def decimate_scene(
    meshes: Sequence,
    target_vertices: Optional[int] = None,
    voxel_size: Optional[float] = None,
    min_vertices_per_fragment: int = 32,
    max_search_steps: int = 12,
    tolerance: float = 0.15,
    mesh_factory=None,
) -> Tuple[List, DecimationInfo]:
    """Decimate a whole scene to (approximately) ``target_vertices`` total.

    Exactly one of ``target_vertices`` / ``voxel_size`` should be given.
    With ``target_vertices``, the voxel size is found by bisection on
    log(voxel size); the search stops as soon as the result is within
    ``tolerance`` of the target, so it typically costs 3-6 trial clusterings
    (each O(V) numpy work, far cheaper than the mesh loading that preceded it).

    INVARIANT: the number of fragments never changes. A fragment that would
    collapse below ``min_vertices_per_fragment`` (or lose all its faces) is
    returned undecimated instead of being dropped, because downstream code
    indexes rotations, transforms and cluster ids by fragment position -- a
    silently shorter list would misalign every one of them.

    THE TARGET IS NOT ALWAYS REACHABLE. A scene of many small fragments can hit
    the per-fragment floor everywhere, in which case nothing is decimated no
    matter how coarse the grid. ``DecimationInfo.target_met`` reports this, and
    ``DecimationInfo.reduced`` says whether the vertex count actually moved.

    Anything that scales a distance by ``info.voxel_size`` -- notably
    ``suggested_correspondence_tol`` -- must gate on ``info.reduced``, not on
    ``info.applied``. A voxel size from a search that decimated nothing is
    meaningless, and feeding it to the correspondence tolerance silently makes
    every vertex match every other vertex.

    One voxel size is used for the whole scene, never per fragment. Two
    coincident interface vertices only collapse to the same representative
    point if both fragments share the grid, so per-fragment voxel sizes would
    reintroduce exactly the interface drift the shared origin exists to
    prevent.
    """
    if mesh_factory is None:
        # Local import so this module's core stays numpy-only. `mesh_factory`
        # lets callers (and tests) supply any mesh-like constructor, which is
        # what keeps the search itself testable without a mesh library.
        import trimesh
        def mesh_factory(v, f):
            return trimesh.Trimesh(v, f, process=False)

    meshes = list(meshes)
    v_before, f_before = (
        int(sum(len(m.vertices) for m in meshes)),
        int(sum(len(m.faces) for m in meshes)),
    )

    if not meshes:
        return meshes, DecimationInfo(False, 0.0, 0, 0, 0, 0)
    if target_vertices is None and voxel_size is None:
        return meshes, DecimationInfo(False, 0.0, v_before, v_before, f_before, f_before)
    if target_vertices is not None and v_before <= target_vertices:
        # Already under budget -- decimating would only lose detail for nothing.
        return meshes, DecimationInfo(False, 0.0, v_before, v_before, f_before, f_before)

    origin = scene_origin(meshes)
    diag = scene_diagonal(meshes)

    def cluster_at(h: float):
        out, kept_original = [], 0
        for m in meshes:
            nv, nf, _ = decimate_arrays(m.vertices, m.faces, origin, h)
            if len(nv) < min_vertices_per_fragment or len(nf) == 0:
                out.append(m)
                kept_original += 1
            else:
                out.append(mesh_factory(nv, nf))
        return out, kept_original

    target_met = True
    if voxel_size is not None:
        chosen = float(voxel_size)
        result, kept_original = cluster_at(chosen)
    else:
        # Search on log(h).
        #
        # The vertex count is NOT monotonic in h, despite the intuition that
        # coarser cells can only merge more. Once h is coarse enough that a
        # fragment would collapse below min_vertices_per_fragment, that
        # fragment is returned UNDECIMATED -- so the total jumps back up to the
        # original count. Past that point, increasing h makes the result
        # strictly worse. Plain bisection on "n > target -> go coarser" walks
        # straight into that region and terminates at the upper bound, having
        # decimated nothing while reporting a voxel size larger than the object
        # (observed on a real 53-fragment scene: 11,303 -> 11,303 vertices at a
        # voxel size of half the scene diagonal).
        #
        # So: probe, but keep the BEST candidate seen rather than the last one,
        # and treat "every fragment was protected" as evidence that h is too
        # coarse regardless of the vertex count.
        lo, hi = diag * 1e-4, diag * 0.5
        best = None               # (score, n, h, meshes, kept)
        chosen, result, kept_original = 0.0, None, 0

        for _ in range(max_search_steps):
            mid = float(np.sqrt(lo * hi))
            cand, kept = cluster_at(mid)
            n = sum(len(m.vertices) for m in cand)
            everything_protected = kept >= len(meshes)

            if n < v_before:
                # Prefer landing at or under budget; among those, closest.
                score = (0 if n <= target_vertices else 1, abs(n - target_vertices))
                if best is None or score < best[0]:
                    best = (score, n, mid, cand, kept)

            if not everything_protected and abs(n - target_vertices) <= tolerance * target_vertices:
                best = ((0, 0), n, mid, cand, kept)
                break

            if everything_protected or n <= target_vertices:
                hi = mid          # too coarse (or already small enough)
            else:
                lo = mid          # too many vertices -> try coarser cells

        if best is not None:
            _score, n_best, chosen, result, kept_original = best
            target_met = abs(n_best - target_vertices) <= tolerance * target_vertices
        else:
            # No probe reduced anything at all: every fragment is at or near
            # the per-fragment floor already. Report honestly rather than
            # returning a voxel size that was never usefully applied.
            result, kept_original = list(meshes), len(meshes)
            chosen, target_met = 0.0, False

    v_after = int(sum(len(m.vertices) for m in result))
    f_after = int(sum(len(m.faces) for m in result))
    return result, DecimationInfo(
        applied=True,
        voxel_size=chosen,
        vertices_before=v_before,
        vertices_after=v_after,
        faces_before=f_before,
        faces_after=f_after,
        fragments_kept_original=kept_original,
        target_met=target_met,
    )
