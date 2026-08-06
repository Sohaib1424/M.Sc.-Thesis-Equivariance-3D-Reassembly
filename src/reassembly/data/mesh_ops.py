"""
Per-mesh geometry operations: visible-shell extraction, face adjacency, and
fracture-surface extraction.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


# --------------------------------------------------------------------------
# Face adjacency
# --------------------------------------------------------------------------
def find_neighbors(
    mesh: "trimesh.Trimesh",
    min_common_vertices: int = 2,
) -> Tuple[csr_matrix, csr_matrix]:
    """Face-face adjacency, as a pair of sparse matrices with identical sparsity.

    Returns ``(W, M)``:
      * ``M[i, j] = 1``  if faces i and j share at least ``min_common_vertices``
      * ``W[i, j]``      the cosine similarity of those two faces' normals

    Computed as a single sparse product on the vertex-face incidence matrix
    (``VF.T @ VF`` counts shared vertices per face pair), replacing an
    O(faces^2) Python double loop.

    ``M`` is deliberately built from ``W``'s own structure rather than
    constructed independently: two separately-built sparse matrices are only
    guaranteed to share a ``.data`` layout if neither drops an explicit zero,
    and downstream code assigns ``A.data = <mask over W.data>``, which silently
    corrupts the result if the layouts ever diverge. Deriving one from the
    other makes that impossible by construction.
    """
    num_faces = mesh.faces.shape[0]
    num_verts = mesh.vertices.shape[0]

    if num_faces == 0:
        empty = csr_matrix((0, 0), dtype=np.float32)
        return empty, empty

    face_ids = np.repeat(np.arange(num_faces), 3)
    vert_ids = np.asarray(mesh.faces).reshape(-1)
    VF = csr_matrix(
        (np.ones(len(vert_ids), dtype=np.float32), (vert_ids, face_ids)),
        shape=(num_verts, num_faces),
    )

    shared = (VF.T @ VF).tocoo()
    keep = (shared.row != shared.col) & (shared.data >= min_common_vertices)
    f1, f2 = shared.row[keep], shared.col[keep]

    normals = np.asarray(mesh.face_normals)
    cos_sim = np.einsum("ij,ij->i", normals[f1], normals[f2]).astype(np.float32)

    W = csr_matrix((cos_sim, (f1, f2)), shape=(num_faces, num_faces))
    M = W.copy()
    M.data = np.ones_like(W.data)      # same sparsity, guaranteed
    return W, M


# --------------------------------------------------------------------------
# Fracture surface
# --------------------------------------------------------------------------
def _select_fracture_faces(
    mesh: "trimesh.Trimesh",
    sharpness_threshold: float,
) -> np.ndarray:
    """Indices of faces classified as fracture surface (may be empty)."""
    if len(mesh.faces) == 0:
        return np.empty((0,), dtype=np.int64)

    W, M = find_neighbors(mesh)
    W_csr, M_csr = W.tocsr(), M.tocsr()

    sharp_mask = np.abs(W_csr.data) < sharpness_threshold
    A_csr = W_csr.copy()
    A_csr.data = sharp_mask.astype(np.float32)

    sharp_counts = np.asarray(A_csr.sum(axis=1)).ravel()
    total_neighbors = np.asarray(M_csr.sum(axis=1)).ravel()

    sharp_faces = np.where(sharp_counts > 0)[0]
    if len(sharp_faces) == 0:
        return np.empty((0,), dtype=np.int64)

    touched = np.unique(np.asarray(mesh.faces)[sharp_faces])
    in_vs = np.zeros(mesh.vertices.shape[0], dtype=bool)
    in_vs[touched] = True

    faces_in_vs = in_vs[np.asarray(mesh.faces)].all(axis=1)      # criterion (a)
    enough_sharp = (total_neighbors // 2) <= sharp_counts        # criterion (b)
    return np.where(faces_in_vs & enough_sharp)[0]


def extract_fractures_with_map(
    mesh: "trimesh.Trimesh",
    sharpness_threshold: float = 0.9,
    min_faces: int = 4,
) -> Tuple["trimesh.Trimesh", np.ndarray]:
    """Extract the fracture surface, plus an old-vertex -> new-vertex index map.

    Heuristic: an edge between two faces whose normals differ sharply
    (|cos| < ``sharpness_threshold``) is a fracture edge. A face is kept when
    BOTH hold:
      (a) every one of its vertices touches some face with a sharp neighbour;
      (b) at least half of its own neighbours are sharp.

    Two bugs this used to have, both silent:

    1. Criterion (b) was computed and never applied, so the returned set was
       (a) alone and over-included faces.

    2. The returned mesh reused ``mesh.vertices`` wholesale while only
       subsetting ``mesh.faces``. Since feature construction reads
       ``mesh.vertices`` directly, the "fracture-surface mesh" had exactly the
       same node count as the full fragment -- only its *edge* count shrank,
       so it saved no memory at all, which was the one thing it was wanted
       for. Pruning is now done explicitly below.

    Doing the prune by hand (rather than calling
    ``remove_unreferenced_vertices()`` afterwards) is what yields
    ``vertex_map`` for free. That map is what allows correspondence to be
    computed ONCE on the full scene and then transferred onto the fracture
    mesh, instead of being recomputed in a second index space that has to be
    kept in sync by hand -- the exact mismatch that made an earlier version of
    this project attach edge cluster ids to the wrong edges.

    ``min_faces`` guards the degenerate case: on a fragment with little
    detectable fracture surface the criteria can select almost nothing, and a
    0-face mesh propagates NaNs into feature construction. Below the
    threshold the FULL mesh is returned unchanged, with an identity map, so
    the fragment (and hence the fragment count) always survives.
    """
    if trimesh is None:  # pragma: no cover
        raise ImportError("extract_fractures_with_map requires trimesh")

    identity = np.arange(len(mesh.vertices), dtype=np.int64)
    selected = _select_fracture_faces(mesh, sharpness_threshold)
    if len(selected) < min_faces:
        return mesh, identity

    selected_faces = np.asarray(mesh.faces)[selected]
    referenced = np.unique(selected_faces)                 # sorted surviving old indices
    if len(referenced) < 3:
        return mesh, identity

    vertex_map = np.full(len(mesh.vertices), -1, dtype=np.int64)
    vertex_map[referenced] = np.arange(len(referenced), dtype=np.int64)

    frac = trimesh.Trimesh(
        np.asarray(mesh.vertices)[referenced].copy(),
        vertex_map[selected_faces],
        process=False,
    )
    if len(frac.faces) < min_faces:
        return mesh, identity
    return frac, vertex_map


def extract_fractures(
    mesh: "trimesh.Trimesh",
    sharpness_threshold: float = 0.9,
    min_faces: int = 4,
) -> "trimesh.Trimesh":
    """Fracture surface only. See :func:`extract_fractures_with_map`."""
    return extract_fractures_with_map(mesh, sharpness_threshold, min_faces)[0]


# --------------------------------------------------------------------------
# Visible shell (unused by training; kept for inspection / the translation
# solver's collision term)
# --------------------------------------------------------------------------
_SQRT2_INV = 1.0 / np.sqrt(2.0)
_SQRT3_INV = 1.0 / np.sqrt(3.0)

_VIEW_DIRECTIONS = np.array(
    [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ]
    + [
        [a * _SQRT2_INV, b * _SQRT2_INV, 0] for a in (1, -1) for b in (1, -1)
    ]
    + [
        [a * _SQRT2_INV, 0, b * _SQRT2_INV] for a in (1, -1) for b in (1, -1)
    ]
    + [
        [0, a * _SQRT2_INV, b * _SQRT2_INV] for a in (1, -1) for b in (1, -1)
    ]
    + [
        [a * _SQRT3_INV, b * _SQRT3_INV, c * _SQRT3_INV]
        for a in (1, -1) for b in (1, -1) for c in (1, -1)
    ],
    dtype=np.float64,
)


def extract_shell(
    mesh: "trimesh.Trimesh",
    chunk_size: int = 1000,
    num_viewpoints: int = 26,
    distance_multiplier: float = 8.5,
) -> "trimesh.Trimesh":
    """Keep only faces visible from outside, by ray-casting from viewpoints.

    ``num_viewpoints``: 6 is enough for convex-ish shapes (bottles, plates),
    18 for moderately concave ones, 26 for the hard cases. Cost scales roughly
    linearly in the count, so start low.
    """
    if trimesh is None:  # pragma: no cover
        raise ImportError("extract_shell requires trimesh")

    centroids = mesh.triangles_center
    face_normals = mesh.face_normals
    visible = np.zeros(len(mesh.faces), dtype=bool)

    radius = float(mesh.extents.max()) * distance_multiplier
    viewpoints = _VIEW_DIRECTIONS[:num_viewpoints] * radius

    for view_pos in viewpoints:
        candidates = np.where(~visible)[0]
        if len(candidates) == 0:
            break

        to_camera = view_pos - centroids[candidates]
        to_camera /= np.linalg.norm(to_camera, axis=1)[:, None]
        facing = np.einsum("ij,ij->i", face_normals[candidates], to_camera) > 0.05
        active = candidates[facing]

        for start in range(0, len(active), chunk_size):
            batch = active[start:start + chunk_size]
            origins = np.tile(view_pos, (len(batch), 1))
            directions = centroids[batch] - view_pos
            directions /= np.linalg.norm(directions, axis=1)[:, None]

            index_tri, index_ray = mesh.ray.intersects_id(
                ray_origins=origins, ray_directions=directions, multiple_hits=False
            )
            aimed_at = batch[index_ray]
            visible[aimed_at[index_tri == aimed_at]] = True

    shell = mesh.copy()
    shell.update_faces(visible)
    shell.remove_unreferenced_vertices()
    return shell
