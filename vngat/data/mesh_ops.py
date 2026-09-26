"""
Per-fragment mesh operations: face adjacency, fracture-surface extraction,
visible-shell extraction.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import csr_matrix


def find_neighbors(mesh: trimesh.base.Trimesh, min_common_vertices: int = 2):
    """
    Face-face adjacency, as a pair of sparse (F, F) matrices:
      W: cosine similarity between adjacent faces' normals
      M: binary adjacency

    Vectorised replacement for the original O(F^2) Python double loop: builds a
    (V, F) vertex-face incidence matrix and gets shared-vertex counts from a
    single sparse product V^T V. Bit-for-bit equivalent to the loop version on
    manifold, non-manifold, and corner-touching-only cases.
    """
    num_faces = mesh.faces.shape[0]
    num_verts = mesh.vertices.shape[0]
    if num_faces == 0:
        empty = csr_matrix((0, 0), dtype=np.float32)
        return empty, empty

    face_ids = np.repeat(np.arange(num_faces), 3)
    vert_ids = mesh.faces.reshape(-1)
    VF = csr_matrix(
        (np.ones(len(vert_ids), dtype=np.float32), (vert_ids, face_ids)),
        shape=(num_verts, num_faces),
    )

    shared = (VF.T @ VF).tocoo()

    mask = (shared.row != shared.col) & (shared.data >= min_common_vertices)
    f1, f2 = shared.row[mask], shared.col[mask]
    cos_sim = np.einsum('ij,ij->i', mesh.face_normals[f1], mesh.face_normals[f2])

    W = csr_matrix((cos_sim.astype(np.float32), (f1, f2)), shape=(num_faces, num_faces))
    M = csr_matrix((np.ones(len(f1), dtype=np.float32), (f1, f2)), shape=(num_faces, num_faces))
    return W, M


def extract_fractures(mesh: trimesh.base.Trimesh, sharpness: float = 0.9) -> trimesh.base.Trimesh:
    """
    Keep only the faces that look like fracture surface, discarding the
    object's original smooth exterior.

    A face is kept when BOTH hold:
      (a) every one of its vertices touches some face that has a "sharp"
          neighbour (adjacent face-normal cosine below `sharpness`), and
      (b) at least half of the face's own neighbours are themselves sharp.

    Two fixes relative to the original implementation:

    1. Criterion (b) was computed and then never used -- only (a) reached the
       returned face set, silently over-including smooth faces that merely
       happened to touch the fracture rim.

    2. The result was built as `trimesh.Trimesh(mesh.vertices, mesh.faces[Fs])`,
       i.e. reusing the FULL vertex array. Since feature construction reads
       `mesh.vertices` directly regardless of which vertices surviving faces
       actually reference, the "fracture-surface mesh" had exactly the same
       node count as the full fragment -- only its edge count shrank. That
       silently defeats the whole point of the frac-vs-full compute
       comparison. Fixed by pruning + reindexing with
       `remove_unreferenced_vertices()`.

    Returns a mesh with zero faces when nothing qualifies; callers are
    expected to handle that degenerate case explicitly.
    """
    if mesh.faces.shape[0] == 0:
        return mesh.copy()

    W, M = find_neighbors(mesh)

    W_csr = W.tocsr()
    M_csr = M.tocsr()

    sharp_mask = np.abs(W_csr.data) < sharpness

    # Copy W (not M) so the sparsity pattern stays aligned with `sharp_mask`.
    A_csr = W_csr.copy()
    A_csr.data = sharp_mask.astype(np.float32)

    sharp_counts = np.asarray(A_csr.sum(axis=1)).ravel()
    total_neighbors = np.asarray(M_csr.sum(axis=1)).ravel()

    sharp_faces = np.where(sharp_counts > 0)[0]
    if len(sharp_faces) == 0:
        uniq_Vs = np.array([], dtype=int)
    else:
        uniq_Vs = np.unique(mesh.faces[sharp_faces])

    in_Vs_mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    in_Vs_mask[uniq_Vs] = True

    faces_in_Vs = in_Vs_mask[mesh.faces].all(axis=1)
    to_include = (total_neighbors // 2) <= sharp_counts

    keep = faces_in_Vs & to_include

    frac_mesh = mesh.copy()
    frac_mesh.update_faces(keep)
    frac_mesh.remove_unreferenced_vertices()
    return frac_mesh


def extract_shell(mesh: trimesh.base.Trimesh, chunk_size: int = 1000) -> trimesh.base.Trimesh:
    """
    Ray-casting extraction of the externally visible shell of a fragment.

    Kept for visualisation/analysis; the training pipeline does not use it
    (it is far too slow for a per-sample data path). `chunk_size` bounds peak
    memory of the ray batches.
    """
    centroids = mesh.triangles_center
    face_normals = mesh.face_normals
    visible_faces_mask = np.zeros(len(mesh.faces), dtype=bool)

    extents = mesh.extents.max() * 8.5
    s2 = (1 / np.sqrt(2)) * extents
    s3 = (1 / np.sqrt(3)) * extents
    axis = [
        (extents, 0, 0), (-extents, 0, 0), (0, extents, 0),
        (0, -extents, 0), (0, 0, extents), (0, 0, -extents),
    ]
    edge = [
        (s2, s2, 0), (-s2, s2, 0), (s2, -s2, 0), (-s2, -s2, 0),
        (s2, 0, s2), (-s2, 0, s2), (s2, 0, -s2), (-s2, 0, -s2),
        (0, s2, s2), (0, -s2, s2), (0, s2, -s2), (0, -s2, -s2),
    ]
    corner = [
        (s3, s3, s3), (-s3, s3, s3), (s3, -s3, s3), (s3, s3, -s3),
        (-s3, -s3, s3), (-s3, s3, -s3), (s3, -s3, -s3), (-s3, -s3, -s3),
    ]
    viewpoints = [np.array(v, dtype=float) for v in (axis + edge + corner)]

    for view_pos in viewpoints:
        candidates = np.where(~visible_faces_mask)[0]
        if len(candidates) == 0:
            break

        to_camera = view_pos - centroids[candidates]
        to_camera /= np.linalg.norm(to_camera, axis=1)[:, np.newaxis]

        facing_camera = np.einsum('ij,ij->i', face_normals[candidates], to_camera) > 0.05
        active_candidates = candidates[facing_camera]

        for i in range(0, len(active_candidates), chunk_size):
            batch_indices = active_candidates[i:i + chunk_size]

            origins = np.tile(view_pos, (len(batch_indices), 1))
            directions = centroids[batch_indices] - view_pos
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

            index_tri, index_ray = mesh.ray.intersects_id(
                ray_origins=origins, ray_directions=directions, multiple_hits=False,
            )

            aimed_at = batch_indices[index_ray]
            was_visible = (index_tri == aimed_at)
            visible_faces_mask[aimed_at[was_visible]] = True

    shell = mesh.copy()
    shell.update_faces(visible_faces_mask)
    shell.remove_unreferenced_vertices()
    return shell
