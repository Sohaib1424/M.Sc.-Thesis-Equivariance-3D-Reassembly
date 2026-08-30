"""
Outer-shell extraction by ray-cast visibility.

Faces visible from outside the fragment -- loosely, the complement of the
fracture surface.

**Nothing in this package uses this module.** It is carried over because it
was in the original ``geometry.py``, where it was likewise only ever called
from its own test. It is not on the dataset path, the extraction path, or the
visualisation path, and deleting the file would break nothing but its test.
It is kept only in case a figure or a sanity check wants it later; do not
build on it without deciding it is actually wanted.

Ray-mesh intersection is inherently slow and no amount of array work fixes
that. Three things make it less slow than the original:

* the viewpoint set is a parameter, not hard-coded. The original always cast
  from all 26 directions even though the comment says 6 usually suffice; the
  last 8 corner views are the expensive ones. ``level=6`` is roughly 4x
  faster than ``level=26``.
* candidate faces are re-filtered after every viewpoint, so once a face is
  known visible it is never tested again.
* ray origins are broadcast instead of tiled, which avoids a full copy of the
  origin array per chunk.

Installing ``pyembree`` or ``embreex`` speeds up ``intersects_id`` by roughly
an order of magnitude and is worth it if shells are needed in bulk.
"""
from __future__ import annotations

import numpy as np

_S2 = 1.0 / np.sqrt(2.0)
_S3 = 1.0 / np.sqrt(3.0)

_AXIS_VIEWS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
], dtype=np.float64)

_EDGE_VIEWS = np.array([
    [_S2, _S2, 0], [-_S2, _S2, 0], [_S2, -_S2, 0], [-_S2, -_S2, 0],
    [_S2, 0, _S2], [-_S2, 0, _S2], [_S2, 0, -_S2], [-_S2, 0, -_S2],
    [0, _S2, _S2], [0, -_S2, _S2], [0, _S2, -_S2], [0, -_S2, -_S2],
], dtype=np.float64)

_CORNER_VIEWS = np.array([
    [_S3, _S3, _S3], [-_S3, _S3, _S3], [_S3, -_S3, _S3], [_S3, _S3, -_S3],
    [-_S3, -_S3, _S3], [-_S3, _S3, -_S3], [_S3, -_S3, -_S3], [-_S3, -_S3, -_S3],
], dtype=np.float64)


def viewpoint_directions(level: int = 26) -> np.ndarray:
    """Unit view directions. ``level`` is 6 (axes), 18 (+edges) or 26 (+corners)."""
    if level == 6:
        return _AXIS_VIEWS
    if level == 18:
        return np.vstack([_AXIS_VIEWS, _EDGE_VIEWS])
    if level == 26:
        return np.vstack([_AXIS_VIEWS, _EDGE_VIEWS, _CORNER_VIEWS])
    raise ValueError(f"level must be 6, 18 or 26, got {level}")


def shell_face_mask(
    mesh,
    level: int = 26,
    chunk_size: int = 4096,
    distance_scale: float = 8.5,
    backface_tol: float = 0.05,
) -> np.ndarray:
    """
    Boolean mask of faces visible from at least one external viewpoint.

    ``chunk_size`` bounds peak memory during ray casting; lower it on a small
    machine. ``backface_tol`` skips faces angled away from the camera, which
    halves the ray count at no cost in accuracy.
    """
    centroids = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    n_faces = len(mesh.faces)
    visible = np.zeros(n_faces, dtype=bool)
    if n_faces == 0:
        return visible

    radius = float(np.max(mesh.extents)) * distance_scale
    for direction in viewpoint_directions(level):
        candidates = np.flatnonzero(~visible)
        if candidates.size == 0:
            break

        view_pos = direction * radius
        to_camera = view_pos - centroids[candidates]
        to_camera /= np.linalg.norm(to_camera, axis=1)[:, None]
        facing = np.einsum("ij,ij->i", normals[candidates], to_camera) > backface_tol
        active = candidates[facing]

        for start in range(0, active.size, chunk_size):
            batch = active[start:start + chunk_size]
            directions = centroids[batch] - view_pos
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            origins = np.broadcast_to(view_pos, directions.shape)

            hit_tri, hit_ray = mesh.ray.intersects_id(
                ray_origins=origins, ray_directions=directions, multiple_hits=False
            )
            if len(hit_ray) == 0:
                continue
            aimed_at = batch[hit_ray]
            visible[aimed_at[hit_tri == aimed_at]] = True

    return visible


def extract_shell(mesh, level: int = 26, chunk_size: int = 4096, **kwargs):
    """The visible-from-outside sub-mesh, with unreferenced vertices dropped."""
    import trimesh

    from ..arrays import compact_indices

    mask = shell_face_mask(mesh, level=level, chunk_size=chunk_size, **kwargs)
    vertices = np.asarray(mesh.vertices)
    kept, new_faces = compact_indices(np.asarray(mesh.faces)[mask], vertices.shape[0])
    return trimesh.Trimesh(vertices[kept], new_faces, process=False)
