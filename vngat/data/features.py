"""
Node and edge feature construction for one fragment mesh.

Node features (per vertex v), 2 vector channels:
    [ x~_v , n_v ]        x~_v = x_v - mean(x)      (centralised position)

Centralisation is the mechanism that isolates rotation from translation: under
a rigid transform x' = A x + t the centroid shifts by exactly A xbar + t, so
x~'_v = A x~_v with the translation cancelled. That algebraic fact is what
makes the rotation target well defined at all.

Edge features (per UNIQUE undirected edge (u, v)):
    scalar:  l_uv = ||x~_u - x~_v||                            (invariant)
    vectors: [ m_uv , n_1 , n_2 ]                              (equivariant)
where m_uv is the midpoint and n_1, n_2 the (up to) two adjacent face normals
(equal to each other on a boundary edge with one incident face; zero if an
edge somehow has none).

Those two adjacent face normals are the specific extra geometric signal this
thesis adds relative to GARF's feature set, so the per-edge face-normal lookup
is on the critical path and is fully vectorised via `edges_unique_inverse`
rather than the original dict-keyed Python loop over every edge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import trimesh

from .graph import FragmentGraph


def _adjacent_face_normals(mesh: trimesh.base.Trimesh, num_edges: int) -> tuple:
    """(n1, n2) float32 arrays of shape (E, 3), aligned with mesh.edges_unique."""
    inv = getattr(mesh, "edges_unique_inverse", None)
    if inv is None:
        # Manual fallback for trimesh builds without the cached property:
        # encode each sorted vertex pair as one integer and binary-search it
        # against the (much smaller) sorted set of unique edges.
        edges_sorted = np.sort(mesh.edges, axis=1)
        edges_unique_sorted = np.sort(mesh.edges_unique, axis=1)
        num_v = mesh.vertices.shape[0]
        keys_all = edges_sorted[:, 0].astype(np.int64) * num_v + edges_sorted[:, 1]
        keys_unique = edges_unique_sorted[:, 0].astype(np.int64) * num_v + edges_unique_sorted[:, 1]
        sort_order = np.argsort(keys_unique)
        pos = np.searchsorted(keys_unique[sort_order], keys_all)
        inv = sort_order[pos]
    inv = np.asarray(inv).reshape(-1)

    face_normals = np.asarray(mesh.face_normals)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    sorted_face_ids = np.asarray(mesh.edges_face)[order]

    counts = np.bincount(sorted_inv, minlength=num_edges)
    starts = np.zeros(num_edges, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    has_any = counts >= 1
    has_two = counts >= 2

    safe_starts = np.where(has_any, starts, 0)
    first_face = sorted_face_ids[safe_starts]

    second_idx = np.clip(starts + 1, 0, max(len(sorted_face_ids) - 1, 0))
    second_face = sorted_face_ids[second_idx]
    second_face_final = np.where(has_two, second_face, first_face)

    n1 = face_normals[first_face].astype(np.float32, copy=True)
    n2 = face_normals[second_face_final].astype(np.float32, copy=True)
    zero_mask = ~has_any
    n1[zero_mask] = 0.0
    n2[zero_mask] = 0.0
    return n1, n2


def _sanitise(array: np.ndarray, name: str, scene: str = "") -> tuple:
    """
    Replace non-finite entries with 0 and report how many there were.

    trimesh derives vertex and face normals by dividing a cross product by its
    own length, so a ZERO-AREA (degenerate) triangle gives 0/0 = NaN. Breaking
    Bad base meshes do contain such triangles: in one 8-object training run,
    three specific objects produced NaN losses repeatedly, under several
    different fracture patterns each, across forty epochs. The NaN was in the
    DATA before the model ever saw it.

    Zeroing is the right repair for a normal: a zero vector contributes nothing
    to the dot products the losses take, which is the correct treatment for a
    face that has no well-defined orientation. Silently dropping it would not
    be -- hence the count, which the dataset surfaces.
    """
    bad = ~np.isfinite(array)
    count = int(bad.sum())
    if count:
        array = array.copy()
        array[bad] = 0.0
    return array, count


def get_features(
    mesh: trimesh.base.Trimesh,
    vertex_cluster_ids: Optional[np.ndarray] = None,
    edge_cluster_ids: Optional[np.ndarray] = None,
) -> FragmentGraph:
    """
    Build one fragment's `FragmentGraph`.

    `vertex_cluster_ids` / `edge_cluster_ids` are the optional correspondence
    supervision, aligned with `mesh.vertices` and `mesh.edges_unique`
    respectively. Pass None to fill with -1 ("nothing to be consistent with").
    """
    # .copy(): trimesh's .vertices / .vertex_normals are cached TrackedArray
    # views into its internal cache and are sometimes flagged non-writable;
    # torch.from_numpy on those warns, and any later in-place write would be
    # genuinely undefined rather than merely noisy. One fragment's worth of
    # vertices is a negligible copy.
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    num_v = verts.shape[0]

    centroid = verts.mean(axis=0) if num_v else np.zeros(3, dtype=np.float64)
    pos = (verts - centroid).astype(np.float32)

    if num_v:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
    else:
        normals = np.zeros((0, 3), dtype=np.float32)

    # Degenerate triangles make trimesh emit NaN normals; repair before they
    # reach a tensor. `pos` is checked too, cheaply, since a NaN vertex would
    # be just as fatal and just as invisible.
    pos, bad_pos = _sanitise(pos, "position")
    normals, bad_normals = _sanitise(normals, "vertex normal")

    node_vec = torch.from_numpy(np.stack([pos, normals], axis=1))     # (V, 2, 3)

    if vertex_cluster_ids is None:
        v_clusters = torch.full((num_v,), -1, dtype=torch.long)
    else:
        v_clusters = torch.from_numpy(np.asarray(vertex_cluster_ids, dtype=np.int64))

    edges_unique = np.asarray(mesh.edges_unique) if num_v else np.zeros((0, 2), dtype=np.int64)
    num_e = int(edges_unique.shape[0])

    if num_e == 0:
        empty = FragmentGraph(
            node_vec=node_vec,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_len=torch.zeros((0, 1), dtype=torch.float32),
            edge_vec=torch.zeros((0, 3, 3), dtype=torch.float32),
            centroid=torch.from_numpy(centroid.astype(np.float32)),
            vertex_cluster_id=v_clusters,
            edge_cluster_id=torch.zeros((0,), dtype=torch.long),
        )
        empty.num_repaired = bad_pos + bad_normals
        return empty

    u = edges_unique[:, 0]
    v = edges_unique[:, 1]
    p_u, p_v = pos[u], pos[v]

    edge_len = np.linalg.norm(p_u - p_v, axis=1, keepdims=True).astype(np.float32)
    edge_mid = ((p_u + p_v) * 0.5).astype(np.float32)
    n1, n2 = _adjacent_face_normals(mesh, num_e)

    n1, bad_n1 = _sanitise(n1, "face normal 1")
    n2, bad_n2 = _sanitise(n2, "face normal 2")
    edge_mid, bad_mid = _sanitise(edge_mid, "edge midpoint")
    edge_len, bad_len = _sanitise(edge_len, "edge length")

    edge_vec = torch.from_numpy(np.stack([edge_mid, n1, n2], axis=1))  # (E, 3, 3)

    if edge_cluster_ids is None:
        e_clusters = torch.full((num_e,), -1, dtype=torch.long)
    else:
        e_clusters = torch.from_numpy(np.asarray(edge_cluster_ids, dtype=np.int64))

    graph = FragmentGraph(
        node_vec=node_vec,
        edge_index=torch.from_numpy(edges_unique.astype(np.int64)).t().contiguous(),
        edge_len=torch.from_numpy(edge_len),
        edge_vec=edge_vec,
        centroid=torch.from_numpy(centroid.astype(np.float32)),
        vertex_cluster_id=v_clusters,
        edge_cluster_id=e_clusters,
    )
    graph.num_repaired = bad_pos + bad_normals + bad_n1 + bad_n2 + bad_mid + bad_len
    return graph
