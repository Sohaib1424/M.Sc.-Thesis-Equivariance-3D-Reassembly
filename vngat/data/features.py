"""
Node and edge features for one fragment mesh.

Node features (per vertex v), 2 vector channels:
    [ x~_v , n_v ]        x~_v = x_v - mean(x)      (centralised position)

Centralisation is what isolates rotation from translation: under a rigid
transform x' = A x + t the centroid moves by exactly A xbar + t, so
x~'_v = A x~_v with the translation cancelled. That is what makes the rotation
target well defined at all.

Edge features (per UNIQUE undirected edge (u, v), u < v), 3 vector channels:
    [ n_1 , n_2 , p_u - p_v ]                                    (equivariant)

n_1, n_2 are the two adjacent face normals in a CANONICAL order (below) and
p_u - p_v is the relative position of the edge's source vertex seen from its
destination. This is the edge input `E:\\My Thesis Work`'s VN-GAT layer uses,
and it replaces the previous `[midpoint, n1, n2]` plus an invariant length.

  * The MIDPOINT went because it is a position, not a relation: it says where
    the edge is in the fragment, which the two endpoint features already say.
  * The RELATIVE POSITION came in because a message is a linear map of the
    source vertex's features, so without it a message can say what the
    neighbour looks like but not which way it lies. The raw difference rather
    than a unit direction: equally equivariant, and its length is an invariant
    the network can read off with a norm, so the separate length scalar is
    redundant too.
  * A fragment's size no longer rides on its coordinates -- see
    `vngat.data.graph.merge_fragments` for the per-scene normalisation and the
    invariant scale feature that replaces it.

WHICH NORMAL GOES IN WHICH SLOT
-------------------------------
The previous feature builder filled the two slots in FACE-ARRAY ORDER, and the
face array is reordered lexicographically by `resolve_duplicated_faces` when
the scene is decompressed. The slot assignment therefore had no geometric
meaning: measured on fragments produced by this project's own decompression
code, it agreed with the geometric rule below on 50.1% of edges (17,176 of
34,276 with non-parallel normals) -- a coin flip. Now the order is fixed by the
scalar triple product

    s = (n_a x n_b) . (x_v - x_u)          keep (n_a, n_b) if s >= 0, else swap

which is "the normal on the left of the edge first". It is invariant under
proper rotation, (R n_a x R n_b) . (R d) = det(R) (n_a x n_b) . d, so rotating
a fragment rotates the two slots without exchanging them -- exactly what an
equivariant layer needs, and what lets `SceneBatch.rotate_per_fragment` stay
exact. The rule is ambiguous only when s -> 0, i.e. when the two normals are
parallel, and then both slots hold the same vector so the order does not
matter. A boundary edge (one incident face) carries that face's normal in both
slots; an edge with no incident face carries zeros.

The reverse directed copy of an edge (v -> u) is built on the device by the
model, as (n_2, n_1, p_v - p_u): the triple product flips sign with the
direction, so the canonical rule itself swaps the slots.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

from .graph import FragmentGraph


def _adjacent_faces(mesh: trimesh.base.Trimesh, num_edges: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every unique edge: its first and second incident face, and how many
    faces it has. Aligned with `mesh.edges_unique`.

    A boundary edge reports its single face twice; an edge with no face
    (impossible for a mesh built from faces, handled anyway) reports face 0 and
    count 0 so the caller can zero it.
    """
    inv = getattr(mesh, "edges_unique_inverse", None)
    if inv is None:
        # Fallback for trimesh builds without the cached property: encode each
        # sorted vertex pair as one integer and binary-search it against the
        # sorted set of unique edges.
        edges_sorted = np.sort(mesh.edges, axis=1)
        edges_unique_sorted = np.sort(mesh.edges_unique, axis=1)
        num_v = mesh.vertices.shape[0]
        keys_all = edges_sorted[:, 0].astype(np.int64) * num_v + edges_sorted[:, 1]
        keys_unique = edges_unique_sorted[:, 0].astype(np.int64) * num_v + edges_unique_sorted[:, 1]
        sort_order = np.argsort(keys_unique)
        pos = np.searchsorted(keys_unique[sort_order], keys_all)
        inv = sort_order[pos]
    inv = np.asarray(inv).reshape(-1)

    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    sorted_face_ids = np.asarray(mesh.edges_face)[order]

    counts = np.bincount(sorted_inv, minlength=num_edges)
    starts = np.zeros(num_edges, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    if len(sorted_face_ids) == 0:
        zeros = np.zeros(num_edges, dtype=np.int64)
        return zeros, zeros, counts
    last = len(sorted_face_ids) - 1
    # Edges with no face at all are clipped to a valid index here and zeroed
    # by the caller (`counts < 1`).
    first_face = sorted_face_ids[np.clip(starts, 0, last)]
    second_face = np.where(counts >= 2, sorted_face_ids[np.clip(starts + 1, 0, last)], first_face)
    return first_face, second_face, counts


def canonical_edge_normals(
    positions: np.ndarray,
    edges: np.ndarray,
    face_normals: np.ndarray,
    first_face: np.ndarray,
    second_face: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (n1, n2), each (E, 3), ordered by the triple-product rule in the module
    docstring. `edges` is (E, 2) with the SOURCE in column 0; the rule uses
    d = x_destination - x_source.
    """
    n_a = face_normals[first_face].astype(np.float64)
    n_b = face_normals[second_face].astype(np.float64)
    d = positions[edges[:, 1]].astype(np.float64) - positions[edges[:, 0]].astype(np.float64)
    s = np.einsum("ij,ij->i", np.cross(n_a, n_b), d)
    swap = (s < 0.0)[:, None]
    n1 = np.where(swap, n_b, n_a)
    n2 = np.where(swap, n_a, n_b)
    none = counts < 1
    n1[none] = 0.0
    n2[none] = 0.0
    return n1.astype(np.float32), n2.astype(np.float32)


def _sanitise(array: np.ndarray, name: str, scene: str = "") -> tuple:
    """
    Replace non-finite entries with 0 and report how many there were.

    trimesh derives vertex normals by dividing a sum of face normals by its own
    length, so a vertex whose incident faces are all ZERO-AREA gets 0/0 = NaN.
    Breaking Bad base meshes do contain such triangles: in one 8-object
    training run, three specific objects produced NaN losses repeatedly, under
    several different fracture patterns each, across forty epochs. The NaN was
    in the DATA before the model ever saw it.

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
    Build one fragment's `FragmentGraph`, in WORLD units.

    Positions are only centred here. Scaling needs the whole scene -- the
    divisor is the largest fragment's radius -- so it happens when fragments
    are merged into a scene (`merge_fragments`).

    `vertex_cluster_ids` / `edge_cluster_ids` are the optional correspondence
    supervision, aligned with `mesh.vertices` and `mesh.edges_unique`
    respectively. Pass None to fill with -1 ("nothing to be consistent with").
    """
    # .copy(): trimesh's .vertices / .vertex_normals are cached TrackedArray
    # views into its internal cache and are sometimes flagged non-writable;
    # torch.from_numpy on those warns, and any later in-place write would be
    # genuinely undefined rather than merely noisy.
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    num_v = verts.shape[0]

    centroid = verts.mean(axis=0) if num_v else np.zeros(3, dtype=np.float64)
    centred = verts - centroid
    radius = float(np.linalg.norm(centred, axis=1).max()) if num_v else 0.0
    pos = centred.astype(np.float32)

    if num_v:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
    else:
        normals = np.zeros((0, 3), dtype=np.float32)

    # Degenerate triangles make trimesh emit NaN normals; repair before they
    # reach a tensor. `pos` is checked too, cheaply, since a NaN vertex would be
    # just as fatal and just as invisible.
    pos, bad_pos = _sanitise(pos, "position")
    normals, bad_normals = _sanitise(normals, "vertex normal")
    if not np.isfinite(radius):
        radius = 0.0

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
            edge_attr=torch.zeros((0, 3, 3), dtype=torch.float32),
            centroid=torch.from_numpy(centroid.astype(np.float32)),
            radius=radius,
            vertex_cluster_id=v_clusters,
            edge_cluster_id=torch.zeros((0,), dtype=torch.long),
        )
        empty.num_repaired = bad_pos + bad_normals
        return empty

    # `edges_unique` rows are sorted, so u < v: the stored copy runs u -> v.
    edges = edges_unique.astype(np.int64)
    # trimesh sets a degenerate face's normal to zero rather than NaN, but the
    # sanitiser runs anyway: a normal is never allowed to reach a tensor unchecked.
    face_normals, bad_faces = _sanitise(np.asarray(mesh.face_normals, dtype=np.float64), "face normal")
    first_face, second_face, counts = _adjacent_faces(mesh, num_e)
    n1, n2 = canonical_edge_normals(centred, edges, face_normals, first_face, second_face, counts)
    delta = (pos[edges[:, 0]] - pos[edges[:, 1]]).astype(np.float32)   # p_source - p_destination

    edge_attr = torch.from_numpy(np.stack([n1, n2, delta], axis=1))    # (E, 3, 3)

    if edge_cluster_ids is None:
        e_clusters = torch.full((num_e,), -1, dtype=torch.long)
    else:
        e_clusters = torch.from_numpy(np.asarray(edge_cluster_ids, dtype=np.int64))

    graph = FragmentGraph(
        node_vec=node_vec,
        edge_index=torch.from_numpy(edges).t().contiguous(),
        edge_attr=edge_attr,
        centroid=torch.from_numpy(centroid.astype(np.float32)),
        radius=radius,
        vertex_cluster_id=v_clusters,
        edge_cluster_id=e_clusters,
    )
    graph.num_repaired = bad_pos + bad_normals + bad_faces
    return graph
