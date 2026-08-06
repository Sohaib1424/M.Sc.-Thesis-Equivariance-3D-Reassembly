"""
Turning one fragment mesh into a PyG ``Data`` graph.

NODE FEATURES (6D per vertex)
    h_v = [ x~_v , n_v ]        x~_v = x_v - centroid(fragment)

Centralization is what isolates rotation from translation: under a rigid
x' = R x + t the centroid moves by exactly the same amount, so x~' = R x~ --
the translation cancels and what remains between the clean and perturbed
fragment is a pure rotation. That algebraic fact is the reason ``R_gt`` is
well-defined at all.

EDGE FEATURES (10D per unique edge, stored bidirectionally)
    e_uv = [ l_uv , m_uv , n1 , n2 ]
    l_uv = ||x~_u - x~_v||        m_uv = (x~_u + x~_v) / 2
    n1, n2 = the (up to) two adjacent face normals

The two adjacent face normals are the specific extra geometric signal this
design carries that GARF's feature set does not -- the mechanism by which the
thesis intends to reach competitive accuracy with a smaller model and fewer
steps.

The backward copy of each edge mirrors the n1/n2 slots so that "the face on
this side" means the same thing regardless of traversal direction.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

EDGE_FEATURE_DIM = 10
NODE_FEATURE_DIM = 6


def _edge_face_normals(mesh, edges_unique: np.ndarray) -> tuple:
    """The (up to) two adjacent face normals per unique edge, vectorized.

    Replaces a pure-Python loop over every edge with a dict keyed by sorted
    vertex pairs. Boundary edges (one incident face) get that face's normal in
    both slots; edges with no incident face get zeros.
    """
    E = len(edges_unique)

    inverse = getattr(mesh, "edges_unique_inverse", None)
    if inverse is None:
        # Manual fallback for trimesh builds without the cached property:
        # encode each sorted vertex pair as one integer and search against the
        # (much smaller) sorted unique-edge key set.
        edges_sorted = np.sort(np.asarray(mesh.edges), axis=1)
        unique_sorted = np.sort(edges_unique, axis=1)
        num_v = int(mesh.vertices.shape[0])
        keys_all = edges_sorted[:, 0].astype(np.int64) * num_v + edges_sorted[:, 1]
        keys_unique = unique_sorted[:, 0].astype(np.int64) * num_v + unique_sorted[:, 1]
        order = np.argsort(keys_unique)
        pos = np.searchsorted(keys_unique[order], keys_all)
        inverse = order[np.clip(pos, 0, len(order) - 1)]
    inverse = np.asarray(inverse).reshape(-1)

    face_normals = np.asarray(mesh.face_normals)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_faces = np.asarray(mesh.edges_face)[order]

    counts = np.bincount(sorted_inverse, minlength=E)
    starts = np.zeros(E, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    has_any = counts >= 1
    has_two = counts >= 2

    first = sorted_faces[np.where(has_any, starts, 0)]
    second_idx = np.clip(starts + 1, 0, max(len(sorted_faces) - 1, 0))
    second = np.where(has_two, sorted_faces[second_idx], first)

    n1 = face_normals[first].copy()
    n2 = face_normals[second].copy()
    n1[~has_any] = 0.0
    n2[~has_any] = 0.0
    return n1, n2


def _empty_graph(
    num_nodes: int,
    x: torch.Tensor,
    vertex_cluster_id: Optional[torch.Tensor],
    with_edge_clusters: bool,
) -> Data:
    data = Data(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32),
        inc_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
    )
    data.is_forward_edge = torch.empty((0,), dtype=torch.bool)
    if vertex_cluster_id is not None:
        data.vertex_cluster_id = vertex_cluster_id
    if with_edge_clusters:
        data.edge_cluster_id = torch.empty((0,), dtype=torch.long)
    return data


def get_features(
    mesh,
    vertex_cluster_ids: Optional[np.ndarray] = None,
    edge_cluster_ids: Optional[np.ndarray] = None,
    dtype: torch.dtype = torch.float32,
) -> Data:
    """Build the graph for one fragment.

    ``vertex_cluster_ids`` / ``edge_cluster_ids`` are the optional
    cross-fragment correspondence labels. ``edge_cluster_ids`` must be aligned
    with ``mesh.edges_unique`` (the undoubled ordering); it is doubled here to
    match the bidirectional edge layout.

    ``.copy()`` on vertices/normals is not cosmetic: trimesh returns cached
    ``TrackedArray`` views that are sometimes flagged non-writable, and
    ``torch.from_numpy`` on those warns and leaves any later in-place write as
    genuinely undefined behaviour. The copy costs one fragment's worth of
    vertices.
    """
    vertices = np.asarray(mesh.vertices)
    num_nodes = len(vertices)

    if num_nodes == 0:
        # Degenerate fragment. Return a structurally valid empty graph rather
        # than letting a NaN centroid propagate silently into training.
        x = torch.zeros((0, NODE_FEATURE_DIM), dtype=dtype)
        data = _empty_graph(0, x, None, edge_cluster_ids is not None)
        data.centroid = torch.zeros((1, 3), dtype=dtype)
        return data

    pos_raw = torch.from_numpy(vertices.copy()).to(dtype)
    centroid = pos_raw.mean(dim=0, keepdim=True)
    pos = pos_raw - centroid

    normals = torch.from_numpy(np.asarray(mesh.vertex_normals).copy()).to(dtype)
    x = torch.cat([pos, normals], dim=-1)

    vertex_cluster_id = None
    if vertex_cluster_ids is not None:
        vertex_cluster_id = torch.from_numpy(
            np.asarray(vertex_cluster_ids, dtype=np.int64)
        ).long()

    edges_unique = np.asarray(mesh.edges_unique)
    if len(edges_unique) == 0:
        data = _empty_graph(num_nodes, x, vertex_cluster_id, edge_cluster_ids is not None)
        data.centroid = centroid
        return data

    E = len(edges_unique)
    u, v = edges_unique[:, 0], edges_unique[:, 1]
    p_u, p_v = pos[u], pos[v]

    edge_lens = torch.norm(p_u - p_v, dim=1, keepdim=True)     # (E, 1) invariant
    edge_mids = (p_u + p_v) / 2.0                              # (E, 3) equivariant

    n1_np, n2_np = _edge_face_normals(mesh, edges_unique)
    edge_n1 = torch.from_numpy(n1_np).to(dtype)
    edge_n2 = torch.from_numpy(n2_np).to(dtype)

    forward = torch.cat([edge_lens, edge_mids, edge_n1, edge_n2], dim=-1)   # (E, 10)
    # Reverse copy: same length/midpoint, face-normal slots swapped so that
    # "n1" keeps meaning the same physical side.
    backward = torch.cat(
        [forward[:, 0:1], forward[:, 1:4], forward[:, 7:10], forward[:, 4:7]], dim=-1
    )

    edge_index_f = torch.from_numpy(edges_unique.astype(np.int64)).t().contiguous()
    edge_index = torch.cat([edge_index_f, edge_index_f.flip(0)], dim=1)     # (2, 2E)
    edge_attr = torch.cat([forward, backward], dim=0)                      # (2E, 10)

    # Explicit forward marker. After several fragments are merged, the layout
    # is per-fragment [fwd_i; bwd_i] blocks end to end -- NOT a clean
    # first-half/second-half split of the whole array. Anything downstream
    # that wants "one entry per undirected edge" must use this mask; slicing
    # positionally is correct for a single fragment and silently wrong for two.
    is_forward_edge = torch.cat(
        [torch.ones(E, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)]
    )

    edge_cluster_id = None
    if edge_cluster_ids is not None:
        ecid = torch.from_numpy(np.asarray(edge_cluster_ids, dtype=np.int64)).long()
        if len(ecid) != E:
            raise ValueError(
                f"edge_cluster_ids has length {len(ecid)} but the mesh has {E} unique "
                f"edges. These must be aligned with mesh.edges_unique."
            )
        edge_cluster_id = torch.cat([ecid, ecid], dim=0)                   # (2E,)

    # Node <-> directed-edge incidence, both endpoints per directed edge.
    num_directed = edge_index.shape[1]
    node_indices = torch.stack([edge_index[0], edge_index[1]], dim=1).reshape(-1)
    edge_ids = torch.arange(num_directed, dtype=torch.long)
    edge_indices = torch.stack([edge_ids, edge_ids], dim=1).reshape(-1)
    inc_index = torch.stack([node_indices, edge_indices], dim=0)           # (2, 4E)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inc_index=inc_index,
        num_nodes=num_nodes,
    )
    data.is_forward_edge = is_forward_edge
    # The centroid subtracted during centralization. Discarding it would make
    # the assembled ground-truth placement unrecoverable from the graph, which
    # is exactly what every translation metric needs.
    data.centroid = centroid
    if vertex_cluster_id is not None:
        data.vertex_cluster_id = vertex_cluster_id
    if edge_cluster_id is not None:
        data.edge_cluster_id = edge_cluster_id
    return data
