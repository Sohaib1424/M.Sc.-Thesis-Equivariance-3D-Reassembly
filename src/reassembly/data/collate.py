"""
Merging fragment graphs into a scene, and scenes into a training batch.

Two levels, both plain disjoint-union concatenation with index offsetting (no
padding, standard PyG practice):

  merge_fragments   per-fragment graphs -> one scene graph, with a per-node
                    ``fragment_id`` local to that scene
  collate_scenes    scene graphs -> one batch, with ``fragment_id`` made
                    globally unique and ``fragment_scene_id`` recording which
                    scene each fragment came from

``fragment_scene_id`` is not decorative. Cross-fragment attention has to be
scoped to "the other fragments of THIS broken object". ``fragment_id`` alone is
unique across the whole batch, so using it as the scope lets fragments of
entirely unrelated scenes exchange information -- which is semantically wrong
and makes the model's output depend on which other scenes happened to land in
the same batch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch_geometric.data import Batch, Data

_PER_NODE_EXTRAS = ("vertex_cluster_id",)
_PER_EDGE_EXTRAS = ("edge_cluster_id", "is_forward_edge")


def merge_fragments(data_list: Sequence[Data]) -> Data:
    """Concatenate one scene's fragment graphs into a single disjoint graph.

    The resulting ``fragment_id`` is *scene-local* (0, 1, 2, ... within this
    scene); :func:`collate_scenes` makes it batch-global.
    """
    if not data_list:
        raise ValueError("merge_fragments got an empty fragment list")

    node_offset = 0
    edge_offset = 0

    xs, edge_indices, edge_attrs, inc_indices, frag_ids = [], [], [], [], []
    has_vertex_cluster = hasattr(data_list[0], "vertex_cluster_id")
    has_edge_cluster = hasattr(data_list[0], "edge_cluster_id")
    vertex_clusters: List[torch.Tensor] = []
    edge_clusters: List[torch.Tensor] = []
    forward_flags: List[torch.Tensor] = []
    centroids: List[torch.Tensor] = []

    for i, data in enumerate(data_list):
        num_nodes = data.x.size(0)
        num_edges = data.edge_attr.size(0)

        edge_indices.append(data.edge_index + node_offset)

        inc = data.inc_index.clone()
        inc[0] += node_offset
        inc[1] += edge_offset
        inc_indices.append(inc)

        frag_ids.append(torch.full((num_nodes,), i, dtype=torch.long))
        xs.append(data.x)
        edge_attrs.append(data.edge_attr)

        # Cluster ids are already scene-wide consistent (assigned once across
        # all fragments), so they concatenate with no offsetting.
        if has_vertex_cluster:
            vertex_clusters.append(data.vertex_cluster_id)
        if has_edge_cluster:
            edge_clusters.append(data.edge_cluster_id)
        forward_flags.append(data.is_forward_edge)
        centroids.append(getattr(data, 'centroid', torch.zeros((1, 3))).reshape(1, 3))

        node_offset += num_nodes
        edge_offset += num_edges

    result = Data(
        x=torch.cat(xs, dim=0).contiguous(),
        edge_index=torch.cat(edge_indices, dim=1).contiguous(),
        edge_attr=torch.cat(edge_attrs, dim=0).contiguous(),
        inc_index=torch.cat(inc_indices, dim=1).contiguous(),
        num_nodes=node_offset,
    )
    result.fragment_id = torch.cat(frag_ids, dim=0).contiguous()
    result.is_forward_edge = torch.cat(forward_flags, dim=0).contiguous()
    # (F, 3): where each fragment sat before centralization. On the CLEAN
    # graph this is the ground-truth assembled placement -- the target the
    # translation solver is scored against.
    result.fragment_centroid = torch.cat(centroids, dim=0).contiguous()

    # EXPLICIT fragment count, rather than fragment_id.max() + 1.
    # A fragment whose graph is empty (possible on the pruned fracture-surface
    # variant) contributes zero nodes and therefore never appears in
    # fragment_id at all -- so max()+1 undercounts, every per-fragment tensor
    # downstream (R_pred, t_matrices, fragment_scene_id) silently misaligns by
    # one, and nothing raises. Carrying the count avoids the whole class of bug.
    result.num_fragments = len(data_list)

    if has_vertex_cluster:
        result.vertex_cluster_id = torch.cat(vertex_clusters, dim=0).contiguous()
    if has_edge_cluster:
        result.edge_cluster_id = torch.cat(edge_clusters, dim=0).contiguous()
    return result


def collate_scenes(graph_list: Sequence[Data]) -> Optional[Batch]:
    """Stitch several scene graphs into one batch.

    Custom attributes are pulled out and re-attached by hand rather than left
    to PyG's name-based auto-batching heuristics, which key off substrings like
    "edge" / "index" in the attribute name and would offset (or fail to offset)
    them by guesswork.
    """
    graph_list = [g for g in graph_list if g is not None]
    if not graph_list:
        return None

    inc_indices, global_frag_ids = [], []
    vertex_clusters, edge_clusters, forward_flags, centroids = [], [], [], []

    node_offset = 0
    edge_offset = 0
    fragment_offset = 0
    # Vertex-cluster and edge-cluster ids are independent numbering schemes, so
    # they get independent offset counters.
    vertex_cluster_offset = 0
    edge_cluster_offset = 0
    fragments_per_scene: List[int] = []

    for g in graph_list:
        if hasattr(g, "inc_index") and g.inc_index is not None:
            inc = g.inc_index.clone()
            inc[0] += node_offset
            inc[1] += edge_offset
            inc_indices.append(inc)
            del g.inc_index

        local_frag_id = g.fragment_id
        global_frag_ids.append(local_frag_id + fragment_offset)
        num_frags = int(getattr(g, "num_fragments", int(local_frag_id.max().item()) + 1))
        fragments_per_scene.append(num_frags)
        del g.fragment_id
        if hasattr(g, "num_fragments"):
            del g.num_fragments

        for name, store, offset_holder in (
            ("vertex_cluster_id", vertex_clusters, "vertex"),
            ("edge_cluster_id", edge_clusters, "edge"),
        ):
            raw = getattr(g, name, None)
            if raw is None:
                continue
            n_clusters = int(raw.max().item()) + 1 if raw.numel() and raw.max() >= 0 else 0
            shifted = raw.clone()
            live = shifted >= 0
            shifted[live] += (
                vertex_cluster_offset if offset_holder == "vertex" else edge_cluster_offset
            )
            store.append(shifted)
            delattr(g, name)
            if offset_holder == "vertex":
                vertex_cluster_offset += n_clusters
            else:
                edge_cluster_offset += n_clusters

        if getattr(g, "is_forward_edge", None) is not None:
            forward_flags.append(g.is_forward_edge)
            del g.is_forward_edge

        if getattr(g, "fragment_centroid", None) is not None:
            centroids.append(g.fragment_centroid)
            del g.fragment_centroid

        node_offset += g.x.size(0)
        edge_offset += g.edge_attr.size(0)
        fragment_offset += num_frags

    batched = Batch.from_data_list(list(graph_list))

    if inc_indices:
        batched.inc_index = torch.cat(inc_indices, dim=1).contiguous()
    batched.fragment_id = torch.cat(global_frag_ids, dim=0).contiguous()
    batched.num_fragments = fragment_offset

    # Which scene each FRAGMENT belongs to. Built directly from the per-scene
    # fragment counts rather than scattered from per-node data, so it is
    # correct even for a fragment that contributed zero nodes.
    batched.fragment_scene_id = torch.repeat_interleave(
        torch.arange(len(fragments_per_scene), dtype=torch.long),
        torch.tensor(fragments_per_scene, dtype=torch.long),
    )

    if vertex_clusters:
        batched.vertex_cluster_id = torch.cat(vertex_clusters, dim=0).contiguous()
    if edge_clusters:
        batched.edge_cluster_id = torch.cat(edge_clusters, dim=0).contiguous()
    if forward_flags:
        batched.is_forward_edge = torch.cat(forward_flags, dim=0).contiguous()
    if centroids:
        batched.fragment_centroid = torch.cat(centroids, dim=0).contiguous()

    return batched


def breaking_bad_collate_fn(batch_list: Sequence[dict]) -> dict:
    """DataLoader ``collate_fn``: several scene samples -> one training batch.

    ``None`` items are dropped, which is how the dataset signals "this scene
    was over the vertex budget / failed to load" without killing the run.
    """
    batch_list = [b for b in batch_list if b is not None]
    if not batch_list:
        return {"graph": None, "num_scenes": 0}

    def gather(key):
        return [item[key] for item in batch_list if item.get(key) is not None]

    t_mats = gather("t_matrices")
    meshes = [m for item in batch_list if item.get("meshes") for m in item["meshes"]]
    diff_meshes = [
        m for item in batch_list if item.get("diffused_meshes") for m in item["diffused_meshes"]
    ]

    return {
        "graph": collate_scenes(gather("graph")),
        "diffused_graph": collate_scenes(gather("diffused_graph")),
        "frac_graph": collate_scenes(gather("frac_graph")),
        "diff_frac_graph": collate_scenes(gather("diff_frac_graph")),
        "t_matrices": torch.cat(t_mats, dim=0) if t_mats else None,
        "meshes": meshes or None,
        "diffused_meshes": diff_meshes or None,
        "num_scenes": len(batch_list),
        "scene_dirs": [item.get("scene_dir") for item in batch_list],
    }
