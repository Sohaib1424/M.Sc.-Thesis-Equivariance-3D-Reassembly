"""
Graph containers and batching.

DESIGN NOTE -- undirected storage, symmetrised on device
--------------------------------------------------------
The original pipeline stored every mesh edge twice (a forward and a backward
directed copy) at *dataset* level, then carried an `is_forward_edge` boolean
through two levels of merging so downstream code could recover "one entry per
undirected edge". That was necessary because merging several fragments lays
the copies out as per-fragment `[fwd_i; bwd_i]` blocks rather than a clean
first-half/second-half split -- a real bug the original code had to be fixed
for once already.

Here, the dataset stores UNDIRECTED edges only (one entry per mesh edge) and
the model symmetrises them on the GPU in a single `cat`. Consequences:
  * the interleaving bug cannot exist, by construction -- there is nothing to
    interleave and no mask to get wrong;
  * per-edge CPU tensors, the pinned-memory copy, and the host->device
    transfer all halve;
  * `edge_cluster_id` lines up with the edges 1:1 with no doubling or masking.

No PyTorch Geometric
--------------------
Batching is plain `torch.cat` with explicit offsets. PyG's `Batch.from_data_list`
was doing the same work through a much slower generic path, brought a heavy and
version-fragile dependency (a recurring source of Kaggle install breakage), and
its attribute-name-based auto-increment heuristics had to be defeated with
`del g.attr` hacks for every custom field. The message passing this model needs
is a scatter-softmax and an index_add, both of which are core torch ops.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional

import torch


@dataclass
class FragmentGraph:
    """One fragment's mesh graph, in its own centralised frame."""

    node_vec: torch.Tensor          # (V, 2, 3)  [centralised position, vertex normal]
    edge_index: torch.Tensor        # (2, E)     undirected, local vertex indices
    edge_len: torch.Tensor          # (E, 1)     invariant scalar
    edge_vec: torch.Tensor          # (E, 3, 3)  [midpoint, face normal 1, face normal 2]
    centroid: torch.Tensor          # (3,)       centroid subtracted from positions
    vertex_cluster_id: torch.Tensor  # (V,)
    edge_cluster_id: torch.Tensor    # (E,)

    @property
    def num_nodes(self) -> int:
        return int(self.node_vec.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class SceneBatch:
    """
    One or more scenes merged into a single disjoint-union graph.

    `node_frag` is a GLOBAL fragment index across the whole batch;
    `frag_scene` says which scene each fragment belongs to. Both are needed:
    fragment-scoped operations (pooling, virtual-node up/down attention) key
    off the first, and the cross-fragment attention stage must be scoped by
    the second or fragments of unrelated objects exchange information.
    """

    node_vec: torch.Tensor           # (N, 2, 3)
    edge_index: torch.Tensor         # (2, E) undirected
    edge_len: torch.Tensor           # (E, 1)
    edge_vec: torch.Tensor           # (E, 3, 3)
    node_frag: torch.Tensor          # (N,)  global fragment id
    frag_scene: torch.Tensor         # (F,)  scene id per fragment
    frag_centroid: torch.Tensor      # (F, 3) centroid in the assembled frame
    vertex_cluster_id: torch.Tensor  # (N,)
    edge_cluster_id: torch.Tensor    # (E,)
    num_fragments: int
    num_scenes: int

    # -- convenience ------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.node_vec.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def pos(self) -> torch.Tensor:
        return self.node_vec[:, 0]

    @property
    def normal(self) -> torch.Tensor:
        return self.node_vec[:, 1]

    @property
    def edge_frag(self) -> torch.Tensor:
        """Fragment id per edge. Mesh edges never cross a fragment boundary
        (fragments are disjoint meshes), so either endpoint gives the same
        answer -- asserted once in tests rather than stored redundantly."""
        return self.node_frag[self.edge_index[0]]

    def to(self, device: torch.device, non_blocking: bool = False) -> "SceneBatch":
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor) else value
            )
        return SceneBatch(**kwargs)

    def pin_memory(self) -> "SceneBatch":
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = value.pin_memory() if isinstance(value, torch.Tensor) else value
        return SceneBatch(**kwargs)

    def rotate_per_fragment(self, rot: torch.Tensor) -> "SceneBatch":
        """
        Apply a per-fragment rotation to every geometric quantity.

        `rot`: (F, 3, 3) acting on column vectors, i.e. v' = R v. Rows are
        stored as row vectors, so the operation is `v @ R^T`.

        This is what replaces re-running mesh transformation + feature
        extraction for the diffused view. It is EXACT, not an approximation:
          - centralised position:  R (x - xbar)             [centroid co-moves]
          - vertex normal:         R n                      [area weights invariant]
          - edge length:           unchanged                [rotations are isometries]
          - edge midpoint:         R m                      [linear in positions]
          - adjacent face normals: R n_f
        Topology (`edge_index`, orderings) depends only on vertex indices and
        is untouched by a rigid transform.
        """
        node_rot = rot[self.node_frag]                    # (N, 3, 3)
        edge_rot = rot[self.edge_frag]                    # (E, 3, 3)
        return SceneBatch(
            node_vec=torch.einsum('nij,ncj->nci', node_rot, self.node_vec),
            edge_index=self.edge_index,
            edge_len=self.edge_len,
            edge_vec=torch.einsum('eij,ecj->eci', edge_rot, self.edge_vec),
            node_frag=self.node_frag,
            frag_scene=self.frag_scene,
            frag_centroid=self.frag_centroid,
            vertex_cluster_id=self.vertex_cluster_id,
            edge_cluster_id=self.edge_cluster_id,
            num_fragments=self.num_fragments,
            num_scenes=self.num_scenes,
        )


def merge_fragments(fragments: List[FragmentGraph]) -> SceneBatch:
    """Disjoint-union one scene's fragments, offsetting vertex indices.

    Cluster ids are already scene-global (assigned once across all fragments
    by `compute_scene_correspondence`), so they are concatenated as-is.
    """
    if not fragments:
        raise ValueError("merge_fragments received an empty fragment list")

    node_offset = 0
    node_vecs, edge_indices, edge_lens, edge_vecs = [], [], [], []
    node_frags, centroids, v_clusters, e_clusters = [], [], [], []

    for i, frag in enumerate(fragments):
        n = frag.num_nodes
        node_vecs.append(frag.node_vec)
        edge_indices.append(frag.edge_index + node_offset)
        edge_lens.append(frag.edge_len)
        edge_vecs.append(frag.edge_vec)
        node_frags.append(torch.full((n,), i, dtype=torch.long))
        centroids.append(frag.centroid)
        v_clusters.append(frag.vertex_cluster_id)
        e_clusters.append(frag.edge_cluster_id)
        node_offset += n

    num_fragments = len(fragments)
    return SceneBatch(
        node_vec=torch.cat(node_vecs, 0),
        edge_index=torch.cat(edge_indices, 1),
        edge_len=torch.cat(edge_lens, 0),
        edge_vec=torch.cat(edge_vecs, 0),
        node_frag=torch.cat(node_frags, 0),
        frag_scene=torch.zeros(num_fragments, dtype=torch.long),
        frag_centroid=torch.stack(centroids, 0),
        vertex_cluster_id=torch.cat(v_clusters, 0),
        edge_cluster_id=torch.cat(e_clusters, 0),
        num_fragments=num_fragments,
        num_scenes=1,
    )


def collate_scenes(scenes: List[SceneBatch]) -> Optional[SceneBatch]:
    """
    Stitch several independent scenes into one training batch.

    Offsets applied: vertex indices, fragment ids, and -- independently --
    the vertex-cluster and edge-cluster id spaces. Entries of -1 ("not shared
    with any other fragment") are left alone so they stay -1 after batching.
    """
    scenes = [s for s in scenes if s is not None]
    if not scenes:
        return None
    if len(scenes) == 1 and scenes[0].num_scenes == 1:
        return scenes[0]

    node_off = frag_off = 0
    v_clu_off = e_clu_off = 0
    node_vecs, edge_indices, edge_lens, edge_vecs = [], [], [], []
    node_frags, frag_scenes, centroids, v_clusters, e_clusters = [], [], [], [], []

    for scene_idx, s in enumerate(scenes):
        node_vecs.append(s.node_vec)
        edge_indices.append(s.edge_index + node_off)
        edge_lens.append(s.edge_len)
        edge_vecs.append(s.edge_vec)
        node_frags.append(s.node_frag + frag_off)
        frag_scenes.append(torch.full((s.num_fragments,), scene_idx, dtype=torch.long))
        centroids.append(s.frag_centroid)

        v_clusters.append(_offset_clusters(s.vertex_cluster_id, v_clu_off))
        e_clusters.append(_offset_clusters(s.edge_cluster_id, e_clu_off))
        v_clu_off += _num_clusters(s.vertex_cluster_id)
        e_clu_off += _num_clusters(s.edge_cluster_id)

        node_off += s.num_nodes
        frag_off += s.num_fragments

    return SceneBatch(
        node_vec=torch.cat(node_vecs, 0),
        edge_index=torch.cat(edge_indices, 1),
        edge_len=torch.cat(edge_lens, 0),
        edge_vec=torch.cat(edge_vecs, 0),
        node_frag=torch.cat(node_frags, 0),
        frag_scene=torch.cat(frag_scenes, 0),
        frag_centroid=torch.cat(centroids, 0),
        vertex_cluster_id=torch.cat(v_clusters, 0),
        edge_cluster_id=torch.cat(e_clusters, 0),
        num_fragments=frag_off,
        num_scenes=len(scenes),
    )


def _num_clusters(cluster_id: torch.Tensor) -> int:
    if cluster_id.numel() == 0:
        return 0
    top = int(cluster_id.max().item())
    return top + 1 if top >= 0 else 0


def _offset_clusters(cluster_id: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0 or cluster_id.numel() == 0:
        return cluster_id
    shifted = cluster_id.clone()
    mask = shifted >= 0
    shifted[mask] += offset
    return shifted
