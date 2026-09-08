"""
Graph containers and batching, NUMPY-backed.

The containers hold numpy arrays rather than framework tensors, and conversion
to `tf.Tensor` happens once at the device boundary in `training/bridge.py`.
That keeps the whole data pipeline framework-agnostic -- it is shared verbatim
with the PyTorch branch apart from this file's dtype -- and avoids constructing
graph tensors inside DataLoader worker processes, which TF does not like.

UNDIRECTED EDGE STORAGE, symmetrised on device. Storing each mesh edge once and
symmetrising in the model makes the "which copy is the forward one" bug class
impossible by construction, and halves every per-edge array and transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional

import numpy as np


@dataclass
class FragmentGraph:
    """One fragment's mesh graph, in its own centralised frame."""

    node_vec: np.ndarray            # (V, 2, 3) [centralised position, vertex normal]
    edge_index: np.ndarray          # (2, E) undirected, local vertex indices
    edge_len: np.ndarray            # (E, 1) invariant scalar
    edge_vec: np.ndarray            # (E, 3, 3) [midpoint, face normal 1, face normal 2]
    centroid: np.ndarray            # (3,)
    vertex_cluster_id: np.ndarray   # (V,)
    edge_cluster_id: np.ndarray     # (E,)
    num_repaired: int = 0
    """Non-finite feature entries replaced with zero when this fragment was
    built. Non-zero means the source mesh has degenerate geometry."""

    @property
    def num_nodes(self) -> int:
        return int(self.node_vec.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class SceneBatch:
    """One or more scenes merged into a single disjoint-union graph."""

    node_vec: np.ndarray
    edge_index: np.ndarray
    edge_len: np.ndarray
    edge_vec: np.ndarray
    node_frag: np.ndarray            # (N,) GLOBAL fragment id
    frag_scene: np.ndarray           # (F,) scene id per fragment
    frag_centroid: np.ndarray        # (F, 3)
    vertex_cluster_id: np.ndarray
    edge_cluster_id: np.ndarray
    num_fragments: int
    num_scenes: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_vec.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def edge_frag(self) -> np.ndarray:
        """Fragment id per edge. Mesh edges never cross a fragment boundary, so
        either endpoint gives the same answer."""
        return self.node_frag[self.edge_index[0]]

    def rotate_per_fragment(self, rot: np.ndarray) -> "SceneBatch":
        """
        Apply a per-fragment rotation to every geometric quantity.

        `rot`: (F, 3, 3) acting on column vectors. Rows are stored as row
        vectors, so the operation is `v @ R^T`.

        EXACT, not an approximation, which is why the diffused view is derived
        this way instead of re-running mesh transformation and feature
        extraction: centralised position and vertex normal rotate, edge length
        is invariant (rotations are isometries), midpoint is linear in
        positions, adjacent face normals rotate, and topology depends only on
        vertex indices. That removes one full `get_features` pass -- the single
        most expensive CPU step -- from every sample.
        """
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        kwargs["node_vec"] = np.einsum('nij,ncj->nci', rot[self.node_frag], self.node_vec)
        kwargs["edge_vec"] = np.einsum('eij,ecj->eci', rot[self.edge_frag], self.edge_vec)
        return SceneBatch(**kwargs)


def merge_fragments(fragments: List[FragmentGraph]) -> SceneBatch:
    """Disjoint-union one scene's fragments, offsetting vertex indices. Cluster
    ids are already scene-global, so they concatenate as-is."""
    if not fragments:
        raise ValueError("merge_fragments received an empty fragment list")

    offset = 0
    nv, ei, el, ev, nf, cen, vc, ec = [], [], [], [], [], [], [], []
    for i, frag in enumerate(fragments):
        nv.append(frag.node_vec)
        ei.append(frag.edge_index + offset)
        el.append(frag.edge_len)
        ev.append(frag.edge_vec)
        nf.append(np.full(frag.num_nodes, i, np.int32))
        cen.append(frag.centroid)
        vc.append(frag.vertex_cluster_id)
        ec.append(frag.edge_cluster_id)
        offset += frag.num_nodes

    return SceneBatch(
        node_vec=np.concatenate(nv, 0), edge_index=np.concatenate(ei, 1),
        edge_len=np.concatenate(el, 0), edge_vec=np.concatenate(ev, 0),
        node_frag=np.concatenate(nf, 0), frag_scene=np.zeros(len(fragments), np.int32),
        frag_centroid=np.stack(cen, 0), vertex_cluster_id=np.concatenate(vc, 0),
        edge_cluster_id=np.concatenate(ec, 0),
        num_fragments=len(fragments), num_scenes=1)


def _num_clusters(cluster_id: np.ndarray) -> int:
    if cluster_id.size == 0:
        return 0
    top = int(cluster_id.max())
    return top + 1 if top >= 0 else 0


def _offset_clusters(cluster_id: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0 or cluster_id.size == 0:
        return cluster_id
    shifted = cluster_id.copy()
    shifted[shifted >= 0] += offset
    return shifted


def collate_scenes(scenes: List[SceneBatch]) -> Optional[SceneBatch]:
    """
    Stitch several independent scenes into one training batch.

    Offsets vertex indices, fragment ids, and -- independently -- the
    vertex-cluster and edge-cluster id spaces. Entries of -1 ("not shared") are
    left alone so they stay -1 after batching.
    """
    scenes = [s for s in scenes if s is not None]
    if not scenes:
        return None
    if len(scenes) == 1 and scenes[0].num_scenes == 1:
        return scenes[0]

    node_off = frag_off = v_off = e_off = 0
    nv, ei, el, ev, nf, fs, cen, vc, ec = [], [], [], [], [], [], [], [], []
    for idx, s in enumerate(scenes):
        nv.append(s.node_vec); ei.append(s.edge_index + node_off)
        el.append(s.edge_len); ev.append(s.edge_vec)
        nf.append(s.node_frag + frag_off)
        fs.append(np.full(s.num_fragments, idx, np.int32))
        cen.append(s.frag_centroid)
        vc.append(_offset_clusters(s.vertex_cluster_id, v_off))
        ec.append(_offset_clusters(s.edge_cluster_id, e_off))
        v_off += _num_clusters(s.vertex_cluster_id)
        e_off += _num_clusters(s.edge_cluster_id)
        node_off += s.num_nodes
        frag_off += s.num_fragments

    return SceneBatch(
        node_vec=np.concatenate(nv, 0), edge_index=np.concatenate(ei, 1),
        edge_len=np.concatenate(el, 0), edge_vec=np.concatenate(ev, 0),
        node_frag=np.concatenate(nf, 0), frag_scene=np.concatenate(fs, 0),
        frag_centroid=np.concatenate(cen, 0),
        vertex_cluster_id=np.concatenate(vc, 0), edge_cluster_id=np.concatenate(ec, 0),
        num_fragments=frag_off, num_scenes=len(scenes))
