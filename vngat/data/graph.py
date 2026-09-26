"""
Graph containers, per-scene normalisation and batching.

DESIGN NOTE -- undirected storage, symmetrised on device
--------------------------------------------------------
The dataset stores UNDIRECTED edges only (one entry per mesh edge, u < v) and
the model builds the reverse copies on the GPU in a single `cat`. Per-edge CPU
tensors, the pinned-memory copy and the host->device transfer all halve, and
`edge_cluster_id` lines up with the stored edges 1:1.

Each stored edge carries `edge_attr = [n1, n2, p_u - p_v]` (see
`vngat.data.features`). The reverse copy is `[n2, n1, p_v - p_u]`, built by
`vngat.models.vn_gat.VNGATModel.symmetrise_edges`.

PER-SCENE NORMALISATION
-----------------------
`merge_fragments` divides every POSITION-like quantity -- the centred vertex
positions and the edges' relative positions -- by one number per scene, the
radius of its largest fragment (`normalize_mode="scene"`, the default), so the
biggest fragment of every scene fits the unit ball. Normals are unit vectors
and are left alone.

Why one number per scene rather than one per fragment: two fracture surfaces
that mate are the same size in world units -- they were one surface before
the object broke -- and a single divisor keeps that. Why normalise at all: the
Breaking Bad objects vary in size by more than an order of magnitude, and a
network that sees raw coordinates has to learn the same geometry at every
scale separately.

What the division throws away -- how big the object really is -- is handed to
the network separately and invariantly: `frag_log_scale` is log(world radius)
per fragment, which the model's `VNScaleGate` turns into a positive
per-channel gain. `frag_unit` keeps the divisor itself, so anything that needs
world units back (Chamfer distance, part accuracy, the translation solver)
multiplies by it.

Rotation never interacts with any of this: a radius is rotation-invariant, so
`rotate_per_fragment` is still exact after normalisation.

No PyTorch Geometric
--------------------
Batching is plain `torch.cat` with explicit offsets. The message passing this
model needs is a scatter-softmax and an index_add, both core torch ops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence

import torch

NORMALIZE_MODES = ("scene", "fragment", "none")

# Edge channels, by meaning. Only the relative position is a length; the two
# normals are directions and are never scaled.
EDGE_N1, EDGE_N2, EDGE_DELTA = 0, 1, 2

# Smallest radius used for division and for the log-scale feature. A fragment
# that is a single point has radius 0; its coordinates are all 0 anyway, and
# log(0) would feed -inf into the scale gate.
_MIN_RADIUS = 1e-6


@dataclass
class FragmentGraph:
    """One fragment's mesh graph, centred at its own centroid, WORLD units."""

    node_vec: torch.Tensor          # (V, 2, 3)  [centred position, vertex normal]
    edge_index: torch.Tensor        # (2, E)     undirected, u < v, local indices
    edge_attr: torch.Tensor         # (E, 3, 3)  [n1, n2, p_u - p_v]
    centroid: torch.Tensor          # (3,)       centroid subtracted from positions
    radius: float                   # max distance of a vertex from the centroid
    vertex_cluster_id: torch.Tensor  # (V,)
    edge_cluster_id: torch.Tensor    # (E,)
    num_repaired: int = 0
    """Non-finite feature entries replaced with zero when this fragment was
    built -- see `vngat.data.features._sanitise`. Non-zero means the source
    mesh has degenerate geometry."""

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
    off the first, and the cross-fragment exchange must be scoped by the
    second or fragments of unrelated objects exchange information.
    """

    node_vec: torch.Tensor           # (N, 2, 3)  normalised position, vertex normal
    edge_index: torch.Tensor         # (2, E)     undirected
    edge_attr: torch.Tensor          # (E, 3, 3)  n1, n2, normalised p_u - p_v
    node_frag: torch.Tensor          # (N,)  global fragment id
    frag_scene: torch.Tensor         # (F,)  scene id per fragment
    frag_centroid: torch.Tensor      # (F, 3) centroid in the assembled frame, WORLD units
    frag_unit: torch.Tensor          # (F,)  world units per normalised unit (the divisor)
    frag_log_scale: torch.Tensor     # (F, 1) log(world radius), the invariant size feature
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

    def world_pos(self) -> torch.Tensor:
        """Centred positions back in world units, (N, 3)."""
        return self.node_vec[:, 0] * self.frag_unit.index_select(0, self.node_frag).unsqueeze(-1)

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
        extraction for the diffused view, and it is EXACT for a rigid
        transform:
          - centred position:      R (x - xbar)      [centroid co-moves]
          - vertex normal:         R n               [weights are invariant]
          - face normals n1, n2:   R n_f             [slot order is invariant --
                                                      the triple product has det R = 1]
          - relative position:     R (p_u - p_v)     [linear in positions]
          - scale, divisor:        unchanged         [radii are invariant]
        Topology (`edge_index`, orderings) depends only on vertex indices and is
        untouched by a rigid transform.
        """
        node_rot = rot[self.node_frag]                    # (N, 3, 3)
        edge_rot = rot[self.edge_frag]                    # (E, 3, 3)
        return SceneBatch(
            node_vec=torch.einsum("nij,ncj->nci", node_rot, self.node_vec),
            edge_index=self.edge_index,
            edge_attr=torch.einsum("eij,ecj->eci", edge_rot, self.edge_attr),
            node_frag=self.node_frag,
            frag_scene=self.frag_scene,
            frag_centroid=self.frag_centroid,
            frag_unit=self.frag_unit,
            frag_log_scale=self.frag_log_scale,
            vertex_cluster_id=self.vertex_cluster_id,
            edge_cluster_id=self.edge_cluster_id,
            num_fragments=self.num_fragments,
            num_scenes=self.num_scenes,
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def scene_divisors(radii: Sequence[float], mode: str = "scene") -> torch.Tensor:
    """
    The divisor for each fragment of ONE scene, in world units.

    "scene"    : the largest fragment radius, for every fragment (the default).
    "fragment" : each fragment's own radius -- GARF's convention; every
                 fragment lands on the unit sphere and relative size is lost
                 from the coordinates (the scale feature still carries it).
    "none"     : 1 -- raw world coordinates, as the previous version used.
    """
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize_mode must be one of {NORMALIZE_MODES}, got {mode!r}")
    r = torch.as_tensor([float(x) for x in radii], dtype=torch.float32)
    if r.numel() == 0 or mode == "none":
        return torch.ones_like(r)
    r = r.clamp_min(_MIN_RADIUS)
    if mode == "scene":
        return torch.full_like(r, float(r.max()))
    return r


def merge_fragments(
    fragments: List[FragmentGraph],
    normalize_mode: str = "scene",
    radii: Optional[Sequence[float]] = None,
) -> SceneBatch:
    """
    Disjoint-union one scene's fragments, offsetting vertex indices, and
    normalise positions per `normalize_mode`.

    `radii` overrides the fragments' own radii. It exists for the
    `input_source="frac"` path: the network is fed each fragment's fracture
    surface but scored on the full fragment, and the two must share ONE
    divisor and ONE scale feature or the prediction and the target would live
    in different units. The full fragments' radii are passed for both.

    Cluster ids are already scene-global (assigned once across all fragments by
    `compute_scene_correspondence`), so they are concatenated as-is.
    """
    if not fragments:
        raise ValueError("merge_fragments received an empty fragment list")
    radii = [f.radius for f in fragments] if radii is None else list(radii)
    if len(radii) != len(fragments):
        raise ValueError(f"{len(radii)} radii for {len(fragments)} fragments")
    unit = scene_divisors(radii, normalize_mode)                              # (F,)
    log_scale = torch.log(torch.as_tensor([max(float(r), _MIN_RADIUS) for r in radii],
                                          dtype=torch.float32)).unsqueeze(-1)  # (F, 1)

    node_offset = 0
    node_vecs, edge_indices, edge_attrs = [], [], []
    node_frags, centroids, v_clusters, e_clusters = [], [], [], []

    for i, frag in enumerate(fragments):
        n = frag.num_nodes
        inv = 1.0 / float(unit[i])
        node_vec = frag.node_vec.clone()
        node_vec[:, 0] *= inv
        edge_attr = frag.edge_attr.clone()
        edge_attr[:, EDGE_DELTA] *= inv
        node_vecs.append(node_vec)
        edge_indices.append(frag.edge_index + node_offset)
        edge_attrs.append(edge_attr)
        node_frags.append(torch.full((n,), i, dtype=torch.long))
        centroids.append(frag.centroid)
        v_clusters.append(frag.vertex_cluster_id)
        e_clusters.append(frag.edge_cluster_id)
        node_offset += n

    num_fragments = len(fragments)
    return SceneBatch(
        node_vec=torch.cat(node_vecs, 0),
        edge_index=torch.cat(edge_indices, 1),
        edge_attr=torch.cat(edge_attrs, 0),
        node_frag=torch.cat(node_frags, 0),
        frag_scene=torch.zeros(num_fragments, dtype=torch.long),
        frag_centroid=torch.stack(centroids, 0),
        frag_unit=unit,
        frag_log_scale=log_scale,
        vertex_cluster_id=torch.cat(v_clusters, 0),
        edge_cluster_id=torch.cat(e_clusters, 0),
        num_fragments=num_fragments,
        num_scenes=1,
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def collate_scenes(scenes: List[SceneBatch]) -> Optional[SceneBatch]:
    """
    Stitch several independent scenes into one batch.

    Offsets applied: vertex indices, fragment ids, scene ids and --
    independently -- the vertex-cluster and edge-cluster id spaces. Entries of
    -1 ("not shared with any other fragment") are left alone so they stay -1.
    """
    scenes = [s for s in scenes if s is not None]
    if not scenes:
        return None
    if len(scenes) == 1:
        return scenes[0]

    node_off = frag_off = scene_off = 0
    v_clu_off = e_clu_off = 0
    parts = {name: [] for name in ("node_vec", "edge_index", "edge_attr", "node_frag",
                                   "frag_scene", "frag_centroid", "frag_unit",
                                   "frag_log_scale", "vertex_cluster_id", "edge_cluster_id")}

    for s in scenes:
        parts["node_vec"].append(s.node_vec)
        parts["edge_index"].append(s.edge_index + node_off)
        parts["edge_attr"].append(s.edge_attr)
        parts["node_frag"].append(s.node_frag + frag_off)
        parts["frag_scene"].append(s.frag_scene + scene_off)
        parts["frag_centroid"].append(s.frag_centroid)
        parts["frag_unit"].append(s.frag_unit)
        parts["frag_log_scale"].append(s.frag_log_scale)
        parts["vertex_cluster_id"].append(_offset_clusters(s.vertex_cluster_id, v_clu_off))
        parts["edge_cluster_id"].append(_offset_clusters(s.edge_cluster_id, e_clu_off))
        v_clu_off += _num_clusters(s.vertex_cluster_id)
        e_clu_off += _num_clusters(s.edge_cluster_id)
        node_off += s.num_nodes
        frag_off += s.num_fragments
        scene_off += s.num_scenes

    return SceneBatch(
        node_vec=torch.cat(parts["node_vec"], 0),
        edge_index=torch.cat(parts["edge_index"], 1),
        edge_attr=torch.cat(parts["edge_attr"], 0),
        node_frag=torch.cat(parts["node_frag"], 0),
        frag_scene=torch.cat(parts["frag_scene"], 0),
        frag_centroid=torch.cat(parts["frag_centroid"], 0),
        frag_unit=torch.cat(parts["frag_unit"], 0),
        frag_log_scale=torch.cat(parts["frag_log_scale"], 0),
        vertex_cluster_id=torch.cat(parts["vertex_cluster_id"], 0),
        edge_cluster_id=torch.cat(parts["edge_cluster_id"], 0),
        num_fragments=frag_off,
        num_scenes=scene_off,
    )


def split_scenes(g: SceneBatch) -> List[SceneBatch]:
    """
    The inverse of `collate_scenes`: one single-scene `SceneBatch` per scene.

    Relies on collation being a plain concatenation in scene order, which makes
    every scene's nodes, edges and fragments a contiguous run -- counted here
    from the index tensors rather than assumed from sizes.
    """
    num_scenes = g.num_scenes
    device = g.node_frag.device
    frags_per = torch.bincount(g.frag_scene, minlength=num_scenes).tolist()
    node_scene = g.frag_scene.index_select(0, g.node_frag)
    nodes_per = torch.bincount(node_scene, minlength=num_scenes).tolist()
    edge_scene = node_scene.index_select(0, g.edge_index[0]) if g.num_edges else node_scene[:0]
    edges_per = torch.bincount(edge_scene, minlength=num_scenes).tolist()

    out: List[SceneBatch] = []
    n0 = e0 = f0 = 0
    for s in range(num_scenes):
        nn_, ne, nf = nodes_per[s], edges_per[s], frags_per[s]
        out.append(SceneBatch(
            node_vec=g.node_vec[n0:n0 + nn_],
            edge_index=g.edge_index[:, e0:e0 + ne] - n0,
            edge_attr=g.edge_attr[e0:e0 + ne],
            node_frag=g.node_frag[n0:n0 + nn_] - f0,
            frag_scene=torch.zeros(nf, dtype=torch.long, device=device),
            frag_centroid=g.frag_centroid[f0:f0 + nf],
            frag_unit=g.frag_unit[f0:f0 + nf],
            frag_log_scale=g.frag_log_scale[f0:f0 + nf],
            vertex_cluster_id=g.vertex_cluster_id[n0:n0 + nn_],
            edge_cluster_id=g.edge_cluster_id[e0:e0 + ne],
            num_fragments=nf,
            num_scenes=1,
        ))
        n0 += nn_
        e0 += ne
        f0 += nf
    return out


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


def log_radius(radius: float) -> float:
    """The scale feature for one radius, exactly as `merge_fragments` computes it."""
    return math.log(max(float(radius), _MIN_RADIUS))
