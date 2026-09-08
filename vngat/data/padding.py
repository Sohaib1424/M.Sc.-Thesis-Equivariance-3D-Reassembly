"""
Pad a `SceneBatch` to fixed shapes, for accelerators that require them.

WHY THIS EXISTS
---------------
XLA compiles a separate program for every distinct set of tensor shapes.
Breaking Bad scenes vary enormously -- 5k to 90k vertices, 2 to 94 fragments --
so an unpadded graph would trigger a fresh compilation almost every step, and
each compilation costs seconds to minutes. Padding into a small ladder of size
buckets bounds the number of compilations to the number of buckets.

HOW PADDING IS MADE HARMLESS
----------------------------
Rather than masking at every use site inside the layers, the padding is given
its own identity:

  * padded vertices belong to one extra PAD FRAGMENT,
  * that pad fragment belongs to one extra PAD SCENE,
  * padded edges have both endpoints on a single PAD VERTEX.

Every segment operation in this model is already scoped by fragment or by
scene, so real and padded data cannot mix -- the isolation falls out of the
existing indexing instead of needing new masks in the message passing.
Attention softmaxes normalise within a segment, so the pad segment normalises
among itself and is then discarded.

Masks ARE still needed in the loss, because a mean over a padded tensor divides
by the padded length. `masked_mean` handles that, and `tests/test_padding.py`
asserts the padded loss equals the unpadded loss to float64 precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .graph import SceneBatch

DEFAULT_NODE_BUCKETS = (8192, 16384, 32768, 65536, 131072)
DEFAULT_EDGE_BUCKETS = (24576, 49152, 98304, 196608, 393216)
DEFAULT_FRAG_BUCKETS = (16, 32, 64, 128)


@dataclass
class PaddedBatch:
    """A `SceneBatch` at fixed shapes, plus the masks the loss needs."""

    graph: SceneBatch
    node_mask: torch.Tensor    # (N_pad,) True for real vertices
    edge_mask: torch.Tensor    # (E_pad,)
    frag_mask: torch.Tensor    # (F_pad,)
    real_nodes: int
    real_edges: int
    real_fragments: int


def choose_bucket(value: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if value <= b:
            return b
    # Beyond the ladder, double rather than fail: one extra compilation beats a
    # crash mid-run.
    size = buckets[-1]
    while size < value:
        size *= 2
    return size


def pad_scene_batch(
    batch: SceneBatch,
    node_buckets: Sequence[int] = DEFAULT_NODE_BUCKETS,
    edge_buckets: Sequence[int] = DEFAULT_EDGE_BUCKETS,
    frag_buckets: Sequence[int] = DEFAULT_FRAG_BUCKETS,
) -> PaddedBatch:
    """
    Pad to the next bucket in each dimension.

    Layout after padding, with N/E/F the real counts:
        vertices  [0 .. N-1] real; index N is the PAD VERTEX
        fragments [0 .. F-1] real; index F is the PAD FRAGMENT
        scenes    [0 .. S-1] real; index S is the PAD SCENE
    """
    n, e, f = batch.num_nodes, batch.num_edges, batch.num_fragments
    s = batch.num_scenes
    device = batch.node_vec.device

    # +1 so the pad vertex / fragment / scene always have somewhere to live,
    # even when the real counts land exactly on a bucket boundary.
    n_pad = choose_bucket(n + 1, node_buckets)
    e_pad = choose_bucket(max(e, 1), edge_buckets)
    f_pad = choose_bucket(f + 1, frag_buckets)

    def grow(t: torch.Tensor, target: int, fill: float = 0.0) -> torch.Tensor:
        if t.shape[0] == target:
            return t
        shape = (target - t.shape[0], *t.shape[1:])
        return torch.cat([t, torch.full(shape, fill, dtype=t.dtype, device=device)], 0)

    pad_vertex, pad_frag, pad_scene = n, f, s

    node_vec = grow(batch.node_vec, n_pad)
    node_frag = grow(batch.node_frag, n_pad, fill=pad_frag)

    edge_index = batch.edge_index
    if e_pad > e:
        filler = torch.full((2, e_pad - e), pad_vertex, dtype=edge_index.dtype, device=device)
        edge_index = torch.cat([edge_index, filler], dim=1)
    edge_len = grow(batch.edge_len, e_pad)
    edge_vec = grow(batch.edge_vec, e_pad)

    frag_scene = grow(batch.frag_scene, f_pad, fill=pad_scene)
    frag_centroid = grow(batch.frag_centroid, f_pad)

    # -1 means "shared with nothing", so padded entries are already excluded
    # from the interface-embedding terms with no extra handling.
    vertex_cluster_id = grow(batch.vertex_cluster_id, n_pad, fill=-1)
    edge_cluster_id = grow(batch.edge_cluster_id, e_pad, fill=-1)

    padded = SceneBatch(
        node_vec=node_vec, edge_index=edge_index, edge_len=edge_len, edge_vec=edge_vec,
        node_frag=node_frag, frag_scene=frag_scene, frag_centroid=frag_centroid,
        vertex_cluster_id=vertex_cluster_id, edge_cluster_id=edge_cluster_id,
        num_fragments=f_pad, num_scenes=s + 1,
    )

    idx_n = torch.arange(n_pad, device=device)
    idx_e = torch.arange(e_pad, device=device)
    idx_f = torch.arange(f_pad, device=device)
    return PaddedBatch(
        graph=padded,
        node_mask=idx_n < n, edge_mask=idx_e < e, frag_mask=idx_f < f,
        real_nodes=n, real_edges=e, real_fragments=f,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over masked entries, computed without any host synchronisation.

    `values[mask].mean()` would need the mask's contents on the host to size the
    result, forcing an XLA sync every step and serialising the pipeline. The
    sum-and-divide form keeps the graph static and stays on device.
    """
    if mask is None:
        return values.mean() if values.numel() else values.sum() * 0.0
    m = mask.to(values.dtype)
    while m.dim() < values.dim():
        m = m.unsqueeze(-1)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def bucket_report(sizes: Sequence[Tuple[int, int, int]],
                  node_buckets: Sequence[int] = DEFAULT_NODE_BUCKETS,
                  edge_buckets: Sequence[int] = DEFAULT_EDGE_BUCKETS,
                  frag_buckets: Sequence[int] = DEFAULT_FRAG_BUCKETS) -> str:
    """How much compute the padding wastes, and how many shapes XLA will see."""
    combos, waste = set(), []
    for n, e, f in sizes:
        nb = choose_bucket(n + 1, node_buckets)
        eb = choose_bucket(max(e, 1), edge_buckets)
        fb = choose_bucket(f + 1, frag_buckets)
        combos.add((nb, eb, fb))
        waste.append(eb / max(e, 1))
    lines = [f"  distinct shape combinations: {len(combos)}  (= XLA compilations)"]
    if waste:
        w = sorted(waste)
        lines.append(f"  edge padding factor: median {w[len(w) // 2]:.2f}x, worst {w[-1]:.2f}x")
    return "\n".join(lines)
