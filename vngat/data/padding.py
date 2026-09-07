"""
Pad a `SceneBatch` to fixed shapes, for accelerators that require them.

WHY THIS EXISTS
---------------
XLA (TPU) compiles a separate program for every distinct set of tensor shapes.
Breaking Bad scenes have wildly varying vertex, edge and fragment counts -- 5k
to 90k nodes, 2 to 94 fragments -- so an unpadded graph would trigger a fresh
compilation almost every step, and compilation costs seconds to minutes. Padding
into a handful of size buckets bounds the number of compilations to the number
of buckets.

HOW PADDING IS MADE HARMLESS
----------------------------
Rather than masking at every use site, the padding is given its own identity:

  * padded vertices belong to one extra PAD FRAGMENT,
  * that pad fragment belongs to one extra PAD SCENE,
  * padded edges have both endpoints on a single PAD VERTEX.

Every segment operation in this model is already scoped by fragment or by scene,
so real fragments and the pad fragment cannot mix -- the isolation falls out of
the existing indexing rather than needing new masks inside the layers. Attention
softmaxes normalise within a segment, so the pad segment normalises among
itself and is then discarded.

The one place masks ARE still needed is the loss, because a mean over a padded
tensor divides by the padded length. `masked_mean` handles that, and
`tests/test_padding.py` asserts the padded loss equals the unpadded loss to
float64 precision.

BUCKETS
-------
Padding to the global maximum would waste ~8x on a median scene. A small
ladder of buckets keeps waste bounded while keeping recompilations few.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
    # Beyond the ladder, round up to the next power of two rather than failing:
    # one extra compilation is far better than a crash mid-run.
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
        vertices  [0 .. N-1] real, [N .. N_pad-1] padding
        the vertex at index N is the PAD VERTEX every padded edge points at
        fragments [0 .. F-1] real, index F is the PAD FRAGMENT
        scenes    [0 .. S-1] real, index S is the PAD SCENE
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

    pad_vertex = n            # first padded slot, reused as the sink
    pad_frag = f
    pad_scene = s

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
    result, which forces an XLA sync every step and destroys throughput. The
    sum-and-divide form keeps the graph static and stays on device.
    """
    if mask is None:
        return values.mean() if values.numel() else values.sum() * 0.0
    m = mask.to(values.dtype)
    while m.dim() < values.dim():
        m = m.unsqueeze(-1)
    total = (values * m).sum()
    count = m.sum().clamp_min(1.0)
    return total / count


def bucket_report(sizes: Sequence[Tuple[int, int, int]],
                  node_buckets: Sequence[int] = DEFAULT_NODE_BUCKETS,
                  edge_buckets: Sequence[int] = DEFAULT_EDGE_BUCKETS,
                  frag_buckets: Sequence[int] = DEFAULT_FRAG_BUCKETS) -> str:
    """How much compute the padding wastes, and how many shapes XLA will see."""
    lines, combos, waste = [], set(), []
    for n, e, f in sizes:
        nb = choose_bucket(n + 1, node_buckets)
        eb = choose_bucket(max(e, 1), edge_buckets)
        fb = choose_bucket(f + 1, frag_buckets)
        combos.add((nb, eb, fb))
        waste.append(eb / max(e, 1))
    lines.append(f"  distinct shape combinations: {len(combos)}  (= XLA compilations)")
    if waste:
        w = sorted(waste)
        lines.append(f"  edge padding factor: median {w[len(w)//2]:.2f}x, worst {w[-1]:.2f}x")
    return "\n".join(lines)
