"""
OPTIONAL angular / triplet features (DimeNet-style), built from ``inc_index``.

``inc_index`` has been correctly computed and offset through the entire
pipeline since the beginning but never read by any model -- this is what it was
for. For each pair of mesh edges meeting at a common vertex, the angle between
them is a rotation-invariant quantity that pure pairwise message passing cannot
see, and fracture surfaces are exactly the kind of geometry where local angular
structure is discriminative.

STATUS AND HONEST CAVEAT
------------------------
This is OFF BY DEFAULT (``--angular``). It is validated for correctness
(invariance, shapes, and that the triplet enumeration matches a brute-force
reference on small graphs) but it has NOT been shown to improve accuracy on
real data -- no such experiment has been run yet. Treat it as an ablation arm
to measure, not as part of the validated baseline.

MEMORY DISCIPLINE
-----------------
Naive enumeration gives ``sum_v deg(v)^2`` triplets. Mesh degrees are usually
~6, so ~36 per vertex, but a single high-valence vertex is quadratic on its
own and Breaking Bad fracture surfaces do contain them. ``max_triplets_per_node``
caps the fan-out with a deterministic stride, keeping the triplet count linear
in N with a known constant.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .vn_layers import VNLinear


def build_triplets(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_triplets_per_node: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate ordered edge pairs ``(e1, e2)`` sharing a target vertex.

    Uses ``edge_index[1]`` (the destination of each directed edge) as the
    common vertex, so a triplet is "two edges arriving at the same vertex".

    Returns ``(edge_a, edge_b, center_node)``, all 1-D and the same length.
    Deterministic: edges are grouped by a stable sort, and the per-node cap is
    applied by a fixed stride rather than random sampling, so two runs on the
    same graph produce identical triplets.
    """
    device = edge_index.device
    dst = edge_index[1]
    E = dst.numel()
    if E == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    order = torch.argsort(dst, stable=True)
    dst_sorted = dst[order]
    counts = torch.bincount(dst_sorted, minlength=num_nodes)
    capped = counts.clamp(max=max_triplets_per_node)

    starts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    starts[1:] = torch.cumsum(counts, 0)[:-1]

    # Take the first `capped[v]` incident edges of each vertex v.
    total_kept = int(capped.sum())
    if total_kept == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    kept_starts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    kept_starts[1:] = torch.cumsum(capped, 0)[:-1]

    node_of_kept = torch.repeat_interleave(torch.arange(num_nodes, device=device), capped)
    # Offset within each node's own list, computed without a Python loop over
    # nodes (which would be N interpreter iterations per layer, per batch).
    keep_offsets = torch.arange(total_kept, device=device) - kept_starts[node_of_kept]
    kept_positions = starts[node_of_kept] + keep_offsets
    kept_edges = order[kept_positions]

    # All ordered pairs within each vertex's kept edge list.
    pair_counts = capped * capped
    center = torch.repeat_interleave(torch.arange(num_nodes, device=device), pair_counts)
    if center.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    pair_starts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    pair_starts[1:] = torch.cumsum(pair_counts, 0)[:-1]
    within = torch.arange(center.numel(), device=device) - pair_starts[center]

    deg = capped[center]
    idx_a = kept_starts[center] + within // deg
    idx_b = kept_starts[center] + within % deg

    # Unordered pairs only. Both `cos` and `va + vb` downstream are symmetric
    # in (a, b), so enumerating (a, b) AND (b, a) computes every message twice
    # -- exactly double the memory for an identical result.
    upper = idx_a < idx_b
    return kept_edges[idx_a[upper]], kept_edges[idx_b[upper]], center[upper]


class AngularBlock(nn.Module):
    """Fold pairwise-edge angular information into node features.

    For a triplet ``(e1, e2)`` at vertex v, the invariant descriptor is the
    cosine between the two edges' projected direction channels. Those cosines
    are passed through a small MLP into a per-triplet invariant gate, which
    scales an equivariant message and scatters it onto v. Invariant gate times
    equivariant vector, summed -- equivariant, by the same argument as the
    attention layers.
    """

    def __init__(self, channels: int, edge_vec_channels: int = 3,
                 hidden: int = 32, max_triplets_per_node: int = 8):
        super().__init__()
        self.max_triplets_per_node = max_triplets_per_node
        self.proj = VNLinear(edge_vec_channels, channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels)
        )
        self.out = VNLinear(channels, channels)

    def forward(self, x, edge_index, edge_vec):
        N = x.size(0)
        edge_a, edge_b, center = build_triplets(
            edge_index, N, self.max_triplets_per_node
        )
        if edge_a.numel() == 0:
            return x

        proj = self.proj(edge_vec)                                  # (E, C, 3)
        va, vb = proj[edge_a], proj[edge_b]                         # (T, C, 3)

        cos = (va * vb).sum(-1) / (
            va.norm(dim=-1).clamp_min(1e-6) * vb.norm(dim=-1).clamp_min(1e-6)
        )                                                            # (T, C) invariant
        gate = torch.tanh(self.gate(cos))                            # (T, C) invariant

        msg = gate.unsqueeze(-1) * (va + vb)                         # equivariant
        agg = proj.new_zeros(N, msg.shape[1], 3)
        agg.index_add_(0, center, msg)
        return x + self.out(agg)
