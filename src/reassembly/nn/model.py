"""
The backbone: intra-fragment attention, cross-fragment attention, rotation head.

Shape of the network
--------------------
Layers alternate. Intra-fragment attention moves information along real mesh
edges; cross-fragment attention lets a fragment's interface tokens hear what
the other fragments in its scene look like; then more intra-fragment layers
carry that back through the mesh. The last part is easy to leave out and
important: a cross layer only updates the *token* vertices, so without intra
layers after it, the rest of the fragment never learns anything about its
neighbours, and the pooled rotation prediction is dominated by vertices that
never heard from anyone.

The default schedule is ``intra, intra, cross, intra, cross, intra`` -- four
intra-fragment layers, matching the layer count carried over from the previous
design, with two cross layers interleaved so each is followed by propagation.

The rotation convention, derived rather than guessed
----------------------------------------------------
Centred, the perturbation is ``v_pert = v_gt Q^T``. Vector features inherit
that rotation, so the head's frame satisfies ``M_pert = Q M_gt``. The label is
the rotation that undoes the perturbation, ``R_gt = Q^T``, so the head returns

    R_pred = M^T                    (M's rows, not its columns)

which makes ``R_pred = M_gt^T Q^T`` -- correct exactly when the network learns
to map an already-assembled fragment to the identity frame, ``M_gt = I``. That
is a fixed, learnable target, because Breaking Bad's assembled pose is a
convention shared by every scene in the dataset.

Applying it: ``v_pert @ R_pred.T`` returns the fragment to its assembled pose.
The transpose here is the single easiest thing to get backwards, and getting it
backwards is invisible at initialisation -- chance is chance either way -- so
``tests/test_model.py`` asserts the round trip on a fragment whose perturbation
is known.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import torch
from torch import Tensor, nn

from .cross import VNCrossFragmentAttention
from .gat import VNGraphAttentionBlock
from .segment import segment_mean
from .vn import VNInvariant, VNLinear, VNScaleGate, gram_schmidt

DEFAULT_SCHEDULE = ("intra", "intra", "cross", "intra", "cross", "intra")


class Prediction(NamedTuple):
    """What one forward pass produces."""
    rotation: Tensor            # (F, 3, 3) maps perturbed -> assembled
    frame: Tensor               # (F, 3, 3) the equivariant frame, rotation's transpose
    vertex_embedding: Tensor    # (N, D) invariant, for correspondence and the
    #                             embedding-consistency loss
    vertex_features: Tensor     # (N, C, 3) equivariant, the backbone's output


class ReassemblyNet(nn.Module):
    """
    SO(3)-equivariant rotation prediction for fractured fragments.

    Every fragment is already a graph -- its mesh -- so nothing is constructed
    inside one. The only built connections are between fragments, among the
    sampled fracture-surface tokens.
    """

    def __init__(
        self,
        channels: int = 32,
        node_channels: int = 2,
        edge_channels: int = 3,
        heads: int = 4,
        head_dim: int = 8,
        embedding_dim: int = 32,
        schedule: Sequence[str] = DEFAULT_SCHEDULE,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.schedule = tuple(schedule)
        unknown = set(self.schedule) - {"intra", "cross"}
        if unknown:
            raise ValueError(f"unknown layer kinds in schedule: {sorted(unknown)}")

        self.embed = VNLinear(node_channels, channels)
        # Fragment scale joins as an invariant gate, never as a third axis of a
        # (C, 3) tensor -- scale has no direction, so there is no legal slot.
        self.scale_gate = VNScaleGate(channels, scalar_features=1)

        self.intra = nn.ModuleList(
            VNGraphAttentionBlock(channels, edge_channels, heads, head_dim, negative_slope)
            for kind in self.schedule if kind == "intra"
        )
        self.cross = nn.ModuleList(
            VNCrossFragmentAttention(channels, heads=heads, head_dim=2 * head_dim)
            for kind in self.schedule if kind == "cross"
        )

        # Pool to a fragment, then two equivariant 3-vectors -> a rotation.
        self.pool_proj = VNLinear(channels, channels)
        self.head = VNLinear(channels, 2)
        # Invariant per-vertex embedding for matching interface points across
        # fragments. Invariant, not equivariant: the vertices being matched sit
        # in differently-perturbed fragments, so equivariant features would be
        # compared across unrelated frames.
        self.readout = VNInvariant(channels, directions=4)
        self.embedding = nn.Sequential(
            nn.Linear(self.readout.out_features, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.GELU(),
            # No bias on the last layer, and this is provable rather than
            # stylistic. The embedding is supervised only by the
            # centroid-variance loss, and a constant added to every embedding
            # shifts them all equally -- the scatter about each cluster centroid
            # does not move. It is unidentifiable for stage two as well, since a
            # global shift leaves every pairwise distance untouched. Keeping it
            # would put a parameter in the model that provably cannot be
            # learned, which then shows up forever as a "dead gradient".
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        vertex_fragment: Tensor,
        num_fragments: int,
        *,
        log_scale: Optional[Tensor] = None,
        token_index: Optional[Tensor] = None,
        token_query: Optional[Tensor] = None,
        token_key: Optional[Tensor] = None,
    ) -> Prediction:
        """
        ``node_features`` ``(N, node_channels, 3)`` -- centred coordinate and
        vertex normal. ``edge_attr`` ``(E, edge_channels, 3)`` -- the two
        canonically ordered face normals and the relative position of the
        source vertex. ``token_index`` selects which vertices
        take part in cross-fragment attention; ``token_query`` and ``token_key``
        are the pair lists from :func:`~reassembly.nn.cross.cross_fragment_index`,
        built once per batch.
        """
        x = self.embed(node_features)
        if log_scale is not None:
            x = self.scale_gate(x, log_scale[vertex_fragment])

        intra = iter(self.intra)
        cross = iter(self.cross)
        for kind in self.schedule:
            if kind == "intra":
                x = next(intra)(x, edge_index, edge_attr)
            else:
                layer = next(cross)
                if token_index is None or token_index.numel() == 0:
                    # No tokens anywhere in the batch. The layer is a no-op
                    # rather than an error: a single-fragment mode has no
                    # cross-fragment structure to attend over.
                    continue
                tokens = layer(x[token_index], token_query, token_key)
                # Residual write-back, so vertices that are not tokens keep
                # their features and token vertices keep theirs too.
                x = x.index_add(0, token_index, tokens - x[token_index])

        pooled = segment_mean(self.pool_proj(x), vertex_fragment, num_fragments)
        frame = gram_schmidt(self.head(pooled))            # (F, 3, 3), columns
        return Prediction(
            rotation=frame.transpose(-1, -2),
            frame=frame,
            vertex_embedding=self.embedding(self.readout(x)),
            vertex_features=x,
        )


def apply_rotation(vertices: Tensor, rotation: Tensor,
                   vertex_fragment: Tensor) -> Tensor:
    """
    ``v @ R.T`` per fragment -- the operation the position and normal losses
    compare against the ground truth.
    """
    return torch.einsum("nij,nj->ni", rotation[vertex_fragment], vertices)
