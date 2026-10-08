"""
Vector-Neuron cross-fragment attention.

This is a Vector Neuron layer throughout: it takes ``(T, C, 3)`` vector
features, returns ``(T, C, 3)`` vector features, and every operation on the
spatial axis is equivariant. What is constrained is not whether vectors are
involved -- they are, on both sides -- but what a message is allowed to depend
on.

The constraint, and why it is about the task and not the architecture
---------------------------------------------------------------------
Take one scene and perturb it twice, changing only fragment *j*'s rotation. The
correct answer for fragment *i* is identical in both cases: its label is the
rotation that undoes *its own* perturbation, and fragment *j* landing at a
different angle does not change it. So a perfect model's output for *i* is a
function that ignores *j*'s orientation. That is forced by the task, before any
architecture is chosen.

An architecture that can see *j*'s orientation would therefore have to *learn*
to ignore it -- from data, by augmentation -- which is precisely the thing this
thesis argues equivariance should make unnecessary. Worse, *j*'s orientation is
not weak signal, it is pure noise injected by the perturbation, feeding straight
into *i*'s prediction. So the requirement is made structural::

    H_i(P_1, ..., P_i A, ..., P_N) = H_i(...) · A          equivariant to its own pose
    H_i(P_1, ..., P_j A, ..., P_N) = H_i(...)   (j != i)   invariant to the others'

Wu et al., *Leveraging SE(3) Equivariance for Learning 3D Geometric Shape
Assembly*, equation 7. Their correlation module is the construction that
satisfies it: ``C_ij = G_j · F_i``, where ``G_j`` is an **invariant matrix**
describing fragment *j* and ``F_i`` is fragment *i*'s **own equivariant**
features. A matrix, not a scalar -- the sender gets to re-mix the receiver's
channels arbitrarily. What it cannot do is contribute a direction of its own,
because a direction would carry its orientation.

What crosses the gap is *shape*. What does not cross is *pose*. Nothing is lost
by that: the relative pose of two arbitrarily tumbled fragments is the
perturbation, not the object.

The layer
---------
Attention with a matrix-valued value, which is the equivariant analogue of a
vector-valued one::

    logits_ij = <q(inv_i), k(inv_j)> / sqrt(d)      invariant to both poses
    alpha     = softmax over j within the scene, excluding j = i's fragment
    out_i     = x_i + ( sum_j alpha_ij G_j ) x_i

``G_j`` is a per-head ``(C/H, C/H)`` channel-mixing matrix read off fragment
*j*'s invariants. Note where the sum sits: because ``G`` is a *linear* function
of those invariants,

    sum_j alpha_ij G(v_j)  =  G( sum_j alpha_ij v_j )

so the aggregation happens on the small invariant vectors and the matrix is
built once per query. That is not a cosmetic rearrangement -- materialising a
matrix per *pair* would be ``(P, H, C/H, C/H)``, gigabytes at the two-million
pairs a large scene reaches, against ``(T, H, C/H, C/H)``, a couple of
megabytes.

Why the scores cannot be vector inner products
----------------------------------------------
The natural VN-attention score is ``<W_q x_i, W_k x_j>``. It is not usable here:
under independent perturbations that inner product becomes
``<A_i a, A_j b>``, which depends on the *relative* rotation ``A_i^T A_j`` --
exactly the quantity that is noise. Scores are therefore inner products of each
token's invariant description. This is forced by the same argument as
everything else, not a convenience.

Which pairs exist
-----------------
Tokens attend across fragments *within one scene*, never inside their own
fragment (the mesh graph already covers that) and never across scenes. The
cross-scene case is not hypothetical: a batch concatenates unrelated objects
and fragment ids are globally unique, so an unmasked layer silently lets a
teapot inform a statue and makes the prediction depend on batch composition.
That bug is in this project's history; :func:`cross_fragment_index` is what
prevents it, and it is built once per batch in the collate rather than per
layer.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from .segment import segment_softmax, segment_sum
from .vn import VNInvariant, VNLayerNorm, VNLinear


def cross_fragment_index(
    token_fragment: Tensor,
    token_scene: Tensor,
    num_scenes: int,
) -> Tuple[Tensor, Tensor]:
    """
    Every ordered token pair in the same scene but different fragments.

    Returns ``(query, key)``, both ``(P,)`` index tensors into the token array.
    Tokens must already be sorted by scene -- the collate sorts by
    ``(scene, fragment)``, which also makes each fragment's tokens contiguous.

    The pair count is exactly what ``patches.cross_fragment_pairs`` predicts and
    what the token budget is chosen to bound, so this is not a hidden cost: it
    is *the* cost, made explicit.
    """
    device = token_fragment.device
    total = token_fragment.numel()
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    counts = torch.bincount(token_scene, minlength=num_scenes)
    scene_start = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])[:-1]

    block = counts[token_scene]                              # scene size per token
    query = torch.repeat_interleave(torch.arange(total, device=device), block)
    # Position within each token's own repeat block, 0 .. block-1.
    block_start = torch.cat([block.new_zeros(1), torch.cumsum(block, 0)])[:-1]
    local = torch.arange(int(block.sum()), device=device) - torch.repeat_interleave(
        block_start, block
    )
    key = torch.repeat_interleave(scene_start[token_scene], block) + local

    keep = token_fragment[query] != token_fragment[key]
    return query[keep], key[keep]


def _pair_logits(query, key, query_index, key_index, heads, head_dim, scale):
    """Attention logits for every pair, without retaining the gathers."""
    q = query[query_index].view(-1, heads, head_dim)
    k = key[key_index].view(-1, heads, head_dim)
    return torch.sum(q * k, dim=-1) * scale


def _pair_pool(value, alpha, query_index, key_index, heads, head_dim, n):
    """Attention-weighted sum of the invariant values, per query token."""
    v = value[key_index].view(-1, heads, head_dim)
    return segment_sum(v * alpha.unsqueeze(-1), query_index, n)


class VNCrossFragmentAttention(nn.Module):
    """
    One layer of Vector-Neuron cross-fragment attention.

    Vector features in, vector features out, with a matrix-valued message: what
    fragment *j* sends fragment *i* is a linear operator on *i*'s channel space,
    which is the most a sender can contribute without contributing a direction.
    See the module docstring for why that ceiling exists.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        head_dim: int = 16,
        directions: int = 4,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.per_head = channels // heads

        self.invariant = VNInvariant(channels, directions=directions)
        d = self.invariant.out_features
        self.query = nn.Linear(d, heads * head_dim)
        self.key = nn.Linear(d, heads * head_dim)
        self.value = nn.Linear(d, heads * head_dim)

        # The sender's channel-mixing matrix, per head. Linear in the attended
        # context, which is what lets the sum over pairs happen before the
        # reshape rather than after -- see the module docstring.
        self.mix = nn.Sequential(
            nn.Linear(heads * head_dim + d, 2 * heads * head_dim),
            nn.LayerNorm(2 * heads * head_dim),
            nn.GELU(),
            nn.Linear(2 * heads * head_dim, heads * self.per_head * self.per_head),
        )
        self.proj = VNLinear(channels, channels)
        self.norm = VNLayerNorm(channels)
        self.scale = 1.0 / math.sqrt(head_dim)
        self.checkpoint = bool(checkpoint)

        # Start near the identity: the mixing matrix begins near zero, so an
        # untrained cross layer passes the intra-fragment representation through
        # rather than injecting noise into a backbone already doing something.
        #
        # *Near* zero, not zero. A zero final weight makes the output constant,
        # which zeroes the gradient into everything upstream -- the attention
        # projections and the invariant readout, i.e. the whole cross-fragment
        # pathway, dead for the first steps and waking only as a bias drifts.
        nn.init.normal_(self.mix[-1].weight, std=1e-2 / math.sqrt(2 * heads * head_dim))
        nn.init.zeros_(self.mix[-1].bias)

    def _maybe_checkpoint(self, function, *args):
        if self.checkpoint and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(function, *args,
                                                     use_reentrant=False)
        return function(*args)

    def forward(
        self,
        x: Tensor,
        query_index: Tensor,
        key_index: Tensor,
    ) -> Tensor:
        """
        ``x`` ``(T, C, 3)`` token features, plus the pair lists from
        :func:`cross_fragment_index`. Returns ``(T, C, 3)``.
        """
        n = x.shape[0]
        scalars = self.invariant(x)                                  # (T, d)

        if query_index.numel() == 0:
            # A scene of one fragment, or a batch where every fragment's tokens
            # are alone. No pairs, no context -- the mixing matrix sees zeros
            # and the layer is the identity, rather than a division by an empty
            # sum. Both cases are real: coincidence labelling reports fragments
            # with no fracture surface at all.
            context = scalars.new_zeros(n, self.heads * self.head_dim)
        else:
            # The pair dimension is this layer's whole memory cost, and it is
            # large: 2048 tokens over 6 fragments is 3.5 million pairs. Holding
            # q, k and v gathered to that dimension retains 1109 bytes per pair
            # -- 3.6 GB for a single scene, 7.2 GB across two cross layers, on
            # a 16 GB card. Measured with saved_tensors_hooks as the slope
            # against pair count, not estimated.
            #
            # Recomputing those gathers in the backward pass rather than storing
            # them leaves 86 bytes per pair, 0.3 GB, for one extra forward of a
            # cheap indexing op. The projections are (T, heads*head_dim) and
            # stay resident either way.
            logits = self._maybe_checkpoint(
                _pair_logits, self.query(scalars), self.key(scalars),
                query_index, key_index, self.heads, self.head_dim, self.scale,
            )
            alpha = segment_softmax(logits, query_index, n)          # (P, heads)
            pooled = self._maybe_checkpoint(
                _pair_pool, self.value(scalars), alpha,
                query_index, key_index, self.heads, self.head_dim, n,
            )
            context = pooled.reshape(n, self.heads * self.head_dim)

        # G: one (per_head, per_head) mixing matrix per head per query token.
        g = self.mix(torch.cat([context, scalars], dim=-1))
        g = g.view(n, self.heads, self.per_head, self.per_head)

        heads = x.view(n, self.heads, self.per_head, 3)
        message = torch.matmul(g, heads).reshape(n, self.channels, 3)
        return self.norm(x + self.proj(message))
