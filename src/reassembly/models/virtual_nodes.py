"""
Cross-fragment communication through K virtual nodes per fragment.

Real mesh edges never join two fragments, so without a separate channel a
fragment's representation can only ever depend on its own geometry -- and the
correct orientation of a shard usually is not determined by that shard alone.
Three stages, per the design document:

  1. Upward pooling      each fragment's K slots attend over its own vertices
  2. Cross-fragment      all slots of the SAME SCENE attend to each other
  3. Downward broadcast  each vertex attends back to its own fragment's slots

TWO DEVIATIONS FROM THE DESIGN DOCUMENT, both forced by equivariance
-------------------------------------------------------------------
The document fixes each slot's spatial coordinates ``x_omega`` symmetrically at
initialization. A literal world-frame coordinate does not rotate when the input
does, which breaks the equivariance the whole backbone exists to provide. Here
each slot instead gets a purely NON-SPATIAL learned identity vector (plain
scalars -- nothing with an xyz axis to rotate wrongly), which gates, by
invariant scalar multiplication, an equivariant seed pooled from the fragment's
own vertices. Same "K distinguishable symmetric slots per fragment" semantics,
every geometric quantity provably equivariant.

Stage 2 must be masked to same-scene fragments. ``fragment_id`` is unique
across a whole batch, not reset per scene, so an unmasked stage 2 lets
fragments of unrelated objects exchange information -- wrong, and it makes the
model's output depend on batch composition.

MEMORY: WHY SEGMENT ATTENTION INSTEAD OF DENSE MASKED ATTENTION
---------------------------------------------------------------
Stages 1 and 3 used to build a dense ``(F*K, heads, N)`` (resp.
``(N, heads, F*K)``) score tensor and then mask away everything outside the
vertex's own fragment -- three tensors of that size (mask, masked_fill,
softmax), all retained for backward. But the mask keeps only ``K*N`` of those
``F*K*N`` scores: the dense form does F times more work and holds F times more
memory than the computation it is expressing. On a 20-fragment scene with 200k
vertices that is 0.48 GiB per tensor instead of 0.02 GiB.

Since each vertex only ever interacts with its own fragment's K slots, the
scores can be indexed directly as ``(N, heads, K)``:

  * stage 1 needs a scatter-softmax over the vertices within each fragment
    (``torch_geometric.utils.softmax`` grouped by ``fragment_id``);
  * stage 3 needs an ordinary softmax over the K axis, no scatter at all.

Verified bit-identical to the dense masked version and equivariant in
``validation/v02_segment_attention.py``; guarded by
``tests/test_virtual_nodes.py::test_segment_matches_dense``.

Stage 2 stays dense: ``F*K`` is a couple of hundred, so its ``(F*K, F*K)``
score matrix is negligible and masking it is the clearest expression.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as scatter_softmax

from .vn_layers import VNLeakyReLU, VNLinear, make_norm


def scatter_mean_vectors(x: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Per-segment mean of equivariant vectors -- uniform (invariant) weights,
    so the result is still equivariant. ``x: (N, C, 3) -> (num_segments, C, 3)``."""
    C = x.shape[1]
    sums = torch.zeros(num_segments, C, 3, device=x.device, dtype=x.dtype)
    counts = torch.zeros(num_segments, device=x.device, dtype=x.dtype)
    sums.index_add_(0, index, x)
    counts.index_add_(0, index, torch.ones_like(index, dtype=x.dtype))
    return sums / counts.clamp_min(1).view(-1, 1, 1)


class VNDenseCrossAttention(nn.Module):
    """Dense equivariant multi-head cross-attention with an optional mask.

    ONE batched matmul, masked, rather than a loop over scenes.

    The mask makes the score matrix block-diagonal, so the cross-scene entries
    are computed and thrown away -- quadratic in batch size rather than linear.
    That looks wasteful, and an earlier version of this code "fixed" it by
    running each scene as its own block. Measured on a real run that was **3.1x
    SLOWER per sample** (0.575 s against 0.184 s): at 8 scenes x 3 layers,
    doubled by gradient checkpointing, it traded one large GPU-saturating
    matmul for ~48 small ones plus a device sync from iterating
    `torch.unique()`.

    Slots are few -- fragments x 8 -- so the wasted FLOPs are cheap and the
    kernel-launch overhead is not. Keep it dense.

    Used only for stage 2, where both sides are the (small) set of virtual
    nodes. Also used by the tests as the reference the segment implementations
    are checked against.
    """

    def __init__(self, in_channels_q: int, in_channels_kv: int, out_channels: int,
                 heads: int = 4, head_dim: Optional[int] = None):
        super().__init__()
        if head_dim is None:
            if out_channels % heads != 0:
                raise ValueError(f"out_channels {out_channels} not divisible by heads {heads}")
            head_dim = out_channels // heads
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.lin_q = VNLinear(in_channels_q, inner)
        self.lin_k = VNLinear(in_channels_kv, inner)
        self.lin_v = VNLinear(in_channels_kv, inner)
        self.lin_out = VNLinear(inner, out_channels)
        self.scale = (head_dim * 3) ** -0.5

    def forward(self, query_x, key_x, mask: Optional[torch.Tensor] = None):
        Nq, Nk = query_x.size(0), key_x.size(0)
        Q = self.lin_q(query_x).view(Nq, self.heads, self.head_dim, 3)
        K = self.lin_k(key_x).view(Nk, self.heads, self.head_dim, 3)
        V = self.lin_v(key_x).view(Nk, self.heads, self.head_dim, 3)

        logits = torch.einsum("qhoc,khoc->qhk", Q, K) * self.scale
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        out = torch.einsum("qhk,khoc->qhoc", alpha, V)
        return self.lin_out(out.reshape(Nq, self.heads * self.head_dim, 3))


class VNSlotAttention(nn.Module):
    """Equivariant attention between vertices and their OWN fragment's K slots.

    ``direction='up'``   queries = slots, keys/values = vertices
                         -> ``(F, K, C, 3)``; softmax over vertices per fragment
    ``direction='down'`` queries = vertices, keys/values = slots
                         -> ``(N, C, 3)``; softmax over the K axis

    Neither direction ever materializes an ``(F*K, N)`` object.
    """

    def __init__(self, in_channels_q: int, in_channels_kv: int, out_channels: int,
                 num_slots: int = 8, heads: int = 4, head_dim: Optional[int] = None,
                 direction: str = "up"):
        super().__init__()
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        if head_dim is None:
            if out_channels % heads != 0:
                raise ValueError(f"out_channels {out_channels} not divisible by heads {heads}")
            head_dim = out_channels // heads

        self.direction = direction
        self.num_slots = num_slots
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim

        self.lin_q = VNLinear(in_channels_q, inner)
        self.lin_k = VNLinear(in_channels_kv, inner)
        self.lin_v = VNLinear(in_channels_kv, inner)
        self.lin_out = VNLinear(inner, out_channels)
        self.scale = (head_dim * 3) ** -0.5

    def forward(self, slots, vertices, fragment_id, num_fragments):
        """slots: ``(F, K, C, 3)``  vertices: ``(N, C, 3)``  fragment_id: ``(N,)``"""
        N = vertices.size(0)
        H, D, K = self.heads, self.head_dim, self.num_slots

        if self.direction == "up":
            Q = self.lin_q(slots.reshape(num_fragments * K, -1, 3)).view(num_fragments, K, H, D, 3)
            Kk = self.lin_k(vertices).view(N, H, D, 3)
            V = self.lin_v(vertices).view(N, H, D, 3)

            # (N, H, K): each vertex scores against its own fragment's slots only
            logits = torch.einsum("nkhoc,nhoc->nhk", Q[fragment_id], Kk) * self.scale
            # Softmax over the vertices *within each fragment*, independently
            # per (head, slot): exactly a scatter-softmax keyed by fragment_id.
            alpha = scatter_softmax(
                logits.reshape(N, H * K), fragment_id, num_nodes=num_fragments
            ).view(N, H, K)

            # Loop over K (8) rather than broadcasting to (N, H, K, D, 3): the
            # broadcast form is K times the peak memory for identical
            # arithmetic, and K is small and fixed.
            #
            # index_add (out-of-place) rather than index_add_ on a view of a
            # shared buffer: in-place accumulation into `out[:, k]` works, but
            # it makes every slot's gradient depend on autograd's view-and-
            # version bookkeeping for a tensor that eight successive in-place
            # ops have mutated. Stacking independent results has no such
            # coupling and costs the same memory.
            zeros = torch.zeros(num_fragments, H, D, 3,
                                device=vertices.device, dtype=vertices.dtype)
            slots_out = [
                zeros.index_add(0, fragment_id, alpha[:, :, k, None, None] * V)
                for k in range(K)
            ]
            out = torch.stack(slots_out, dim=1)                    # (F, K, H, D, 3)
            return self.lin_out(out.reshape(num_fragments * K, H * D, 3)).view(
                num_fragments, K, -1, 3
            )

        # direction == "down"
        Q = self.lin_q(vertices).view(N, H, D, 3)
        Kk = self.lin_k(slots.reshape(num_fragments * K, -1, 3)).view(num_fragments, K, H, D, 3)
        V = self.lin_v(slots.reshape(num_fragments * K, -1, 3)).view(num_fragments, K, H, D, 3)

        logits = torch.einsum("nhoc,nkhoc->nhk", Q, Kk[fragment_id]) * self.scale
        alpha = torch.softmax(logits, dim=-1)                      # plain softmax over K

        # Gather one slot at a time: V[fragment_id] would materialise
        # (N, K, H, D, 3), which is K times larger than anything else here and
        # would give back a chunk of the memory this formulation exists to save.
        out = torch.zeros(N, H, D, 3, device=vertices.device, dtype=vertices.dtype)
        for k in range(K):
            out = out + alpha[:, :, k, None, None] * V[:, k][fragment_id]
        return self.lin_out(out.reshape(N, H * D, 3))


class VNVirtualNodeInit(nn.Module):
    """Initial per-fragment slot queries, with no fixed spatial anchor.

    ``seed[f]``  = mean over fragment f's vertices of ``VNLinear(x)`` (equivariant)
    ``gate[k]``  = sigmoid(Linear(slot_identity[k]))                (invariant)
    ``query[f,k] = gate[k] * seed[f]``   (invariant scalar x equivariant vector)
    """

    def __init__(self, in_channels: int, out_channels: int, num_slots: int = 8,
                 identity_dim: int = 16):
        super().__init__()
        self.num_slots = num_slots
        self.out_channels = out_channels
        self.seed_proj = VNLinear(in_channels, out_channels)
        self.slot_identity = nn.Parameter(torch.randn(num_slots, identity_dim) * 0.1)
        self.gate_map = nn.Linear(identity_dim, out_channels)

    def forward(self, x, fragment_id, num_fragments) -> torch.Tensor:
        """Returns ``(num_fragments, num_slots, out_channels, 3)``."""
        seed = scatter_mean_vectors(self.seed_proj(x), fragment_id, num_fragments)
        gate = torch.sigmoid(self.gate_map(self.slot_identity))     # (K, C), invariant
        return gate.view(1, self.num_slots, self.out_channels, 1) * seed.unsqueeze(1)


class VirtualNodeCommunicationBlock(nn.Module):
    """The full three-stage cross-fragment message-passing loop."""

    def __init__(self, channels: int, num_slots: int = 8, heads: int = 4,
                 head_dim: Optional[int] = None, identity_dim: int = 16,
                 norm: str = "layer"):
        super().__init__()
        self.num_slots = num_slots
        self.channels = channels

        self.init_query = VNVirtualNodeInit(channels, channels, num_slots, identity_dim)
        self.upward = VNSlotAttention(channels, channels, channels, num_slots, heads,
                                      head_dim, direction="up")
        self.global_attn = VNDenseCrossAttention(channels, channels, channels, heads, head_dim)
        self.downward = VNSlotAttention(channels, channels, channels, num_slots, heads,
                                        head_dim, direction="down")

        self.norm = make_norm(norm, channels)
        self.act = VNLeakyReLU(channels)

    def forward(
        self,
        x: torch.Tensor,
        fragment_id: torch.Tensor,
        num_fragments: int,
        fragment_scene_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x: (N, C, 3)`` vertex features, possibly for a batch of scenes.

        ``fragment_scene_id: (num_fragments,)`` is REQUIRED whenever more than
        one scene may be present -- i.e. essentially always during batched
        training. Passing ``None`` means "assume a single scene" and is only
        appropriate for a hand-built single-scene graph.
        """
        K = self.num_slots

        # Stage 1: upward pooling, scoped per fragment.
        queries = self.init_query(x, fragment_id, num_fragments)          # (F, K, C, 3)
        slots = self.upward(queries, x, fragment_id, num_fragments)       # (F, K, C, 3)

        # Stage 2: cross-fragment attention within each scene.
        flat = slots.reshape(num_fragments * K, self.channels, 3)
        if fragment_scene_id is not None:
            slot_scene = fragment_scene_id.repeat_interleave(K)
            mask = slot_scene.unsqueeze(1) == slot_scene.unsqueeze(0)
        else:
            mask = None
        flat = self.global_attn(flat, flat, mask=mask)
        slots = flat.view(num_fragments, K, self.channels, 3)

        # Stage 3: downward broadcast, fused as an equivariant residual.
        context = self.downward(slots, x, fragment_id, num_fragments)     # (N, C, 3)
        return self.act(self.norm(x + context))
