"""
Cross-fragment communication through K virtual nodes per fragment.

Real mesh edges never join two fragments, so without this the network could
only ever reason about one fragment at a time -- and orientation is only
determined relative to the rest of the object.

Three stages, matching the design document:
  1. Upward pooling  : each fragment's vertices -> that fragment's K slots.
  2. Global exchange : invariant descriptors of all slots of the SAME SCENE
                       attend to each other, gating each fragment's own vectors.
  3. Downward cast   : each vertex attends back to its own fragment's slots,
                       fused in as an equivariant residual.

DEVIATION FROM THE DESIGN DOCUMENT (equivariance, not preference)
-----------------------------------------------------------------
The document gives each slot a FIXED spatial coordinate x_omega, set
symmetrically at initialisation. A literal world-frame coordinate does not
rotate when the input fragment does, which breaks the equivariance the whole
backbone exists to provide. Instead each slot gets a fixed NON-SPATIAL learned
identity embedding (plain scalars -- no xyz axis, so nothing can rotate
wrongly), which gates, by invariant scalar multiplication, an equivariant
"seed" vector pooled from the fragment's own vertices. The intended semantics
(K distinguishable, symmetric, per-fragment slots) survive; the fixed anchor
does not.

PER-FRAGMENT EQUIVARIANCE -- WHY STAGE 2 EXCHANGES INVARIANTS ONLY
------------------------------------------------------------------
Diffusion rotates EVERY FRAGMENT INDEPENDENTLY, so the property the model
actually needs is

    G_f(A_1 x_1, ..., A_F x_F) = G_f(x_1, ..., x_F) A_f^T,

i.e. fragment f's output must be equivariant to ITS OWN rotation and
INVARIANT to every other fragment's. That is what lets a single learned
canonicalisation, G_f(clean) = I, produce the target A_f^T for every draw --
the entire argument in `predict_rotation`.

Letting slots of different fragments attend to each other as VECTORS destroys
it. An attention logit <q from fragment a, k from fragment b> becomes
<A_a q, A_b k>, which is only invariant when A_a = A_b. A block that mixes
equivariant vectors across fragments is therefore equivariant to a GLOBAL
rotation of the whole scene but not to the per-fragment rotations that
training actually applies -- a distinction that is invisible to a global-only
equivariance check while quietly removing the guarantee.

So everything that crosses a fragment boundary here is invariant: slots are
reduced to rotation-invariant descriptors, those attend to each other, and the
resulting invariant context produces a per-channel GATE that rescales each
fragment's own equivariant slot vectors. Invariant scalar times equivariant
vector is equivariant, so the update rotates with that fragment and with
nothing else.

Nothing useful is lost. The relative orientation between two scattered
fragments is, by construction, uniformly random noise -- being invariant to it
is correct, not a limitation. What genuinely determines which pieces mate is
invariant interface shape, which is exactly what still crosses.

SCENE SCOPING
-------------
Stage 2 must also be restricted to fragments of the SAME SCENE. A training
batch concatenates independent scenes -- usually different objects entirely --
and fragment ids are unique across the whole batch rather than reset per
scene. Unmasked, fragments of unrelated objects would exchange information,
which is both semantically wrong and makes a scene's output depend on what
else happened to land in the batch.

MEMORY
------
Stages 1 and 3 are SEGMENT attention over (vertex, slot) pairs, not dense
masked attention over (vertex, every-slot-in-the-batch). The dense form
allocated an (N, F*K) logit tensor per head -- three times over, for the raw
logits, the masked copy and the softmax -- which is where most of the
out-of-memory pressure on a 16 GB T4 came from. The segment form only ever
forms the N*K pairs the mask would have kept: a ~500x reduction at realistic
fragment counts, and numerically identical (see `tests/test_segment_ops.py`).

Stage 2 is dense within each scene and never across scenes, evaluated one
contiguous scene-block at a time, so it is quadratic in slots-per-scene and
strictly linear in batch size. It now attends over invariant tokens of width
C rather than vectors of width C*3, which makes it cheaper again.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .segment_ops import blockwise_softmax, segment_mean, segment_softmax, segment_sum
from .vn_layers import VNInvariant, VNLeakyReLU, VNLinear, make_norm


class VirtualNodeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_slots: int = 8,
        heads: int = 4,
        identity_dim: int = 16,
        norm: str = "layer",
        invariant_bottleneck: int = 8,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels = channels
        self.num_slots = num_slots
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = (self.head_channels * 3) ** -0.5
        self.token_scale = (channels // heads) ** -0.5

        # Slot identities: pure scalars, no xyz axis. `gate_map` is a plain
        # Linear because both its input and output are invariant.
        self.slot_identity = nn.Parameter(torch.randn(num_slots, identity_dim) * 0.1)
        self.gate_map = nn.Linear(identity_dim, channels)
        self.seed_proj = VNLinear(channels, channels)

        self.up_q = VNLinear(channels, channels)
        self.up_k = VNLinear(channels, channels)
        self.up_v = VNLinear(channels, channels)

        # Stage 2 crosses fragment boundaries, so everything that travels is
        # INVARIANT (see the module docstring). The only equivariant object
        # involved is each fragment's own slot vectors, rescaled by a gate.
        self.slot_invariant = VNInvariant(channels, bottleneck=invariant_bottleneck)
        self.token_in = nn.Linear(self.slot_invariant.out_features, channels)
        self.token_q = nn.Linear(channels, channels)
        self.token_k = nn.Linear(channels, channels)
        self.token_v = nn.Linear(channels, channels)
        self.token_out = nn.Linear(channels, channels)
        self.glob_out = VNLinear(channels, channels)

        self.down_q = VNLinear(channels, channels)
        self.down_k = VNLinear(channels, channels)
        self.down_v = VNLinear(channels, channels)
        self.down_out = VNLinear(channels, channels)

        self.norm = make_norm(norm, channels)
        self.act = VNLeakyReLU(channels)

    # ------------------------------------------------------------------
    def _slot_queries(self, x: torch.Tensor, node_frag: torch.Tensor, num_fragments: int) -> torch.Tensor:
        """(F, K, C, 3) per-fragment, per-slot query vectors."""
        seed = segment_mean(self.seed_proj(x), node_frag, num_fragments)   # (F, C, 3)
        gate = torch.sigmoid(self.gate_map(self.slot_identity))            # (K, C) invariant
        return gate.reshape(1, self.num_slots, self.channels, 1) * seed.unsqueeze(1)

    def _scene_block_sizes(self, frag_scene: torch.Tensor, K: int):
        """
        Slot-count per scene, or None when a single dense block is correct.

        `collate_scenes` lays fragments out scene by scene, so each scene's
        slots are a contiguous run and scoping is a slice rather than a mask.
        Returns None if there is only one scene (nothing to scope) or if the
        layout is unexpectedly non-contiguous (the caller then falls back to
        masked dense attention rather than silently mixing scenes).
        """
        if frag_scene is None or frag_scene.numel() == 0:
            return None
        if frag_scene.numel() > 1 and not bool((frag_scene[1:] >= frag_scene[:-1]).all()):
            return "non_contiguous"
        sizes = [int(c) * K for c in torch.bincount(frag_scene).tolist() if int(c) > 0]
        return None if len(sizes) <= 1 else sizes

    def _exchange(self, q, k, v, frag_scene, K):
        """
        Scaled dot-product attention over INVARIANT slot tokens, restricted to
        slots of the same scene.

        Evaluated one contiguous scene-block at a time. The masked-dense
        alternative still ALLOCATES the full batch-wide (Q, H, Q) score matrix
        and then discards every cross-scene entry, so its memory grows with the
        square of the batch size for no benefit.
        """
        sizes = self._scene_block_sizes(frag_scene, K)

        def attend(qs, ks, vs):
            logits = torch.einsum('qhd,khd->qhk', qs, ks) * self.token_scale
            return torch.einsum('qhk,khd->qhd', torch.softmax(logits, dim=-1), vs)

        if sizes is None:
            return attend(q, k, v)
        if sizes == "non_contiguous":
            logits = torch.einsum('qhd,khd->qhk', q, k) * self.token_scale
            alpha = blockwise_softmax(logits, frag_scene.repeat_interleave(K))
            return torch.einsum('qhk,khd->qhd', alpha, v)

        out, start = [], 0
        for size in sizes:
            stop = start + size
            out.append(attend(q[start:stop], k[start:stop], v[start:stop]))
            start = stop
        return torch.cat(out, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        node_frag: torch.Tensor,
        num_fragments: int,
        frag_scene: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:          (N, C, 3) vertex features, possibly spanning several scenes
        node_frag:  (N,) global fragment id per vertex
        frag_scene: (F,) scene id per fragment. Required whenever the batch may
                    hold more than one scene; None means "trust that this is a
                    single scene" and is only for single-scene debugging.
        """
        N = x.shape[0]
        K, H, Ch, C = self.num_slots, self.heads, self.head_channels, self.channels

        # ---- Stage 1: vertices -> their own fragment's slots -------------
        queries = self._slot_queries(x, node_frag, num_fragments)          # (F, K, C, 3)
        q_up = self.up_q(queries).reshape(num_fragments, K, H, Ch, 3)
        k_up = self.up_k(x).reshape(N, H, Ch, 3)
        v_up = self.up_v(x).reshape(N, H, Ch, 3)

        q_node = q_up.index_select(0, node_frag)                           # (N, K, H, Ch, 3)
        logits = torch.einsum('nkhci,nhci->nkh', q_node, k_up) * self.scale
        alpha = segment_softmax(logits, node_frag, num_fragments)          # (N, K, H)
        slots = segment_sum(
            alpha[..., None, None] * v_up.unsqueeze(1), node_frag, num_fragments
        )                                                                  # (F, K, H, Ch, 3)

        # ---- Stage 2: cross-fragment exchange, through invariants only ----
        flat = slots.reshape(num_fragments * K, C, 3)
        tokens = self.token_in(self.slot_invariant(flat))                  # (Q, C) invariant
        q = self.token_q(tokens).reshape(-1, H, C // H)
        k = self.token_k(tokens).reshape(-1, H, C // H)
        v = self.token_v(tokens).reshape(-1, H, C // H)
        context = self._exchange(q, k, v, frag_scene, K).reshape(-1, C)
        gate = torch.tanh(self.token_out(context))                         # (Q, C) invariant
        # The only equivariant quantity here is the fragment's OWN slots,
        # rescaled per channel by an invariant gate -- so the update rotates
        # with that fragment and with nothing else.
        flat = flat + gate.unsqueeze(-1) * self.glob_out(flat)
        slots = flat.reshape(num_fragments, K, C, 3)

        # ---- Stage 3: slots -> vertices of the same fragment --------------
        dq = self.down_q(x).reshape(N, H, Ch, 3)
        dk = self.down_k(slots.reshape(-1, C, 3)).reshape(num_fragments, K, H, Ch, 3)
        dv = self.down_v(slots.reshape(-1, C, 3)).reshape(num_fragments, K, H, Ch, 3)

        dk_node = dk.index_select(0, node_frag)                            # (N, K, H, Ch, 3)
        dv_node = dv.index_select(0, node_frag)
        down_logits = torch.einsum('nhci,nkhci->nkh', dq, dk_node) * self.scale
        # Softmax over the K slots of this vertex's own fragment: a plain
        # softmax over dim 1 IS the fragment-scoped one, because dk_node was
        # gathered by fragment -- no mask needed at all.
        down_alpha = torch.softmax(down_logits, dim=1)
        context = (down_alpha[..., None, None] * dv_node).sum(dim=1).reshape(N, C, 3)

        out = x + self.down_out(context)
        return self.act(self.norm(out))
