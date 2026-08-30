# Design notes — batching, features, and open decisions

Working notes for the current redesign. Everything here is either implemented
and tested, or an explicit recommendation with the reasoning attached.

---

## 1. Where cross-fragment attention runs (standardised or not)

**Your mask idea is right, and it settles the part it addresses.** Fracture
membership is an *index* set. Centring subtracts a constant and standardising
multiplies by a scalar; neither reorders vertices, so a boolean mask over vertex
indices survives both untouched. `fracture_patches` takes exactly that mask, and
`test_partition_is_invariant_to_translation_and_scale` pins it down.

But identification was only half the question. The other half is what
coordinates the attention *scores* are computed from, and there the answer is
not free.

**Rotation is scale-free**, so the pipeline you described is sound:

1. centre every fragment (both copies) at its own centroid;
2. standardise the diffused copy per fragment, feed that to the network;
3. apply the predicted `R` to the **centred, non-standardised** fragments;
4. compute the geometric losses against the centred ground truth.

`R` does not depend on scale, so predicting it from standardised geometry and
applying it to unstandardised geometry is exactly right.

**The catch is cross-fragment comparison.** Two fracture surfaces that mate are
the *same size* in world units. Standardising each fragment by its own extent
destroys that: a small chip and the large body it broke from arrive at the
attention layer rescaled by different factors, so their surfaces no longer look
like they fit. The per-fragment scale feature carries the information needed to
undo this, but the network has to *learn* to undo it — it is not automatic, and
it is a strange thing to ask a network to learn when you could simply not break
it.

**Recommendation: standardise per scene, not per fragment.** Divide every
fragment of a scene by one common factor (the assembled object's extent, or the
largest fragment's). Then:

- all fragments share a metric frame, so mating surfaces stay the same size and
  cross-fragment attention compares like with like;
- the network still never sees absolute world scale;
- you keep a per-fragment scale feature *as well*, which now means "how big is
  this fragment relative to the object" — a genuinely useful signal (GARF's
  ablation found large fragments easier and worth up-weighting).

GARF standardises per fragment because its encoder processes fragments
independently — there is no cross-fragment geometry in that stage. This design
does cross-fragment attention on raw geometry, so it has a reason to differ.

**Implemented**, in `reassembly.data.transforms`, defaulting to per scene:

```python
norm = normalize_fragments(fragments)                    # mode="scene" (default)
norm = normalize_fragments(fragments, mode="fragment")   # GARF's convention
norm = normalize_fragments(fragments, rescale=False)     # centre only
```

`Normalized` carries `vertices`, `centroid`, `divisor` and `radius`.
`denormalize_fragments` inverts it exactly — stage two solves translation in the
world frame, so a normalisation that could not be undone would destroy the
centroids the solver is trying to recover. `radius` stays in **world units**
whichever mode is used, so the invariant scale feature means the same thing
either way.

`tests/test_transforms.py` asserts the property that decides the default: with
`mode="scene"` a body and a chip keep their 10:1 world size ratio, and with
`mode="fragment"` that ratio collapses to 1:1.

---

## 2. Batch layout

Yes — adopt the `cu(ℓ)` idea, but keep **both** representations, because the two
kinds of operation want different ones:

| representation | shape | what needs it |
|---|---|---|
| `*_batch` (segment id per element) | `(N,)` | scatter / segment-softmax (PyG message passing) |
| `*_ptr` (cumulative offsets) | `(S+1,)` | varlen attention kernels — this is GARF's `cu(ℓ)` |

They are one line apart, so store both rather than converting on the hot path:

```python
ptr   = torch.cat([zeros(1), bincount(batch).cumsum(0)])
batch = torch.repeat_interleave(arange(len(ptr) - 1), ptr.diff())
```

### The tensors a batch carries

```
vertex_pos        (V, 3)      centred (and standardised) coordinates
vertex_normal     (V, 3)
vertex_frag       (V,)        fragment id, globally unique across the batch
vertex_ptr        (Nf+1,)     cu(l) over fragments        <- varlen attention
edge_index        (2, 2E)     COO, endpoints already offset by vertex_ptr
edge_n1, edge_n2  (E, 3)      canonically ordered (see §3)
edge_frag         (E,)        fragment id per edge
frag_scene        (Nf,)       scene id per fragment
frag_scale        (Nf,)       invariant scalar feature
frag_ptr          (Ns+1,)     cu(l) over scenes
patch_frag        (P,)        fragment id per fracture patch
patch_scene_ptr   (Ns+1,)     cu(l) over scenes, for cross-fragment attention
```

### Three rules that are not optional

**Sort by `(scene, fragment)` before building any `ptr`.** Varlen kernels
require each segment's elements to be contiguous. A `ptr` built over a
non-contiguous segmentation is silently wrong — it will index a valid range of
the wrong rows, produce finite numbers, and train.

**Cross-fragment attention is segmented by *scene*, not by batch.** A batch
concatenates unrelated objects, and fragment ids are globally unique, so
unmasked global attention lets fragments of different scenes exchange
information. This exact bug is in the project's history (`handoff.pdf` §7.3),
found only because someone asked how variable-size batches were handled. It
makes the prediction depend on batch composition. `patch_scene_ptr` is the fix:
attend within scene segments, and additionally mask same-fragment pairs, since
intra-fragment communication is the mesh graph's job.

**Both copies of a scene must share one topology.** The loader produces a
ground-truth scene and a diffused scene, and the losses match *corresponding*
vertices, edges and face normals by index. So build the topology **once** and
transform only the coordinates. Building the two copies independently invites
`resolve_duplicated_faces` — which reorders faces lexicographically — to
produce two different vertex orderings, after which every per-vertex loss is
comparing unrelated rows. Nothing raises; the loss simply never converges.

**Implemented in `data/features.py`, and by construction rather than by
assertion.** `build_scene` computes topology and normals *once* on the assembled
copy and derives the perturbed one by rotating coordinates — including the
normals and edge normals, which are rotated rather than recomputed so the two
copies cannot differ even by round-off in a quantity the loss compares
directly. There is nothing left for an `assert gt.edge_index == diffused.edge_index`
to catch, because there is only one `edge_index`.

`Batch` carries the tensors listed above under implementation names, plus
`token_query` / `token_key`: the cross-fragment pair list, built once per batch
by `cross_fragment_index` rather than per layer.

### Intra-fragment connectivity

To be explicit, since it came up: **nothing new is built inside a fragment.**
The mesh edges are the intra-fragment graph, and message passing runs along
them via `edge_index`. The only *new* connections are inter-fragment, between
fracture patches. `scripts/analyze_graph_cost.py` counts them separately for
this reason: `intra_edges` is the mesh edge count, not a constructed graph.

---

## 3. Edge face-normal ordering — implemented

`reassembly.mesh.orientation`, with `tests/test_orientation.py`.

For edge `(u, v)` with `u < v` and `d = x_v − x_u`, order the two adjacent
normals by the sign of `s = (n_a × n_b) · d`. Invariant under proper rotation
because `det(R) = 1`; a reflection *does* exchange the slots, which is correct
and is asserted so the assumption stays visible.

The degeneracy is benign: for a manifold edge `d` is parallel to `n_a × n_b`,
so `|s| = sin(angle) · ‖d‖`, and the order becomes ambiguous exactly when the
normals are parallel — i.e. exactly when both slots hold the same vector and
the swap is a no-op. Ambiguous edges are flagged, not silently ordered.

Reversing the edge negates `d`, so the reverse directed copy gets the swapped
pair *from the rule itself*. `directed_edge_normals` returns both copies with
the slots already mirrored; the mirroring is a property, not a convention that
a caller has to remember.

The test suite includes `test_naive_face_index_order_is_NOT_stable`, which
demonstrates the bug being prevented: taking normals in `faces_per_edge` order
does change under a face permutation.

---

## 4. The graph, as agreed

**The configuration this project is building, stated once so it is not implicit:**

| level | what participates |
|---|---|
| nodes | **every** mesh vertex of every fragment — nothing is dropped |
| intra-fragment edges | **every** mesh edge, already in the mesh, nothing constructed |
| inter-fragment edges | among vertices **sampled by FPS from the dihedral-masked subset only** |

So the fracture mask is a *gate on the sampling pool*, not a filter on the graph.
A vertex that is not fracture-labelled still has node features, still sends and
receives messages along its mesh edges, and still contributes to its fragment's
representation — it is simply not eligible to be chosen as a cross-fragment token.

```python
mask   = fracture_face_mask(V, F).face_mask          # dihedral
tokens = fracture_patches(V, F, vertex_mask, mode="sample", max_patches=K)
```

That is exactly what `mode="sample"` with the default `eligible_mask` does.

**One real advantage of dihedral over coincidence here**, which cuts against the
earlier recommendation: dihedral is what is available at *inference*, so training
on it means train and test see the same labelling. Coincidence labels are more
accurate but only exist at training time, so a model trained on them meets a
different (worse) input distribution at test. That mismatch tends to surface as a
validation gap misattributed to the architecture. Training on dihedral trades
label quality for train/test consistency — a defensible trade, and one worth
stating in the thesis rather than leaving implicit.

`--method coincidence` remains the way to *measure* how good the dihedral labels
are, and the switch if the consistency argument turns out not to pay for itself.

### Reducing the inter-fragment graph — three modes, implemented

`reassembly.mesh.patches`, with `tests/test_patches.py`. All three produce the
same structure, so they are swappable and benchmarkable:

| mode | token is | keeps | budget |
|---|---|---|---|
| `vertex` | every fracture vertex | everything | none — the baseline |
| `patch` | mean of a group of vertices | all vertices, averaged | `max_patches` |
| `sample` | one real mesh vertex | a subset, discards the rest | `max_patches` |

`sample` is the vertex-level analogue of GARF's Poisson-disk sampling. It uses
farthest-point selection rather than dart-throwing, deliberately: FPS on a fixed
point set gives approximately the same blue-noise coverage, but it is
**deterministic and exactly rotation-invariant** (ties broken by vertex index),
whereas dart-throwing depends on an RNG stream that has nothing to do with the
geometry. Two views of one fragment must select the same vertices.

**Pooling versus selection is a real trade.** `patch` uses information from
every vertex, but its tokens sit at centroids, which are not points on the
surface — and for a design whose second stage matches *interface points*, a
token that is not on the interface is a real drawback. `sample` keeps genuine
surface points but throws most of them away. Which wins is empirical, which is
why both are here.

### What "budget" means, and why it takes two limits

**Terminology first.** A *token* is one thing that participates in cross-fragment
attention — one vertex, or one pooled patch. A *budget* is simply a cap on how
many tokens you allow. Attention connects every token to every token in another
fragment, so the connection count grows with the square of the token count: 100
tokens cost 4x what 50 cost.

There are two places to put that cap, and they behave very differently.

**A per-fragment cap** — "at most 32 tokens from each fragment" — does not fix
the cost, because it says nothing about how many fragments there are:

| scene | fragments | tokens | cross-fragment connections |
|---|---|---|---|
| small | 2 | 64 | 1,024 |
| typical | 8 | 256 | 28,672 |
| large | 20 | 640 | 194,560 |

Same cap, 190x spread. And most fragments never reach the cap anyway — the median
Breaking Bad fragment has only 125 fracture vertices — so the cap binds on the
large fragments and does nothing at all on the small ones.

**A per-scene total** — "at most 256 tokens for this whole scene, divided among
its fragments in proportion to their size" — makes the count a constant. Every
scene contributes 256 tokens whether it has 2 fragments or 20, so the connection
count depends only on the configuration, never on the mesh. That is what
`allocate_budget(weights, total)` does, and it is the same structure GARF uses: a
fixed M = 5000 points per object, shared out by fragment surface area.

**The per-scene total is the only limit needed** — and this section previously
argued otherwise, so the correction is worth stating plainly.

For a budget `T` split into `t_i`, the undirected cross-fragment pair count is

```
(T² − Σ t_i²) / 2   <   T² / 2
```

for **any** split. So `total = 2048` can never exceed ~2.1 million pairs — 0.03×
GARF's six-layer stack — however lopsided the allocation. A per-fragment cap does
not tighten that ceiling by a single pair.

| fragments | worst case (even split) | most concentrated |
|---|---|---|
| 3 | 1,398,101 | 4,093 |
| 20 | 1,992,292 | 38,722 |
| 99 | 2,075,958 | 195,853 |

What a cap *does* do is destroy the proportionality the area weighting exists to
provide. A head statue with both ears, the nose and a piece of hair broken off:
the head carries four mating surfaces, each small piece carries one.

| piece | proportional to area | after a 128 cap |
|---|---|---|
| head | 1024 | 128 |
| ear | 256 | 128 |
| ear | 256 | 128 |
| nose | 256 | 128 |
| hair | 256 | 128 |

The head has four times the fracture surface of an ear and ends up describing it
with the same number of tokens — the piece with the most interface to match gets
the least resolution per unit of it. And the capped allocation spends 640 of its
2048 tokens, throwing the rest away.

**Where the earlier argument went wrong.** It observed, correctly, that "a
per-scene budget alone starves the tail — 256 tokens across 99 fragments is 2
each", and concluded that a second limit was needed. The conclusion does not
follow: a cap cannot cure starvation, because capping at 128 when every fragment
already receives 20 changes nothing. The fix for a starved tail is a larger
`total`, which is why the default is 2048 and not 256.

```python
scene_tokens(fragments, masks, total=2048, mode="sample")
```

`max_per_fragment` remains available for an ablation or a hard memory bound. It
defaults to `None` and should stay there without a specific reason.

### But measured on the real labels, you probably need neither

This machinery was designed against the *dihedral* mask, which over-labels by
~3.9x. Against the coincidence ground truth the fracture surface is far smaller —
median **125 fracture vertices per fragment**, median **3 fragments per scene** —
and the unreduced per-vertex graph is already cheap:

| fragments | fracture verts each | cross-fragment connections | vs GARF's 6-layer stack |
|---|---|---|---|
| 3 (median scene) | 125 | 46,875 | 0.001x |
| 8 (mean scene) | 164 | 753,088 | 0.01x |
| 20 | 125 | 2,968,750 | 0.04x |
| 99 (largest) | 125 | 75,796,875 | **1.01x** |

Cost is `7812.5 · n(n-1)` for n fragments at the median size, so it crosses GARF's
whole stack at **n ~ 99** — exactly the largest scene in the dataset — and stays
under 10% of it up to **n ~ 32**.

**So start with `mode="vertex"` and no budget at all.** It is the simplest option,
affordable for the overwhelming majority of scenes, and — the reason that matters
most — it preserves exact vertex identity, which the embedding-consistency loss
and correspondence discovery both depend on. Reach for a budget only above ~32
fragments, where the tail would otherwise blow up — and when you do, one number
is enough: `total=2048` bounds the graph on its own.

The `patch` and `sample` modes stay worth having: for that tail, for the
gate-versus-feature ablation, and as the answer if mesh resolution ever increases.
But they are no longer load-bearing, and the earlier framing here — that reduction
was necessary — came from measuring against a 3.9x-inflated mask.

### Gate or feature: the decision that makes precision matter

`eligible_mask` says which vertices may become tokens.

* **default (gate)** — tokens come from the fracture surface only. A wrong
  label is an edge that should not exist, or one that is missing; the model
  cannot undo either.
* **`eligible_mask=all` (feature)** — tokens are drawn from the whole fragment
  surface and `fracture_fraction` carries fracture-ness as a per-token channel.
  A wrong label is one noisy input among several, which attention can discount.

Measured on the full dataset, the dihedral mask is **3.9x larger** than the true
fracture surface at threshold 0.9 (precision 0.24, recall 0.93). That is
disqualifying for a gate and largely survivable for a feature — the same
labelling, two very different consequences.

**GARF takes the feature route**, which is worth being precise about because it
is easy to assume otherwise. Its encoder produces `F = E(P)` over **all** M
sampled points, and its global attention runs with `ℓ = M` — every point, not
the fracture subset. Fracture segmentation is a *pretraining objective* shaping
those features (`L_Seg`, Dice loss), never a mask on the attention graph. Gating
the graph by fracture-ness is a stronger commitment than the SOTA makes.

There is a substantive argument for the feature route beyond robustness: the
original sharp features carry assembly information of their own. A rim broken
across three fragments still has to close into one circle; a handle split in two
still has to rejoin. Those constraints are real and a fracture-only graph
discards them by construction. What the fracture surface uniquely provides is
*exact mating*; what the original features provide is *global continuity*. They
are different signals, and there is no reason the model should only get one.

The two are cleanly separable in this implementation because the token budget is
fixed either way — same token count, same cross-attention cost, different
surface sampled. That makes it a genuine ablation rather than a confound:

```python
gated   = fracture_patches(V, F, frac_mask, mode="sample", max_patches=64)
feature = fracture_patches(V, F, frac_mask, mode="sample", max_patches=64,
                           eligible_mask=np.ones(len(V), bool))
```

Supervise them differently, though. The embedding-consistency loss is defined
over *coincident* vertices, so it only supervises the fracture channel. Tokens
sampled from the original surface get no direct supervision and are shaped only
by the rotation loss — include them knowingly, not by accident.

### What sampling does not fix

Sampling bounds the *count*; it does nothing for the *contamination*. If 20% of
what the extractor labels is actually the fragment's original exterior — the rim
effect measured in §9 — then roughly 20% of the sampled tokens are rim, and
attention is being asked to match surfaces that never mated. Precision is the
threshold's job (`scripts/tune_sharp_threshold.py`), not the sampler's.

### One consequence to design around

Breaking Bad stores mating fragments with **coincident vertices** on the
fracture surface — that is exactly what the `coincidence` labelling exploits.
Sampling each fragment independently breaks that: fragment A's 40 selected
vertices and fragment B's 40 will generally not be the same points, even where
the surfaces coincide exactly.

That is fine for attention, which does not need coincidence. It is **not** fine
for the embedding-consistency loss or for correspondence discovery, both of
which are defined over vertices that share a location. Keep them on the full
vertex set and use the sampled set only for the attention graph. The two need
not be the same set, and conflating them would quietly destroy the supervision
signal that motivated the interface embeddings in the first place.

## 5. Vector features, and where scale goes

A Vector Neuron feature is `(C, 3)`; rotation acts on the last axis, `x → x Rᵀ`.
Your features map cleanly:

| | channels | type |
|---|---|---|
| node | centred coordinate, vertex normal | `(2, 3)` vector |
| edge | `n1`, `n2` | `(2, 3)` vector |
| fragment | scale | **scalar, invariant** |

Scale cannot be concatenated onto a `(C, 3)` tensor — it has no direction, and
inventing one breaks equivariance. Keep a parallel invariant stream and use it
to *condition* the vector stream:

```
inv   = VNInvariant(x)                    # Gram matrix, rotation-invariant
h     = MLP(concat[inv, log_scale])       # ordinary scalars
gain  = softplus(Linear(h))               # (C,)
x     = gain[..., None] * x               # still equivariant
```

An equivariant vector times an invariant scalar is equivariant, so this is safe.
It is the same pattern the existing `VNGATLayer` already uses for the edge
scalar's attention bias.

**Use `log(scale)`, standardised — not raw scale.** Your run shows fragments
from 4 to 83,039 vertices, so linear extent spans orders of magnitude. A raw
heavy-tailed scalar will dominate whatever it is concatenated with. GARF applies
a positional encoding `PE(s)` for the same reason; log-and-standardise is the
cheaper equivalent.

---

## 6. Applying a rotation to a 64-dim embedding

You were right that this needs care, and the resolution is that the question
dissolves if the embedding stays vector-typed.

A VN embedding is not `(64,)` — it is `(64, 3)`. `R` acts on the 3-axis for any
channel count, so `z → z Rᵀ` is well-defined, and the embedding-consistency loss
you described works as written.

**The trap:** if any layer collapses to invariants (a Gram matrix, channel
norms), `R` acts *trivially* on the result. The consistency loss is then
satisfied automatically, contributes no gradient, and looks like a term that
converged instantly. Check it the way this project checks everything else:
feed a random `R`, assert `‖z(Rx) − z(x)Rᵀ‖ < 1e-12` in float64, and separately
assert the loss is **not** already zero at initialisation.

And keep the centroid-variance form of the loss, not `‖Σz‖²` — the latter is
minimised by embeddings that cancel rather than agree. That degeneracy is
already documented in `handoff.pdf` §3.1 with a worked counterexample.

---

## 7. `arccos` in the geodesic loss

`DesignV4.pdf` §1.5.1 still specifies

```
θ = arccos((tr(RᵀR_gt) − 1) / 2)
```

`arccos` has unbounded derivative at ±1, so **the gradient diverges as the model
converges** — the term crowds out every other loss exactly when it should be
handing over. Domain-guarding with `clamp(−1+ε, 1−ε)` turns that into an error
floor instead: a perfect prediction never reports zero.

Use the `atan2` form, which is bounded everywhere and exact at zero:

```python
M     = R_pred.transpose(-1, -2) @ R_gt
cos_t = (M.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
sin_t = 0.5 * torch.stack([M[..., 2, 1] - M[..., 1, 2],
                           M[..., 0, 2] - M[..., 2, 0],
                           M[..., 1, 0] - M[..., 0, 1]], -1).norm(dim=-1)
theta = torch.atan2(sin_t, cos_t)
```

This is already logged as bug #11 in the handoff. Update the design document so
it does not get re-implemented from the spec.

**Implemented in `nn/losses.py`, with one correction to the sketch above.** The
axis norm needs a guard: at a *perfect* prediction the axis is exactly zero and
`.norm()`'s gradient there is NaN, which propagates through the whole graph. The
obvious guard, `sqrt(s + 1e-8)`, is wrong for a different reason — it perturbs
every norm, not just the near-zero ones, and left a 5e-9 rad floor at both 0°
and 180°. Clamping the squared sum is exact away from zero, and this path can
take a far smaller floor than the rest of the package because
`d·atan2/d‖axis‖ → ½` exactly cancels the `1/‖axis‖` in the norm's own
derivative: the composite gradient is bounded by ½ whatever the floor is.

---

## 8. Reference values

Check every number against these before concluding anything.

All of these are now *measured* by Monte Carlo in `tests/test_losses.py` rather
than asserted here.

| quantity | value |
|---|---|
| chance geodesic error | 126.48° = π/2 + 2/π |
| chance Euler RMSE, **random** prediction | 86.29° |
| chance Euler RMSE, **identity** prediction | 83.14° |
| axis correct, azimuth random — geodesic | 89.9° |
| untrained `L_normal`, `L_face` | 1.0, 2.0 |
| `L_position` on the unit sphere | 4/3 |
| perfect prediction | geodesic exactly 0; the rest to the 1e-8 norm floor |

**The 83.20° this table used to give as "chance Euler RMSE" is the wrong
baseline.** It is the score for predicting the *identity* every time, and
predicting the identity beats guessing randomly by three degrees. Since
collapsing towards identity is the cheapest early way to reduce a rotation loss,
a model can appear to beat chance on GARF's headline metric while having learned
nothing at all. Geodesic error reads 126.5° for both — it does not pay for the
collapse — which is the argument for making it the primary number.

The two metrics also disagree on the same prediction: a 30° single-axis error is
30° geodesic but ≈17.3° Euler RMSE. GARF's tables are Euler RMSE. Always say
which.

---

## 9. Empty fracture masks — diagnosed and fixed

`frac_vertices` min = 0 in the full run, which is impossible: every fragment of
a broken object tore away from a neighbour. So it was the heuristic failing.

**Mechanism, measured.** The dihedral test asks whether adjacent faces differ by
more than ~26° (`|cos| < 0.9`). Roughness is relative to triangle size; the
threshold is absolute. On a small, gently curved shard *no* adjacency clears it,
no face has a sharp neighbour, and the mask is empty. A 6-face shard goes from
6/6 labelled to 0/6 as its tilt drops from 0.5 to 0.2 — the geometry barely
changes, the label collapses.

A second, distinct failure shows up on *smooth* interfaces: the mask does not
empty, it moves. On a flat interface inside a box, the only sharp edges are
where the interface meets the box walls, so the extractor labels the rim at a
steady ~8% regardless of interface roughness. Worth knowing when reading
per-fragment percentages: a low number can mean "found the rim", not "found
less".

**Fix, in two parts.**

`--min-fracture-faces 1` relaxes the threshold for an offending fragment alone,
stepping through quantiles of *that fragment's own* dihedral distribution until
something is labelled. Fragments that already worked come out bit-identical
(asserted in `tests/test_fracture_nonempty.py`), so it cannot invalidate results
that were already sensible. Lowering the global threshold instead would relabel
everything, sweeping the smooth exterior in with it.

`scripts/tune_sharp_threshold.py` picks the global threshold against the
coincidence ground truth — precision, recall, F1 and empty-rate per candidate.
That is the number to quote in the thesis, not a default.

**Still open:** the dihedral heuristic is a stand-in for what GARF *learns* (a
PTv3 segmentation head trained on coincidence labels). If tuning does not get F1
high enough, the honest options are to train a small segmenter on the
coincidence labels, or to use coincidence labels at training time and accept
that inference needs the learned version.

### Are the remaining zeros extraneous fragments?

A fragment with no fracture surface would, in the real world, be a piece that
does not belong to this object — GARF studies exactly this (§4.4, Table 4): it
builds an *extraneous parts* subset by injecting fragments from **other objects**
in the same category, and reports that performance degrades gracefully (PA 83.4
complete → 79.2 with 20% extraneous, against PF++ dropping 49.4 → 40.6). Their
fracture-aware pretraining is what buys that robustness: a piece with no matching
fracture surface simply gets no strong correspondence, and the model leaves it
alone.

**But vanilla Breaking Bad contains no extraneous fragments.** Every piece comes
from the physics simulation of one object, which is why GARF had to *construct*
that subset. So a zero here is not an extraneous piece — the likeliest cause is a
mode with a single fragment, where `fracture_vertex_masks` returns all-false by
construction (`len(points) < 2`), or a piece whose vertices missed the tolerance.
It needs counting before it is designed around.

The design consequence is the same either way, and worth adopting regardless of
cause: **a fragment with no fracture surface should be excluded from the
reassembly**, not fed in with zero inter-fragment edges. Excluding it is honest;
including it silently gives the rotation head a fragment it has no information
about, which can only add noise to the loss. Once the count is known, this is a
filter in the loader.

---

## 10. The network — built and verified

`reassembly.nn`, with `tests/test_nn_equivariance.py`, `tests/test_losses.py`
and `tests/test_model.py`, plus `tests/test_training.py` for the engine.
292 tests total, clean under `-W error`.

`torch` installs from the **default** PyPI index. The earlier failure was
`download.pytorch.org` being blocked, not torch being unavailable — so
everything in §5–§8 that was previously "a reasoned recommendation, not an
executed result" has now been executed.

### What may cross between fragments

The constraint that decides the cross-fragment layer's whole shape, and the
thing most likely to be got wrong by someone reimplementing it.

Each fragment is perturbed by its own rotation, so the label for fragment *i*
does not move when fragment *j* lands at a different angle. The prediction must
not either. Writing `A` for a rotation on one fragment's input:

```
H_i(P_1, …, P_i A, …, P_N) = H_i(…) · A          equivariant to its own pose
H_i(P_1, …, P_j A, …, P_N) = H_i(…)   (j ≠ i)    invariant to everyone else's
```

This is equation 7 of *Leveraging SE(3) Equivariance for Learning 3D Geometric
Shape Assembly*, and it is why their correlation module is `C_ij = G_j · F_i` —
an **invariant** matrix from the sender multiplying the **receiver's own**
equivariant features. Nothing directional crosses the gap.

**This is not "no vectors between fragments".** `nn/cross.py` is a Vector
Neuron layer throughout — `(T, C, 3)` in, `(T, C, 3)` out, equivariant on the
spatial axis at every step. The constraint is on what a message may *depend
on*, and within it the message is as rich as it can be: a full `(C/H, C/H)`
**channel-mixing matrix** per head, read off the sender's invariants and applied
to the receiver's own vector features.

```
logits_ij = ⟨q(inv_i), k(inv_j)⟩ / √d        invariant to both poses
alpha     = softmax over j, same scene, different fragment
out_i     = x_i + ( Σ_j α_ij G_j ) x_i       G_j invariant, x_i equivariant
```

The sender re-mixes the receiver's channels arbitrarily — that is the full
`G_j · F_i` of the paper, under attention. What it cannot do is contribute a
direction of its own, because a direction carries an orientation. Nothing is
lost: the relative orientation of two arbitrarily tumbled fragments *is* the
perturbation, not the object. Shape is the signal and shape survives.

**Where the sum sits matters.** `G` is *linear* in the invariants, so

```
Σ_j α_ij G(v_j)  =  G( Σ_j α_ij v_j )
```

— aggregate the small invariant vectors, then build one matrix per query.
Materialising a matrix per *pair* would be `(P, H, C/H, C/H)`: gigabytes at the
two million pairs a large scene reaches, against a couple of megabytes per
token.

**The scores cannot be vector inner products.** The natural VN-attention score
`⟨W_q x_i, W_k x_j⟩` becomes `⟨A_i a, A_j b⟩` under independent perturbations,
which depends on the relative rotation `A_iᵀA_j` — precisely the noise. So Q
and K are linear maps of each token's invariant description. Forced by the same
argument, not a shortcut.

Measured: own-pose equivariance 8.9e-16, other-pose invariance 8.9e-16 against
outputs of order 1; the mixing matrix is full rank with off-diagonal magnitude
0.8× the diagonal, so it is a genuine channel mix rather than a per-channel
gain.

### Verified numerically, in float64

| check | result |
|---|---|
| every VN primitive, `L(xRᵀ) = L(x)Rᵀ` | ≤ 1e-15 |
| `VNInvariant`, invariance | 1.8e-15 |
| Gram-Schmidt head, orthonormal and det = +1 | exact to 1e-16 |
| intra-fragment attention, equivariance | 1.3e-15 |
| cross-fragment, own-pose equivariance | 4.4e-16 |
| cross-fragment, other-pose **invariance** | 3.3e-16 |
| cross-scene leakage | exactly 0 |
| whole model, per-fragment equivariance | ≤ 1e-9 through 6 layers |
| rotation label round trip | exact |
| every parameter receives gradient | 77 / 77 |

### Two bugs the verification caught

Both would have trained without complaining.

**A "near-identity" initialisation that was actually a dead subnetwork.**
Zeroing a gate's final weight makes its output constant, so the gradient into
*everything upstream of it* is exactly zero. That was the entire cross-fragment
pathway plus the scale gate — 31 of 77 parameters — contributing nothing to the
first optimiser steps and waking only as a bias drifted. A small random final
weight keeps the layer within 0.14% of the identity with every gradient path
alive. `test_every_parameter_receives_gradient` is the regression.

**An epsilon in the wrong place.** `sqrt(s + 1e-8)` perturbs every norm, not
just the near-zero ones. The head came out orthonormal only to 1e-8 with
det = 0.99999995, and the geodesic loss floored at 5e-9 rad. Clamping the
squared sum instead is exact wherever `‖x‖ > eps`.

### The transpose

Centred, the perturbation is `v_pert = v_gt Qᵀ`, so the head's frame satisfies
`M_pert = Q M_gt` and the head returns **`R_pred = Mᵀ`**. The network's job
reduces to mapping an already-assembled fragment to the identity frame — a fixed
target, because the assembled pose is a dataset-wide convention.

Getting this backwards is invisible at initialisation, since chance is chance in
either direction, so it would surface only as a model that never converges.
`test_model.py` asserts the round trip against a known perturbation.

### The layer schedule is interleaved, not stacked

`intra, intra, cross, intra, cross, intra`. A cross layer updates only the
*token* vertices, so without intra layers after it the rest of the fragment
never hears about its neighbours and the pooled rotation is decided by vertices
that learned nothing from the cross-attention.

### No `torch_scatter`

Every scatter this needs exists natively as `scatter_reduce_` or `index_add_`
(`nn/segment.py`). `torch_scatter` needs a compiler and a matching CUDA toolkit
at install time, which is a real obstacle on the Windows machine this trains
from, for no functional gain.

---

## 11. The token budget, as wired into the loader

`data/features.py` calls `patches.scene_tokens`, so the pipeline does what §4
says rather than something adjacent to it:

| | |
|---|---|
| nodes | **every** vertex of every fragment |
| intra-fragment edges | **every** mesh edge |
| token pool | fracture vertices only (**dihedral** mask, for the first run) |
| token selection | farthest-point sampling *within* each fragment |
| token count | `tokens_per_scene=2048`, no per-fragment cap |

Measured on synthetic scenes of 642-vertex fragments with 200 fracture vertices
each:

| fragments | `total=512` | `total=2048` | no budget at all |
|---|---|---|---|
| 3 | 384 (capacity binds) | 384 | 384 |
| 8 | **512** | 1024 | 1024 |
| 20 | **512** | **2048** | 2560 |
| 40 | **512** | **2048** | 5120 |
| 99 | **512** | **2048** | 12,672 |

The scene total is what makes the inter-fragment connection count a property of
the *configuration* instead of whatever mesh arrived, and it does so alone.

At the tail the split gets thin: 99 fragments sharing 2048 tokens by
fracture-surface area gives roughly 1 to 97 tokens per fragment depending on how
uneven the areas are. A fragment holding a single token contributes almost
nothing to matching. That is the intended trade — those scenes are cost-bound,
and the alternative starves every scene — but it is a number to watch in the
first training run, not a settled question.

If the total is smaller than the fragment count, some fragments get **zero**
tokens and no cross-fragment edges. `allocate_budget` serves the largest fracture
surfaces first and spends exactly `total`; it does not quietly overspend to give
everyone one, which is what it used to do.

### Geodesic farthest-point, not Euclidean, and not dart-throwing

Two independent choices, often conflated.

**Selection rule: farthest-point, not dart-throwing.** GARF uses Poisson-disk
sampling, which throws darts and rejects those landing inside an existing
disk. FPS gives approximately the same blue-noise coverage on a fixed point set
and is **deterministic and exactly rotation-invariant**, ties broken by vertex
index. Dart-throwing depends on an RNG stream unrelated to the geometry, so two
views of one fragment would select different vertices — unusable when the whole
design rests on a fragment producing the same tokens in every pose.

**Metric: geodesic, i.e. distance along the fracture surface itself.**
Shortest path on the mesh edges whose *both* endpoints lie on the break, so a
path never shortcuts across the original exterior. This is the discrete
stand-in for a true geodesic; it overestimates, because a path must follow
edges rather than cross faces, by a roughly uniform factor that FPS — which
only ever compares distances — is indifferent to.

The metric matters because a fracture surface is a thin strip wrapping a curved
fragment. Straight-line distance measures *through the material*, so two points
on opposite arms of a folded strip read as neighbours and the sampler declines
to place tokens on both. Measured on a U-shaped strip whose arms are 0.10 apart
in space and up to 2.24 apart along the surface:

| tokens | metric | min separation | coverage radius |
|---|---|---|---|
| 6 | euclidean | 0.271 | 0.588 |
| 6 | **geodesic** | **0.576** | **0.545** |
| 12 | euclidean | 0.169 | 0.396 |
| 12 | **geodesic** | **0.291** | **0.286** |
| 24 | euclidean | 0.136 | 0.162 |
| 24 | **geodesic** | **0.158** | **0.158** |

Minimum separation is the disk radius — it is the property "disk sampling"
names — and geodesic roughly doubles it where the budget is tight, which is
exactly the regime a per-scene budget puts a large scene in.

**Honest caveat.** On a *near-closed ring* — a shard whose break runs almost the
whole way round — Euclidean measured slightly better coverage at low token
counts (0.449 vs 0.743 at k=8), while geodesic still won on minimum separation
at higher counts. Greedy FPS on a one-dimensional path subdivides dyadically, so
gaps come out in a 2:1 ratio rather than uniform; on a ring the Euclidean metric
happens not to be path-like and dodges that. `metric="euclidean"` stays
available as the ablation.

**Cost is not the reason to prefer either.** Dijkstra is `O(E log V)` per seed
while Euclidean FPS recomputes an `O(n)` distance column per seed, so geodesic
is *cheaper* on large fragments and only slower on small ones, where everything
is fast:

| fragment | fracture verts | tokens | euclidean | geodesic |
|---|---|---|---|---|
| median, 296 v | 125 | 20 | 0.2 ms | 0.9 ms |
| p75, 642 v | 200 | 128 | 1.2 ms | 3.2 ms |
| large, 2562 v | 1200 | 128 | 4.6 ms | 5.3 ms |
| p99, 10k v | 4000 | 128 | 13.4 ms | **7.9 ms** |

A whole median 3-fragment `build_scene` goes from 2.6 ms to 3.2 ms.

**Disconnected patches fall out for free.** A fragment that broke from three
neighbours has three separate fracture patches, at infinite surface distance
from one another, so `argmax` seeds every patch before refining any — the right
priority, with no special case. Euclidean needs an explicit connected-component
split to get the same behaviour, which is what `mode="patch"` does and
`mode="sample"` did not.

### Sampling is per fragment, never over the pooled scene

The obvious-looking alternative — pool every fracture vertex in the scene and
run one farthest-point pass over the union — is wrong here. At input time the
fragments sit in arbitrary perturbed poses, so a global pass would select points
according to where the pieces happened to have been thrown: a different sample
for every perturbation of the same scene, and no equivariance left. The budget
is *split* per fragment and the sampling runs *inside* each one, which keeps
every choice a function of that fragment's own shape.

---

## 12. Training — what the engine guards against

`reassembly.training`, one module: config, dataset, loops, schedule,
checkpoints, metrics, two-GPU DDP. `tests/test_training.py` covers it.

### The test that matters

`test_the_model_can_actually_learn` memorises six **synthetic** scenes —
icospheres with random dents, not Breaking Bad — and drives the *training*
rotation error from 126.5° to 23.8°.

That separates "are the conventions self-consistent and can the architecture fit
anything at all" from "does it generalise", and only the first is answerable
without the dataset. A transposed rotation label, a head whose frame points the
wrong way, or a loss comparing misaligned rows fails here and passes every other
test in the repo. That is its whole value.

**What it is not.** An earlier draft of this section described 23.8° as clearing
"the 89.9° axis-only floor the previous design never cleared". That was wrong,
and wrong in the specific way this document keeps warning about: the previous
VN-GAT reached **43° on training data** and stalled at ~94° on *validation*.
89.9° is a validation floor. Comparing a training number to it is meaningless —
training error was never what was stuck there, and the old model also got well
below 89.9° in training. The claim has been withdrawn. Whether the
cross-fragment redesign actually clears that floor is unmeasured, and only a
validation run on Breaking Bad can answer it.

### Choices made against a specific failure

| choice | the failure it prevents |
|---|---|
| Chance printed in the banner | An epoch at 126.5° reads as "needs more training" instead of "has learned nothing" |
| Loss-at-init checked against references | A term far from its reference is measuring something other than its name, and shows up later only as slow convergence |
| Schedule is a pure function of the step | `CosineAnnealingLR` is *periodic* — past `T_max` the rate climbs back — and its `T_max` lives in its own state dict, so `--epochs` on resume may not change it |
| `last.pt` written unconditionally, `best.pt` separately | A policy-gated checkpoint can freeze while training continues; resume then discards the gap silently |
| Checkpoint carries optimiser, RNG, step, history | Weights alone restart the schedule and throw away the moments |
| Dropped samples tallied **by name** | An item failing every epoch has been removed from the dataset; a per-batch counter cannot see it, since a batch of four with one bad item still collates |
| fp32 default, AMP opt-in | In fp16 an attention product overflows to `inf` past ~256, and the max-subtracting softmax then computes `inf − inf = NaN` — the stability trick manufactures the NaN |
| `find_unused_parameters=True` under DDP | A batch whose scenes all lack a fracture surface skips the cross layers, leaving their parameters gradient-less — a hang six hours in |
| Verdicts that refuse to flatter | A result at chance, one at the axis-only floor, and one still descending at its cutoff are each perfectly reportable numbers meaning something else |

### Two things the build surfaced

**There is no translation setting, and that is deliberate.** `build_scene`
applies rotation only and then centres every fragment at its own centroid, which
removes any translation that had been applied. A `translation_std` knob would
change nothing. Stage one predicts rotation; translation is solved
geometrically in stage two from the centroids `Normalized` keeps.

**The embedding head is trained by exactly one term.** With
`supervise_embedding=False`, or on a scene whose fragments share no vertices,
the embedding-consistency loss is absent and the readout plus its MLP receive
*no gradient at all* — while the rotation loss descends exactly as before. It
matters because stage two matches interface points by mutual nearest neighbours
in embedding space, so the result is a rotation model that works and a
translation solver with nothing to use. `Config` says so at startup and
`test_without_coincidence_labels_the_embedding_head_is_untrained` pins it.

### Multi-GPU on Kaggle

`devices=2` spawns one process per T4. It must be launched from a **file**:
torch's spawner re-imports `__main__` in each child, and a notebook cell has no
file to import, so the children die with a `FileNotFoundError` on `<stdin>`
that says nothing about the cause. `train()` detects that and raises with the
`%%writefile` recipe instead of letting the spawn fail obscurely.

### Surviving Kaggle's 12-hour cap

A full training set is far more than one session, so the run has to be a chain
of sessions rather than one long job. Six things make that work, and each exists
because its absence loses hours silently.

| | |
|---|---|
| `max_hours = 11.0` | Stops cleanly inside the batch loop with a checkpoint written, rather than being killed at 12 with the epoch's work gone |
| `checkpoint_every_minutes = 30` | An epoch over the full set can outlast a session. Checkpointing only at epoch boundaries would then never checkpoint at all |
| `completed` flag | A mid-epoch save records the epoch as *incomplete*, so resume re-runs it instead of skipping the part that never ran |
| `resume_from` | Loads from a different directory than it saves to. On Kaggle these are always different: you write to `/kaggle/working`, and the next session mounts that output read-only under `/kaggle/input` |
| SIGTERM handler | Kaggle terminates the process. The handler only sets a flag — saving from inside a signal handler risks a half-written file and would desync DDP ranks — and the loop checks it at the next batch boundary |
| `elapsed` | Cumulative training seconds across *all* sessions, so a run spanning four of them still knows how long it has actually trained |

**The resume point is copied into the new `out_dir` immediately.** Two failures
otherwise break the chain, both silently: a session killed during its first
epoch leaves `out_dir` empty, and a session with nothing left to do writes
nothing at all. Either way the *next* session resumes from an empty directory,
starts from scratch, and discards every hour spent — announcing only "starting
from scratch", one line into a long log.

**A config change on resume is caught, and the two kinds fail differently.** A
changed `channels` makes `load_state_dict` raise a shape error naming a tensor,
which takes a while to trace back to the flag; that is refused outright. A
changed `label_method` or `tokens_per_scene` raises *nothing* — training simply
continues on a different problem than the weights were trained for, and the loss
curve has a step in it that reads as noise. That is warned about loudly. A
changed `epochs` rescales the whole learning-rate curve, since the schedule is a
function of `(step, total_steps)`; also warned.

**RNG state is restored, not merely saved.** It was saved and never restored for
a while, which is the same as not saving it: a resumed run draws a different
perturbation stream from an uninterrupted one, so the two diverge and neither is
reproducible. Restoration is best-effort — a checkpoint from another torch build
can carry a state this one rejects — and says so when it fails, because the
stream then genuinely differs.

**A partial epoch appears twice in the history**, once as the interrupted
attempt and once as the re-run. Both rows are kept, because the interrupted
epoch's metrics are real measurements, and the first carries `partial = 1`.
Filter on `partial == 0` for a clean curve.

The session banner projects what remains once an epoch has been timed:

```
  ~1:04:12/epoch, 31 epoch(s) left = ~33:10:12  (4 more session(s) at 11h)
```

### Preflight

`--preflight` exists because the test suite and the target machine share almost
nothing. The suite runs on synthetic icospheres on CPU; Kaggle runs on real
Breaking Bad meshes on two T4s. Everything that differs is a place a session can
be lost, and every one of them is cheap to check first:

- the dataset path finds nothing, or finds it but no official split file, so the
  run quietly uses a hashed split and produces numbers that cannot be compared
  to anything published;
- an 83,039-vertex fragment that the tests never see does not fit in 16 GB;
- the dihedral mask comes out empty on real geometry, so the cross-fragment
  layers have nothing to attend over and the architecture's whole contribution
  is silently absent;
- an epoch turns out to take four hours, which changes the plan entirely and
  cannot be guessed before one is timed.

It found a real defect on its first run against synthetic data:
`embedding.3.bias` received no gradient. That turned out not to be a bug but a
**provable redundancy** — the embedding is supervised only by the
centroid-variance loss, and adding a constant to every embedding shifts them all
equally, so the scatter about each cluster centroid does not move. It is
unidentifiable for stage two as well, since a global shift leaves every pairwise
distance untouched. The bias was removed: a parameter that cannot be learned
would otherwise show up as a dead gradient in every check forever.

### Two bugs the first real preflight found

Both on a 1,006-object local dataset, in the three minutes before a session
would have started.

**16 objects appeared in both train and val.** `load_official_split` took
`Path(line).name` — the object directory name, discarding the category.
Breaking Bad lists `everyday/<category>/<object>` and object directory names
are *not* unique across categories, so a name present in both lists put the
same scene in both splits. That is the worst shape a bug can take: validation
becomes partly a memorisation test, so the number it produces is **better** than
the honest one and reads as success. Matching is now on `(category, object)`,
with a per-entry fallback for split files that list bare names, and
`BreakingBadScenes` raises if any pair of splits still intersects — checked in
the loader rather than only in preflight, because a preflight is skippable and
this must not be.

**Out of memory at `batch_size=4` on a 15.6 GB T4.** Measured where it went:
on a batch of 81k vertices and 488k directed edges at `channels=64`, one
intra-fragment attention layer retained **2,961 MB** for the backward pass, and
1.16 GB of that was pure concatenation temporary — `cat([x_src, x_dst,
edge_attr])` at 768 MB and `cat([x_src, edge_attr])` at 393 MB, built only to be
multiplied.

`VNLinear` has no bias, so `W [a;b;c] == W_a a + W_b b + W_c c` exactly. The
layer now applies separate projections and sums them, which removes both
concatenations *and* lets each projection run on the N **nodes** before being
gathered to the E edges rather than after — one sixth of the work on a triangle
mesh. Measured: **2,961 → 2,091 MB per layer**, 29%, 3.5 GB across the four
intra layers. `fan_in` on `VNLinear` keeps each piece's initialisation identical
to the single wide layer; without it, splitting silently rescales it.

Two things follow that the arithmetic alone would not have given:

- The saving is real but not sufficient on its own. Preflight now **halves the
  batch until something fits** and reports the largest that does, with the
  `--accumulate` needed to keep the effective batch — "out of memory" alone
  leaves the useful question unanswered.
- Preflight samples a dozen scenes; the dataset's largest single fragment is
  83,039 vertices, several times anything it will have seen. So the training
  loop now catches `OutOfMemoryError`, empties the cache, and skips that batch
  with a count and a size — one pathological scene must not end an eleven-hour
  session. It is counted and reported rather than swallowed: if it fires often,
  the batch size is wrong and the largest objects are being dropped from
  training.

### What the second preflight run found — the real ceiling

With the split fixed, preflight got further and reported something more useful
than a pass: `batch_size=1` fits, at **11.61 GB of 15.6 GB (74%)**, and then the
next step died allocating 5.16 GiB inside the cross-fragment layer. Two separate
problems, one mine and one architectural.

**The crash was a preflight bug.** Step 5 measures the largest batch that fits;
step 6 then timed an epoch — but step 5's batch, its loss, and the autograd
graph holding both were still alive when step 6 allocated its own. The tool
built to catch out-of-memory was itself leaking a batch. It now frees the batch,
zeroes the gradients and empties the cache between steps, and sizes step 6 from
what step 5 proved fits rather than guessing.

**The 74% was the real finding.** Where it goes, measured with
`saved_tensors_hooks` as the slope against pair count rather than as a total —
a total is dominated by the `(T, C)` tensors at small sizes and would have said
nothing about the case that matters:

```
  q, k, v gathered to the pair dimension   1109 B/pair
  logits and weights only                    86 B/pair
```

At 2048 tokens over 6 fragments — 3,496,618 pairs — that is **3.6 GB against
0.3 GB per cross layer**, 7.2 GB against 0.6 GB across the two of them, for a
*single scene* on a 16 GB card. The cross-fragment layers were most of the
budget.

The fix is `torch.utils.checkpoint` around the two gathers, so the backward pass
recomputes them. What is recomputed is indexing and a dot product, not the
projections — those are `(T, heads·head_dim)` and stay resident either way — so
the compute cost is close to nothing while the memory falls by 13×. Outputs and
gradients are **bitwise** identical, and own-pose equivariance and other-pose
invariance are unchanged at 1e-16; all three are pinned by tests, along with the
bytes-per-pair saving itself, so a future edit that reintroduces a pair-sized
retained tensor fails rather than silently OOMing on Kaggle.

It is on by default (`checkpoint_cross=True`) because the trade is that
lopsided. The intra-fragment equivalent is off (`checkpoint_intra=False`):
there, recomputation re-runs the projections themselves, which is real work.

#### The retraction that goes with it

`tokens_per_scene = 2048` was chosen by comparing the resulting pair count to
the token counts GARF's stack processes. That is a **FLOPs** argument, and GARF
runs on 4×H100. It was never checked against T4 *memory*, which is the binding
constraint here and is quadratic in exactly the quantity being set. The number
survives — with the gathers recomputed, 2048 tokens fit — but it survived by
luck, not because the check had been done. The ordering to prefer, when
`batch_size=1` still will not fit, is `--checkpoint-intra` first (free but for
compute), then `--channels 32`, and only then `--tokens-per-scene 1024`, because
the last one changes what the model can represent while the first two do not.

### The third preflight run, and the bug that kept coming back

With the cross layers checkpointed, `batch_size=2` fits at **11.96 GB of
15.6 GB** — the two cross layers had been most of the previous 11.61 GB at
`batch_size=1`. Then step 7 died in `loss.backward()`.

It was the same bug for the third time, in a third place. The shape of it:

| | |
|---|---|
| Step 6, run 2 | Allocated while step 5's batch and graph were still alive |
| Step 7, run 3 | Timed at `config.batch_size`, which step 5 had *just proved* does not fit |
| Step 6, run 3 | Sized from `fitted` — then doubled it, to `fitted * 2` |

The third one was mine, introduced by the fix for the first. The reasoning was
that step 6 runs under `no_grad` and so stores no autograd graph, which is true
and still not a proof; `fitted * 2` died the moment `fitted` was 2. **Preflight
is the one function in this project that must not estimate memory** — it exists
because estimates are what fail on Kaggle.

So the rule is now explicit rather than repeatedly rediscovered: after step 5,
`fitted` is a ceiling that no later step may exceed, and each step that
allocates catches `OutOfMemoryError` and degrades instead of raising. A
preflight that dies of its own memory use is worse than no preflight, because
it throws away every check that had already passed.

The reason this survived three rounds is that **`preflight` was the one function
the test suite could not reach**: it needs a dataset on disk in the real
Breaking Bad layout, so every bug in it was found by spending a Kaggle session.
`tests/test_training.py` now builds one — a handful of icospheres labelled into
angular sectors, each fine vertex its own cell so the cell matrix is the
identity — and runs the whole of preflight against it with `_forward`
monkeypatched to raise `OutOfMemoryError` above a chosen batch size. CPU has no
memory ceiling to hit, but the control flow under test is the real one, and it
fails on all three bugs above.

One more thing it found, unrelated to memory: when a split came out empty,
preflight printed advice about fixing `--root` — but `--root` was *correct*, or
no scenes would have been found at all. Wrong advice sends someone the wrong
way for longer than no advice does. The two cases are now distinguished.

#### The defaults changed

`batch_size` is now **2** with `accumulate=2`, rather than 4 with no
accumulation. The effective batch is identical — `2 × 2 × 2 devices = 8`, the
same as `4 × 1 × 2` — and the LR schedule is expressed in forward passes rather
than optimizer steps, so warmup does not shift. The old default was not chosen
by measurement; the new one is, on the actual card.

What is *not* settled: 11.96 GB is 76% of the card, measured on the largest of
twelve **sampled** scenes at 20,777 vertices, and the dataset's largest single
fragment is 83,039. Vertex-linear terms grow with it, though the cross-attention
term does not (tokens are capped at 2048 regardless of mesh size). So OOM-skips
on the biggest objects are likely. Training counts and reports them rather than
swallowing them; if that count is more than a percent or two of an epoch, the
answer is `--batch-size 1 --accumulate 4`, because silently dropping the largest
objects from training is a bias in the result, not a performance detail.
