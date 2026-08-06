# Architecture

## The two-stage split

A rigid placement is a rotation and a translation. This project learns only the
rotation and computes the translation classically.

```
scattered fragments
        │
        ▼
  ┌───────────────────────────────┐
  │ STAGE 1 — learned             │   VN-GAT backbone, SO(3)-equivariant
  │   per-fragment rotation       │   → R_pred  (F, 3, 3)
  │   per-vertex/edge embedding   │   → invariant descriptors
  └───────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────┐
  │ STAGE 2 — classical           │   mutual-NN matching in embedding space
  │   interface correspondences   │   → weighted Laplacian solve + IRLS
  │   per-fragment translation    │   → t  (F, 3)
  └───────────────────────────────┘
        │
        ▼
   assembled object
```

**Why split it.** Once fragments are correctly oriented, translation has a
closed-form optimum given correspondences (`docs/CHANGES.md` §5) — exact to
1e-15, no optimiser, no local minima. Spending network capacity regressing a
quantity that can be solved exactly is a poor trade. It also makes failure
legible: a bad result is attributable to stage 1 or stage 2, and
`scripts/evaluate.py` has a mode that isolates each.

---

## The algebraic fact everything rests on

Vertex positions are **centralised per fragment** — each fragment's centroid is
subtracted before the network sees anything.

Under a rigid transform `x' = R·x + t`, the centroid moves by exactly the same
amount, so in centralised coordinates the translation cancels:

```
x̃' = R·x̃
```

Two consequences:

1. The rotation target is well defined independently of translation. `R_gt` is
   exactly `R_diffuse᷀ᵀ`.
2. The network never sees absolute position, so it cannot learn dataset-specific
   placement shortcuts.

`scripts/smoke_test.py` step 2 checks this holds on real data (expect ~1e-5 in
fp32). If it does not, the rotation supervision is not what the code assumes and
nothing downstream is trustworthy.

Centralisation discards the centroid, which is the ground-truth *placement* —
so it is carried alongside as `fragment_centroid`, without which no translation
metric is computable.

---

## Stage 1: the backbone

### Vector Neurons

Standard networks map scalars to scalars. Vector Neurons lift every feature to a
list of 3-vectors, `(N, C, 3)`, and restrict operations to those commuting with
rotation:

- **Linear** — mixes the `C` axis only, never the `3` axis. `f(Q·x) = Q·f(x)`.
- **LeakyReLU** — the nonlinearity acts on the component of `x` along a learned
  direction, both rotating together, so the relationship is preserved.
- **LayerNorm** — normalises per-channel magnitudes, which are invariant.
- **Invariant readout** — inner products between vector features. Rotations
  preserve inner products, so the output does not change under rotation. This is
  what produces the matching descriptors.

Equivariance is a structural property here, not something trained for.
`tests/test_vn_layers.py` asserts it for each primitive.

### Message passing (`models/vn_gat.py`)

A GAT layer over the mesh graph. The richer edge features are one of the
thesis's contributions over GARF:

```
edge_attr = [ length | midpoint | face_normal_1 | face_normal_2 ]
              1        3          3               3               = 10
```

Attention logits are computed from **invariant** quantities (norms and inner
products); messages are **equivariant** vectors. Attention is a scalar weight
over vector messages, so a rotated input yields a rotated output with unchanged
weights.

Edges are stored bidirectionally with an explicit `is_forward_edge` mask. The
backward copy swaps the two face-normal slots — the "left" and "right" face of
an edge depend on traversal direction. The mask is **not** a positional half
slice: after merging fragments the layout is per-fragment `[fwd; bwd]` blocks,
so code slicing the first half is silently wrong for two or more fragments.
`tests/test_pipeline.py::test_is_forward_edge_is_not_a_positional_half_slice`
pins this down.

### Cross-fragment communication (`models/virtual_nodes.py`)

Mesh edges never cross a fragment boundary, so message passing alone can never
let fragments see each other. Virtual nodes provide that channel in three
stages:

```
stage 1 (up)     vertices  →  K slots per fragment      within-fragment
stage 2 (across) slots     →  slots                      within-SCENE
stage 3 (down)   slots     →  vertices                   within-fragment
```

Stages 1 and 3 use **segment attention** — each virtual node attends only over
its own fragment's vertices via scatter-softmax, so the `(F·K, heads, N)` score
tensor is never built. Bit-identical to the dense masked version it replaced,
at 20× less memory (`docs/CHANGES.md` §2).

Stage 2 is masked by `fragment_scene_id`. This is load-bearing: without it, one
scene's fragments attend to another scene's in the same batch, and the model
gets information it will not have at inference.
`tests/test_virtual_nodes.py` tests both that the leak is absent when scene ids
are supplied *and* that it is present when they are not — the second test
documents that passing `None` is safe only for genuinely single-scene input.

### The rotation head

Two predicted vectors → Gram–Schmidt → a proper rotation. The head returns the
**transpose** of that frame, because the frame is left-equivariant while the
target transforms on the right. This is the subject of `docs/CHANGES.md` §1 and
is the single most important line in the model.

---

## Stage 2: assembly

Pure numpy and scipy — no torch. `reassembly.assembly` and
`reassembly.evaluation` import and run in a torch-free environment, which is why
63 tests execute without a GPU or a deep-learning install.

```
R_pred + invariant embeddings
        │
        ▼
  mutual nearest neighbours in embedding space
  + Lowe ratio test
  + normal-opposition filter          ← E_normal, applied where it has force
  + distance ceiling                  ← anchored on the true-match scale
  + min matches per pair              ← mutual-NN is never empty
        │
        ▼
  connected components of the correspondence graph
        │
        ▼
  weighted Laplacian solve  L·t = c   ← closed form, one anchor per component
  + IRLS reweighting                  ← 130× better at 40% outliers
  + optional collision refinement     ← genuinely non-quadratic, so last
        │
        ▼
  translations, and an honest report of what was underdetermined
```

`TranslationResult` carries `num_components`, `fully_constrained`, and
`residual_rms`. A fragment nothing matched against is *unconstrained*, not
badly estimated, and the distinction is reported rather than hidden behind a
plausible-looking zero.

---

## Invariants the code depends on

Breaking any of these produces silent wrongness rather than an exception. Each
has a test.

| Invariant | Why it matters | Test |
|---|---|---|
| Edges never cross a fragment boundary | `build_predictions` uses it instead of carrying a per-edge fragment id | `test_pipeline.py::test_edges_never_cross_a_fragment_boundary` |
| `num_fragments` is explicit, never `max()+1` | an empty fragment never appears in `fragment_id`; every per-fragment tensor would misalign | `test_pipeline.py::test_merge_carries_an_explicit_fragment_count` |
| `fragment_scene_id` is present for multi-scene batches | otherwise stage 2 leaks across scenes | `test_virtual_nodes.py::test_no_cross_scene_leakage_when_scene_ids_are_supplied` |
| Positions are centralised | the rotation target is only well defined in centralised coordinates | `test_pipeline.py::test_centralization_isolates_rotation_from_translation` |
| The rotation head returns `Fᵀ` | equivariance must match the target's transform law | `test_model.py::test_rotation_head_matches_target_law` |
| Every parameter receives a gradient | DDP with `find_unused_parameters=False` refuses otherwise | `test_model.py::test_every_parameter_receives_a_gradient` |
| Fragment count is stable across `full`/`frac` | fragment indices are positional; dropping one misaligns rotations | guard in `data/dataset.py` |

---

## Memory model

Peak memory is roughly

```
  vertices × hidden_channels × 3 × layers × bytes_per_element
+ edges    × hidden_channels × 3 × layers × bytes_per_element
+ fragments × slots × hidden_channels × 3
```

with mesh graphs having ~6 directed edges per vertex, so the edge term
dominates.

The four levers, in order of effect:

| lever | effect | cost |
|---|---|---|
| `data.decimate_to` | linear in vertex count | geometric detail |
| `model.gradient_checkpointing` | ~4× less activation memory | ~30% more compute |
| `train.amp` (fp16) | 2× on every activation | numerical range |
| `model.hidden_channels` | linear | capacity |

`scripts/profile_memory.py --sweep` measures all of them on your GPU rather
than trusting the estimate. It was written because the original OOM was not
reproducible on demand — it depended on which scene got drawn — and a
measurement beats an anecdote.

T4 is Turing: **fp16 yes, bf16 no**. `configs/kaggle_t4x2.yaml` sets
`amp_dtype: fp16` for this reason.

---

## Where things live

```
src/reassembly/
  config.py              typed config, YAML + --set overrides
  data/
    splits.py            deterministic hash split, cached scene index
    scene_io.py          mesh loading, duplicate-face resolution
    decimate.py          shared-grid voxel decimation
    mesh_ops.py          face adjacency, fracture-surface extraction
    correspondence.py    cross-fragment vertex/edge matching
    features.py          node and edge feature construction, centralisation
    augment.py           the scattering transform
    collate.py           fragment merge and scene batching
    dataset.py           the Dataset itself
  models/
    vn_layers.py         Vector Neuron primitives, rotation head, rotation losses
    vn_gat.py            equivariant GAT layer
    virtual_nodes.py     segment attention, three-stage communication
    angular.py           triplet/angular block (off by default)
    vn_gat_model.py      the assembled model
  training/
    losses.py            composite loss, cluster consistency
    bridge.py            dataset ↔ model tensor plumbing
    engine.py            epoch loop, AMP, accumulation, OOM recovery
    distributed.py       DDP setup, collective vote, metric reduction
  assembly/              stage 2 — pure numpy
    matching.py          mutual-NN correspondence discovery
    translation.py       Laplacian solve, IRLS, collision refinement
  evaluation/
    metrics.py           RMSE(R) both conventions, RMSE(T), PA, CD
  utils/
    memory.py            AMP context, OOM detection, memory estimates
```
