# What is worth taking from `E:\Thesis 1` (VN-GAT)

Full read of all 71 source files of the previous design. Written as a
decision list; several items have since been implemented (see
`docs/PROJECT-STATE.md` §10), and §1.1 has since been measured and corrected. The virtual-node machinery is excluded
per instruction — but note that `vngat/models/virtual_nodes.py` contains one
*argument* (not a mechanism) that our cross-attention already relies on, and it
is worth reading for the statement of it, not for the code.

Ranking is by expected effect on the current symptom (405 objects, train ≈ 4.15,
val ≈ 5.35, rotation error at chance), not by how much work each item is.

---

## Tier 0 — defects in *our* current code that the old version already fixed

These are not "ideas to consider". They are bugs I can point at in
`src/reassembly/training.py` today.

### 0.1 A non-finite loss reaches `backward()` before we check it
`training.py` runs the forward *and* `scaled.backward()` inside the `try`, and
only then tests `if not torch.isfinite(loss)` (line ≈782). By that point the
NaN has already been written into `.grad` for every parameter, and under DDP a
syncing micro-batch has already all-reduced it to the peer. The counter says
"skipped"; the optimizer state says otherwise, and Adam's moments never
recover. The old trainer checks finiteness *before* the backward, and — because
it ran AMP — retried the same scene in full precision before giving up, on the
principle that skipping "KEEPS THE DATA, instead of silently excluding whole
objects".

**Take:** move the finiteness check above the backward. (The fp32 retry is moot
for us: we do not use autocast at all.)

### 0.2 An OOM zeroes the whole accumulation group but leaves `group_weight` intact
In the OOM handler we call `model.zero_grad(set_to_none=True)` (line ≈760),
which discards the gradients of every micro-batch accumulated so far in the
group — but `group_weight` is only reset at the optimizer step (line 803). The
next step therefore divides the surviving gradients by a denominator that
includes fragments whose contribution was thrown away, silently shrinking the
update by up to the discarded fraction. Single-GPU only (DDP stops instead).

**Take:** reset `group_weight = 0.0` in the OOM branch, or don't zero the group.

### 0.3 `E_normal` does not depend on translation — DesignV5 §1.10 is wrong as written
`vngat/assembly/translation.py`, module docstring: normals are translation-
invariant, so `∂E_normal/∂t ≡ 0`. Minimising `E_pos + λ₁E_normal + λ₂E_collision`
jointly over `t` is the same optimisation as minimising `E_pos` alone, and
reporting the normal term as part of a "solved" energy is misleading. What the
term actually measures is whether a *correspondence* is plausible — matching
fracture surfaces should face each other — so it belongs as a per-match weight:

```
normal_compatibility = ((1 - cos(n_src, n_dst)) * 0.5).clamp(0, 1)
```

With that, `E_pos` alone is a weighted graph Laplacian with a closed-form
solution (gauge-fixed by `L + 1/N` plus a zero-mean projection, in float64,
robustified by 5 IRLS iterations with a Huber weight). No alternation, no
learning, no gradient descent.

I reproduced V4's joint energy into DesignV5 §1.10 without checking it. That
section needs rewriting whatever else is decided.

### 0.4 Object double-counting in our split
Our preflight reports 809 train / 181 val for Everyday; the official partition
is 407 / 91. Cause confirmed in our code: `Config.subsets` defaults to `None`,
so `find_scenes` accepts `volume_constrained-everyday_compressed` alongside
`everyday_compressed`, and `filter_by_split` matches both copies against the
same official entry. We *have* `object_key`/`base_subset` dedup in
`data/paths.py` — it is simply not applied on this path. The old
`load_official_split` **raises** when one entry matches more than one
directory, with the message that the vanilla and volume-constrained copies
share object ids.

**Take:** either default `subsets=["everyday_compressed"]` or dedup by
`object_key` after `filter_by_split`, and make an ambiguous match an error, not
a silent duplication. Every per-epoch number we have quoted is over a dataset
twice the size we thought.

### 0.5 There is no official test split
The old loader raises on `split == "test"`: Breaking Bad ships train and val
lists only. My earlier command list told you to run `--evaluate --split test`.
That falls back to our hash split, which is not comparable to anything
published.

### 0.6 The Euler-RMSE convention we report is not GARF's
Ours: `_euler_xyz(pred) − _euler_xyz(target)`, componentwise.
Theirs (`vngat/evaluation/metrics.py`): Euler angles of the **residual**
`Rᵀ_pred R_gt`.

Measured over 200k Haar-random pairs:

| convention | random vs random | identity prediction |
|---|---|---|
| ours | 86.26° | 83.08° |
| residual (GARF) | 83.22° | 83.08° |

The "predicting the identity beats random guessing by 3°" trap — which is a
centrepiece of our reference table, DesignV5 §1.8 and the training banner —
is an artefact of our convention and vanishes under the residual one.

### 0.7 The 89.9° axis-only floor is refuted by measurement
Their `metrics.py` carries the counter-argument explicitly, and a scaling run
on 8 Everyday objects (bottles, bowls, mugs — surfaces of revolution) reached
**30.90°** training error, well below the floor we quote. The reason:

> A FRAGMENT of a symmetric object is not itself symmetric: its fracture
> boundary is jagged and unique.

**Take:** drop `SYMMETRY_FLOOR_*` from the reference table, or demote it to "a
floor for the *intact* shape, not for a fragment".

---

## Tier 1 — candidates for the current plateau (1.1 since measured and ruled out)

### 1.1 ~~The embedding term owns the gradient budget~~ — measured, and it does not

*Corrected Sept 2026.* This section originally claimed that our InfoNCE term,
at 56% of the loss **value** at initialisation, dominated the gradient budget
the way the old design's unnormalised embedding loss had. That was an argument
from value share, not a measurement, and the measurement contradicts it.

Each term back-propagated alone, gradient norm taken on the parameters every
term shares (everything below the rotation and embedding heads), median over
six model seeds, on two closed solids sharing a rough fracture interface:

| term | share of the loss value | share of the backbone gradient |
|---|---|---|
| rotation | 17% | **54%** |
| face normals | 15% | 22% |
| embedding (InfoNCE) | **54%** | 13% |
| vertex normal | 7% | 8% |
| position | 6% | 5% |

So at initialisation rotation already drives the shared layers, and the
embedding term is not why rotation stays at chance. Two limits on that
conclusion: it is one synthetic geometry, and it is a statement about the
start of training — the shares can move once the heads have learned something.
The old design's own 63%/15% figure was about its *unnormalised* loss, before
its fix; after the fix its two embedding terms carried about 1% of the backbone
gradient on the same geometry.

### 1.2 Three named degeneracies of an interface-embedding loss
From `vngat/losses/composite.py`, all three *observed in training*:

1. `‖Σ z‖²` (the V4 formula) is minimised by embeddings that **cancel**, not
   ones that agree.
2. Plain within-cluster variance is zero for **any constant** embedding — a
   real run collapsed to a single vector within two epochs.
3. Adding a `push` term on **unnormalised** embeddings just moves the
   degeneracy: separating centroids by inflating magnitude is easier than
   arranging them. A real run drove mean centroid norm to **565** while the
   loss still read 0.56 → squaring overflows fp16 → NaN bursts from epoch 19.

Fix: L2-normalise to the unit sphere first. Distances then lie in [0,2], the
loss is bounded by construction and therefore *cannot* dominate the gradient
budget, and it is the right space for the inference-time matcher anyway.

We use InfoNCE, which is immune to (1) and (2), so this is mostly confirmation
that our choice was right. It is **not** bounded the way theirs is, and it has a
property worth knowing: a collapsed embedding is *not* its worst value.
Measured on the same 260 clustered vertices, old pull/push against InfoNCE:

| embeddings | old pull/push | new InfoNCE |
|---|---|---|
| random (untrained) | 0.39 | **7.29** |
| collapsed (all identical) | 1.00 | **5.56** |
| perfect clusters | 0.00 | 0.05 |

Random embeddings score *above* collapse, so an InfoNCE value falling early in
training is compatible with the embeddings collapsing. `match@1` is the number
that cannot be fooled that way, and it is already reported every epoch.

Also worth stealing verbatim: `push` is subsampled above 512 clusters (the only
O(C²) term), and it deliberately does **not** use `torch.cdist(x, x)`, whose
diagonal is an exact zero distance with a 0/0 gradient — masking the diagonal
out of the *forward* does not stop the backward producing NaN.

### 1.3 `lr_monitor: rot`, not `total`
> *"Watching the composite total is what killed a real run: face + norm + rot
> made up 4.95 of a 5.40 total and none of them moved, so every epoch
> registered as a plateau and the rate was halved nine times, ending 512× below
> where it started."*

Our LR is a pure function of the step (linear warmup → cosine), so we are
immune to this particular failure — but the diagnosis applies to *us reading
the number*: a total dominated by terms that cannot move is not a progress
signal. Our train 4.15 / val 5.35 may be almost entirely `face + norm`.

### 1.4 `steps_per_epoch` / `val_steps` decouple "epoch" from dataset size
Their configs use `steps_per_epoch: 80`, `val_steps: 16`. An epoch is a fixed
number of optimiser steps drawn at random, not a pass over 405 objects. This
makes checkpoint cadence, LR schedule and time projection independent of how
many objects are in the split — and would have made our time-projection bug
(2.7× under-estimate) impossible to write.

### 1.5 `input_source: 'full' | 'frac'` as a first-class ablation axis
Feed the pruned fracture surface, *always evaluate on the full mesh*, so the
two modes are directly comparable. One flag, one config file, one table row.

---

## Tier 2 — diagnostics we do not have

### 2.1 `swing_twist_error` — the single most valuable one
Splits the residual rotation into **tilt** off the object's symmetry axis and
**twist** about it. Reported, never optimised. It separates two failure modes
that look identical in a mean geodesic error:

- tilt ≈ 90 → the model has not learned the axis either (an optimisation
  problem: keep training, change something)
- tilt ≈ 0, twist ≈ 90 → the axis is learned and the azimuth is genuinely
  unidentifiable (a structural limit: no amount of compute moves it)

That distinction decides whether to redesign or keep training. Their trainer
prints a verdict line and `scripts/evaluate.py --by_category` breaks it down by
object category, so "symmetric categories score worse than asymmetric ones"
becomes direct evidence rather than a hypothesis. Implementation note: the tilt
must **not** go through `arccos`, whose clamp reports 0.026° where the true
value is 0 — precisely the regime the metric exists to detect.

### 2.2 `head_cos` — |cosine| between the rotation head's two output channels
Computed under `no_grad`, printed in the epoch table, ~free.

> *"Two seeds of an otherwise identical 8-object run ended 74 deg apart (30.9
> vs 104.7), and this is the cheapest measurement that separates 'bad basin'
> from 'still descending'."*

A stalling run shows |cos| near 1 (the two channels have become parallel, so
Gram–Schmidt is resolving a near-degenerate frame); their healthy runs sat at
0.78–0.90. This is directly relevant to us: our head is the same 6D →
Gram–Schmidt construction.

### 2.3 Part Accuracy + Chamfer — the actual Breaking Bad headline metrics
We have neither. Chamfer must be computed in float64 on squared distances:
`cdist`'s matmul expansion scores *identical* clouds at ~2e-7 against a 0.01
threshold, so a naive implementation reports part accuracy that looks
suspiciously good. Their `part_accuracy` also ignores NaN entries rather than
propagating them.

### 2.4 Euler conversion details
Shepperd's branch selection for matrix→quaternion (stable near 180°, where the
naive trace formula loses precision), and an explicit gimbal-lock branch in
`matrix_to_euler_xyz`. Both are pinned by tests.

### 2.5 The epoch table itself
Fixed-width columns printed with plain `print` (not through tqdm), one reusable
progress bar never nested, `mininterval=10s` on a non-TTY. The learning rate is
a column, because *"a schedule that quietly decays to nothing looks exactly
like a model that has stopped learning."* Kaggle-specific and entirely
transferable.

---

## Tier 3 — engine and infrastructure

### 3.1 `rotate_per_fragment` — directly attacks our 645 ms/scene CPU bottleneck
`get_features` runs **once** per fragment, on the clean mesh. The diffused view
is then derived by rotating the extracted feature vectors on the GPU, which is
exact for a rigid transform. Edge lengths are invariant and reused verbatim.
`prepare_scene` asserts `diffused_input is diffused_target` — no duplicate work.

This is the single biggest throughput idea in the old repo and it is
independent of everything else.

### 3.2 `split_batch_by_scene` — micro-batching so peak memory tracks the largest *scene*
> *"That is what makes a 16 GB T4 able to train at any batch size."*

`batch_size` (scenes per optimiser step, per rank) and `micro_batch_scenes`
(scenes forwarded at once) are separate, and `Config.validate()` requires
`batch_size % micro_batch_scenes == 0` *"so every rank performs the same number
of backward passes — unequal counts deadlock DDP's gradient all-reduce."*
We already accumulate; the round-trip test (`test_split_round_trips_exactly`)
is worth copying regardless.

### 3.3 `_sync_gradients` — strictly better than what I implemented
Theirs wraps **every real micro-batch** in `no_sync()`, then fires the
all-reduce with a fixed zero-loss backward on a 2-vertex dummy scene. The
collective count per optimiser step is therefore **constant** — unaffected by
OOM retries, skipped scenes or uneven scene counts.

I implemented the pattern they rejected: sync on the last *real* micro-batch.
That is the pattern that produces exactly the deadlock we hit.

### 3.4 `all_ranks_agree` — every early exit is voted on
Time budget, stop signal, OOM: each is put to a collective vote *before* any
backward runs. Our deadline check and stop-signal are local decisions, which
is a latent hang: one rank exiting the loop while its peer enters `backward`
waits forever.

### 3.5 The three-step OOM ladder
retry with gradient checkpointing → substitute the dummy scene to preserve rank
symmetry → raise. Tested on CPU without DDP
(`tests/test_trainer_recovery.py`), including the subtle one: `DDP.no_sync()`
returns a **single-use** `_GeneratorContextManager` that deletes its own
`args`/`kwds` on `__enter__`, so a retry loop must build a *fresh* one. There
is a test that pins even that assumption so it cannot rot.

### 3.6 Small DDP details
- `find_unused_parameters=False`, made safe by every loss returning a
  graph-connected zero (`values.sum() * 0.0`) rather than a bare `0.0`, so
  every parameter stays reachable and all ranks build identical buckets.
- `broadcast_buffers=(norm == "batch")`.
- `dist.init_process_group(device_id=device)` — mutes the exact barrier warning
  we currently see.
- `reduce_metrics` sorts keys and checks tensor sizes across ranks, guarding
  against one rank's epoch lacking a key the other has.

### 3.7 Numerical guards worth porting verbatim
- **Gram–Schmidt pre-normalisation.** Normalise both channels *before*
  orthogonalising. Mathematically identical (the frame depends only on
  directions) but scale-invariant in floating point: "4e-15 across twelve
  orders of magnitude, against 2e-3 before". Worst exactly in the
  high-`head_cos`, small-activation regime an untrained network sits in.
- **`at_least_float32`, not `.float()`.** An unconditional cast downcasts
  float64 and silently drops equivariance-test residuals from ~1e-15 to ~1e-8.
- **`_accum_dtype`** — fp16 running sums silently drop terms past ~2048.
- **`segment_softmax` clamps before the max-shift** — degrades to a hard argmax
  instead of NaN.
- **`_sanitise`** — degenerate zero-area triangles give `0/0 = NaN` normals in
  the **base meshes**. Three Breaking Bad objects produced NaN losses across
  forty epochs, under several fracture patterns each; the NaN was in the data
  before the model saw it. They repair *and count* it, and deliberately do
  **not** filter trimesh's "invalid value encountered in divide" warning,
  because that warning is the signal.
- **Repeat-offender scene naming** — the trainer tallies NaN per scene and
  names repeat offenders, because *"repeat offenders are effectively excluded
  from training."*

### 3.8 Checkpointing
- `last.pt` **always** written, separate from the policy-governed
  `checkpoint.pt` ("replace unless the stored one is better on *both* losses").
  In a real run `checkpoint.pt` froze at epoch 69 while training reached 100 —
  resuming would have silently discarded 31 epochs.
- `best_val` updated **before** the state dict is built, so both files record
  it. Otherwise resuming from `checkpoint.pt` restores `best_val = inf` and the
  first validation of the new session overwrites `best.pt` — destroying exactly
  what keeping it separate is for.
- Atomic writes (temp + `os.replace`).
- `_RESTORED_FIELDS`: a checkpoint is self-sufficient, so `--resume auto` needs
  no flags at all, and an explicitly typed flag still wins. The field list
  deliberately includes the **data definition** (`split_source`, `data_subsets`,
  `steps_per_epoch`, `val_steps`, `max_scenes`, `input_source`, `val_frac`,
  `split_seed`) and deliberately excludes platform fields (`batch_size`,
  `num_gpus`, `checkpoint_dir`, ...). Dropping `--split_source official` on a
  resume would silently revert to `hash`: *"validation objects leak into
  training and every number afterwards is meaningless."* There is an ordering
  test asserting the adoption happens before the dataloaders are built.
- `history.truncate_to(epoch)` on resume — a checkpoint at epoch 20 may sit
  beside a history that ran to 27.

### 3.9 Scheduler traps (informational for us — our LR is step-based)
- `CosineAnnealingLR` is **periodic**: a resumed run went 1.0e-5 → 6.9e-4 over
  75 epochs. `_CosineOnce` is a callable *class*, not a closure, because
  `LambdaLR.state_dict()` serialises a lambda's `__dict__` only for
  non-plain-functions — with a closure the horizon is never written to the
  checkpoint.
- `restart_schedule` must set **both** `lr` and `initial_lr`; setting `lr`
  alone is silently undone on the scheduler's first step (a restart requesting
  2e-4 ran at the checkpoint's 5e-4 while the log said 2e-4).
- Scheduler state is restored only when the **type** matches, because
  `LRScheduler.load_state_dict` is `self.__dict__.update(...)` and will happily
  inject a ReduceLROnPlateau state into a cosine scheduler.

### 3.10 AMP: measured and rejected
Same seed, same config, 8 objects, only precision differing:

| epoch | AMP | fp32 |
|---|---|---|
| 62 | 79.11 | 74.06 |
| 70 | 85.24 | 62.82 |
| 85 | 65.84 | **43.18** |
| best | 54.26 @ 160 ep | **43.18 @ 90 ep** |

AMP also produced non-finite losses from epoch 66 on six of the eight training
objects. The gradient corruption **preceded** the visible NaN — fp32 was
already 5–19° ahead at epochs 62–65. Cost of fp32 is only 10–20% wall clock,
not 2×, because the model is scatter/gather-dominated rather than GEMM-bound.

We already run fp32. This is the citation for why, and it belongs in the
thesis.

### 3.11 Process hygiene
`limit_blas_threads(1)` called **before** numpy/torch import, at the top of
every entry point. Kaggle's 2×T4 exposes ~4 vCPUs; without this, (2 ranks × N
workers) each spin up a full-width OpenMP pool and the geometry work in the
loader workers ends up *slower* than single-threaded.
`dataloader_worker_init` re-seeds python `random` and numpy's global RNG per
worker — PyTorch seeds only `torch`, so without it every worker draws the
**identical** stream of scenes and rotations, cutting effective data diversity
by `num_workers` while looking fine.

---

## Tier 4 — tooling scripts worth having

- **`benchmark_data.py`** — per-stage cost of one scene, then real DataLoader
  throughput, with the reading rule stated: an epoch costs
  `max(data/workers, compute)`, not their sum. *"Throughput assumptions are the
  easiest thing in this project to get wrong by an order of magnitude."*
- **`check_scene.py`** — for one named scene, walks mesh → features → model in
  fp32 → model under AMP and names the first stage that goes non-finite.
  `--locate` installs forward hooks on every module and prints the first
  module whose output goes non-finite *plus the activation-magnitude profile
  leading up to it*, with a "how to read this" block. It also refuses to be
  useful without `--checkpoint` and says so: an untrained model has O(1)
  activations and cannot reproduce an overflow.
- **`inspect_data.py`** — assumption-free walk of the dataset tree, compared
  against what the project's own scanner finds, plus `--debug_split` which
  prints the exact keys the official-split matcher builds and looks up (and the
  md5 of the loaded `splits.py`, because a stale file is indistinguishable from
  a logic bug from the outside).
- **`scaling_sweep.py`** — object-count scaling with `--steps_per_object` as
  the control rather than a fixed total budget, because an equal total budget
  degrades error with N by arithmetic alone. Flags truncated runs **INVALID**
  rather than reporting them as comparable, reports tail slope so a
  mid-descent number is not read as a ceiling, and requires ≥3 seeds because
  the between-seed spread at fixed N was **73.8°**, larger than the effect
  being measured.
- **`dump_prediction.py` + `visualize_reassembly.py` + `render_gif.py`** — a
  self-contained `.npz` (flat arrays + offsets, no `allow_pickle`) produced on
  Kaggle and animated locally. The GIF renderer holds a **fixed camera**
  computed from the union of every frame, keeps per-fragment colours stable
  across frames and panels, and ping-pongs with a hold at each end — all
  because *"an autofit camera would make the object appear to breathe as the
  fragments converge, motion that is an artefact of the renderer, not of the
  model."* This is how the thesis figure gets made.
- **`check_version.py`** — 52 (file, marker) pairs asserting a deployed copy
  actually contains every fix, plus a duplicate-test-name check and a
  config-vs-dataclass field check. Exists because the project is edited in one
  place and executed in another, and a partial copy produces a traceback whose
  line numbers do not match the source.

---

## Tier 5 — smaller things, noted for completeness

- **`symmetric_edge_features`** — edge embeddings must be symmetric in their
  endpoints, because vertex numbering is arbitrary across fragments. Pinned by
  a test that flips `edge_index` and demands bit-identical output.
- **`edge_scalar_to_bias`** — edge length used as an **invariant scalar
  attention bias**. We dropped length entirely; it does have a legal slot.
- **`symmetrise_edges`** — forward block then reverse block, with the two face
  normals swapped in the reverse half, and a test pinning the exact layout.
- **`extract_fractures`** — two bugs they found and fixed: criterion (b) was
  computed and never used, and reusing the full vertex array meant the
  "fracture mesh" had the same node count as the full mesh, so pruning saved
  nothing.
- **`load_official_split`** — suffix matching over 1..3 path components,
  most-specific-first; `rglob` rather than `glob` because the release nests the
  lists one level deeper (`data_split/data_split/`), and a non-recursive glob
  found nothing and reported it as an id mismatch.
- **`resolve_duplicated_faces`** exact short-circuit guarded by an env var
  (`VNGAT_VERIFY_RESOLVE=1`) so the expensive check is available but not paid
  for.
- **`_mean` that returns `values.sum() * 0.0` on an empty tensor** —
  `torch.empty(0).mean()` is NaN, and one NaN in a weighted total destroys a
  12-hour run.
- **`configure_warnings(strict=True)`** — message-scoped filters, one call per
  category (a *tuple* passed as `category=` silently does nothing), with the
  test suite running everything not on the known-benign list as an error.
- **`test_no_duplicate_test_names`** — two same-named test functions in one
  module: Python keeps the last, so the earlier never runs. That silently
  reverted an updated assertion to a stale one.

---

## Benchmark framing (affects what we claim, not what we build)

GARF reports two tables. The one our numbers are comparable to is the **vanilla
Everyday supplementary** table, where SE(3)-Equiv is **79.30° RMSE(R)** and
GARF-mini (Everyday only) is **10.41°**. Our docs quote 6.1° throughout, which
is the wrong row.

---

## One thing to be careful about in the premise

The old version reaching total loss ≈ 0.6 on 8 / 16 / 32 objects with no
validation improvement is the signature of **memorisation**, not of a better
architecture — and the old repo says so itself. Its `scaling_sweep.py`
docstring records that the **between-seed spread at a fixed object count was
73.8°**, larger than the difference between object counts (7.1°), so a single
seed per point "would have supported whichever conclusion it happened to land
on". Three seeds of an identical 8-object run finished at 30.9°, 54.3° and
104.7°.

So: worth mining for the items above, but the 0.6 figure is not a target our
current run is failing to reach. The comparable number is a *validation*
number, and the old version did not have one.
