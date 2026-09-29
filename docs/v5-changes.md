# Thesis v5 — what changed from Thesis 1, and why

v5 started as a file-for-file copy of `E:\Thesis 1` (no git history was
carried over; this document is the record instead). It keeps Thesis 1's
design — virtual nodes for cross-fragment communication, the Gram-Schmidt
rotation head, the classical translation solver, the `input_source=frac`
option — and brings in the mesh-edge attention layer and edge features of
`E:\My Thesis Work`, plus the fixes found while doing so. My Thesis Work's
fracture-surface extraction logic was deliberately **not** brought over.

Everything below is covered by the test suite: 218 passed, 3 skipped
(Thesis 1: 159 passed, 3 skipped), including two- and three-process DDP tests.

---

## 1. The model

### The mesh-edge layer is My Thesis Work's

`vngat/models/gat_layer.py` is now `VNGraphAttention` / `VNGraphAttentionBlock`,
the same layer as `src/reassembly/nn/gat.py`, built on this package's own
Vector Neuron primitives:

```
q_i  = W_q x_i
k_ij = W_ks x_j + W_kd x_i + W_ke e_ij          (source, destination, edge)
a_ij = softmax_j(<q_i, k_ij> / sqrt(3 head_dim))
m_ij = W_vs x_j + W_ve e_ij
out  = act(norm(W_o(sum_j a_ij m_ij + W_self x_i)))      block: x + out
```

**Equivalence, measured.** With weights copied across, float64, random graph:

| quantity | max abs difference |
|---|---|
| layer output | 2.9e-6 (relative 1.4e-7) |
| gradient w.r.t. input | 4.5e-6 |
| residual block output | 3.8e-6 |
| layer output, both sides given the **same** epsilons | **2.2e-16** |

The whole residual is the two primitive sets' numerical guards (this
package's LeakyReLU clamps `|d|^2` at 1e-6 where My Thesis Work adds 1e-8;
its LayerNorm clamps at 1e-5 where My Thesis Work adds 1e-8). With those
matched, the layers agree to round-off. Per-fragment equivariance: 3e-15.

Three differences from Thesis 1's `VNGATLayer`:

- **The score reads the edge.** Thesis 1's key depended on the source vertex
  only, so replacing every face normal left the attention weights exactly
  unchanged — the normals reached the message but never decided whom to
  listen to (`test_attention_score_reads_the_edge`).
- **A vertex keeps its own state.** No self term and no residual made an
  isolated vertex output exactly zero; `W_self x_i` and the block residual fix
  both (`test_isolated_vertex_keeps_its_own_state`).
- **Scores have their own width.** `heads × head_dim` (new `head_dim`, default
  8) is independent of `hidden_channels`. No edge-length attention bias.

### Fragment size, through a gate

Positions are now normalised per scene (section 2), so the absolute size of an
object is gone from the coordinates. It comes back as `log(world radius)` per
fragment through `VNScaleGate`: invariant features of the vertex plus the scale
→ a positive per-channel gain (`softplus`, initialised at 1.0 with a small —
not zero — final weight so the gate is near-identity but not dead). Applied
right after the input projection, as in My Thesis Work.

### Smaller changes

- Edge embedding head input: `h_u + h_v` plus **`n1 + n2`** — the only
  endpoint-symmetric linear form of the new edge channels (reversing an edge
  swaps the normals and negates the relative position).
- 376,592 → 394,128 parameters at the default width.

---

## 2. The data

### Edge features: `[n1, n2, p_u − p_v]`

Each stored undirected edge (u < v) carries its two adjacent face normals and
the relative position of its source. The midpoint and the invariant length are
gone (the length is recoverable as a norm; the midpoint is a position, which
the endpoints already carry). The reverse copy built on the device is
`[n2, n1, p_v − p_u]`.

**Which normal goes in which slot.** Thesis 1 filled the two slots in
face-array order, which `resolve_duplicated_faces` reorders lexicographically.
Measured through this project's own decompression code, that order agreed with
a geometric rule on **50.1%** of edges (17,176 of 34,276 with non-parallel
normals) — a coin flip. v5 orders them by the triple product
`(n_a × n_b)·(x_v − x_u) ≥ 0`, which is invariant under rotation (so the
diffused view can still be made by rotating features) and independent of face
order (`tests/test_features.py`).

### Per-scene normalisation (`normalize_mode`)

`merge_fragments` divides positions and relative positions by the largest
fragment radius of the scene (`scene`, default), by each fragment's own radius
(`fragment`, GARF's convention) or not at all (`none`, Thesis 1's behaviour).
`frag_unit` keeps the divisor so world units can be recovered;
`frag_log_scale` carries the world radius to the scale gate. Under
`input_source=frac` the fracture-surface input uses the **full** fragments'
divisor and scale, so prediction and target share units.

Consequence for reading logs: the position loss is now in normalised units, so
its values are not comparable with Thesis 1's.

### What is drawn: the catalogue (`vngat/data/catalog.py`)

- **Objects, not directories.** The `volume_constrained-*` copies of a shape
  are grouped with the vanilla copy into one object with the union of their
  break patterns, so a shape is counted once and can never sit in two splits.
  (Thesis 1 raised on the ambiguity under the official split, and
  double-counted under the hash split.) Official split lists are matched per
  object.
- **`split_by: fracture`** holds out break patterns of known objects —
  positional partition after a deterministic per-object shuffle, so every
  object contributes exactly its share; objects with fewer than 3 patterns are
  train-only. `fracture_pool: train` (default) uses only the object split's
  training objects, so the held-out shapes stay clean for the real benchmark
  number. Unlike My Thesis Work's version, this holds under the hash split too.
- **`balance: category`** (off by default) draws categories equally often,
  softened by `balance_temperature`. `none` is Thesis 1's behaviour (every
  object equally likely). The startup banner prints category shares and the
  effective number of objects.
- **A fixed validation set** (`val_fixed`, default on): the same scenes and the
  same rotations every epoch, one break pattern per object by default
  (`val_scenes: 0`), interleaved across categories, sharded — not repeated —
  across GPUs. Thesis 1 drew fresh random validation scenes every epoch, so
  epoch-to-epoch changes were partly sampling noise and `best.pt` was chosen
  on it.

---

## 3. The losses

The user-chosen option: Thesis 1's terms, fixed.

- **Every fragment counts once.** Position, vertex-normal and face-normal terms
  are averaged over each fragment's vertices/edges first, then over fragments
  (`fragment_mean`). Thesis 1 averaged over all vertices, so a fragment's weight
  was its vertex count while the rotation term weighted fragments equally.
- **The midpoint term is gone** (it re-counted the position term). Six terms
  remain: rot, pos, node, face, emb_v, emb_e. `w_mid` is removed from the
  config.

---

## 4. The training loop

- **Fragment-weighted steps, exact under DDP.** Each micro-batch is weighted by
  its fragment count, and the accumulated gradient is rescaled once by
  `fragments_on_this_rank × world / fragments_contributed_everywhere` **before**
  the all-reduce. Result: the step's gradient is the exact mean over every
  fragment that contributed, on every rank, and a dropped micro-batch shrinks
  the denominator instead of the step. Rescaling after the all-reduce would
  make the two replicas step differently and drift apart; `tests/test_ddp.py`
  runs two and three real processes and checks both the gradient and that the
  replicas stay bit-identical (and fails when the rescale is moved after the
  sync).
- **An OOM retry no longer counts the scene twice.** The accumulated gradient
  is set aside for each micro-step and merged back only on success; a partial
  gradient from a failed backward is discarded.
- **A finite loss with a non-finite gradient** now drops only that
  micro-batch (fp32; under AMP the GradScaler keeps that job).
- **Two OOMs skip the scene** instead of back-propagating a placeholder's real
  loss into the step.
- **A step with no data does not move the weights** (AdamW would, on momentum
  and decay).
- **Validation runs outside the DDP wrapper**, and all metrics are reduced as
  sums and counts, fragment-weighted, identical on every rank.
- **Per-category validation** every epoch, plus the category mean.
- **`best.pt` is judged every epoch** on `checkpoint_monitor` (val rotation by
  default), not only on `save_every` epochs.
- **Resume draws distinct data per GPU.** The checkpoint stores rank 0's RNG
  state and every rank restored it — after which both GPUs (and their data
  workers) drew identical scenes for the rest of the run. v5 re-seeds each rank
  after loading. *(This bug is in Thesis 1 too.)*
- **No checkpoint is written from non-finite weights** (Thesis 1's stop path
  saved them into `last.pt`).
- `config.yaml` is written after a resumed checkpoint's settings are adopted;
  `history.json` is flushed every epoch; changing `checkpoint_monitor` on
  resume resets the best-so-far; a Thesis 1 checkpoint is refused with a
  message.
- The tilt/twist verdict no longer calls the axis-only pattern a "structural
  floor" — it is a landmark (a measured run went well below it).
- `batch_size` no longer has to be a multiple of `micro_batch_scenes`. The rule
  said unequal backward counts would deadlock DDP, but every real micro-step
  already ran under `no_sync()`; only the one sync step talks to the other
  rank, and fragment weighting handles a short last micro-batch.

---

## 5. Scripts

| script | change |
|---|---|
| `evaluate` | Rebuilds the data from the **checkpoint's** config. Thesis 1's omitted `split_source` and `data_subsets`, so an officially-split model was scored on the hash split over every subset on disk — partly on official training objects. Now: fixed deterministic scene set, default `--split val` (there is no official test split), Chamfer / part accuracy / translation solver in world units, collision radii actually passed, landmark wording, per-category table |
| `dump_prediction` | `--list_pool` reads the run's own data definition from its checkpoint |
| `visualize` | prediction mode builds the scene the way the checkpoint was trained; random scenes come from the held-out split |
| `check_scene` | new features and normalisation |
| `benchmark_data` | catalogue draws; `--config`; reports the median edge relative-position length |
| `inspect_data` | adds the object-level view: variants grouped, categories, split counts per source |
| `plot_history` | no midpoint panel; tilt, twist, head conditioning, per-category validation, gradient norm |
| `scaling_sweep` | keeps validation to one random batch (the sweep reads training error) |
| `check_version` | markers for every v5 change |

---

## 6. Config

New: `split_by`, `fracture_pool`, `balance`, `balance_temperature`,
`normalize_mode`, `head_dim`, `val_fixed`, `val_scenes`, `checkpoint_monitor`.
Removed: `w_mid`, `log_every` (never read). All are restored on resume. The
YAMLs are generated from the dataclass; both Kaggle configs now use the
**official Everyday** split (the benchmark setting).

---

## 7. One GPU, two, or more

As in Thesis 1: `--num_gpus N` spawns one process per GPU, `--num_gpus 1` runs
in the calling process, and asking for more GPUs than are visible uses the
ones that are (the CPU when there are none). Nothing in the step, the
validation sharding or the reductions assumes two. Verified beyond two, on CPU
processes over gloo:

- `tests/test_ddp.py` with 2 and 3 ranks, including a rank that drops a
  micro-batch -- the reduced gradient is the global fragment mean and the
  replicas stay bit-identical; a rank left waiting fails the test on a timeout
  instead of hanging it;
- a whole training run on 3 processes (fixed validation sharded 3 ways), resumed
  on 3 processes, then resumed again on 1.

## 8. Worth knowing before a long run

- Loss weights are untuned (all 1.0), as in Thesis 1.
- The edge relative position is short next to the unit face normals it travels
  with; the layer has to learn any gain between them.
  `python -m scripts.benchmark_data --config configs/kaggle_2xt4_full.yaml`
  prints the median on the real data.
- `--split_by fracture` is the quickest test of the "it cannot handle unseen
  shapes" hypothesis: run it next to the object split and compare.

---

## 9. The largest fragment as the anchor (Sept 2026)

Added after the official-split runs of Thesis 1 and My Thesis Work stayed at
chance on validation while training fell (My Thesis Work, one epoch: 123.0 deg
against 126.5 chance; tilt 87.0, twist 87.8; `match@1` 0.67). The label is each
fragment's rotation into its object's stored frame, which an unseen, mostly
round shape does not determine. The benchmark fixes one fragment instead: GARF
trains with anchors "with identity rotations and zero translations" and
evaluates with the largest fragment fixed (supplementary C.3).

**Compatibility, checked before porting.** v5 uses the column convention
(`x_pred = R_pred x_diffused`, `R_gt = A^T`), so a global rotation of a
predicted assembly multiplies on the left -- the same correction as My Thesis
Work's. Every batch carries `frag_scene` and `frag_log_scale`, and the largest
radius is already what `normalize_mode: scene` divides by. Every
rotation-dependent term is built from `R_pred` in one place (`build_predictions`)
and averaged per fragment (`fragment_mean`). The model has the same symmetry as
My Thesis Work's -- equivariant to its own fragment's pose, invariant to the
others' (`test_rotating_one_fragment_leaves_the_others_alone`) -- and is not
touched.

**What changed.**

- `vngat/evaluation/anchor.py` (new): per scene `C = R_gt[a] R_pred[a]^T`, on
  the left of every prediction; in float32 outside autocast. Anchor: largest
  `frag_log_scale`, ties to the first; left out of every average.
- `_forward_loss`: under `rotation_target: anchor` (the default) the aligned
  prediction feeds `build_predictions` and the loss, with
  `targets["fragment_keep"]` leaving the anchors out of rot, pos, node and face.
  The anchor-protocol figures ride along under underscored keys either way.
- `run_phase`: micro-batches weighted by the fragments the loss scores (one
  fewer per scene under the anchor target), the step rescale and the epoch
  averages included; `anchor_deg`, `absolute_deg` and the anchor's tilt/twist
  summed per fragment and divided once after the reduction over ranks.
- Config: `rotation_target` (restored on resume; a checkpoint without it is
  read as `absolute`, announced), `checkpoint_monitor` default `anchor_deg`,
  `anchor_deg` and `absolute_deg` monitorable. A resumed monitor that changed
  meaning -- a loss term after the target changed, or a protocol figure from a
  checkpoint older than the protocol -- resets the best-so-far.
- `scripts/evaluate.py` and `evaluate_scene`: aligned rotations, translations
  measured from the anchor's, the anchor left out of every per-fragment number,
  `absolute_geodesic_deg` beside them. `dump_prediction` and `visualize` show
  the aligned prediction and name the anchor.
- YAMLs: `rotation_target: anchor`, `checkpoint_monitor: anchor_deg`.

**Verified.** 238 passed, 3 skipped (223 before, with `test_ddp` now run under
both targets). Mutation checks -- correction on the right, anchor per batch,
anchor left in the average, anchor not trained, weights still counting the
anchor, loss ignoring the target, old checkpoints read with the new default,
`fragment_mean` ignoring `keep` -- each fails at least one test. `check_version`:
73 markers.

**Two things found on the way.** v5's `geodesic_rotation_loss` has a
`sqrt(1e-12)` floor, so an exact prediction reads ~3e-5 deg, not 0. And
`tests/conftest.random_rotation` is not uniform on SO(3) (QR without the sign
fix: two draws are 100.8 deg apart on average, not 126.5); harmless where a test
needs *a* rotation, so it is left alone, and the chance test uses the data
pipeline's sampler instead.

