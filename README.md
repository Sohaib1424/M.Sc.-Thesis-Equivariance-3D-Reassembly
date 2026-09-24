# Equivariant 3D Fracture Reassembly

Reassembling a shattered object from its fragments, with an SO(3)-equivariant
graph network trained on commodity hardware.

M.Sc. thesis project, built on the [Breaking Bad](https://breaking-bad-dataset.github.io/)
dataset. The data and geometry layer, the equivariant network and the training
engine are all here; the translation solver is not. Nothing has been trained on
the real dataset yet — see [Status](#status) before you plan around it.

**New here, or picking the project back up?** Start with
[`docs/PROJECT-STATE.md`](docs/PROJECT-STATE.md) — the pipeline, what the dataset
turned out to contain, every design decision and its evidence, and what is still
missing. [`docs/design-notes.md`](docs/design-notes.md) holds the technical
detail, and [`docs/figure-review.md`](docs/figure-review.md) checks the
explanatory GAT / VN / V-GAT posters against what the layers actually have to
satisfy — the V-GAT one specifies a layer that is **not** equivariant, so read
that before implementing anything from it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/fracture_reduction_dark.png">
  <img alt="Share of each fragment kept as fracture surface, and the spread across fragments" src="docs/fracture_reduction.png">
</picture>

> Rendered from the full dataset with
> `python -m scripts.plot_fracture_stats --stats out/stats_true.csv --both-themes`.

Over the full dataset — 1,096,825 fragments across 1,442 objects, extracted in
about an hour on 4 workers — the fracture surface is a small part of each mesh:

| | of all vertices | of all faces | of all edges |
|---|---|---|---|
| **removed** (original exterior) | **89.4%** | **93.8%** | **92.4%** |
| kept as fracture surface | 10.6% | 6.2% | 7.6% |

Size-weighted totals (`1 − Σkept / Σoriginal`) from the **coincidence** ground
truth: a vertex is fracture surface exactly when another fragment has a vertex
at the same point, which Breaking Bad makes decidable because fragments are
stored in their assembled frame.

The *per-fragment mean* is much lower — 49.9% of vertices — because most
fragments are small and small fragments are mostly fracture surface. Both are
reported because averaging percentages answers "what happens to a typical
fragment", not "how much data survives". The median fragment keeps **125
fracture vertices** out of 296.

> The `dihedral` heuristic, which is what you can compute at inference on a
> fragment in isolation, over-labels by **~3.9×** at its default threshold
> (precision 0.24, recall 0.93 against this ground truth). It also lights up
> handles, rims and feet — sharp because the object was *designed* that way. See
> [`docs/design-notes.md`](docs/design-notes.md) §4 and §9.

## What problem this solves

Given the fragments of a broken object, each in an arbitrary pose, recover the
rigid transform that puts every fragment back where it belongs.

The state of the art, [GARF](https://arxiv.org/abs/2504.05400), reaches 6.1°
rotation RMSE — but takes 4×H100 for 72 hours. **The thesis question is whether
equivariance can substitute for scale**: whether a network that is
SO(3)-equivariant *by construction* needs to see far fewer examples than one
that must learn rotational invariance from data, and can therefore be trained
on a 2×T4 budget.

Two commitments follow from that:

**Equivariance instead of augmentation.** A conventional network sees two
rotated copies of one fragment as unrelated inputs and must learn the same
geometry many times over. An equivariant layer satisfies `L(R·x) = R·L(x)` as an
algebraic property, so the rotation is free.

**Rotation and translation are separate problems.** Orientation needs semantic
understanding of how fragments relate. Translation, once orientations are known,
is a well-posed geometric optimisation over matched interface points — it does
not need to be learned at all.

## Why fracture surfaces matter

A fragment's surface is two different things glued together: the object's
**original exterior** (smooth, and shared with no one) and the **fracture
surface** where it tore away from its neighbours (jagged, and matching exactly
one other fragment).

The fracture surface carries the *exact mating* information — one surface fits
exactly one other. Separating it out is what `reassembly.mesh.fracture` does.

It is not the only useful signal, though: a rim broken across three fragments
still has to close into one circle, and a handle split in two still has to
rejoin. Those constraints live on the *original* surface. So fracture-ness can
either **gate** the cross-fragment graph or ride along as a per-token
**feature** — GARF does the latter, attending over all its sampled points and
using fracture segmentation only as a pretraining objective. Both are
implemented; see [`docs/design-notes.md`](docs/design-notes.md) §4.

Two independent labellings are available:

| method | what it uses | available at inference? |
|---|---|---|
| `dihedral` | sharp dihedral variation within one fragment | **yes** — one fragment in isolation |
| `coincidence` | vertices another fragment also occupies | no — needs the assembled scene |

`coincidence` is ground truth (this is how GARF derives its supervision);
`dihedral` is the estimate a model can compute at test time.
`scripts/tune_sharp_threshold.py` scores one against the other across a sweep,
so `--sharp-threshold` is chosen by measurement rather than by eye. It reports
recall-weighted **F-beta** rather than F1, because a missed fracture face is an
edge the model never sees while a spurious one is only noise.

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `libigl`, `trimesh` — see [`requirements.txt`](requirements.txt)
- `torch` ≥ 2.2 for the network and training (install from the **default** PyPI
  index; `download.pytorch.org` is blocked on some networks and the CUDA wheels
  are on PyPI anyway)
- `pandas` for the summary tables and figures; `matplotlib` for the figures;
  `pillow` for the reassembly GIFs; `pyglet<2` only for an on-screen viewer window
- The Breaking Bad dataset (not redistributable — [download it here](https://breaking-bad-dataset.github.io/))

**The test suite does not need `libigl`.** It builds its meshes in
`conftest.py`, so if libigl gives you trouble on your platform the tests still
run with `numpy scipy trimesh torch pytest` alone. Without `torch` the geometry
tests still pass and the network tests skip.

## Install

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest                                        # 445 passed
```

Installing is optional — `pytest.ini` sets `pythonpath = src .` and each script
prepends `src` itself, so everything runs from a clone.

### Dataset layout

Point `--root` at the directory *containing* the subset folders:

```
<root>/everyday_compressed/everyday_compressed/<category>/<object>/
<root>/artifact_compressed/artifact_compressed/<object>/
    compressed_mesh.obj                       intact fine mesh
    compressed_data.npz                       sparse cell → fine-vertex matrix
    fractured_<k>/compressed_fracture.npy     cell → piece label
    mode_<k>/compressed_fracture.npy
```

Scene discovery walks until it finds directories holding `compressed_mesh.obj`,
so extra nesting is tolerated. A wrong `--root` gives
`no scene directories found under ...` rather than a wrong answer.

## Usage

### Look at a scene

```bash
python -m scripts.visualize --root /path/to/data
```

A random scene, fracture mode and perturbation each run — and the seed is
printed, so any view can be pinned:

```
scene     everyday_compressed/Bottle/obj_F
mode      fractured_1  (4 available)
seed      599617297   (--seed 599617297 to see this exact scene again)
fragments 2
load      mesh+matrix     15.4 ms   (once per scene, amortised over 4 modes)
          fragments        1.8 ms   (per sample; 0.9 ms/fragment)
          total           17.2 ms
```

| | |
|---|---|
| `--show fracture` | draw only the extracted fracture surfaces |
| `--show both` | fragments and surfaces side by side |
| `--method coincidence` | draw the ground-truth labelling instead of the heuristic |
| `--diffuse` | apply a random SE(3) transform per fragment |
| `--seed N` | pin the scene, mode and perturbation |
| `--scene NAME --mode NAME` | pin them directly |
| `--export out/scene.glb` | write a file instead of opening a window |
| `--no-show` | print the table and timings only |
| `--repeat N` | re-time the load N times, report best and median |

An interactive window needs `pip install "pyglet<2"`; `--export` and
`--no-show` do not.

The load timing is split because the halves behave differently under a batch
loader: `mesh+matrix` is parsed once per scene and reused across its ~100 modes,
while `fragments` is the per-sample cost. Both `igl` and `trimesh` are imported
before the clock starts — measured cold, the first load pays ~500 ms of
`import trimesh` against ~2 ms of actual work.

### Extract fracture surfaces over the whole dataset

```bash
python -m scripts.extract_fracture_surfaces --root /path/to/data --dry-run
python -m scripts.extract_fracture_surfaces --root /path/to/data --workers 4 \
    --min-fracture-faces 1 \
    --masks-dir out/masks --stats-csv out/stats.csv --summary-csv out/summary.csv
```

`--min-fracture-faces 1` guarantees a non-empty labelling. Every fragment of a
broken object tore away from a neighbour, so it has fracture surface by
definition — but the dihedral test is an *absolute* angle threshold while
roughness is relative to triangle size, so a small, gently curved shard can have
no adjacency above the threshold and come out empty. The flag relaxes the
threshold for **those fragments only**, leaving every other fragment
bit-identical. Pick the global threshold with evidence instead of by eye:

```bash
python -m scripts.tune_sharp_threshold --root /path/to/data --limit 200 --workers 4
```

which sweeps thresholds against the coincidence ground truth and reports
precision, recall, F1 and the empty-fragment rate for each.

Deterministic order, resumable (`--resume`), interrupt-safe, and bounded by
`--time-budget` for session-limited machines. `--limit 20` gives a number in a
minute.

Storing every fracture surface as a mesh would be ~10 GB; the default writes
**packed-bit masks** instead — about one bit per face, under a gigabyte for the
whole dataset, from which the surfaces reconstruct exactly:

```python
from scripts.extract_fracture_surfaces import load_masks
masks = load_masks("out/masks/everyday_compressed__cat__obj/fractured_3.npz")
```

### Figures and analysis

```bash
python -m scripts.plot_fracture_stats  --stats out/stats.csv --both-themes
python -m scripts.analyze_graph_cost   --stats out/stats.csv --plot docs/graph_cost.png
```

`analyze_graph_cost` measures how many attention connections the fracture-surface
graph creates per scene, against GARF's cost as a reference. It exists because
cross-fragment attention over fracture vertices is **quadratic in fracture-vertex
count**, while GARF samples a fixed 5000 points per object regardless of mesh
resolution — so the comparison decides whether the design needs sub-sampling.

## Code tree

```
src/reassembly/
├── arrays.py                      int64 key encoding, grouping, index compaction
├── mesh/
│   ├── topology.py                half-edges, adjacency, unique edges, safe normals
│   ├── repair.py                  resolve_duplicated_faces (vectorised)
│   ├── fracture.py            ★   fracture-surface extraction, both methods
│   ├── orientation.py             canonical edge face-normal ordering
│   ├── patches.py                 fracture-surface patches for cross-attention
│   ├── correspondence.py          cross-fragment coincidence matching
│   └── shell.py                   ray-cast visibility (unused; kept for figures)
├── data/
│   ├── paths.py                   scene discovery, subsets, official splits
│   ├── catalog.py             ★   one shape is one shape; split modes; balancing
│   ├── sampling.py                weighted + distributed + reproducible sampler
│   ├── scene.py                   SceneReader — caches the intact mesh across modes
│   ├── transforms.py              SE(3) perturbation, centring, normalisation
│   └── features.py                meshes → tensors, and the batch collate
├── training.py                 ★  config, dataset, loops, checkpoints, metrics
├── distributed.py                 1, 2 or N GPUs: one all-reduce per optimizer step
├── assembly/
│   ├── translation.py             stage two: match in embedding space, solve for t
│   └── scoring.py                 RMSE(T), Chamfer, part accuracy, in world units
├── nn/
│   ├── vn.py                  ★   Vector Neuron primitives, Gram-Schmidt head
│   ├── segment.py                 scatter reductions (no torch_scatter)
│   ├── gat.py                     intra-fragment attention along mesh edges
│   ├── cross.py               ★   cross-fragment attention — read its docstring
│   ├── losses.py                  the composite objective, verified chance values
│   └── model.py                   the backbone and the rotation convention
├── evaluation/metrics.py          reported-only: tilt/twist, head|cos|, Chamfer, PA
└── viz/
    ├── scene.py                   scene assembly for rendering
    └── reassembly.py              poses and frames for animating a prediction

scripts/
├── train.py                       train, resume, preflight, evaluate
├── config_flags.py                one flag per Config field, shared by the tools
├── benchmark_data.py              where the data pipeline's time goes
├── check_scene.py                 why a named scene fails (--locate: which module)
├── dump_prediction.py             one prediction + its assembly, to a small .npz
├── visualize_reassembly.py        watch a dump reassemble
├── render_gif.py                  a dump as a GIF or PNG figure
├── scaling_sweep.py               error vs number of objects, confounds controlled
├── check_version.py               is this copy the edited copy?
├── extract_fracture_surfaces.py   exhaustive pass over the dataset
├── plot_fracture_stats.py         the README figure
├── analyze_graph_cost.py          graph size vs GARF
├── tune_sharp_threshold.py        pick --sharp-threshold by F1
└── visualize.py                   render or describe one scene

tests/                             445 tests, no skips
```

Only `reassembly` is packaged; `scripts/` and `tests/` are entry points and
checks, run from the repository root.

## Status

**Implemented and tested:** data loading, scene discovery, official splits,
mesh topology, fracture-surface extraction (both methods), cross-fragment
correspondence, SE(3) perturbation, visualisation, the exhaustive extraction
pass — and the network: Vector Neuron primitives, intra-fragment graph
attention, cross-fragment attention, the composite loss, feature construction
and the batch collate.

Equivariance is checked numerically in float64, not asserted in prose. Every
primitive satisfies `L(xRᵀ) = L(x)Rᵀ` to ≤1e-15; the whole model is equivariant
to a fragment's own pose and *invariant* to every other fragment's, which is
the property that makes a per-fragment rotation label learnable at all.

**Training** is in `reassembly.training` — one module holding the config, the
dataset, the loops, checkpointing, the schedule and the metrics. It prints every
loss term separately each epoch alongside the total, with chance in the banner
so a number can be read against something:

```bash
python -m scripts.train --root_dir /path/to/breaking_bad --epochs 40
python -m scripts.train --root_dir /path/to/breaking_bad --evaluate
```

**The flags are Thesis 1's** wherever the two projects share a setting, with
Thesis 1's meaning, so one set of habits drives both:

| flag | meaning |
|---|---|
| `--root_dir`, `--checkpoint_dir` | the dataset; where `last.pt` / `best.pt` go |
| `--resume auto\|none\|PATH` | continue from `--checkpoint_dir` (default); start fresh; or load from elsewhere |
| `--num_gpus` | `-1` every visible GPU (default), `N` the first N, `0` the CPU |
| `--data_subsets`, `--fracture_pattern`, `--split_source` | which subsets; which break patterns (`fractured_`, the default); `official`, `hash` or `auto` |
| `--batch_size` | scenes per optimizer step per GPU |
| `--micro_batch_scenes` | scenes per forward pass (default 1) — the only batch setting peak memory depends on |
| `--steps_per_epoch`, `--val_steps` | optimizer steps per epoch; `val_steps x batch_size` validation scenes (default: all) |
| `--lr`, `--lr_min`, `--lr_schedule`, `--lr_warmup_epochs` | the rate, its floor, `cosine` or `constant`, the warmup in epochs |
| `--hidden_channels`, `--heads`, `--embed_dim` | the width, attention heads, embedding size |
| `--grad_checkpointing True` | recompute every layer in the backward pass (below) |
| `--num_workers`, `--save_every`, `--time_budget_hours` | loader processes per GPU; `last.pt` every N epochs; stop cleanly after this long |

Settings only this project has keep their own names in the same style
(`--tokens_per_scene`, `--modes_per_scene`, `--max_objects`, `--schedule`, ...);
`python -m scripts.train --help` lists them. Booleans take a value as in Thesis 1
(`--amp False`). A Thesis 1 flag with no counterpart here (`--num_vn_slots`,
`--num_layers`, `--max_scenes`, `--config`, ...) stops with what to use instead.

It refuses to print a verdict a run did not earn: a result at chance, one
parked at the ~90° axis-only landmark, and one still descending at its cutoff
are each flagged for what they are.

### One GPU, two, or more

`--num_gpus` picks them: `-1` (the default) every visible GPU, `1` one, `N` the
first N, `0` the CPU; asking for more than are visible uses what is there, and
says so. More than one runs one process per GPU — from a *file*,
not a notebook cell (`scripts/train.py` explains why) — and each draws its own
share of every epoch.

The GPUs meet only where every one of them is guaranteed to arrive: once per
optimizer step (how many fragments contributed on each, whether any wants to
stop, then one all-reduce of the summed gradient) and once per epoch (the
summaries, so the logged numbers cover every GPU's data). Between those, a GPU
can skip a batch that is too large, drop one whose loss or gradient is not
finite, or retry one that ran out of memory, and no other GPU is waiting on it.
The step is the exact fragment-weighted mean over every fragment that
contributed on any GPU, and every replica applies the identical update.

That replaced a `DistributedDataParallel` path with two faults, both silent. It
rescaled each GPU's gradient by its *own* fragment count after the all-reduce,
so the replicas took different steps and drifted apart for good; and its skips
were decided per GPU while DDP all-reduces inside every syncing backward, so a
GPU that skipped met its peer one time fewer and the last all-reduce of the
epoch waited forever. `tests/test_distributed.py` runs two and three real
processes, checks every step against a one-process reference and the replicas
bit for bit, and fails on a hang instead of stalling. (Mutation-checked: moving
the rescale back to the old place fails it.)

Stopping — the time budget, or SIGTERM — is voted at the step, so all GPUs stop
at the same one and rank 0 checkpoints. Validation runs on every GPU over
disjoint, unpadded shards, so each scene is scored once.

### An epoch is a fixed amount of training

`--steps_per_epoch 800` (the default) makes an epoch 800 optimizer steps per
GPU, whatever the split size, `--modes_per_scene`, balancing or GPU count. At
the default `--batch_size 2` on two T4s that is 3,200 scenes — about one pass
over the Everyday training split; on one GPU it is 1,600 scenes and the same
number of optimizer steps. `--steps_per_epoch 0` goes back to one full pass. The
banner prints what an epoch is in steps, scenes and passes. For a smoke test
with `--limit_train`, use `--steps_per_epoch 0` or a small number — 800 steps
over 40 scenes is forty passes.

### What happens to a batch that fails

| | |
|---|---|
| out of memory | retried once with gradient checkpointing forced on every layer (same gradient, less memory, more time); skipped only if that fails too |
| non-finite loss | dropped before the backward |
| finite loss, non-finite gradient | dropped (fp32; under AMP the GradScaler does this job) |
| more than `--max_vertices_per_batch` | skipped before it is attempted (optional) |

Each is counted, and named by the scenes in it. A tally of failed scenes
survives across epochs and sessions (in the checkpoint and `offenders.json`);
a scene that fails in more than one epoch is printed as a **repeat offender**
— it is effectively excluded from training — with the command that diagnoses
it: `python -m scripts.check_scene --scene <name> --locate`. Degenerate
geometry repaired to zero normals is counted every epoch too, and a scene with
non-finite coordinates is skipped by name.

### Before you spend a session: preflight

```bash
python -m scripts.train --root_dir /kaggle/input/breaking-bad --preflight
```

Runs in a few minutes, trains nothing, and exits non-zero if it is not safe to
proceed. The local test suite proves the maths, but it runs on synthetic meshes
on CPU — none of the failures that actually cost a Kaggle session are visible
from there. Preflight checks the things that are:

| | |
|---|---|
| GPUs, memory, CPU count | `devices=2` on a one-GPU machine, workers oversubscribed |
| `out_dir` writable *now* | Pointing at `/kaggle/input` otherwise fails at the first checkpoint, hours in |
| Objects, samples, split source | **No official split file** means a silent fallback to a hashed one, and numbers that are not comparable to published results |
| Train/val object overlap | Validation measuring memorisation |
| Real scene sizes | Fragments, vertices and tokens per scene — and whether the fracture mask came out empty |
| Forward + backward on the **largest** sampled scene | Peak GPU memory against the card, and OOM caught here rather than mid-epoch |
| Every parameter receives gradient | A layer silently not in the model |
| Loss at initialisation vs chance | A term far from its reference is measuring something other than its name |
| Measured seconds/step | Projects epoch time, total time, **and how many sessions it will take** |

**Memory.** Of the batch settings, only `--micro_batch_scenes` (default 1)
changes the peak: `--batch_size` is how many scenes go into an optimizer step,
and they go through the GPU `--micro_batch_scenes` at a time with their
gradients summed — the same step, bit for bit, as one big pass. Watch the
out-of-memory counts in the first epoch. A batch that runs out of memory is
retried with gradient checkpointing and skipped only if that fails too, and the
epoch summary says how many were rescued and how many skipped; more than a
percent or two skipped is a bias in the result (the largest objects are the
ones dropped), not a performance detail.

**`--grad_checkpointing True`** recomputes every layer in the backward pass, as
in Thesis 1, so each layer keeps only its `(N, C, 3)` input instead of its
edge-sized insides — a mesh has about six directed edges per vertex. Measured on
CPU at `--hidden_channels 128`, one forward and backward, four fragments,
2,048 tokens:

| vertices | per-layer switches of the old version, both on | `--grad_checkpointing True` |
|---|---|---|
| 2,568 | 0.87 GB held after the forward, 1.06 GB peak | 0.22 GB held, 0.55 GB peak |
| 10,248 | 3.3 GB held; the backward exceeded the 6.5 GB free and was killed | 0.43 GB held, 3.7 GB peak |
| 40,968 | — | 1.0 GB held, 4.1 GB peak |

About 20 KB per vertex is held instead of about 250 KB. Most of the ~3 GB peak is
the cross-attention's pair tensors, rebuilt for one layer at a time in the
backward pass; it depends on the tokens (`--tokens_per_scene`) and not on the
vertices. The price is time — about one extra forward pass (+19% on the one size
measured both ways here; Thesis 1 measured ~35%). The cross layers recompute
their pair gathers whatever the setting (1109 bytes per pair stored against 86
recomputed, bitwise identical), so there is nothing to switch there.

If preflight reports that even `--micro_batch_scenes 1` will not fit, turn the
knobs in this order — the first two do not change what the model can represent,
the third does:

```bash
--grad_checkpointing True   # recompute every layer in the backward pass
--hidden_channels 32        # a narrower model
--tokens_per_scene 1024     # fewer cross-fragment tokens; changes the model
```

### Training across Kaggle's 12-hour cap

A full training set takes more than one session, so a run is a *chain* of them.
Session one:

```python
from reassembly.training import Config, train
if __name__ == "__main__":                       # required for devices=2
    train(Config(root="/kaggle/input/breaking-bad",
                 out_dir="/kaggle/working/vgat", devices=2, epochs=40))
```

Save the notebook version so `/kaggle/working` becomes an output, add that
output as an input dataset to the next session, and point at it:

```python
train(Config(root="/kaggle/input/breaking-bad",
             resume_from="/kaggle/input/<previous-output-name>",
             out_dir="/kaggle/working/vgat", devices=2, epochs=40))
```

`out_dir` must stay under `/kaggle/working`; `/kaggle/input` is read-only, and
the run checks that before training rather than at the first checkpoint.

It stops at `max_hours=11` with a checkpoint written, checkpoints every 30
minutes in case one epoch outlasts a session, catches the SIGTERM Kaggle sends,
restores optimiser moments and RNG state, refuses to resume onto a different
architecture, warns loudly if the *data* settings changed (that one raises
nothing on its own), and tracks cumulative training time so the banner can
project how many more sessions are left:

```
  ~1:04:12/epoch, 31 epoch(s) left = ~33:10:12  (4 more session(s) at 11h)
```

### Stage two and the benchmark numbers

`--evaluate` assembles as well as rotates: the translation solver
(`reassembly.assembly`) matches fracture vertices across fragments in the
network's invariant embedding, weights each match by how opposed its normals
are, and solves the translations by weighted least squares with a Huber
reweighting. Scores are Breaking Bad's — RMSE(T), Chamfer distance, part
accuracy (per-fragment Chamfer below 0.01) — in world units, per scene then
over scenes, with a per-category table. `--no_assemble` reports rotation only,
and says so. The model and, by default, the data definition come from the
checkpoint itself, with every setting that differs from the flags printed
(`--data_from_flags` keeps the flags' data settings).

```bash
python -m scripts.train --root_dir /path/to/breaking_bad --evaluate --checkpoint best.pt
python -m scripts.dump_prediction --root_dir ... --checkpoint best.pt --scene <object>/<mode> --out pred.npz
python -m scripts.visualize_reassembly --dump pred.npz --mode compare
python -m scripts.render_gif --dump pred.npz --out reassembly.gif
```

A dump is a small `.npz` with plain arrays, so it can be copied off Kaggle and
watched on any machine with a clone and trimesh (no torch).

### Tools

| | |
|---|---|
| `scripts.benchmark_data` | per-stage data cost, loader throughput, and where to finish a batch — CPU or GPU, measured |
| `scripts.check_scene` | why a named scene fails: the mesh, fp32, AMP, or the gradient; `--locate` names the module |
| `scripts.scaling_sweep` | error against the number of objects, with a fixed budget *per object* and truncated runs excluded |
| `scripts.check_version` | a copy of the project that compiles and carries every fix, checked in a second |

Every tool takes the same `Config` flags as training (`scripts/config_flags.py`).

An earlier VN-GAT implementation fit small subsets (43° train error against a
126.5° chance baseline) but did not generalise — validation stayed near 94–102°
at every dataset size, which is close to the ~90° level expected if a model
recovers a fragment's axis but not its azimuth. That result is what motivated
the current redesign, in which cross-fragment attention runs over
fracture-surface vertices rather than virtual nodes.

That ~90° level is a **landmark, not a floor**, and the earlier design's own
later measurements refute reading it as one: a fragment of a symmetric object
is not itself symmetric, because its fracture boundary is jagged and unique,
and a scaling run on eight Everyday objects reached 30.9°. The tilt/twist split
in `reassembly.evaluation.metrics` distinguishes the two readings directly.

Reference values used throughout, worth checking any number against:

| quantity | value |
|---|---|
| chance geodesic error (uniform residual) | 126.48° = π/2 + 2/π |
| chance Euler RMSE, **random** prediction | 83.25° |
| chance Euler RMSE, **identity** prediction | 83.18° |
| "axis correct, azimuth random" geodesic | 90.0° |
| untrained `L_normal`, `L_face` | 1.0, 2.0 |
| `L_position` on the unit sphere | 4/3 |

All measured by Monte Carlo in `tests/test_losses.py`.

Two things worth knowing about these. Euler RMSE and geodesic error are **not**
the same metric on the same prediction — a 30° single-axis error is 30°
geodesic but ≈17.3° Euler RMSE, and GARF's tables use Euler RMSE, so always say
which. And the comparable GARF row is the **vanilla Everyday supplementary**
table — SE(3)-Equiv 79.30°, GARF-mini 10.41° — not the 6.1° headline.

`euler_rmse` measures the Euler angles of the **residual** rotation
`predictedᵀ · target`, not the componentwise difference of two Euler triples.
The difference is not cosmetic: Euler angles are chart coordinates, so
subtracting two charts is not a metric on SO(3) and it manufactured a 3° gap in
which a model collapsing to the identity scored *better than guessing* while
having learned nothing. Measuring the residual closes that gap to 0.07°, which
is Monte-Carlo noise — the two rows above are the same number, and that is the
correct behaviour.

## Notes on the implementation

- `topology.unique_edges` defines the canonical edge ordering for the package.
  `trimesh.edges_unique` returns the same set in a different order; mixing the
  two silently misaligns every per-edge attribute.
- `resolve_duplicated_faces` returns faces ordered by sorted vertex triple, not
  in input order. Per-face artifacts (the packed masks) index by position, so
  this is part of the contract — do not "tidy" it by sorting.
- `diffuse_fragments` takes an explicit `numpy.random.Generator`. Drawing from
  global `np.random` gives correlated streams across DataLoader workers.
- `Scene.object_key` collapses fracture-mode variants, so `count_objects()`
  reports distinct shapes (746) rather than scene directories (1,442).
  `data.catalog.build_catalog` applies that grouping to the dataset itself,
  merging a shape's break patterns across variant directories. Without it a
  shape is two objects, and one copy can land in train while the other lands in
  val — so the object split is not an object split, and nothing raises.
- Two ways to hold data out, `--split_by object` (the benchmark) and
  `--split_by fracture` (unseen break patterns of known shapes). The second is
  a diagnostic; its numbers are not comparable with published results and the
  config prints that at startup.

## Citation

Built on Breaking Bad (Sellán et al., NeurIPS 2022 Datasets & Benchmarks) and
informed by GARF (Li et al., 2025), Vector Neurons (Deng et al., ICCV 2021), and
Tensor Field Networks (Thomas et al., 2018).
