# VN-GAT v5 — SO(3)-equivariant fracture reassembly on commodity GPUs

Predicts the rotation that reassembles the fragments of a broken 3D object
(Breaking Bad dataset), using a Vector-Neuron graph attention network, then
recovers translations with a classical closed-form solver.

**v5** is the Thesis 1 design (virtual nodes for cross-fragment communication)
with the mesh-edge attention layer and edge features of `E:\My Thesis Work`,
per-scene normalisation, per-fragment losses, held-out-object *and*
held-out-break-pattern splits, optional category balancing and a fixed
validation set. What changed and why, with the measurements behind it:
[`docs/v5-changes.md`](docs/v5-changes.md). Thesis 1 checkpoints do not load
into v5 (the layer's parameters differ); the trainer says so instead of failing
on a missing key.

The thesis claim is that the equivariance does the work that GARF needs
4×H100-scale compute for, so competitive accuracy should be reachable on 2×T4.

```
scattered fragments ──► VN-GAT (equivariant) ──► per-fragment rotation
                                                        │
                        invariant interface embeddings ──┴──► translation solver ──► assembled object
```

---

## Quick start

```bash
pip install -r requirements.txt

# check everything before spending GPU hours
pytest tests -q

# what training will see: objects, categories, how each split partitions them
python -m scripts.inspect_data --root data

# measure real throughput before planning a long run
python -m scripts.benchmark_data --config configs/kaggle_2xt4_full.yaml --root_dir data --num_scenes 25

# train on 2 GPUs (official Everyday split); --num_gpus 1 for one, N for more
python -m scripts.train --config configs/kaggle_2xt4_full.yaml --root_dir data

# curves, metrics, pictures -- evaluation reads the data definition from the checkpoint
python -m scripts.plot_history --checkpoint_dir checkpoints --out curves.png
python -m scripts.evaluate --checkpoint checkpoints/best.pt --split val --num_scenes 0
python -m scripts.visualize --mode prediction --checkpoint checkpoints/best.pt --out pred.glb
```

`notebooks/kaggle_train.ipynb` has the same thing as Kaggle cells.

Expected data layout (either source alone is fine):

```
data/
├── everyday_compressed/…/<category>/<shape>/{compressed_mesh.obj, compressed_data.npz, fractured_*/}
└── artifact_compressed/…/<shape>/{…}
```

---

## Layout

```
vngat/
├── config.py              one dataclass drives YAML + CLI (CLI > YAML > default)
├── data/
│   ├── io.py              igl decompression — LOGIC UNCHANGED from the original
│   ├── splits.py          scene discovery, official split lists, the stable hash
│   ├── catalog.py         objects (variants grouped), split modes, balancing, fixed val set
│   ├── mesh_ops.py        face adjacency, fracture extraction, shell extraction
│   ├── correspondence.py  vectorised cross-fragment interface matching
│   ├── features.py        mesh → node features; edges = canonical (n1, n2) + p_u − p_v
│   ├── graph.py           FragmentGraph / SceneBatch, per-scene normalisation, collation
│   └── dataset.py         BreakingBadDataset, collate, micro-batch splitting
├── models/
│   ├── vn_layers.py       VN primitives, the scale gate, the Gram-Schmidt rotation head
│   ├── segment_ops.py     scatter softmax/sum/mean (replaces PyTorch Geometric)
│   ├── gat_layer.py       the mesh-edge attention layer (same as My Thesis Work's)
│   ├── virtual_nodes.py   cross-fragment communication via K slots per fragment
│   ├── heads.py           invariant interface embeddings
│   └── vn_gat.py          the model (4 layers by default)
├── losses/composite.py    6-term objective, geometric terms averaged per fragment
├── training/              bridge, trainer, DDP, checkpointing, Drive, history
├── assembly/translation.py closed-form weighted-Laplacian solver + IRLS
└── evaluation/metrics.py  Euler RMSE, geodesic, RMSE(T), Chamfer, Part Accuracy
scripts/    train · evaluate · plot_history · benchmark_data · visualize
tests/      pytest suite (start with test_rotation_convention.py)
configs/    default · kaggle_2xt4_full · kaggle_2xt4_frac · smoke
```

---

## The rotation-convention fix

Every VN primitive is **left**-equivariant: `F(Ax) = A·F(x)`.
The training target is the rotation that undoes the scattering, `R_gt = Aᵀ`.

Returning the Gram-Schmidt frame `F` directly would require
`A·F(x_clean) = Aᵀ`, i.e. `F(x_clean) = (A²)ᵀ` — for **every** random `A`, from
the same clean geometry. Impossible. The target is unlearnable, and it fails
*silently*: loss plateaus near chance with no error.

Returning `Fᵀ` makes the head **right**-equivariant, `G(Ax) = G(x)·Aᵀ`. Now a
single orientation-independent thing has to be learned — `G(x_clean) = I` — and
`R_gt = Aᵀ` then follows automatically for every `A`.

Proved in `tests/test_rotation_convention.py`; derived in full in the
`predict_rotation` docstring.

**Equivariance has to be per-fragment, not global.** Diffusion rotates each
fragment independently, so fragment *f*'s output must be equivariant to its own
rotation and *invariant* to every other fragment's. Cross-fragment attention
over equivariant vectors satisfies the global check and fails this one — the
logit `<A_a q, A_b k>` is only invariant when `A_a = A_b`. Everything crossing a
fragment boundary in the virtual-node stage is therefore invariant, gating each
fragment's own vectors. Nothing is lost: relative orientation between scattered
fragments is random noise by construction. Guarded by
`test_virtual_node_block_is_equivariant_per_fragment`.

## Why batch_size=2 used to OOM on a 16 GB T4

| cause | fix | effect |
|---|---|---|
| dense masked attention over (17k vertices × all slots in batch), allocated 3× | segment attention over allowed pairs only | ~500× fewer logits, numerically identical |
| every head given the full hidden width | split width across heads | ÷ `heads` on the widest tensors |
| un-bottlenecked Gram matrix (95k edges × 131²) | equivariant bottleneck to 16 channels | 6.5 GB → ~100 MB |
| peak memory scaled with `batch_size` | micro-batching + gradient accumulation | peak depends on the largest *single scene* |
| cross-fragment attention allocated a batch-wide (Q, H, Q) score matrix | evaluated one scene block at a time, over invariant tokens | quadratic per scene, linear in batch |

Plus AMP, and gradient checkpointing engaged automatically only for a scene
that would otherwise not fit.

## Data-path constraints honoured

No caching. No decimation — the model sees every vertex and edge. `igl`
decompression untouched. Cost was removed instead by calling `get_features`
**once** per fragment (the diffused view is derived by rotating the extracted
vectors on GPU — exact for a rigid transform), storing edges undirected and
symmetrising on device, and vectorising correspondence.

## Reporting caveat

`rmse_R_euler_deg` (what GARF's tables use) and `geodesic_deg` (what the loss
uses) are different numbers — a 30° single-axis error is 30° geodesic but
≈17.3° as an Euler RMSE. Both are reported separately; state which one your
tables quote.

## Multi-session training on Kaggle

Checkpoints save every 10 epochs and replace the previous one **unless** the
previous is better on *both* train and validation `checkpoint_monitor` (rotation
by default); `best.pt` is written whenever that validation term improves,
checked every epoch -- not only on save epochs -- against a FIXED validation set
(same scenes, same rotations every epoch), so "best" is not chosen on sampling
noise. Training stops cleanly at
`time_budget_hours` so a 12-hour session ends resumable rather than killed.
Set `drive_folder_id` + `drive_credentials` to mirror to Google Drive — see the
setup steps in `vngat/training/drive.py`, especially sharing the folder with
the service account, without which uploads fail with `storageQuotaExceeded`.
Re-running the same command with `resume: auto` continues where it left off.

## Precision: train in fp32, not AMP

Measured, not assumed. Same seed, same configuration, only precision differing,
on 8 objects:

| epoch | AMP (fp16) | fp32 |
|---|---|---|
| 62 | 79.11 | 74.06 |
| 70 | 85.24 | 62.82 |
| 85 | 65.84 | **43.18** |

AMP produced non-finite losses from epoch 66 onward on six of the eight
training objects, silently excluding them; fp32 ran 90 epochs with none. And
the corruption preceded the visible NaN -- fp32 was already 5-19 degrees ahead
at epochs 62-65.

Cost is ~10-20% wall clock, not the ~2x one might expect: this model is
dominated by scatter/gather and many small matmuls rather than the large GEMMs
tensor cores accelerate. Peak memory goes from ~1.8 GB to ~4 GB.

`amp` therefore defaults to False, and the trainer warns if it is turned on.

## Numerical notes

Segment reductions accumulate in float32 even under AMP: they run over whole
fragments, and float16 silently drops terms once a running sum exceeds ~2048.
The Gram–Schmidt head normalises by a *clamped* norm rather than `norm + eps`,
which keeps `R_pred` orthogonal to machine precision instead of drifting by
`eps/‖a‖` — 3% at small activations, which is what an untrained network emits.

## Benchmark protocol

The Breaking Bad release ships its own partition in `data_split/*.txt`
(everyday: 407 train / 91 val; artifact: 164 / 40). Published leaderboards are
computed on that exact partition, so `split_source: official` is required for
any number quoted against them; the hash split is for when those lists are
absent. There is no official TEST split -- evaluate on `val`.

GARF reports two tables: the main one on the volume-constrained version, and a
supplementary one on the vanilla version "to align with the settings of
previous methods". The vanilla Everyday table is the relevant comparison for
this project, and the nearest baseline in it is SE(3)-Equiv at 79.30 degrees
RMSE(R). GARF-mini -- trained, like this model, on the Everyday subset alone --
reaches 10.41.

`volume_constrained-*` contains the SAME objects under a different fracture
mode, not additional shapes. `data_subsets` excludes it by default; if it is
included, the catalogue groups each shape's copies into ONE object with the
union of their break patterns, so nothing is double-counted and one shape can
never sit in two splits.

### The benchmark's 2-20 pieces

Breaking Bad's break patterns run from 2 to 100 pieces (GARF, Table 1). GARF
trains and reports on those of 2 to 20 (section 4.5: "only been trained on data
with 2-20 fragments"; supplementary C.5 calls it the common setting), so a
number is comparable with its tables only from a run on the same range:

```bash
python -m scripts.train --config configs/kaggle_2xt4_full.yaml --max_fragments 20
python -m scripts.evaluate --checkpoint checkpoints/best.pt --split val --max_fragments 20
```

* `max_fragments: 0` (the default) is no limit: every pattern, as before.
* The limit filters the list of break patterns when the run starts, reading
  each one's piece count from its `compressed_fracture.npy` -- one small file,
  not a decompression. Objects left with no pattern drop out, and the banner
  prints what each split kept.
* Applied after the split, so no pattern moves between train and validation,
  and before `max_scenes`, which then counts objects that still have one.
* Restored on resume. A resume onto a different limit validates on different
  scenes, so the best-so-far is reset -- better to start a fresh run.
* `scripts.evaluate --max_fragments N` overrides the checkpoint's own value:
  an existing model can be scored on 2-20 without retraining.

## The largest fragment as the anchor

`R_gt` is each fragment's rotation back into the frame its object is *stored*
in. For a shape the model has never seen, that frame cannot be read off the
input -- most Everyday objects are round, so the turn of the stored object
about its axis is arbitrary -- and Thesis 1 and My Thesis Work both sat at
chance on the official-split validation while their training error fell. The
benchmark asks something else: GARF and PuzzleFusion++ fix each scene's largest
fragment at its true pose and score the others relative to it. This version
does the same (`vngat/evaluation/anchor.py`):

* **Metrics, always.** Every epoch reports `anchor_deg` (each scene's
  prediction turned so its largest fragment is exact; the other fragments'
  error) and `absolute_deg` (every fragment in its stored frame), on one
  `[val ]` line. Chance is 126.47 deg for both. tilt/twist and the
  per-category table are the anchor's; `best.pt` is chosen on `anchor_deg`.
  `scripts.evaluate` scores this way, translations measured from the anchor's.
* **Loss, by default** (`--rotation_target anchor`): rot, pos, node and face
  compare the anchor-aligned prediction, over every fragment but the anchors.
  `--rotation_target absolute` is Thesis 1's target.
* **Unchanged:** the model, the embedding terms, the data, every other flag.
* **Resuming a checkpoint written before this** continues it on the absolute
  target it was trained on, monitoring what it monitored; pass
  `--rotation_target anchor` to switch it (announced, and best-so-far reset).

```bash
# score an existing checkpoint both ways, no retraining
python -m scripts.evaluate --checkpoint checkpoints/best.pt --split val
```

## Split modes, balancing, validation

| setting | what it does |
|---|---|
| `split_by: object` (default) | validation holds out whole shapes -- the benchmark's question |
| `split_by: fracture` | validation holds out *break patterns* of known shapes -- easier; a diagnostic, never a benchmark number. `fracture_pool: train` (default) keeps the held-out shapes clean |
| `balance: category` | draws each category equally often (`balance_temperature` 0..1 softens it); the banner prints the resulting shares and effective object count |
| `val_fixed: true` (default) | same validation scenes and rotations every epoch; `val_scenes: 0` = one pattern per object, sharded across GPUs |

At chance on the object split but good on the fracture split: the model has
learned per-shape orientations that do not transfer. At chance on both: the
problem is upstream of generalisation. The validation log breaks rotation error
down by category every epoch.

## Known open items

- No v5 run to convergence yet; loss weights are all 1.0 and untuned.
- The edge relative position is short next to the unit face normals it travels
  with (mesh resolution over the largest fragment's radius); the layer has to
  learn any gain between them. `scripts.benchmark_data` prints the median on
  your data.
- `resolve_collisions` uses bounding spheres, not true volumetric overlap.
- The translation solver depends on the invariant embeddings being good;
  `num_matches` in the evaluation output is the diagnostic to watch.
