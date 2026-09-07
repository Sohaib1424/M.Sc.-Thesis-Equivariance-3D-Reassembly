# VN-GAT — SO(3)-equivariant fracture reassembly on commodity GPUs

Predicts the rotation that reassembles the fragments of a broken 3D object
(Breaking Bad dataset), using a Vector-Neuron graph attention network, then
recovers translations with a classical closed-form solver.

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

# measure real throughput before planning a long run
python -m scripts.benchmark_data --root_dir data --num_scenes 25 --batch_size 2

# train on 2 GPUs
python -m scripts.train --config configs/kaggle_2xt4_full.yaml --root_dir data

# curves, metrics, pictures
python -m scripts.plot_history --checkpoint_dir checkpoints --out curves.png
python -m scripts.evaluate --checkpoint checkpoints/best.pt --split test --num_scenes 200
python -m scripts.visualize --mode prediction --checkpoint checkpoints/best.pt --out pred.glb
```

`notebooks/kaggle_train.py` has the same thing as copy-paste Kaggle cells.

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
│   ├── splits.py          deterministic hash-based train/val/test
│   ├── mesh_ops.py        face adjacency, fracture extraction, shell extraction
│   ├── correspondence.py  vectorised cross-fragment interface matching
│   ├── features.py        mesh → node/edge features
│   ├── graph.py           FragmentGraph / SceneBatch, merging, collation
│   └── dataset.py         BreakingBadDataset, collate, micro-batch splitting
├── models/
│   ├── vn_layers.py       VN primitives + the Gram-Schmidt rotation head
│   ├── segment_ops.py     scatter softmax/sum/mean (replaces PyTorch Geometric)
│   ├── gat_layer.py       equivariant multi-head graph attention
│   ├── virtual_nodes.py   cross-fragment communication via K slots per fragment
│   ├── heads.py           invariant interface embeddings
│   └── vn_gat.py          the model (4 layers by default)
├── losses/composite.py    7-term objective
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
previous is better on *both* train and validation loss; `best.pt` is kept
separately on validation improvement. Training stops cleanly at
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
mode, not additional shapes, which is why `data_subsets` excludes it by
default: mixing the two variants would double-count objects and make the
official split lists ambiguous.

## Known open items

- No run to convergence yet; loss weights are all 1.0 and untuned.
- `resolve_collisions` uses bounding spheres, not true volumetric overlap.
- The translation solver depends on the invariant embeddings being good;
  `num_matches` in the evaluation output is the diagnostic to watch.
