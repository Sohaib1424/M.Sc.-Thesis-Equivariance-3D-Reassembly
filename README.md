# VN-GAT — TensorFlow / Colab TPU branch

SO(3)-equivariant graph network for 3D fracture reassembly. This branch exists
so the model can run on a Colab TPU without torch_xla, whose Colab install is
version-fragile. The PyTorch branch remains the reference for GPU work.

## Quick start

```bash
pip install -q libigl trimesh pyyaml            # TF ships with Colab
python -m pytest tests -q                       # 31 tests

python -m scripts.train_tpu --report_buckets --root_dir data    # measure FIRST
python -m scripts.train_tpu --config configs/colab_tpu.yaml --root_dir data \
    --checkpoint_dir /content/drive/MyDrive/vngat/tpu1
```

`notebooks/colab_tpu.ipynb` has the whole thing as runnable cells.

## What is verified, and how

The port is not assumed correct; each property is asserted numerically in
float64, where a correct implementation gives ~1e-15 and float32 would give
~1e-7 for a correct *and* a broken one.

| property | residual |
|---|---|
| VNLinear / VNLeakyReLU / VNLayerNorm equivariant | ~5e-16 |
| VNInvariant invariant, and bottlenecked | 5e-15 |
| Gram-Schmidt gives proper rotations (det +1) | 1e-15 |
| head is RIGHT-equivariant, `G(Ax) = G(x)Aᵀ` | 1e-15 |
| head recovers `Aᵀ` for every `A` | 4e-16 |
| VNGATLayer equivariant | 9e-16 |
| VirtualNodeBlock **per-fragment** equivariant | 6e-16 |
| full model per-fragment equivariant | 2.5e-15 |
| padded loss == unpadded loss | 0.0 to 6e-16 |
| perfect prediction ⇒ zero geometric loss | 1e-31 |
| **parameter count vs PyTorch branch** | **376,592 = 376,592** |

Chance levels reproduce theory: geodesic 126.29 against an analytic 126.47,
Euler RMSE 83.17 against 83.20, tilt 90.03.

## Differences from the PyTorch branch

* **Static shapes.** XLA compiles per shape combination, so batches are padded
  into buckets (`vngat/data/padding.py`). Measured on Breaking Bad sizes: ~64
  distinct shapes, median edge padding 1.50x.
* **`TPUStrategy`** replaces DDP; no OOM ladder, since TPU memory is allocated
  per compiled program.
* **Data containers hold numpy**, converting to tensors once at the device
  boundary in `training/bridge.py`. That keeps the whole data pipeline shared
  with the PyTorch branch.
* **No AMP flag.** Half precision measured both unstable and worse in the
  PyTorch branch (43.2 vs 54.3 degrees at matched seed and steps).
* Config drops the torch-only knobs (`micro_batch_scenes`, `amp`, `num_gpus`,
  `master_port`, `grad_checkpointing`, `restart_schedule`) — 54 fields.

## One improvement that should go back to PyTorch

`gram_schmidt_frame` now **pre-normalises both channels to unit length**.
Mathematically identical, since the frame depends only on directions, but it
makes the routine scale-invariant. Measured before the change: the frame drifts
2e-9 from orthogonal at ‖a‖ ~ 1e-3 and **2e-3** at ‖a‖ ~ 1e-6, worst for pairs
whose channels are nearly parallel — exactly the high-`head_cos` regime the real
training runs sit in. After: 4e-15 across twelve orders of magnitude.

## Reading a run

Chance is **126.47 deg** geodesic. Reference levels are in
`vngat/evaluation/metrics.py`.

The `[val ] tilt / twist` line each epoch is the diagnostic that matters:
tilt ~ 0 with twist ~ 90 means the axis is learned but the azimuth is not
recoverable (structural — no tuning moves it); tilt ~ 90 means the axis is not
learned either and there is real headroom.

`hcos` is the Gram-Schmidt head's conditioning. Real runs sat at 0.8–0.9 while
learning nothing and dropped to ~0.2 exactly when they started making progress.
