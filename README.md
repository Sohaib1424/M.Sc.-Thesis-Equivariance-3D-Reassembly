# Equivariant fracture reassembly

SO(3)-equivariant graph networks for reassembling broken 3D objects, built to
train on commodity GPUs rather than a datacenter.

The reference system (GARF) trains an SE(3)-equivariant flow-matching
transformer on 4×H100 for 72 hours. This project targets competitive accuracy
at a small fraction of that, through three choices:

1. **Richer edge features** — length, midpoint, and both adjacent face normals,
   rather than positions alone.
2. **A two-stage split** — rotation is learned; translation is solved exactly by
   classical geometric optimisation.
3. **Vector Neurons** instead of a heavy equivariant transformer.

---

## Read this before running anything

**Nothing here has been trained.** The pipeline is verified as correctly
*wired*, not as *effective*. There is no convergence curve, no accuracy number,
and no confirmation that the loss weights are sensible.

**The torch layer has never been executed.** It was built in an environment
without torch, torch-geometric, or trimesh. What backs it:

- 66 tests **actually run and pass** — the numpy layer (splits, mesh IO,
  decimation, correspondence, the translation solver, all metrics).
- 6 numerical validations of the underlying mathematics, run as numpy
  prototypes before the torch code was written.
- Static analysis: 49/49 files compile, 30/30 modules import, 193/193
  cross-module names resolve.
- Hand-tracing of every shape and index convention in the new attention code.
- **One real scene** (a Breaking Bad bottle, 100 fracture patterns) driven
  through loading, decimation, and correspondence detection. That run found a
  live bug that silently destroyed the interface supervision — see
  `docs/CHANGES.md` §13. Everything downstream of correspondence detection is
  still unexercised on real data.

That is real evidence, and it is not the same as running. **Run `pytest` in your
environment first.** Five test modules covering the model, losses, virtual
nodes, and pipeline are written and waiting; they will execute the moment torch
is present.

---

## Install

Install torch and torch-geometric **first**, matching your CUDA version — they
are not in `requirements.txt` because the correct wheel depends on your driver.

```bash
# check your CUDA version first
nvidia-smi

# example: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

# then everything else
pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

---

## Quickstart

```bash
# 1. does the whole pipeline hold together on real data?
python scripts/smoke_test.py --root-dir /path/to/breaking-bad

# 2. what actually fits on this GPU?
python scripts/profile_memory.py --sweep

# 3. train
python scripts/train.py --config configs/default.yaml --root-dir /path/to/data

# 4. score it
python scripts/evaluate.py --checkpoint checkpoints/best.pt --rotation-only
python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

Run the smoke test first. It prints shapes and values at every stage and checks
the equivariance law, the centralisation identity, and gradient coverage — a
clean run means the pipeline is correctly wired end to end.

---

## Kaggle: 2×T4, 16 GB each

```bash
python scripts/train.py --config configs/kaggle_t4x2.yaml \
    --root-dir /kaggle/input/<your-dataset-slug> \
    --cache-dir /kaggle/working/scene_cache \
    --num-gpus 2
```

`--root-dir` overrides whatever path is in the YAML, so you do not have to edit
the config to point at your dataset. `--num-gpus 2` spawns one process per GPU
via DDP; run it as `!python scripts/train.py ...` from a notebook cell (a
subprocess), not by importing `main()`, because `mp.spawn` needs a real process
to fork from.

Sessions cap at 12 hours. The Kaggle config sets `resume: auto` and
`time_budget_hours: 11.5`, so training stops cleanly before the cap and picks
itself up next time — **re-run the same cell, nothing to edit**. Full state
(optimizer, scheduler, AMP scale, RNG, history) is carried across, and the loss
curve stays continuous. See `docs/RESUMING.md`.

What makes it fit, in order of effect:

| setting | value | why |
|---|---|---|
| `data.decimate_to` | 15000 | caps vertices per scene, so peak memory follows the config rather than the largest scene in the dataset |
| `model.gradient_checkpointing` | true | ~4× less activation memory for ~30% more compute |
| `train.amp` / `amp_dtype` | true / **fp16** | halves every activation. T4 is Turing — fp16 works, **bf16 does not** |
| `train.batch_size` × `accum_steps` | 2 × 4 | effective batch of 8 per rank (16 across both GPUs) at the memory cost of 2 |
| `data.num_workers` | 2 | 2 ranks × 2 workers on a ~4-core box; scene loading is CPU-bound mesh work |
| `data.cache_dir` | `/kaggle/working/scene_cache` | preprocessing measured at 49% of epoch wall-clock, and it is deterministic — so it is paid once |

If it still OOMs, lower `decimate_to` first — it is the lever with the most
headroom and the least effect on the model.

Training prints a `data=` versus `compute=` split each epoch. If `data` dominates,
the bottleneck is mesh loading, not the GPU, and shrinking the model will not help.

---

## Scripts

| script | what it does |
|---|---|
| `smoke_test.py` | full pipeline on real data, with per-stage diagnostics |
| `train.py` | training, single-GPU or DDP |
| `evaluate.py` | GARF-comparable metrics, in three modes |
| `profile_memory.py` | measured peak memory on synthetic graphs of a chosen size |
| `benchmark_compute.py` | throughput, memory, projected wall-clock and GPU-hours |
| `plot_history.py` | loss curves, with run-comparison overlay |
| `vis_prediction.py` | a trained model's reassembly next to the ground truth |
| `visualize/` | scene, scattered scene, and fracture-surface viewers |

From a notebook, call `main([...])` explicitly rather than relying on `sys.argv`
— in Jupyter that holds the kernel's launch arguments and argparse exits with a
confusing `SystemExit: 2`:

```python
from scripts.train import main
main(["--config", "configs/kaggle_t4x2.yaml", "--num-gpus", "2"])
```

---

## Evaluation: use all three modes

```bash
python scripts/evaluate.py --checkpoint ckpt.pt --rotation-only     # stage 1 alone
python scripts/evaluate.py --checkpoint ckpt.pt                     # full assembly
python scripts/evaluate.py --checkpoint ckpt.pt --oracle-rotation   # stage 2 alone
```

Reporting only the middle one leaves "is the remaining error from the network or
from the solver?" unanswered, and that is the first question any reader has.

**On rotation error.** `RMSE_R_euler_deg` is the number comparable to published
Breaking Bad and GARF tables. `RMSE_R_geodesic_deg` is a different convention —
measured at roughly 0.6–0.7× the geodesic value, distribution-dependent. Do not
mix them in one table. Confirm the Euler *sequence* (`xyz` vs `zyx`) against the
GARF reference implementation before quoting numbers.

**On the matcher.** The correspondence filter defaults are uncalibrated against
real trained embeddings. Run `--match-report` to get the distance quantiles from
your own model, then set `max_distance` explicitly.

---

## Substantiating the efficiency claim

The claim has two halves and a loss curve measures only one:

```bash
python scripts/evaluate.py --checkpoint ckpt.pt --out results/accuracy.json
python scripts/benchmark_compute.py --config configs/kaggle_t4x2.yaml --out results/cost.json
```

`benchmark_compute.py` reports parameters, seconds per step, scenes per second,
peak memory, and projected GPU-hours against GARF's 288 H100-GPU-hours. It
measures **compute only**, on synthetic graphs — it excludes data loading and
says nothing about accuracy. State both halves together.

---

## Experiments the code supports

```bash
# input source: full mesh vs fracture surface only
python scripts/train.py --config configs/frac_input.yaml
python scripts/plot_history.py --checkpoint-dir checkpoints --compare checkpoints/frac

# the vertex-budget / accuracy trade-off as a curve
python scripts/benchmark_compute.py --sweep-budget
```

Two components are **off by default and unvalidated for accuracy**:
`model.angular` (the triplet block) and `loss.auto_balance`. Both are
equivariant and shape-correct, neither has been shown to help on this task. They
are ablation arms, not part of the baseline. The angular block is also the most
memory-hungry optional component — budget for roughly double activation memory.

---

## Documentation

- **`docs/CHANGES.md`** — every change, why it mattered, and the evidence, with
  explicit confidence labels. The rotation-convention finding (§1) is
  thesis-relevant, not cosmetic; §13 covers what the first real scene exposed.
- **`docs/ARCHITECTURE.md`** — how the model works and the invariants the code
  depends on.
- **`docs/RESUMING.md`** — training across 12-hour sessions: what is saved,
  how the history stays continuous, and how to check a resume worked.
- **`docs/TRAINING_BUDGET.md`** — how long training actually takes, derived
  from measured throughput: epochs, wall-clock, FLOPs, and the GARF comparison
  with its assumptions stated.

---

## Project layout

```
configs/     default, kaggle_t4x2, smoke, frac_input
scripts/     CLI entry points
src/reassembly/
  data/        splits, mesh IO, decimation, correspondence, features, batching
  models/      Vector Neurons, equivariant GAT, virtual nodes, rotation head
  training/    losses, bridge, engine, distributed
  assembly/    stage 2 — matching and the translation solver (pure numpy)
  evaluation/  GARF-comparable metrics (pure numpy)
tests/       12 modules; 66 run without torch
docs/        CHANGES.md, ARCHITECTURE.md
```

`reassembly.assembly` and `reassembly.evaluation` import without torch, which is
why most of the test suite runs anywhere.
