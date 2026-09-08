#!/usr/bin/env python
"""
Train on a Colab TPU with TensorFlow.

    # Runtime -> Change runtime type -> TPU
    python -m scripts.train_tpu --report_buckets --root_dir data      # measure first
    python -m scripts.train_tpu --config configs/colab_tpu.yaml --root_dir data \
        --checkpoint_dir /content/drive/MyDrive/vngat/tpu1

`notebooks/colab_tpu.ipynb` has this as runnable cells, including Drive mounting
and the dataset setup.

WHY TENSORFLOW HERE. torch_xla's Colab install is version-fragile; TF talks to
Cloud TPUs through `TPUStrategy` with no extra package. The model, losses and
metrics are a direct port of the PyTorch branch and are verified against the
same numerical properties -- equivariance to ~1e-15 in float64, chance levels,
perfect-prediction-gives-zero-loss.

WHAT DIFFERS FROM THE GPU (PYTORCH) BRANCH
------------------------------------------
* STATIC SHAPES. XLA compiles per shape combination, so every batch is padded
  into buckets by `vngat.data.padding`. Without it, varying vertex, edge and
  fragment counts would trigger a recompilation nearly every step.
* `TPUStrategy` replaces DDP. The gradient reduction is handled by the strategy.
* NO OOM LADDER. TPU memory is allocated per compiled program, so a bucket that
  does not fit is a configuration error rather than something to retry.
* fp32 by default. Half precision measured both unstable and WORSE on this model
  in the PyTorch branch (43.2 vs 54.3 degrees at matched seed and steps), and
  produced non-finite losses on six of eight training objects.

MEASURE BEFORE COMMITTING A SESSION
-----------------------------------
`--report_buckets` prints how many distinct shapes XLA will compile and how much
compute the padding wastes. On Breaking Bad sizes that is ~64 compilations and a
median edge padding factor of 1.5x, so the TPU must be more than 1.5x faster on
this workload just to break even. The model is small (376k parameters) and
scatter/gather-heavy, which is not where TPUs are strong -- treat the comparison
as an open question, not a foregone win.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from vngat.config import parse_config  # noqa: E402
from vngat.data.dataset import BreakingBadDataset, collate_fn  # noqa: E402
from vngat.data.padding import bucket_report  # noqa: E402
from vngat.losses.composite import CompositeLoss  # noqa: E402
from vngat.models.vn_gat import VNGATModel  # noqa: E402
from vngat.training.bridge import (  # noqa: E402
    build_predictions, build_targets, prepare_scene, to_tensors,
)
from vngat.training.checkpoint import CheckpointManager  # noqa: E402
from vngat.training.history import History  # noqa: E402
from vngat.utils.env import seed_everything  # noqa: E402

_LOSS_KEYS = ("total", "rot", "rot_deg", "pos", "node", "mid", "face",
              "emb_v", "emb_e", "head_cos", "tilt", "twist")


def _subsets(cfg):
    return [s for s in cfg.data_subsets.split(",") if s.strip()] or None


def _make_dataset(cfg, split, length):
    return BreakingBadDataset(
        root_dir=cfg.root_dir, split=split, val_frac=cfg.val_frac, test_frac=cfg.test_frac,
        split_seed=cfg.split_seed, max_scenes=cfg.max_scenes, subsets=_subsets(cfg),
        split_source=cfg.split_source, fracture_pattern=cfg.fracture_pattern or None,
        input_source=cfg.input_source, with_correspondence=cfg.correspondence,
        correspondence_tol=cfg.correspondence_tol, min_fragments=cfg.min_fragments,
        nominal_length=length)


def report_buckets(cfg, num_scenes: int = 60) -> int:
    ds = _make_dataset(cfg, "train", num_scenes)
    sizes = []
    for i in range(num_scenes):
        g = ds[i]["target"]
        sizes.append((g.num_nodes, g.num_edges, g.num_fragments))
    n = np.array([s[0] for s in sizes]); e = np.array([s[1] for s in sizes])
    print(f"sampled {len(sizes)} scenes")
    print(f"  nodes median {np.median(n):,.0f}  max {n.max():,}")
    print(f"  edges median {np.median(e):,.0f}  max {e.max():,}")
    print(bucket_report(sizes))
    print("\nA large padding factor means the TPU spends most of its time on padding.")
    print("Compare epoch time against the GPU branch before committing a session.")
    return 0


def resolve_strategy(force_cpu: bool = False):
    """TPU if one is attached, otherwise the default strategy, so the same
    script runs unchanged on CPU for smoke tests."""
    if force_cpu:
        return tf.distribute.get_strategy(), "cpu"
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver), "tpu"
    except Exception as exc:  # noqa: BLE001
        print(f"no TPU ({type(exc).__name__}); falling back to the default strategy")
        return tf.distribute.get_strategy(), "cpu"


def run_epoch(model, loss_fn, dataset, optimizer, cfg, train: bool, pad: bool):
    totals = {k: 0.0 for k in _LOSS_KEYS}
    count = 0
    for _ in range(cfg.steps_per_epoch if train else cfg.val_steps):
        samples = [dataset[i] for i in range(cfg.batch_size)]
        batch = collate_fn(samples)
        scene = prepare_scene(batch, pad=pad)
        inputs = to_tensors(scene["diffused_input"])
        targets = build_targets(scene["clean_target"], scene["rot"], scene["diffused_input"])
        for key in ("node_mask", "edge_mask", "frag_mask"):
            if key in scene:
                targets[key] = scene[key]

        def forward():
            out = model(**inputs, training=train)
            merged = dict(R_pred=out["R_pred"],
                          vertex_embedding=out["vertex_embedding"],
                          edge_embedding=out["edge_embedding"],
                          **build_predictions(scene["diffused_target"], out["R_pred"]))
            losses = loss_fn(merged, targets)
            losses["head_cos"] = out["head_cos"]
            return losses

        if train:
            with tf.GradientTape() as tape:
                losses = forward()
            grads = tape.gradient(losses["total"], model.trainable_weights)
            if cfg.grad_clip > 0:
                grads, _ = tf.clip_by_global_norm(
                    [g if g is not None else tf.zeros_like(w)
                     for g, w in zip(grads, model.trainable_weights)], cfg.grad_clip)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        else:
            losses = forward()

        for k in _LOSS_KEYS:
            totals[k] += float(losses[k])
        count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--report_buckets", action="store_true")
    pre.add_argument("--cpu", action="store_true", help="Force the CPU strategy.")
    pre.add_argument("--no_pad", action="store_true",
                     help="Skip bucket padding. Only sensible off-TPU.")
    known, rest = pre.parse_known_args(argv)
    cfg = parse_config(rest)

    if known.report_buckets:
        return report_buckets(cfg)

    strategy, kind = resolve_strategy(known.cpu)
    pad = not known.no_pad and kind == "tpu"
    seed_everything(cfg.seed)

    train_set = _make_dataset(cfg, "train", cfg.steps_per_epoch * cfg.batch_size)
    val_set = _make_dataset(cfg, "val", cfg.val_steps * cfg.batch_size)

    with strategy.scope():
        model = VNGATModel(
            hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
            num_vn_slots=cfg.num_vn_slots, heads=cfg.heads, embed_dim=cfg.embed_dim,
            gram_bottleneck=cfg.gram_bottleneck, norm=cfg.norm)
        replicas = strategy.num_replicas_in_sync
        floor = cfg.lr_min / max(cfg.lr, 1e-12)
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.lr * replicas,
            decay_steps=max(cfg.epochs, 1), alpha=floor)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule, weight_decay=cfg.weight_decay)

    manager = CheckpointManager(cfg.checkpoint_dir, cfg.save_every, None, cfg.tag)
    history = History()
    start_epoch = 0
    if cfg.resume not in ("", "none", "None"):
        path = manager.locate(cfg.resume)
        if path is not None:
            # One forward pass first, so the weights exist to load into.
            probe = prepare_scene(collate_fn([train_set[0]]), pad=pad)
            model(**to_tensors(probe["diffused_input"]))
            meta = manager.load(path, model, optimizer)
            start_epoch = int(meta.get("epoch", -1)) + 1
            history = History.from_dict(meta.get("history") or {})
            history.truncate_to(start_epoch)
            print(f"  [ckpt] resumed from {path} at epoch {start_epoch}")
            print(f"  [ckpt] --epochs {cfg.epochs} is a TOTAL, so "
                  f"{max(0, cfg.epochs - start_epoch)} epoch(s) remain")

    loss_fn = CompositeLoss(
        w_rot=cfg.w_rot, w_pos=cfg.w_pos, w_node=cfg.w_node, w_mid=cfg.w_mid,
        w_face=cfg.w_face, w_emb_v=cfg.w_emb_v, w_emb_e=cfg.w_emb_e,
        emb_pull_margin=cfg.emb_pull_margin, emb_push_margin=cfg.emb_push_margin,
        symmetry_axis=cfg.symmetry_axis)

    print(f"VN-GAT (TensorFlow) on {kind} | {replicas} replica(s) | padding={pad}")
    print("chance level: geodesic 126.47 deg")
    if pad:
        print("first epochs are dominated by XLA compilation; judge speed from epoch 5 on")

    wall = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        tr = run_epoch(model, loss_fn, train_set, optimizer, cfg, True, pad)
        va = run_epoch(model, loss_fn, val_set, optimizer, cfg, False, pad)
        history.append("train", tr)
        history.append("val", va)
        lr = float(optimizer.learning_rate(optimizer.iterations)
                   if callable(optimizer.learning_rate) else optimizer.learning_rate)
        history.append_meta(lr=lr, epoch_seconds=time.perf_counter() - wall)

        print(f"  epoch {epoch:>4} train total {tr['total']:8.4f} deg {tr['rot_deg']:7.2f} "
              f"hcos {tr['head_cos']:.2f} | val deg {va['rot_deg']:7.2f} "
              f"tilt {va['tilt']:6.2f} twist {va['twist']:6.2f} | lr {lr:.2e}")
        verdict = ("  <- axis learned, azimuth NOT (structural floor)"
                   if va["tilt"] < 25 and va["twist"] > 60 else
                   "  <- axis not learned either (headroom remains)"
                   if va["tilt"] > 60 else "")
        if verdict:
            print(f"  [val ] tilt {va['tilt']:6.2f} twist {va['twist']:6.2f}{verdict}")

        if manager.should_save(epoch, cfg.epochs):
            wrote = manager.save(model, optimizer, epoch, tr["total"], va["total"],
                                 history.to_dict(), cfg.to_dict())
            print(f"  [ckpt] epoch {epoch}: wrote {', '.join(k for k, v in wrote.items() if v)}")
        if (time.perf_counter() - wall) > cfg.time_budget_hours * 3600:
            print(f"  [budget] stopping cleanly at epoch {epoch}")
            manager.save(model, optimizer, epoch, tr["total"], va["total"],
                         history.to_dict(), cfg.to_dict(), force=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
