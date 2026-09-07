#!/usr/bin/env python
"""
Train on a Kaggle TPU (v3-8), via torch_xla.

    pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    python -m scripts.train_tpu --config configs/kaggle_tpu.yaml --root_dir data \
        --checkpoint_dir /kaggle/working/tpu1 --data_subsets everyday_compressed \
        --split_source official --epochs 300 --batch_size 8 --lr 1e-3

A SEPARATE ENTRY POINT, deliberately. The GPU trainer took a long time to get
right (DDP collective ordering, the OOM ladder, NaN handling, checkpoint
policy); XLA needs different versions of several of those, and interleaving
both into one loop would put the working path at risk for the sake of the
experimental one.

WHAT IS DIFFERENT FROM THE GPU PATH
-----------------------------------
* STATIC SHAPES. XLA compiles per shape combination, so every batch is padded
  into buckets by `vngat.data.padding`. Without this, the varying vertex, edge
  and fragment counts would trigger a recompilation nearly every step.
* NO `.item()` IN THE LOOP. Reading a scalar forces a host sync and serialises
  the pipeline. Metrics are accumulated as device tensors and pulled once per
  epoch.
* `xm.optimizer_step` replaces `optimizer.step`, and performs the cross-replica
  gradient reduction.
* NO AMP. XLA uses bfloat16 through its own flags, not `torch.cuda.amp`; and
  fp32 was measured better for this model regardless.
* NO OOM LADDER. TPU memory is allocated per compiled program; the recovery
  ladder has nothing to retry differently, so a too-large bucket is a
  configuration error rather than something to catch.

CHECK THE PADDING COST FIRST
----------------------------
    python -m scripts.train_tpu --report_buckets --root_dir data

If the reported edge padding factor is large, the TPU is doing mostly wasted
work and the GPU path will be faster despite the slower chip.
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
import torch  # noqa: E402

from vngat.config import parse_config  # noqa: E402
from vngat.data.dataset import BreakingBadDataset, collate_fn  # noqa: E402
from vngat.data.padding import (  # noqa: E402
    DEFAULT_EDGE_BUCKETS, DEFAULT_FRAG_BUCKETS, DEFAULT_NODE_BUCKETS,
    bucket_report, pad_scene_batch,
)
from vngat.losses.composite import CompositeLoss  # noqa: E402
from vngat.models.vn_gat import VNGATModel  # noqa: E402
from vngat.training.bridge import (  # noqa: E402
    build_model_inputs, build_predictions, build_targets,
)
from vngat.training.checkpoint import CheckpointManager  # noqa: E402
from vngat.training.history import History  # noqa: E402
from vngat.utils.env import dataloader_worker_init, seed_everything  # noqa: E402

_LOSS_KEYS = ("total", "rot", "rot_deg", "pos", "node", "mid", "face",
              "emb_v", "emb_e", "head_cos", "tilt", "twist")


def report_buckets(cfg, num_scenes: int = 60) -> int:
    """Measure the padding overhead on real scenes before committing to a run."""
    ds = BreakingBadDataset(
        root_dir=cfg.root_dir, split="train",
        subsets=[s for s in cfg.data_subsets.split(",") if s.strip()] or None,
        split_source=cfg.split_source, fracture_pattern=cfg.fracture_pattern or None,
        nominal_length=num_scenes,
    )
    sizes = []
    for i in range(num_scenes):
        sample = ds[i]
        g = sample["target"]
        sizes.append((g.num_nodes, g.num_edges, g.num_fragments))
    n = np.array([s[0] for s in sizes]); e = np.array([s[1] for s in sizes])
    print(f"sampled {len(sizes)} scenes")
    print(f"  nodes median {np.median(n):,.0f}  max {n.max():,}")
    print(f"  edges median {np.median(e):,.0f}  max {e.max():,}")
    print(bucket_report(sizes))
    print("\nA large padding factor means the TPU spends most of its time on")
    print("padding. Compare against ~340 s/epoch on the 2xT4 before switching.")
    return 0


def _pad_and_move(batch, device):
    """Pad both graphs to bucket shapes and move everything to the TPU core."""
    target = pad_scene_batch(batch["target"])
    source = batch["input"]
    padded_in = target if source is None else pad_scene_batch(source)
    rot = batch["rot"]

    # Rotations must cover the padded fragment count; the extra entries are
    # identity and are masked out of every loss term.
    f_pad = target.graph.num_fragments
    if rot.shape[0] < f_pad:
        pad = torch.eye(3).unsqueeze(0).repeat(f_pad - rot.shape[0], 1, 1)
        rot = torch.cat([rot, pad], 0)

    g = target.graph.to(device)
    gi = padded_in.graph.to(device) if padded_in is not target else g
    rot = rot.to(device)
    diffused = g.rotate_per_fragment(rot)
    diffused_in = diffused if gi is g else gi.rotate_per_fragment(rot)
    return {
        "clean_target": g, "diffused_target": diffused, "diffused_input": diffused_in,
        "rot": rot,
        "node_mask": target.node_mask.to(device),
        "edge_mask": target.edge_mask.to(device),
        "frag_mask": target.frag_mask.to(device),
    }


def _worker(index, cfg):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = xm.xla_device()
    is_main = xm.is_master_ordinal()
    seed_everything(cfg.seed, index)

    subsets = [s for s in cfg.data_subsets.split(",") if s.strip()] or None
    common = dict(
        root_dir=cfg.root_dir, val_frac=cfg.val_frac, test_frac=cfg.test_frac,
        split_seed=cfg.split_seed, max_scenes=cfg.max_scenes, subsets=subsets,
        split_source=cfg.split_source, fracture_pattern=cfg.fracture_pattern or None,
        input_source=cfg.input_source, with_correspondence=cfg.correspondence,
        correspondence_tol=cfg.correspondence_tol, min_fragments=cfg.min_fragments,
    )
    train_set = BreakingBadDataset(
        split="train", nominal_length=cfg.steps_per_epoch * cfg.batch_size, **common)
    val_set = BreakingBadDataset(
        split="val", nominal_length=cfg.val_steps * cfg.batch_size, **common)
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn,
                         num_workers=cfg.num_workers, drop_last=True,
                         worker_init_fn=dataloader_worker_init)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=cfg.prefetch_factor)
    train_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_kwargs)

    model = VNGATModel(
        hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
        num_vn_slots=cfg.num_vn_slots, heads=cfg.heads, embed_dim=cfg.embed_dim,
        gram_bottleneck=cfg.gram_bottleneck, norm=cfg.norm,
        grad_checkpointing=cfg.grad_checkpointing,
    ).to(device)
    loss_fn = CompositeLoss(
        w_rot=cfg.w_rot, w_pos=cfg.w_pos, w_node=cfg.w_node, w_mid=cfg.w_mid,
        w_face=cfg.w_face, w_emb_v=cfg.w_emb_v, w_emb_e=cfg.w_emb_e,
        emb_pull_margin=cfg.emb_pull_margin, emb_push_margin=cfg.emb_push_margin,
        symmetry_axis=cfg.symmetry_axis)
    # Scale the rate by the replica count, as is standard when the effective
    # batch is multiplied by the number of cores.
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr * xm.xrt_world_size(),
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: max(cfg.lr_min / cfg.lr,
                                 0.5 * (1 + np.cos(np.pi * min(e, cfg.epochs) / cfg.epochs))))

    manager = CheckpointManager(cfg.checkpoint_dir, cfg.save_every, None, cfg.tag)
    history = History()
    start_epoch = 0
    if cfg.resume not in ("", "none", "None"):
        path = manager.locate(cfg.resume)
        if path is not None and path.is_file():
            state = manager.load(path, model, optimizer, scheduler, None, map_location="cpu")
            start_epoch = int(state.get("epoch", -1)) + 1
            history = History.from_dict(state.get("history") or {})
            history.truncate_to(start_epoch)
            model = model.to(device)
            if is_main:
                print(f"  [ckpt] resumed at epoch {start_epoch}")

    if is_main:
        print(f"VN-GAT on TPU | {model.num_parameters():,} params | "
              f"{xm.xrt_world_size()} cores | batch {cfg.batch_size}/core")
        print("chance level: geodesic 126.47 deg")

    wall = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            device_loader = pl.MpDeviceLoader(loader, device)
            # Accumulated ON DEVICE. Calling .item() per step would sync the
            # host with the TPU every iteration and serialise the pipeline.
            totals = {k: torch.zeros((), device=device) for k in _LOSS_KEYS}
            count = torch.zeros((), device=device)

            for batch in device_loader:
                scene = _pad_and_move(batch, device)
                if phase == "train":
                    optimizer.zero_grad()
                context = torch.enable_grad() if phase == "train" else torch.no_grad()
                with context:
                    out = model(**build_model_inputs(scene["diffused_input"]))
                    merged = dict(R_pred=out["R_pred"],
                                  vertex_embedding=out["vertex_embedding"],
                                  edge_embedding=out["edge_embedding"],
                                  **build_predictions(scene["diffused_target"], out["R_pred"]))
                    targets = build_targets(scene["clean_target"], scene["rot"],
                                            scene["diffused_input"])
                    targets.update(node_mask=scene["node_mask"],
                                   edge_mask=scene["edge_mask"],
                                   frag_mask=scene["frag_mask"])
                    merged["head_cos"] = out["head_cos"]
                    losses = loss_fn(merged, targets)
                    losses["head_cos"] = out["head_cos"]
                if phase == "train":
                    losses["total"].backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    xm.optimizer_step(optimizer)      # includes the all-reduce
                for k in _LOSS_KEYS:
                    totals[k] = totals[k] + losses[k].detach()
                count = count + 1

            metrics = {k: (totals[k] / count.clamp_min(1)) for k in _LOSS_KEYS}
            metrics = {k: float(xm.mesh_reduce(f"m_{k}", v.item(), lambda x: sum(x) / len(x)))
                       for k, v in metrics.items()}
            if is_main:
                history.append(phase, metrics)
                print(f"  epoch {epoch:>4} {phase:<5} total {metrics['total']:8.4f}  "
                      f"deg {metrics['rot_deg']:7.2f}  tilt {metrics['tilt']:6.2f}  "
                      f"twist {metrics['twist']:6.2f}  hcos {metrics['head_cos']:.2f}")

        scheduler.step()
        if is_main:
            history.append_meta(lr=optimizer.param_groups[0]["lr"],
                                epoch_seconds=time.perf_counter() - wall)
            if manager.should_save(epoch, cfg.epochs):
                # `_maybe_cpu` keeps the saved tensors framework-portable, so a
                # TPU checkpoint can be evaluated or resumed on a GPU.
                cpu_model = VNGATModel(
                    hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
                    num_vn_slots=cfg.num_vn_slots, heads=cfg.heads,
                    embed_dim=cfg.embed_dim, gram_bottleneck=cfg.gram_bottleneck,
                    norm=cfg.norm)
                cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
                manager.save(cpu_model, optimizer, scheduler, None, epoch,
                             history.data["train"]["total"][-1],
                             history.data["val"]["total"][-1],
                             history.to_dict(), cfg.to_dict())

        if (time.perf_counter() - wall) > cfg.time_budget_hours * 3600:
            if is_main:
                print(f"  [budget] stopping cleanly at epoch {epoch}")
            break


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--report_buckets", action="store_true")
    pre.add_argument("--num_cores", type=int, default=8)
    known, rest = pre.parse_known_args(argv)

    cfg = parse_config(rest)
    if cfg.amp:
        print("note: --amp has no effect on XLA; use XLA_USE_BF16=1 instead. Ignoring.")
        cfg.amp = False

    if known.report_buckets:
        return report_buckets(cfg)

    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        print("torch_xla is not installed. On Kaggle:\n"
              "  pip install torch_xla[tpu] -f "
              "https://storage.googleapis.com/libtpu-releases/index.html")
        return 1

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(str(Path(cfg.checkpoint_dir) / "config.yaml"))
    xmp.spawn(_worker, args=(cfg,), nprocs=known.num_cores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
