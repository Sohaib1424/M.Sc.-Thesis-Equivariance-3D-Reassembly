#!/usr/bin/env python
"""
Training entry point.

    # single GPU / CPU
    python scripts/train.py --config configs/kaggle_t4x2.yaml

    # two GPUs via DistributedDataParallel
    python scripts/train.py --config configs/kaggle_t4x2.yaml --num-gpus 2

    # override anything from the config on the command line
    python scripts/train.py --config configs/default.yaml \
        --set model.hidden_channels=64 --set data.decimate_to=10000

From a notebook, call ``main([...])`` explicitly rather than relying on
``sys.argv`` -- in Jupyter that holds the KERNEL's launch arguments (``-f
/path/to/connection.json``), which argparse tries to parse and then exits with
a confusing ``SystemExit: 2``:

    from scripts.train import main
    main(["--config", "configs/kaggle_t4x2.yaml", "--num-gpus", "2"])
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler

from reassembly.config import Config
from reassembly.data.collate import breaking_bad_collate_fn
from reassembly.data.dataset import BreakingBadDataset
from reassembly.models.vn_gat_model import VNGATModel
from reassembly.training.distributed import cleanup_distributed, setup_distributed
from reassembly.training.engine import fit
from reassembly.training.losses import CompositeLoss
from reassembly.utils.memory import estimate_activation_bytes


def _parse_value(text: str):
    """Coerce a --set value into the obvious Python type."""
    lowered = text.lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def build_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value",
                        help="Override a config value (repeatable)")
    # A few high-traffic shortcuts, so the common cases stay short.
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--input-source", type=str, default=None, choices=["full", "frac"])
    parser.add_argument("--decimate-to", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()

    overrides = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects section.key=value, got {item!r}")
        key, _, value = item.partition("=")
        overrides[key.strip()] = _parse_value(value.strip())

    for flag, dotted in (
        ("root_dir", "data.root_dir"), ("checkpoint_dir", "train.checkpoint_dir"),
        ("epochs", "train.epochs"), ("batch_size", "train.batch_size"),
        ("num_gpus", "train.num_gpus"), ("device", "train.device"),
        ("resume", "train.resume"), ("input_source", "data.input_source"),
        ("decimate_to", "data.decimate_to"),
    ):
        value = getattr(args, flag)
        if value is not None:
            overrides[dotted] = value

    return cfg.apply_overrides(overrides)


def build_dataloaders(cfg: Config):
    common = dict(
        root_dir=cfg.data.root_dir,
        val_frac=cfg.data.val_frac, test_frac=cfg.data.test_frac,
        split_seed=cfg.data.split_seed,
        input_source=cfg.data.input_source,
        correspondence_tol=cfg.data.correspondence_tol,
        max_vertices=cfg.data.max_vertices,
        decimate_to=cfg.data.decimate_to,
        min_vertices_per_fragment=cfg.data.min_vertices_per_fragment,
        fracture_pattern=cfg.data.fracture_pattern,
        seed=cfg.data.seed,
    )
    train_set = BreakingBadDataset(split="train", **common)
    val_set = BreakingBadDataset(split="val", **common)

    def loader(dataset, steps):
        # persistent_workers matters a lot on Windows: without it the whole
        # worker pool is torn down and respawned every epoch, and Windows has
        # no fork(), so that spawn cost is substantial across many epochs.
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            sampler=RandomSampler(dataset, num_samples=steps * cfg.train.batch_size,
                                  replacement=True),
            collate_fn=breaking_bad_collate_fn,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    return loader(train_set, cfg.train.steps_per_epoch), loader(val_set, cfg.train.val_steps)


def build_model_and_optim(cfg: Config, device):
    model = VNGATModel(
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        num_vn_slots=cfg.model.num_vn_slots,
        heads=cfg.model.heads,
        head_dim=cfg.model.head_dim,
        embed_dim=cfg.model.embed_dim,
        norm=cfg.model.norm,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        angular=cfg.model.angular,
        max_triplets_per_node=cfg.model.max_triplets_per_node,
    ).to(device)

    loss_fn = CompositeLoss(
        w_rot=cfg.loss.rot, w_pos=cfg.loss.pos, w_node=cfg.loss.node,
        w_mid=cfg.loss.mid, w_face=cfg.loss.face,
        w_embv=cfg.loss.embv, w_embe=cfg.loss.embe,
        rot_loss=cfg.loss.rot_loss, auto_balance=cfg.loss.auto_balance,
    ).to(device)

    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr,
                                  weight_decay=cfg.optim.weight_decay)

    if cfg.optim.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.optim.plateau_factor,
            patience=cfg.optim.plateau_patience,
        )
    elif cfg.optim.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.train.epochs, 1)
        )
    else:
        scheduler = None

    return model, loss_fn, optimizer, scheduler


def worker(rank: int, world_size: int, cfg: Config):
    distributed = world_size > 1
    if distributed:
        device = setup_distributed(rank, world_size, cfg.train.master_port)
    else:
        device = torch.device(cfg.train.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch.cuda.is_available() is False. "
                "This is a torch/driver install issue, not a bug in this project."
            )

    torch.manual_seed(cfg.train.seed + rank)

    if rank == 0:
        print(f"world_size={world_size}, device={device}")
        print(cfg.describe())

    model, loss_fn, optimizer, scheduler = build_model_and_optim(cfg, device)

    if rank == 0:
        est = estimate_activation_bytes(
            num_nodes=cfg.data.decimate_to or 50_000,
            num_directed_edges=6 * (cfg.data.decimate_to or 50_000),
            hidden_channels=cfg.model.hidden_channels,
            num_layers=cfg.model.num_layers,
            num_vn_slots=cfg.model.num_vn_slots,
            amp=cfg.train.amp,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
        print(f"model parameters: {model.num_parameters():,}")
        print(f"estimated activations for one scene: {est.gib:.2f} GiB "
              f"(dominant: {est.dominant_term}); "
              f"x{cfg.train.batch_size} scenes/batch")

    if distributed:
        model = DDP(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
            # find_unused_parameters stays FALSE: the embedding heads always
            # receive a gradient now, because cluster_consistency_loss returns
            # a gradient-connected zero when a batch has no shared vertices
            # instead of a detached constant. See training/losses.py.
            find_unused_parameters=False,
            # Safe together with use_reentrant=False checkpointing, and it lets
            # DDP skip its per-iteration graph traversal.
            static_graph=True,
        )

    train_loader, val_loader = build_dataloaders(cfg)
    try:
        fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, cfg)
    finally:
        if distributed:
            cleanup_distributed()


def main(argv=None):
    cfg = build_config(argv)

    available = torch.cuda.device_count()
    if cfg.train.num_gpus > 1 and available < cfg.train.num_gpus:
        raise RuntimeError(
            f"num_gpus={cfg.train.num_gpus} requested but only {available} CUDA device(s) "
            f"are visible. Check nvidia-smi / CUDA_VISIBLE_DEVICES."
        )

    if cfg.train.num_gpus > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(cfg.train.master_port))
        mp.spawn(worker, args=(cfg.train.num_gpus, cfg), nprocs=cfg.train.num_gpus, join=True)
    else:
        worker(0, 1, cfg)


if __name__ == "__main__":
    main()
