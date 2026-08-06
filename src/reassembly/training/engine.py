"""
The training loop.

Everything memory-related that the loop is responsible for lives here:
mixed precision, gradient accumulation, and OOM recovery that keeps DDP ranks
in lockstep instead of hanging.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm.auto import tqdm

from ..config import Config
from ..utils.memory import (
    AmpContext,
    cuda_memory_summary,
    is_oom_error,
    release_cuda_memory,
)
from .bridge import build_model_inputs, build_predictions, build_targets, select_input_graph
from .distributed import all_ranks_agree, get_world_size, is_main_process, reduce_metrics
from .losses import LOSS_KEYS

METRIC_KEYS = ("total", *LOSS_KEYS, "rot_deg")

# Kept short deliberately. With all eight terms in the live postfix, a long
# prefix pushes the redrawn line past what a Kaggle/Colab output cell renders,
# which looks like the numbers are missing rather than merely truncated.
_SHORT = {"total": "tot", "rot": "rot", "pos": "pos", "node": "nrm", "mid": "mid",
          "face": "fac", "embv": "ev", "embe": "ee", "rot_deg": "deg"}


def _to_device(batch: Dict, device: torch.device, keys) -> Dict:
    out = dict(batch)
    for k in keys:
        if out.get(k) is not None and hasattr(out[k], "to"):
            out[k] = out[k].to(device, non_blocking=True)
    return out


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device: torch.device,
    cfg: Config,
    amp: AmpContext,
    train: bool,
    epoch: int = 0,
    phase: str = "train",
) -> Dict[str, float]:
    model.train(train)
    totals: Dict[str, float] = {}
    num_batches = 0
    skipped = 0
    consecutive_oom = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    max_nodes = 0

    main = is_main_process()
    prefix = f"E{epoch:03d} {phase}"
    pbar = tqdm(loader, desc=f"{prefix} load", leave=False, unit="batch", disable=not main)

    if train:
        optimizer.zero_grad(set_to_none=True)

    data_start = time.perf_counter()
    for step, batch in enumerate(pbar):
        data_seconds += time.perf_counter() - data_start
        compute_start = time.perf_counter()

        # An entirely empty batch happens when every scene in it was rejected
        # by the vertex budget. All ranks must still vote, or they desync.
        local_ok = batch.get("graph") is not None
        losses = None
        nodes_here = 0

        if local_ok:
            try:
                pbar.set_description(f"{prefix} \u2192dev")
                batch = _to_device(batch, device,
                                   ("graph", "diffused_graph", "frac_graph",
                                    "diff_frac_graph", "t_matrices"))

                full_diffused = batch["diffused_graph"]
                input_graph = select_input_graph(batch, cfg.data.input_source, diffused=True)
                nodes_here = int(input_graph.x.shape[0])

                model_inputs = build_model_inputs(input_graph)
                targets = build_targets(batch["graph"], batch["t_matrices"],
                                        input_graph=input_graph)

                with torch.set_grad_enabled(train):
                    pbar.set_description(f"{prefix} fwd")
                    with amp.autocast():
                        outputs = model(**model_inputs)

                    # Geometry predictions and losses stay in fp32: they feed
                    # squared-distance terms where fp16 rounding is a real
                    # error, and they are cheap relative to the backbone.
                    R_pred = outputs["R_pred"].float()
                    predicted = build_predictions(full_diffused, R_pred)

                    pbar.set_description(f"{prefix} loss")
                    losses = loss_fn(
                        dict(
                            R_pred=R_pred,
                            **predicted,
                            vertex_embedding=outputs["vertex_embedding"].float(),
                            edge_embedding=outputs["edge_embedding"].float(),
                        ),
                        targets,
                    )

                    if train:
                        pbar.set_description(f"{prefix} bwd")
                        amp.backward(losses["total"] / cfg.train.accum_steps)

            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                local_ok = False
                consecutive_oom += 1
                if main:
                    tqdm.write(
                        f"!! OOM on {device} at {phase} step {step} "
                        f"({nodes_here} nodes). {cuda_memory_summary(device)}"
                    )
                losses = None
                release_cuda_memory()
                if train:
                    # Whatever partial gradient the failed backward left behind
                    # is meaningless and would otherwise be applied at the next
                    # optimizer step.
                    optimizer.zero_grad(set_to_none=True)
                if consecutive_oom >= cfg.train.max_consecutive_oom:
                    raise RuntimeError(
                        f"{consecutive_oom} consecutive OOMs -- the configuration does not fit "
                        f"this GPU at all, rather than being unlucky on one scene. Lower "
                        f"data.decimate_to / train.batch_size / model.hidden_channels, or "
                        f"enable model.gradient_checkpointing."
                    ) from exc

        # Every rank votes, unconditionally: this is what stops a single rank's
        # OOM from leaving the others blocked forever at the next collective.
        step_ok = all_ranks_agree(local_ok, device)

        if train and step_ok and (step + 1) % cfg.train.accum_steps == 0:
            amp.step(optimizer, grad_clip=cfg.optim.grad_clip, parameters=model.parameters())
            optimizer.zero_grad(set_to_none=True)
        elif train and not step_ok:
            optimizer.zero_grad(set_to_none=True)

        compute_seconds += time.perf_counter() - compute_start

        if step_ok and losses is not None:
            consecutive_oom = 0
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            num_batches += 1
            max_nodes = max(max_nodes, nodes_here)
            pbar.set_postfix({
                _SHORT.get(k, k): f"{v / num_batches:.3f}" for k, v in totals.items()
            })
        else:
            skipped += 1

        pbar.set_description(f"{prefix} load")
        data_start = time.perf_counter()

    pbar.close()

    if num_batches == 0:
        if main:
            tqdm.write(
                f"!! {phase} epoch {epoch} processed ZERO usable batches "
                f"({skipped} skipped). Reporting NaN rather than crashing on a "
                f"missing key. Check steps_per_epoch/val_steps and the vertex budget."
            )
        metrics = {k: float("nan") for k in METRIC_KEYS}
    else:
        metrics = {k: v / num_batches for k, v in totals.items()}

    metrics = reduce_metrics(metrics, device)

    if main:
        tqdm.write(
            f"  {phase}: data={data_seconds:.1f}s compute={compute_seconds:.1f}s "
            f"over {num_batches} batches ({skipped} skipped) on this rank; "
            f"largest input graph {max_nodes} nodes; {cuda_memory_summary(device)}"
        )
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_val: float, cfg: Config) -> None:
    """Full training state, not just weights -- an interrupted run should
    resume, not restart."""
    underlying = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": underlying.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "config": cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    map_location="cpu") -> Dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    underlying = model.module if hasattr(model, "module") else model
    underlying.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
) -> Dict:
    amp = AmpContext(cfg.train.amp, device_type=device.type, dtype=cfg.train.amp_dtype)
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    main = is_main_process()

    if main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(str(ckpt_dir / "config.yaml"))

    history_path = ckpt_dir / "loss_history.json"
    history: Dict[str, Dict[str, list]] = {"train": {}, "val": {}}
    start_epoch, best_val = 0, float("inf")

    if cfg.train.resume:
        state = load_checkpoint(cfg.train.resume, model, optimizer, scheduler,
                                map_location=device)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val = float(state.get("best_val", float("inf")))
        if main:
            print(f"Resumed from {cfg.train.resume} at epoch {start_epoch} "
                  f"(best_val={best_val:.4f})")

    if main and cfg.train.resume_history and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                  cfg, amp, train=True, epoch=epoch, phase="train")
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device,
                                cfg, amp, train=False, epoch=epoch, phase="val")

        if scheduler is not None:
            if cfg.optim.scheduler == "plateau":
                target = val_metrics.get("total", float("nan"))
                # ReduceLROnPlateau treats NaN as "no improvement" forever and
                # would keep cutting the LR; skip the step instead.
                if target == target:
                    scheduler.step(target)
            else:
                scheduler.step()

        elapsed = time.perf_counter() - epoch_start

        if main:
            tqdm.write(
                f"[epoch {epoch:04d}] train={train_metrics.get('total', float('nan')):.4f} "
                f"val={val_metrics.get('total', float('nan')):.4f} | "
                f"rot={val_metrics.get('rot', float('nan')):.4f} "
                f"({val_metrics.get('rot_deg', float('nan')):.2f}deg) "
                f"pos={val_metrics.get('pos', float('nan')):.4f} "
                f"node={val_metrics.get('node', float('nan')):.4f} "
                f"mid={val_metrics.get('mid', float('nan')):.4f} "
                f"face={val_metrics.get('face', float('nan')):.4f} "
                f"embv={val_metrics.get('embv', float('nan')):.4f} "
                f"embe={val_metrics.get('embe', float('nan')):.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e} [{elapsed:.1f}s]"
            )
            for k, v in train_metrics.items():
                history["train"].setdefault(k, []).append(v)
            for k, v in val_metrics.items():
                history["val"].setdefault(k, []).append(v)
            # Written every epoch, not only at the end: a crash partway through
            # a long run should not cost the curves so far.
            with open(history_path, "w") as f:
                json.dump(history, f)

            if (epoch + 1) % cfg.train.save_every == 0:
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler,
                                epoch, best_val, cfg)
            current = val_metrics.get("total", float("inf"))
            if current == current and current < best_val:
                best_val = current
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                                epoch, best_val, cfg)

    return history
