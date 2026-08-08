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
from typing import Dict, Optional, Sequence

import torch
from tqdm.auto import tqdm

from ..config import Config
from ..utils.console import MetricTable, format_epoch_line
from ..utils.memory import (
    AmpContext,
    cuda_memory_summary,
    is_oom_error,
    release_cuda_memory,
)
from .bridge import build_model_inputs, build_predictions, build_targets, select_input_graph
from .distributed import all_ranks_agree, get_world_size, is_main_process, reduce_metrics
from .losses import LOSS_KEYS
from .session import (
    SessionLimit, load_rng_state, merge_history, rng_state,
)

METRIC_KEYS = ("total", *LOSS_KEYS, "rot_deg")

#: Where `resume: auto` looks when the working checkpoint directory is empty.
#: Kaggle mounts a previous notebook's saved output read-only under
#: /kaggle/input, which is the durable way to carry a run across sessions.
DEFAULT_RESUME_ROOTS = ("/kaggle/input",)

# Column order for the live table and the per-epoch summary. Kept identical
# between the two so the numbers line up vertically as an epoch finishes.
COLUMNS = ("total", "rot", "pos", "node", "mid", "face", "embv", "embe", "rot_deg")


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
    pbar = MetricTable(loader, COLUMNS, desc=f"{prefix} load", disable=not main,
                       log_every=max(cfg.train.log_every, 1) * 10)

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
                pbar.set_desc(f"{prefix} \u2192dev")
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
                    pbar.set_desc(f"{prefix} fwd")
                    with amp.autocast():
                        outputs = model(**model_inputs)

                    # Geometry predictions and losses stay in fp32: they feed
                    # squared-distance terms where fp16 rounding is a real
                    # error, and they are cheap relative to the backbone.
                    R_pred = outputs["R_pred"].float()
                    predicted = build_predictions(full_diffused, R_pred)

                    pbar.set_desc(f"{prefix} loss")
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
                        pbar.set_desc(f"{prefix} bwd")
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
            pbar.update_metrics({k: v / num_batches for k, v in totals.items()})
        else:
            skipped += 1

        pbar.set_desc(f"{prefix} load")
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
        cache_note = ""
        dataset = getattr(loader, "dataset", None)
        for attr, label in (("base_cache", "base"), ("cache", "scene")):
            c = getattr(dataset, attr, None)
            if c is not None and getattr(c, "enabled", False):
                looked = c.hits + c.misses
                if looked:
                    cache_note += f" {label}-cache={100 * c.hits / looked:.0f}%"
        tqdm.write(
            f"  {phase}: data={data_seconds:.1f}s compute={compute_seconds:.1f}s "
            f"over {num_batches} batches ({skipped} skipped) on this rank; "
            f"largest input graph {max_nodes} nodes;{cache_note} "
            f"{cuda_memory_summary(device)}"
        )
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_val: float, cfg: Config, amp=None, history=None,
                    loss_fn=None) -> None:
    """Everything needed to continue, not just to evaluate.

    Weights alone restart the optimizer from zero momentum and reset the AMP
    scale factor. Across a dozen capped sessions that produces a visible bump
    in the loss at every boundary -- which is indistinguishable, on a plot,
    from a model that is failing to converge.

    Written to a temporary file and renamed, so a session killed mid-write
    leaves the previous checkpoint intact rather than a truncated one.
    """
    underlying = model.module if hasattr(model, "module") else model
    payload = {
        "model": underlying.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg.to_dict(),
        # The loss module holds learnable parameters when auto_balance is on.
        "loss_fn": loss_fn.state_dict() if loss_fn is not None else None,
        # Scale factor: dropping it costs a few skipped steps while AMP
        # re-discovers it, every single session.
        "scaler": amp.state_dict() if amp is not None else None,
        # So the data order and augmentations continue rather than replay.
        "rng": rng_state(),
        # Carried inside the checkpoint as well as in loss_history.json, so
        # weights and curves cannot drift apart.
        "history": history or {},
        "version": 2,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    map_location="cpu", amp=None, loss_fn=None,
                    restore_rng: bool = True) -> Dict:
    """Restore as much as the checkpoint carries.

    Older checkpoints (version 1) have no scaler, RNG or history; those fields
    are simply absent and are skipped, so an older file still resumes -- just
    without the extras.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    underlying = model.module if hasattr(model, "module") else model
    underlying.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if loss_fn is not None and state.get("loss_fn"):
        loss_fn.load_state_dict(state["loss_fn"])
    if amp is not None and state.get("scaler"):
        amp.load_state_dict(state["scaler"])
    state["_rng_restored"] = load_rng_state(state.get("rng")) if restore_rng else False
    return state


def rotate_snapshots(ckpt_dir: Path, keep: int) -> None:
    """Keep only the newest ``keep`` snapshot files.

    Snapshots are named ``snapshot_e00120.pt`` so they sort chronologically by
    name -- no reliance on mtime, which is unreliable across a filesystem that
    may be remounted between sessions.
    """
    if keep <= 0:
        return
    snapshots = sorted(ckpt_dir.glob("snapshot_e*.pt"))
    for stale in snapshots[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def find_resume_checkpoint(spec: Optional[str], checkpoint_dir: Path,
                           extra_roots: Optional[Sequence[str]] = None) -> Optional[str]:
    """Resolve ``train.resume``.

    ``"auto"`` picks up ``last.pt`` from the checkpoint directory if it exists
    and starts fresh otherwise, so the SAME command can be re-run each session
    without editing a path by hand. That matters when a run spans a dozen
    12-hour sessions: editing the command every time is how a run ends up
    resumed from the wrong checkpoint.
    """
    if not spec:
        return None
    if str(spec).lower() != "auto":
        return str(spec)

    candidate = checkpoint_dir / "last.pt"
    if candidate.exists():
        return str(candidate)

    # Nothing in the working directory. On Kaggle an interactive session does
    # NOT reliably persist /kaggle/working -- the durable route is Save Version,
    # then attach that notebook's output as a data source in the next session,
    # which mounts it read-only under /kaggle/input. Search there too, so the
    # same command resumes either way instead of silently starting over.
    for root in (extra_roots or DEFAULT_RESUME_ROOTS):
        base = Path(root)
        if not base.exists():
            continue
        found = sorted(base.glob("*/checkpoints/last.pt")) + \
            sorted(base.glob("*/last.pt")) + \
            sorted(base.glob("*/checkpoints/snapshot_e*.pt"))
        if found:
            return str(found[-1])
    return None


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

    resume_path = find_resume_checkpoint(cfg.train.resume, ckpt_dir)
    if resume_path:
        state = load_checkpoint(resume_path, model, optimizer, scheduler,
                                map_location=device, amp=amp, loss_fn=loss_fn)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val = float(state.get("best_val", float("inf")))

        # Prefer the history carried inside the checkpoint: it cannot have
        # drifted from the weights. Fall back to the JSON file for older
        # checkpoints that predate it.
        loaded = state.get("history") or {}
        if not loaded and history_path.exists():
            try:
                with open(history_path) as f:
                    loaded = json.load(f)
            except (OSError, ValueError):
                loaded = {}
        history = merge_history(loaded, start_epoch)

        if main:
            epochs_recorded = len(history.get("train", {}).get("total", []))
            print(
                f"resumed from {resume_path}\n"
                f"  epoch          : {start_epoch} of {cfg.train.epochs}\n"
                f"  best val total : {best_val:.6f}\n"
                f"  history        : {epochs_recorded} epochs carried forward\n"
                f"  optimizer/sched: restored\n"
                f"  amp scaler     : {'restored' if state.get('scaler') else 'not in checkpoint'}\n"
                f"  rng stream     : {'continued' if state.get('_rng_restored') else 'reset'}"
            )
    elif cfg.train.resume and str(cfg.train.resume).lower() == "auto" and main:
        print(f"resume=auto: no {ckpt_dir / 'last.pt'} yet, starting fresh")

    if start_epoch >= cfg.train.epochs:
        if main:
            print(f"nothing to do: already at epoch {start_epoch} of "
                  f"{cfg.train.epochs}. Raise train.epochs to continue.")
        return history

    session = SessionLimit(cfg.train.time_budget_hours, main=main)
    if main and session.budget:
        print(f"session budget: {cfg.train.time_budget_hours:.2f} h "
              f"-- will stop cleanly rather than be killed mid-epoch")

    stop_reason = ""
    last_epoch_seconds = 0.0

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
            tqdm.write(format_epoch_line(
                epoch, train_metrics, val_metrics, COLUMNS,
                lr=optimizer.param_groups[0]["lr"], seconds=elapsed,
            ))
            for k, v in train_metrics.items():
                history["train"].setdefault(k, []).append(v)
            for k, v in val_metrics.items():
                history["val"].setdefault(k, []).append(v)
            # Written every epoch, not only at the end: a crash partway through
            # a long run should not cost the curves so far.
            with open(history_path, "w") as f:
                json.dump(history, f)

            current = val_metrics.get("total", float("inf"))
            if current == current and current < best_val:
                best_val = current
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                                epoch, best_val, cfg, amp=amp, history=history,
                                loss_fn=loss_fn)

            if (epoch + 1) % cfg.train.save_every == 0:
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler,
                                epoch, best_val, cfg, amp=amp, history=history,
                                loss_fn=loss_fn)

            # Rotating snapshots. `last.pt` is a single file that is
            # overwritten; snapshots keep a few numbered generations, so a
            # checkpoint that turns out to be bad (NaN weights after a bad
            # step, a truncated write on a filesystem without atomic rename)
            # is not the only thing standing between you and starting over.
            if cfg.train.snapshot_every and (epoch + 1) % cfg.train.snapshot_every == 0:
                save_checkpoint(ckpt_dir / f"snapshot_e{epoch:05d}.pt", model,
                                optimizer, scheduler, epoch, best_val, cfg,
                                amp=amp, history=history, loss_fn=loss_fn)
                rotate_snapshots(ckpt_dir, cfg.train.keep_snapshots)

        last_epoch_seconds = elapsed

        # Whether to stop is decided on the main rank and then agreed by all.
        # One rank leaving the loop while the others continue hangs the job at
        # the next collective, with no error and no progress.
        stop, reason = session.should_stop(last_epoch_seconds)
        if not all_ranks_agree(not stop, device):
            stop_reason = reason or "another rank requested a stop"
            if main:
                # last.pt may be several epochs stale if save_every > 1, and
                # this epoch is the one worth keeping.
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler,
                                epoch, best_val, cfg, amp=amp, history=history,
                                loss_fn=loss_fn)
                with open(history_path, "w") as f:
                    json.dump(history, f)
                tqdm.write(f"\nstopping after epoch {epoch}: {stop_reason}")
            break

    session.restore()

    if main and stop_reason:
        remaining = cfg.train.epochs - (epoch + 1)
        print(
            f"\nPAUSED, not finished -- {remaining} of {cfg.train.epochs} epochs left.\n"
            f"  reason         : {stop_reason}\n"
            f"  {session.describe()}\n"
            f"  best val total : {best_val:.6f}\n"
            f"  saved          : {ckpt_dir / 'last.pt'}  (resume point)\n"
            f"                   {ckpt_dir / 'best.pt'}  (lowest validation)\n"
            f"                   {history_path}\n\n"
            f"continue with the SAME command -- train.resume: auto picks up\n"
            f"last.pt by itself, so nothing needs editing between sessions."
        )
        return history

    if main:
        # Written unconditionally when the loop finishes, separate from the
        # every-N-epochs `last.pt` and from `best.pt`. Three files, three
        # different questions:
        #   best.pt   lowest validation loss seen        -> evaluate this one
        #   last.pt   periodic snapshot                  -> resume from this one
        #   final.pt  the weights training ended on      -> reproducibility
        # `best` and `final` differ whenever the run overfits after its best
        # epoch, which is exactly when you want to be able to tell them apart.
        final_path = ckpt_dir / "final.pt"
        save_checkpoint(final_path, model, optimizer, scheduler,
                        cfg.train.epochs - 1, best_val, cfg, amp=amp,
                        history=history, loss_fn=loss_fn)
        underlying = model.module if hasattr(model, "module") else model
        weights_path = ckpt_dir / "final_weights.pt"
        torch.save(underlying.state_dict(), weights_path)
        print(
            f"\ntraining complete\n"
            f"  best validation total : {best_val:.6f}\n"
            f"  {final_path}          full state (model + optimizer + scheduler)\n"
            f"  {weights_path}   weights only, for inference\n"
            f"  {ckpt_dir / 'best.pt'}           lowest-validation checkpoint\n"
            f"  {ckpt_dir / 'loss_history.json'} curves\n\n"
            f"evaluate with:\n"
            f"  python scripts/evaluate.py --checkpoint {ckpt_dir / 'best.pt'}\n"
            f"visualise with:\n"
            f"  python scripts/vis_prediction.py --checkpoint {ckpt_dir / 'best.pt'} "
            f"--root-dir <data>"
        )

    return history
