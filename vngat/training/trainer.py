"""
Training loop.

The memory strategy, which is the point of this file
----------------------------------------------------
Peak activation memory is made independent of `batch_size` by processing
`micro_batch_scenes` scenes at a time and accumulating gradients. The largest
thing on the GPU is then the largest SINGLE SCENE, which is a property of the
dataset, not of a tunable that the user would otherwise have to guess. Layered
on top:

  * per-head channel splitting in the attention layer  (factor `heads`)
  * segment attention instead of dense masked attention (factor ~F*K/K)
  * a Gram bottleneck in the embedding heads           (factor (C/16)^2)
  * mixed precision                                    (factor ~2)
  * gradient checkpointing, engaged automatically only for a scene that would
    otherwise not fit

Together these take a batch-size-2 step on a 16 GB T4 from "out of memory"
to a few GB, with the fallbacks reserved for genuinely pathological scenes.

Every early exit is voted on across ranks before any backward pass, and the
gradient all-reduce is fired from a fixed placeholder step rather than from
whichever real micro-batch happened to be last -- see `distributed.py` and
`_sync_gradients` for why neither is optional.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import BreakingBadDataset, collate_fn, split_batch_by_scene
from ..data.graph import SceneBatch, collate_scenes
from ..losses.composite import CompositeLoss
from ..models.vn_gat import VNGATModel
from ..utils.env import dataloader_worker_init, seed_everything
from ..utils.progress import make_bar, table_header, table_row, write
from . import distributed as D
from .bridge import build_model_inputs, build_predictions, build_targets, prepare_scene
from .checkpoint import CheckpointManager
from .drive import DriveSync
from .history import History

_LOSS_KEYS = ("total", "rot", "rot_deg", "pos", "node", "mid", "face", "emb_v", "emb_e")


# ---------------------------------------------------------------------------
# torch.amp compatibility (the API moved in torch 2.4)
# ---------------------------------------------------------------------------
def make_autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.0
        return torch.cuda.amp.autocast()


def make_scaler(device: torch.device, enabled: bool):
    use = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=use)


# ---------------------------------------------------------------------------
def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, BreakingBadDataset]:
    common = dict(
        root_dir=cfg.root_dir,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        split_seed=cfg.split_seed,
        max_scenes=cfg.max_scenes,
        fracture_pattern=cfg.fracture_pattern or None,
        input_source=cfg.input_source,
        with_correspondence=cfg.correspondence,
        correspondence_tol=cfg.correspondence_tol,
        min_fragments=cfg.min_fragments,
    )
    train_set = BreakingBadDataset(
        split="train", nominal_length=cfg.steps_per_epoch * cfg.batch_size, **common
    )
    val_set = BreakingBadDataset(
        split="val", nominal_length=cfg.val_steps * cfg.batch_size, **common
    )

    # NOTE: no DistributedSampler. It exists to stop ranks from drawing the
    # same INDICES of a fixed dataset, but __getitem__ ignores its index and
    # draws a fresh random scene every call, so there is no index space to
    # partition. Each rank is seeded differently (seed_everything(seed, rank)),
    # which is what actually keeps the ranks from duplicating work.
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        worker_init_fn=dataloader_worker_init,
    )
    if cfg.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
    return DataLoader(train_set, **loader_kwargs), DataLoader(val_set, **loader_kwargs), train_set


def build_model(cfg: Config, device: torch.device) -> VNGATModel:
    model = VNGATModel(
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        num_vn_slots=cfg.num_vn_slots,
        heads=cfg.heads,
        embed_dim=cfg.embed_dim,
        gram_bottleneck=cfg.gram_bottleneck,
        norm=cfg.norm,
        grad_checkpointing=cfg.grad_checkpointing,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
def _dummy_scene(device: torch.device) -> Dict:
    """
    A minimal 1-fragment, 2-vertex, 1-edge scene.

    Used only as the last resort when a real scene cannot be made to fit. Its
    purpose is RANK SYMMETRY, not learning: it produces a real forward and
    backward through every parameter, so this rank performs exactly as many
    backward passes with exactly the same reduction buckets as its peers. A
    rank that simply skipped the step would leave the others blocked on an
    all-reduce that never comes.
    """
    graph = SceneBatch(
        node_vec=torch.zeros(2, 2, 3, device=device),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long, device=device),
        edge_len=torch.ones(1, 1, device=device),
        edge_vec=torch.zeros(1, 3, 3, device=device),
        node_frag=torch.zeros(2, dtype=torch.long, device=device),
        frag_scene=torch.zeros(1, dtype=torch.long, device=device),
        frag_centroid=torch.zeros(1, 3, device=device),
        vertex_cluster_id=torch.full((2,), -1, dtype=torch.long, device=device),
        edge_cluster_id=torch.full((1,), -1, dtype=torch.long, device=device),
        num_fragments=1,
        num_scenes=1,
    )
    # A tiny non-degenerate geometry so normalisations do not divide by zero.
    graph.node_vec[0, 0, 0] = -0.5
    graph.node_vec[1, 0, 0] = 0.5
    graph.node_vec[:, 1, 2] = 1.0
    rot = torch.eye(3, device=device).unsqueeze(0)
    return {
        "clean_target": graph,
        "diffused_target": graph,
        "diffused_input": graph,
        "rot": rot,
        "trans": torch.zeros(1, 3, device=device),
    }


def _forward_loss(model, loss_fn, scene: Dict, device: torch.device, amp: bool):
    with make_autocast(device, amp):
        outputs = model(**build_model_inputs(scene["diffused_input"]))
        predictions = build_predictions(scene["diffused_target"], outputs["R_pred"])
        merged = dict(
            R_pred=outputs["R_pred"],
            vertex_embedding=outputs["vertex_embedding"],
            edge_embedding=outputs["edge_embedding"],
            **predictions,
        )
        targets = build_targets(scene["clean_target"], scene["rot"], scene["diffused_input"])
        losses = loss_fn(merged, targets)
    return losses


_OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", None)


def _is_oom(err: Exception) -> bool:
    """String match as well as the type: `OutOfMemoryError` only exists on
    newer torch, and some allocator failures still surface as plain
    RuntimeError."""
    if _OOM_ERROR is not None and isinstance(err, _OOM_ERROR):
        return True
    text = str(err).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def run_phase(
    model,
    loader: DataLoader,
    loss_fn: CompositeLoss,
    optimizer,
    scaler,
    device: torch.device,
    cfg: Config,
    train: bool,
    epoch: int,
    bar,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """One train or validation pass. Returns (metrics, diagnostics)."""
    model.train(train)
    raw_model = model.module if isinstance(model, DDP) else model

    totals = {k: 0.0 for k in _LOSS_KEYS}
    num_micro = 0
    data_seconds = compute_seconds = 0.0
    max_nodes = max_edges = 0
    oom_recoveries = 0
    fallback_steps = 0

    phase = "train" if train else "val"
    data_start = time.perf_counter()

    for batch in loader:
        data_seconds += time.perf_counter() - data_start
        compute_start = time.perf_counter()

        max_nodes = max(max_nodes, batch["target"].num_nodes)
        max_edges = max(max_edges, batch["target"].num_edges)

        micro_batches = _make_micro_batches(batch, cfg.micro_batch_scenes)
        if train:
            optimizer.zero_grad(set_to_none=True)

        for micro_idx, micro in enumerate(micro_batches):
            # EVERY real micro-step runs under no_sync -- including the last.
            #
            # The obvious scheme (sync on the final micro-step) is unsafe here:
            # if that step runs out of memory PART WAY THROUGH its backward,
            # some gradient buckets have already been all-reduced. Retrying then
            # re-fires those reductions, so this rank performs more collectives
            # than its peers and the job hangs until NCCL times out. Deferring
            # the reduction to a step that cannot fail makes the OOM ladder
            # genuinely recoverable instead of merely appearing to be.
            sync_ctx = (
                model.no_sync() if (train and isinstance(model, DDP)) else nullcontext()
            )
            losses, recovered, used_fallback = _run_micro_step(
                model, raw_model, loss_fn, micro, device, cfg, train, scaler, sync_ctx,
                scale=1.0 / len(micro_batches),
            )
            oom_recoveries += int(recovered)
            fallback_steps += int(used_fallback)
            for key in _LOSS_KEYS:
                totals[key] += float(losses[key].detach())
            num_micro += 1

        if train:
            if isinstance(model, DDP):
                _sync_gradients(model, loss_fn, device, cfg, scaler)
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        compute_seconds += time.perf_counter() - compute_start
        bar.update(batch["num_scenes"])
        if num_micro:
            bar.set_postfix_str(
                f"{phase} tot={totals['total'] / num_micro:.3f} "
                f"rot={totals['rot_deg'] / num_micro:.1f}deg"
            )
        data_start = time.perf_counter()

    if num_micro == 0:
        write(f"!! {phase} epoch {epoch} processed zero batches -- check steps_per_epoch/val_steps.")
        metrics = {k: float("nan") for k in _LOSS_KEYS}
    else:
        metrics = {k: v / num_micro for k, v in totals.items()}

    diagnostics = {
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "max_nodes": float(max_nodes),
        "max_edges": float(max_edges),
        "oom_recoveries": float(oom_recoveries),
        "fallback_steps": float(fallback_steps),
    }
    return metrics, diagnostics


def _make_micro_batches(batch: Dict, micro_scenes: int) -> List[Dict]:
    if micro_scenes >= batch["num_scenes"]:
        return [batch]
    scenes = split_batch_by_scene(batch)
    groups = []
    for start in range(0, len(scenes), micro_scenes):
        chunk = scenes[start:start + micro_scenes]
        if len(chunk) == 1:
            groups.append(chunk[0])
        else:
            groups.append({
                "target": collate_scenes([c["target"] for c in chunk]),
                "input": (collate_scenes([c["input"] for c in chunk])
                          if chunk[0]["input"] is not None else None),
                "rot": torch.cat([c["rot"] for c in chunk], 0),
                "trans": torch.cat([c["trans"] for c in chunk], 0),
                "scene_dirs": [d for c in chunk for d in c["scene_dirs"]],
                "num_scenes": len(chunk),
            })
    return groups


def _sync_gradients(model, loss_fn, device, cfg, scaler) -> None:
    """
    Trigger DDP's gradient all-reduce exactly once per optimiser step, from a
    step that cannot fail.

    Runs a forward/backward on the two-vertex placeholder scene with the loss
    multiplied by ZERO. Backward still traverses the whole graph, so every
    parameter's DDP hook fires and every bucket is reduced -- carrying the
    gradients accumulated during the preceding `no_sync()` micro-steps, because
    DDP reduces the contents of `param.grad`, which accumulation has already
    filled in. Multiplying by zero means this step contributes nothing to the
    gradient itself.

    Two properties follow, and both matter for a multi-day unattended run:
      * the number and order of collectives per step is FIXED, independent of
        scene sizes, OOM retries or how many fragments a rank happened to draw;
      * the reduction runs on a graph small enough that it cannot itself OOM,
        so a recoverable failure never leaves the collectives half-finished.

    Cost is one forward/backward over a 2-node graph -- kernel-launch bound and
    negligible beside a real scene.
    """
    scene = _dummy_scene(device)
    losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
    scaler.scale(losses["total"] * 0.0).backward()


def _run_micro_step(model, raw_model, loss_fn, micro, device, cfg, train, scaler, sync_ctx, scale):
    """
    Forward/backward for one micro-batch, with a two-step OOM ladder:
      1. retry the same scene with gradient checkpointing on;
      2. if it still will not fit, substitute the dummy scene so this rank
         performs the same number of backward passes as its peers.
    """
    recovered = False
    used_fallback = False
    checkpoint_was = raw_model.grad_checkpointing

    for attempt in range(3):
        try:
            scene = _dummy_scene(device) if attempt == 2 else prepare_scene(micro, device)
            if attempt == 1:
                raw_model.grad_checkpointing = True
            with sync_ctx:
                if train:
                    losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
                    scaler.scale(losses["total"] * scale).backward()
                else:
                    with torch.no_grad():
                        losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
            raw_model.grad_checkpointing = checkpoint_was
            return losses, recovered, used_fallback
        except RuntimeError as err:
            raw_model.grad_checkpointing = checkpoint_was
            if not _is_oom(err) or attempt == 2:
                _report_batch_failure(micro, device, err)
                raise
            recovered = True
            used_fallback = attempt == 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write(
                f"  [oom] scene with {micro['target'].num_nodes} nodes / "
                f"{micro['target'].num_edges} edges did not fit; "
                + ("retrying with gradient checkpointing" if attempt == 0
                   else "substituting a placeholder step to keep ranks in lockstep")
            )
    raise RuntimeError("unreachable")


def _report_batch_failure(micro: Dict, device: torch.device, err: Exception) -> None:
    """
    Diagnostics that can never themselves become the raised exception.

    If the original error already poisoned the CUDA context, even
    `torch.cuda.memory_allocated()` can throw -- and if that happened here it
    would replace the real error with a confusing secondary traceback.
    """
    try:
        mem = ""
        if device.type == "cuda":
            try:
                mem = (f", allocated={torch.cuda.memory_allocated(device) / 1e9:.2f}GB"
                       f", reserved={torch.cuda.memory_reserved(device) / 1e9:.2f}GB")
            except RuntimeError:
                mem = " (memory stats unavailable -- CUDA context likely broken)"
        write(
            f"!! batch failed on {device}: {type(err).__name__}: {err}\n   "
            f"{micro['target'].num_nodes} nodes, {micro['target'].num_edges} edges, "
            f"{micro['target'].num_fragments} fragments, {micro['num_scenes']} scene(s){mem}"
        )
        write(f"   scenes: {micro.get('scene_dirs')}")
    except Exception as diag_err:  # noqa: BLE001
        print(f"(failed to print failure diagnostics: {diag_err!r})", flush=True)


# ---------------------------------------------------------------------------
def run_worker(rank: int, world_size: int, cfg: Config) -> None:
    distributed = world_size > 1
    is_main = rank == 0

    if distributed:
        device = D.setup(rank, world_size, cfg.master_port)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    seed_everything(cfg.seed, rank)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, val_loader, train_set = build_dataloaders(cfg)
    model = build_model(cfg, device)

    if is_main:
        write(f"VN-GAT | {model.num_parameters():,} parameters | hidden={cfg.hidden_channels} "
              f"layers={cfg.num_layers} heads={cfg.heads} slots={cfg.num_vn_slots} norm={cfg.norm}")
        write(f"input_source={cfg.input_source} | train pool={train_set.pool_size} scenes | "
              f"world_size={world_size} | batch={cfg.batch_size} (micro={cfg.micro_batch_scenes}) | "
              f"amp={cfg.amp}")

    if distributed:
        model = DDP(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
            # OFF on purpose: the loss keeps every parameter connected, so no
            # traversal is needed and every rank builds identical buckets.
            find_unused_parameters=False,
            # VNLayerNorm has no buffers; skipping the broadcast saves a
            # collective per step. (With norm='batch' this must be True.)
            broadcast_buffers=(cfg.norm == "batch"),
        )

    loss_fn = CompositeLoss(
        w_rot=cfg.w_rot, w_pos=cfg.w_pos, w_node=cfg.w_node, w_mid=cfg.w_mid,
        w_face=cfg.w_face, w_emb_v=cfg.w_emb_v, w_emb_e=cfg.w_emb_e,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
    )
    scaler = make_scaler(device, cfg.amp)

    drive = DriveSync(cfg.drive_folder_id, cfg.drive_credentials, enabled=is_main)
    manager = CheckpointManager(cfg.checkpoint_dir, cfg.save_every, drive, cfg.tag)
    history = History()
    start_epoch = 0

    # Only rank 0 talks to Drive (DriveSync is constructed with enabled=is_main),
    # so it downloads first; the barrier makes the other ranks wait for the file
    # to actually be on disk before they look for it.
    resume_path = None
    if cfg.resume not in ("", "none", "None"):
        if is_main:
            resume_path = manager.locate(cfg.resume)
        if distributed:
            dist.barrier()
            if not is_main:
                resume_path = manager.locate(cfg.resume)
    if resume_path is not None and resume_path.is_file():
        state = manager.load(resume_path, model, optimizer, scheduler, scaler,
                             map_location=str(device))
        start_epoch = int(state.get("epoch", -1)) + 1
        history = History.from_dict(state.get("history") or {})
        history.truncate_to(start_epoch)
        if is_main:
            write(f"  [ckpt] resumed from {resume_path} at epoch {start_epoch} "
                  f"(best val {manager.best_val:.4f})")
    elif is_main and cfg.resume not in ("", "none", "None"):
        write("  [ckpt] no checkpoint found -- starting from scratch")

    if is_main:
        write(table_header())

    total_steps = cfg.steps_per_epoch * cfg.batch_size + cfg.val_steps * cfg.batch_size
    bar = make_bar(total_steps, "epoch", enabled=is_main)
    wall_start = time.perf_counter()
    budget_seconds = cfg.time_budget_hours * 3600.0

    for epoch in range(start_epoch, cfg.epochs):
        bar.reset(total=total_steps)
        bar.set_description(f"epoch {epoch}")

        train_metrics, train_diag = run_phase(
            model, train_loader, loss_fn, optimizer, scaler, device, cfg,
            train=True, epoch=epoch, bar=bar,
        )
        val_metrics, val_diag = run_phase(
            model, val_loader, loss_fn, optimizer, scaler, device, cfg,
            train=False, epoch=epoch, bar=bar,
        )

        if distributed:
            train_metrics = D.reduce_metrics(train_metrics, device)
            val_metrics = D.reduce_metrics(val_metrics, device)

        scheduler.step(val_metrics["total"])

        if is_main:
            write(table_row(epoch, "train", train_metrics,
                            train_diag["data_seconds"], train_diag["compute_seconds"]))
            write(table_row(epoch, "val", val_metrics,
                            val_diag["data_seconds"], val_diag["compute_seconds"]))
            history.append("train", train_metrics)
            history.append("val", val_metrics)
            history.append_meta(
                lr=optimizer.param_groups[0]["lr"],
                train_data_seconds=train_diag["data_seconds"],
                train_compute_seconds=train_diag["compute_seconds"],
                max_nodes=train_diag["max_nodes"],
                oom_recoveries=train_diag["oom_recoveries"],
                epoch_seconds=time.perf_counter() - wall_start,
            )

        elapsed = time.perf_counter() - wall_start
        out_of_time = elapsed > budget_seconds
        last_epoch = epoch + 1 >= cfg.epochs
        # Vote BEFORE the checkpoint/exit decision so every rank leaves the
        # loop together. A rank that decided alone would strand the others.
        stop_now = not D.all_ranks_agree(not (out_of_time or last_epoch), device)

        if is_main and (manager.should_save(epoch, cfg.epochs) or stop_now):
            wrote = manager.save(
                model, optimizer, scheduler, scaler, epoch,
                train_metrics["total"], val_metrics["total"],
                history.to_dict(), cfg.to_dict(),
                force=stop_now,
            )
            tags = [k for k, v in wrote.items() if v] or ["nothing (older was better on both)"]
            write(f"  [ckpt] epoch {epoch}: wrote {', '.join(tags)}")

        if stop_now:
            if is_main and out_of_time:
                write(f"  [budget] {elapsed / 3600:.2f}h used of {cfg.time_budget_hours}h "
                      f"-- stopping cleanly at epoch {epoch}. Re-run with the same "
                      f"--checkpoint_dir/--drive_folder_id to resume.")
            break

    bar.close()
    if distributed:
        D.cleanup()


def launch(cfg: Config) -> None:
    """Entry point: spawn one process per GPU, or run inline for 1 GPU / CPU."""
    cfg.validate()
    available = torch.cuda.device_count()
    num_gpus = cfg.num_gpus
    if num_gpus > available:
        write(f"  [setup] {num_gpus} GPUs requested but {available} visible -- using {available or 1}.")
        num_gpus = available
    if num_gpus <= 1:
        run_worker(0, 1, cfg)
        return
    import torch.multiprocessing as mp

    mp.spawn(run_worker, args=(num_gpus, cfg), nprocs=num_gpus, join=True)
