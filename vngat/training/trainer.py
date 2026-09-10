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

_LOSS_KEYS = ("total", "rot", "rot_deg", "pos", "node", "mid", "face", "emb_v", "emb_e",
              "head_cos", "tilt", "twist")


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
        subsets=[x for x in cfg.data_subsets.split(',') if x.strip()] or None,
        split_source=cfg.split_source,
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


class _CosineOnce:
    """
    Cosine decay to a floor, clamped so it never rises again.

    A class rather than a closure so `LambdaLR.state_dict()` captures `t_max`
    and `floor` -- see the note in `build_scheduler`. `__dict__` is what gets
    serialised, so both must be plain attributes.
    """

    def __init__(self, t_max: int, floor: float):
        self.t_max = max(1, int(t_max))
        self.floor = float(floor)

    def __call__(self, epoch: int) -> float:
        import math

        phase = min(epoch, self.t_max) / self.t_max      # clamped: never rises
        return self.floor + (1.0 - self.floor) * (1.0 + math.cos(math.pi * phase)) / 2.0


def build_scheduler(cfg: Config, optimizer, span: int | None = None):
    """
    Returns (scheduler, needs_metric).

    `plateau` watches `lr_monitor` on VALIDATION -- by default `rot`, the
    primary objective, not `total`. Watching the composite total is what killed
    a real run: face + norm + rot made up 4.95 of a 5.40 total and none of them
    moved, so every epoch registered as a plateau and the rate was halved nine
    times, ending 512x below where it started while the log gave no sign.
    """
    # `span` is the number of epochs the schedule should cover. It differs from
    # cfg.epochs only on a restarted resume, where the schedule must anneal over
    # the epochs that REMAIN rather than the run's total.
    span = max(1, span if span is not None else cfg.epochs)
    kind = (cfg.lr_schedule or "plateau").lower()
    if kind == "constant":
        return None, False
    if kind == "cosine":
        # CosineAnnealingLR is PERIODIC: past T_max it climbs back toward the
        # base rate. A real run resumed past its horizon did exactly that -- the
        # rate went 1.0e-5 -> 6.9e-4 over 75 epochs and the loss rose with it.
        # `_CosineOnce` clamps the phase so it anneals once and stays down.
        #
        # It is a CALLABLE CLASS, not a closure, and that is load-bearing:
        # LambdaLR.state_dict() serialises a lambda's __dict__ only when the
        # lambda is not a plain function. With a closure the horizon is never
        # written to the checkpoint, so on resume it silently reverts to
        # cfg.epochs while last_epoch continues -- a 23-epoch anneal became a
        # 60-epoch one and never reached its floor.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _CosineOnce(span, cfg.lr_min / max(cfg.lr, 1e-12))), False

    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.lr_min), True
    raise ValueError(f"lr_schedule must be 'plateau', 'cosine' or 'constant', got {kind!r}")


_RESTORED_FIELDS = (
    # --- what is being learned: schedule and objective -------------------
    "lr", "lr_min", "lr_schedule", "lr_monitor", "lr_patience", "lr_factor",
    "lr_warmup_epochs", "epochs", "weight_decay", "grad_clip",
    "w_rot", "w_pos", "w_node", "w_mid", "w_face", "w_emb_v", "w_emb_e",
    "emb_pull_margin", "emb_push_margin", "symmetry_axis",
    # --- WHAT IT IS BEING LEARNED FROM -----------------------------------
    # These define the dataset and the train/val split. Letting them fall back
    # to defaults mid-run is the worst failure mode available: dropping
    # `--split_source official` silently reverts to `hash`, which is a
    # DIFFERENT SPLIT -- validation objects leak into training and every
    # number afterwards is meaningless, with nothing in the log to say so.
    "data_subsets", "split_source", "val_frac", "test_frac", "split_seed",
    "max_scenes", "fracture_pattern", "input_source", "correspondence",
    "correspondence_tol", "min_fragments",
    # --- what an "epoch" means -------------------------------------------
    # The schedule is indexed in epochs, so changing these mid-run rescales the
    # horizon: 150 epochs of 80 steps is not 150 epochs of 50.
    "steps_per_epoch", "val_steps",
    # --- architecture ------------------------------------------------------
    # Restored like everything else, but ALSO checked: weights of one shape
    # cannot load into another, so an explicit mismatch must fail loudly rather
    # than be silently overridden.
    "hidden_channels", "num_layers", "num_vn_slots", "heads", "embed_dim",
    "gram_bottleneck", "norm",
)
_ARCHITECTURE_FIELDS = (
    "hidden_channels", "num_layers", "num_vn_slots", "heads", "embed_dim",
    "gram_bottleneck", "norm",
)
# Deliberately NOT restored -- genuinely per-machine, and the caller must be
# free to change them between platforms:
#   batch_size, num_gpus, num_workers, save_every, time_budget_hours,
#   checkpoint_dir, root_dir, device, amp, grad_checkpointing,
#   micro_batch_scenes, pin_memory, prefetch_factor, tag, resume, drive_*
# `batch_size` is per RANK, so 16 on two GPUs and 32 on one are the same
# effective batch -- which is why it must stay caller-controlled.


def adopt_checkpoint_config(cfg: Config, resume_path, is_main: bool) -> None:
    """
    Make a checkpoint SELF-SUFFICIENT, so resuming needs no flags.

    `--resume auto --checkpoint_dir X` continues a run exactly as it was --
    same schedule, same horizon, same loss weights -- on Kaggle, on Colab, or
    moving between them. Anything typed on the command line still wins, and is
    logged when it differs.

    Called BEFORE the model is built, so the architecture check below fails with
    a readable message instead of a shape error deep inside load_state_dict.
    """
    if resume_path is None or not resume_path.is_file():
        return
    stored = (torch.load(resume_path, map_location="cpu", weights_only=False)
              .get("config") or {})
    if not stored:
        return

    explicit = getattr(cfg, "_explicit", frozenset())
    for name in _ARCHITECTURE_FIELDS:
        # Only an EXPLICIT mismatch is an error. Saying nothing means "use the
        # checkpoint's architecture", which is restored below.
        if name in explicit and name in stored and stored[name] != getattr(cfg, name):
            raise SystemExit(
                f"Architecture mismatch: the checkpoint has {name}={stored[name]}, "
                f"this run specifies {getattr(cfg, name)}. Weights of one shape cannot "
                f"load into another. Drop the flag to use the checkpoint's value, or "
                f"start fresh with --resume none."
            )

    restored, overridden = [], []
    for name in _RESTORED_FIELDS:
        if name not in stored:
            continue
        if name in explicit:
            if stored[name] != getattr(cfg, name):
                overridden.append(f"{name}: {stored[name]} -> {getattr(cfg, name)}")
        else:
            setattr(cfg, name, stored[name])
            restored.append(name)
    if is_main and restored:
        write(f"  [ckpt] using the checkpoint's settings for: {', '.join(restored)}")
    for line in overridden:
        if is_main:
            write(f"  [ckpt] command line overrides {line}")


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
def _first_non_finite_parameter(model) -> str | None:
    """Name of the first parameter containing NaN/inf, or None."""
    target = model.module if isinstance(model, DDP) else model
    for name, param in target.named_parameters():
        if not bool(torch.isfinite(param).all()):
            return name
    return None


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
        losses = dict(losses)
        losses["head_cos"] = outputs["head_cos"]
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
    nan_skips = 0
    nan_scenes: Dict[str, int] = {}

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
        # A factory: `model.no_sync` and `nullcontext` are both callables that
        # return a FRESH context manager, which the OOM ladder requires.
        make_sync = model.no_sync if (train and isinstance(model, DDP)) else nullcontext

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
            losses, recovered, used_fallback, skipped = _run_micro_step(
                model, raw_model, loss_fn, micro, device, cfg, train, scaler, make_sync,
                scale=1.0 / len(micro_batches),
            )
            oom_recoveries += int(recovered)
            fallback_steps += int(used_fallback)
            if skipped:
                # A NaN in the accumulator would make every reported metric NaN
                # for the rest of the epoch, hiding whether training recovered.
                nan_skips += 1
                for name in micro.get("scene_dirs", []):
                    nan_scenes[name] = nan_scenes.get(name, 0) + 1
                continue
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
        "nan_skips": float(nan_skips),
        "nan_scenes": nan_scenes,
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


def _run_micro_step(model, raw_model, loss_fn, micro, device, cfg, train, scaler, make_sync, scale):
    """
    Forward/backward for one micro-batch, with a two-step OOM ladder:
      1. retry the same scene with gradient checkpointing on;
      2. if it still will not fit, substitute the dummy scene so this rank
         performs the same number of backward passes as its peers.

    `make_sync` is a FACTORY, not a context manager. `DistributedDataParallel.
    no_sync()` returns a `contextlib._GeneratorContextManager`, which deletes
    its own args/kwds/func on `__enter__` and therefore cannot be entered
    twice -- re-entering one raises
    "'_GeneratorContextManager' object has no attribute 'args'". Since this
    loop may enter it up to three times, a fresh instance has to be built per
    attempt. (`nullcontext` IS reusable, which is why the single-instance
    version worked everywhere except the one path that matters: DDP plus an
    actual out-of-memory retry.)
    """
    recovered = False
    used_fallback = False
    checkpoint_was = raw_model.grad_checkpointing

    for attempt in range(3):
        try:
            scene = _dummy_scene(device) if attempt == 2 else prepare_scene(micro, device)
            if attempt == 1:
                raw_model.grad_checkpointing = True
            with make_sync():
                if train:
                    losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
                    # Check BEFORE backward. A non-finite loss produces
                    # non-finite gradients, and although GradScaler normally
                    # skips such a step, letting it near the optimiser at all
                    # risks poisoning the weights -- after which every
                    # subsequent forward is NaN and the run silently burns
                    # hours producing nothing. Skipping is safe for DDP here
                    # because the gradient all-reduce is fired by a separate
                    # placeholder step, so the collective count per optimiser
                    # step does not depend on how many micro-steps ran.
                    if not bool(torch.isfinite(losses["total"])):
                        # Retry in FULL PRECISION before giving up on the scene.
                        #
                        # The failure is an activation-magnitude overflow: it
                        # appears only once training has grown the weights, and
                        # it lands on whichever scene is drawn at the time, not
                        # on a fixed set of bad meshes. Recomputing the same
                        # step in float32 costs one extra forward/backward on a
                        # small fraction of steps and KEEPS THE DATA, instead of
                        # silently excluding whole objects from training.
                        if cfg.amp:
                            losses = _forward_loss(model, loss_fn, scene, device, amp=False)
                        if bool(torch.isfinite(losses["total"])):
                            scaler.scale(losses["total"] * scale).backward()
                            raw_model.grad_checkpointing = checkpoint_was
                            return losses, recovered, True, False
                        bad = [k for k, v in losses.items()
                               if not bool(torch.isfinite(v).all())]
                        write(f"  [nan] non-finite loss {bad} in float32 too -- skipping. "
                              f"{micro['target'].num_nodes} nodes, "
                              f"{micro['target'].num_fragments} fragments, "
                              f"scenes {micro.get('scene_dirs')}")
                        raw_model.grad_checkpointing = checkpoint_was
                        return losses, recovered, used_fallback, True
                    scaler.scale(losses["total"] * scale).backward()
                else:
                    with torch.no_grad():
                        losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
                    # Validation has no backward to protect, but a single
                    # non-finite micro-step still poisons the running average
                    # and reports the WHOLE epoch as nan -- hiding whether the
                    # rest of validation was fine.
                    if not bool(torch.isfinite(losses["total"])) and cfg.amp:
                        with torch.no_grad():
                            losses = _forward_loss(model, loss_fn, scene, device, amp=False)
                    if not bool(torch.isfinite(losses["total"])):
                        write(f"  [nan] non-finite validation loss on "
                              f"{micro.get('scene_dirs')} -- excluded from the average")
                        raw_model.grad_checkpointing = checkpoint_was
                        return losses, recovered, used_fallback, True
            raw_model.grad_checkpointing = checkpoint_was
            return losses, recovered, used_fallback, False
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
        device = D.resolve_device(cfg.device, rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    seed_everything(cfg.seed, rank)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, val_loader, train_set = build_dataloaders(cfg)
    # Peek at the checkpoint first: its stored config decides the architecture
    # and the schedule, so nothing may be constructed before this.
    _probe = CheckpointManager(cfg.checkpoint_dir, cfg.save_every, None, cfg.tag)
    _resume_path = (_probe.locate(cfg.resume)
                    if cfg.resume not in ("", "none", "None") else None)
    adopt_checkpoint_config(cfg, _resume_path, is_main)

    model = build_model(cfg, device)

    if is_main:
        write("chance level: geodesic 126.47 deg. A matched-budget scaling run reached "
              "30.9 deg on 8 objects, so there is no known structural floor above that.")
        write(f"VN-GAT | {model.num_parameters():,} parameters | hidden={cfg.hidden_channels} "
              f"layers={cfg.num_layers} heads={cfg.heads} slots={cfg.num_vn_slots} norm={cfg.norm}")
        write(f"input_source={cfg.input_source} | train pool={train_set.pool_size} scenes | "
              f"world_size={world_size} | batch={cfg.batch_size} (micro={cfg.micro_batch_scenes}) | "
              f"amp={cfg.amp}")
        if cfg.amp:
            write("!! amp=True: a controlled comparison found half precision both unstable "
                  "(non-finite losses on 6 of 8 objects from epoch 66) and WORSE than fp32 "
                  "(54.3 vs 43.2 deg). Use --amp false unless you have a specific reason.")

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
        emb_pull_margin=cfg.emb_pull_margin, emb_push_margin=cfg.emb_push_margin,
        symmetry_axis=cfg.symmetry_axis,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler, scheduler_needs_metric = build_scheduler(cfg, optimizer)
    scaler = make_scaler(device, cfg.amp)

    drive = DriveSync(cfg.drive_folder_id, cfg.drive_credentials, enabled=is_main)
    if is_main and drive.enabled:
        # One real round trip now. Authentication succeeding says nothing about
        # whether writes land, and discovering otherwise at the first checkpoint
        # means ten epochs of a multi-hour run were never mirrored.
        if drive.verify():
            write(f"  [drive] mirroring verified ({drive.credential_kind})")
        else:
            write("!! [drive] configured but NOT working -- training will continue with "
                  "LOCAL checkpoints only.")
            write("   Run `python -m scripts.check_drive --folder_id ... ` for the diagnosis.")
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
            remaining = cfg.epochs - start_epoch
            if remaining <= 0:
                write(f"!! this checkpoint is already at epoch {start_epoch} of "
                      f"{cfg.epochs}, so there is nothing left to run. Pass a larger "
                      f"--epochs to extend the schedule.")
            else:
                write(f"  [ckpt] --epochs {cfg.epochs} is a TOTAL, so {remaining} epoch(s) "
                      f"remain from here")
    elif is_main and cfg.resume not in ("", "none", "None"):
        write("  [ckpt] no checkpoint found -- starting from scratch")

    if start_epoch > 0 and cfg.restart_schedule:
        remaining = max(1, cfg.epochs - start_epoch)
        for group in optimizer.param_groups:
            # BOTH keys. `LambdaLR`/`CosineAnnealingLR` read their base rates
            # from `initial_lr` and only write that key when it is ABSENT --
            # and `optimizer.load_state_dict` has already restored it from the
            # checkpoint. Setting `lr` alone is silently undone on the
            # scheduler's first step, so a restart requesting 2e-4 actually ran
            # at the checkpoint's 5e-4 while the log reported 2e-4.
            group["lr"] = cfg.lr
            group["initial_lr"] = cfg.lr
        scheduler, scheduler_needs_metric = build_scheduler(cfg, optimizer, span=remaining)
        if is_main:
            actual = optimizer.param_groups[0]["lr"]
            write(f"  [lr  ] schedule restarted at {actual:.2e}, annealing over the "
                  f"{remaining} remaining epoch(s) to {cfg.lr_min:.2e}")
            if abs(actual - cfg.lr) > 1e-12:
                # Report what the optimiser actually holds, never what was asked
                # for -- the previous message printed the request and was wrong.
                write(f"!! requested --lr {cfg.lr:.2e} but the optimiser holds "
                      f"{actual:.2e}; the schedule is NOT what you asked for")

    if is_main:
        write(table_header())

    total_steps = cfg.steps_per_epoch * cfg.batch_size + cfg.val_steps * cfg.batch_size
    bar = make_bar(total_steps, "epoch", enabled=is_main)
    wall_start = time.perf_counter()
    budget_seconds = cfg.time_budget_hours * 3600.0

    for epoch in range(start_epoch, cfg.epochs):
        # Linear warmup, applied before any scheduler step so the two do not
        # fight over param_groups.
        if cfg.lr_warmup_epochs > 0 and epoch < cfg.lr_warmup_epochs:
            warm = (epoch + 1) / cfg.lr_warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * (0.1 + 0.9 * warm)

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

        if scheduler is not None and epoch >= cfg.lr_warmup_epochs:
            if scheduler_needs_metric:
                monitored = val_metrics.get(cfg.lr_monitor, val_metrics["total"])
                scheduler.step(monitored)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            write(table_row(epoch, "train", train_metrics,
                            train_diag["data_seconds"], train_diag["compute_seconds"], current_lr))
            write(table_row(epoch, "val", val_metrics,
                            val_diag["data_seconds"], val_diag["compute_seconds"], current_lr))
            tilt_v, twist_v = val_metrics.get("tilt"), val_metrics.get("twist")
            if tilt_v is not None:
                verdict = ""
                if tilt_v < 25 and twist_v > 60:
                    verdict = "  <- axis learned, azimuth NOT (structural floor)"
                elif tilt_v > 60:
                    verdict = "  <- axis not learned either (headroom remains)"
                write(f"  [val ] tilt {tilt_v:6.2f}  twist {twist_v:6.2f}"
                      f"  (chance 90/90){verdict}")
            if train_diag["fallback_steps"]:
                write(f"  [amp] {int(train_diag['fallback_steps'])} step(s) recomputed in "
                      f"float32 after a half-precision overflow (data kept)")
            if train_diag["nan_skips"]:
                skipped = int(train_diag["nan_skips"])
                attempted = skipped + cfg.steps_per_epoch * cfg.batch_size // max(cfg.micro_batch_scenes, 1)
                write(f"  [nan] skipped {skipped} micro-step(s) this epoch "
                      f"({100.0 * skipped / max(attempted, 1):.1f}%) with non-finite losses")
                # Name the repeat offenders. Dropping a step is safe for the
                # optimiser but NOT neutral for the data: if the same scenes
                # fail every epoch they are effectively excluded from training,
                # which is a silent bias rather than a transient loss.
                worst = sorted(train_diag["nan_scenes"].items(), key=lambda kv: -kv[1])[:3]
                for name, hits in worst:
                    write(f"     {hits}x  {name}")
                if worst and worst[0][1] > 1:
                    write("   Repeat offenders are effectively excluded from training. "
                          "Diagnose one with: python -m scripts.check_scene --scene <path>")
                if skipped > 0.2 * attempted:
                    write("   A skip rate this high is systematic, not an unlucky scene. "
                          "Most likely activation overflow under AMP: retry with --amp false "
                          "to confirm, and lower --lr.")
            history.append("train", train_metrics)
            history.append("val", val_metrics)
            history.append_meta(
                lr=current_lr,
                train_data_seconds=train_diag["data_seconds"],
                train_compute_seconds=train_diag["compute_seconds"],
                max_nodes=train_diag["max_nodes"],
                oom_recoveries=train_diag["oom_recoveries"],
                nan_skips=train_diag["nan_skips"],
                epoch_seconds=time.perf_counter() - wall_start,
            )

        # Weights themselves, not just the reported loss. Once a parameter is
        # NaN every later forward is NaN and the run produces nothing -- there
        # is no recovery, so continuing wastes the remainder of the session.
        # Checked once per epoch: ~200 tiny kernels, unmeasurable.
        corrupted = _first_non_finite_parameter(model)
        if corrupted is not None:
            write(f"!! [nan] parameter '{corrupted}' is non-finite -- the model is unrecoverable.")
            write(f"   Stopping at epoch {epoch}. The last good checkpoint is still in "
                  f"{cfg.checkpoint_dir}; resume from it with a lower --lr, and see the "
                  f"[nan] skip lines above for which scenes produced non-finite losses.")

        elapsed = time.perf_counter() - wall_start
        out_of_time = elapsed > budget_seconds
        last_epoch = epoch + 1 >= cfg.epochs or corrupted is not None
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
            tags = [k for k, v in wrote.items() if v]
            write(f"  [ckpt] epoch {epoch}: wrote {', '.join(tags)}"
                  + ("" if wrote["rolling"] else "  (checkpoint.pt held: older better on both)"))

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
