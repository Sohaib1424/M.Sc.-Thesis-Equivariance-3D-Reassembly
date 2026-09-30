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
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from ..config import Config
from ..data.catalog import describe, fracture_split_report
from ..data.dataset import (
    BreakingBadDataset, collate_fn, dataset_kwargs, merge_micro_batches, split_batch_by_scene,
)
from ..data.graph import SceneBatch
from ..evaluation.anchor import anchor_alignment, scored_fragments
from ..evaluation.metrics import swing_twist_error
from ..losses.composite import CompositeLoss
from ..models.vn_gat import VNGATModel
from ..models.vn_layers import geodesic_rotation_loss
from ..utils.env import dataloader_worker_init, seed_everything
from ..utils.progress import make_bar, table_header, table_row, write
from . import distributed as D
from .bridge import build_model_inputs, build_predictions, build_targets, prepare_scene
from .checkpoint import CheckpointManager
from .drive import DriveSync
from .history import History

_LOSS_KEYS = ("total", "rot", "rot_deg", "pos", "node", "face", "emb_v", "emb_e",
              "head_cos", "tilt", "twist")

# The rotation metrics every epoch reports whatever the training target, summed
# per fragment and divided once after the reduction over ranks (exact):
#   anchor_deg    each scene's largest fragment set to its true pose, the other
#                 fragments' geodesic error -- the benchmark's protocol
#   absolute_deg  every fragment against its object's stored frame
# The protocol's tilt/twist replace the loss-side ones in the returned metrics.
# See `vngat.evaluation.anchor`.
_PROTOCOL_SUMS = ("anchor_deg", "absolute_deg", "tilt", "twist")


def loss_fragments(graph: SceneBatch, rotation_target: str) -> int:
    """
    The fragments one micro-batch's loss is a mean over, and so its weight in
    the step and in the epoch's averages: every fragment under the absolute
    target, all but one per scene under the anchor target. The step divides by
    the total of these over every rank, so each scored fragment counts once.
    """
    if rotation_target == "anchor":
        return scored_fragments(graph)
    return int(graph.num_fragments)


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
class ShardSampler(Sampler):
    """
    Indices `rank, rank + world, rank + 2*world, ...` of a FIXED dataset.

    `DistributedSampler` pads the shards to equal length by repeating items,
    which would count those scenes twice in every validation number. Unequal
    shards are harmless here: validation runs outside DDP (no collective per
    batch), and its metrics are reduced once, as sums and counts.
    """

    def __init__(self, length: int, rank: int = 0, world: int = 1):
        self.indices = list(range(rank, length, max(world, 1)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_dataloaders(cfg: Config, rank: int = 0, world: int = 1):
    """Returns (train_loader, val_loader, train_set, val_set)."""
    common = dataset_kwargs(cfg)
    train_set = BreakingBadDataset(
        split="train", nominal_length=cfg.steps_per_epoch * cfg.batch_size,
        balance=cfg.balance, balance_temperature=cfg.balance_temperature, **common,
    )
    val_set = BreakingBadDataset(
        split="val", nominal_length=cfg.val_steps * cfg.batch_size,
        fixed=cfg.val_fixed, fixed_count=cfg.val_scenes, **common,
    )

    # Training draws ignore their index, so there is no index space for a
    # DistributedSampler to partition. Each rank is seeded differently
    # (seed_everything(seed, rank)), which is what keeps ranks from duplicating
    # each other's scenes.
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=dataloader_worker_init,
    )
    if cfg.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
    train_loader = DataLoader(train_set, shuffle=False, drop_last=True, **loader_kwargs)
    if cfg.val_fixed:
        # The fixed set is SHARDED, not repeated: each scene is validated once
        # per epoch, on exactly one rank.
        val_loader = DataLoader(val_set, sampler=ShardSampler(len(val_set), rank, world),
                                drop_last=False, **loader_kwargs)
    else:
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_kwargs)
    return train_loader, val_loader, train_set, val_set


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
    "w_rot", "w_pos", "w_node", "w_face", "w_emb_v", "w_emb_e",
    "emb_pull_margin", "emb_push_margin", "symmetry_axis", "checkpoint_monitor",
    # A checkpoint from before this setting existed was trained on the absolute
    # target and is restored as such (`adopt_checkpoint_config`), so resuming
    # an old run continues it unchanged.
    "rotation_target",
    # --- WHAT IT IS BEING LEARNED FROM -----------------------------------
    # These define the dataset and the train/val split. Letting them fall back
    # to defaults mid-run is the worst failure mode available: dropping
    # `--split_source official` silently reverts to `hash`, which is a
    # DIFFERENT SPLIT -- validation objects leak into training and every
    # number afterwards is meaningless, with nothing in the log to say so.
    # `split_by` swaps the validation set for break patterns of objects the
    # model trained on; `balance` changes what the training distribution is;
    # `normalize_mode` changes the units the network was trained in.
    "data_subsets", "split_source", "split_by", "fracture_pool", "val_frac",
    "test_frac", "split_seed", "max_scenes", "fracture_pattern", "input_source",
    "correspondence", "correspondence_tol", "min_fragments", "max_fragments", "balance",
    "balance_temperature", "normalize_mode", "val_fixed", "val_scenes",
    # --- what an "epoch" means -------------------------------------------
    # The schedule is indexed in epochs, so changing these mid-run rescales the
    # horizon: 150 epochs of 80 steps is not 150 epochs of 50.
    "steps_per_epoch", "val_steps",
    # --- architecture ------------------------------------------------------
    # Restored like everything else, but ALSO checked: weights of one shape
    # cannot load into another, so an explicit mismatch must fail loudly rather
    # than be silently overridden.
    "hidden_channels", "num_layers", "num_vn_slots", "heads", "head_dim",
    "embed_dim", "gram_bottleneck", "norm",
)
_ARCHITECTURE_FIELDS = (
    "hidden_channels", "num_layers", "num_vn_slots", "heads", "head_dim",
    "embed_dim", "gram_bottleneck", "norm",
)
# Deliberately NOT restored -- genuinely per-machine, and the caller must be
# free to change them between platforms:
#   batch_size, num_gpus, num_workers, save_every, time_budget_hours,
#   checkpoint_dir, root_dir, device, amp, grad_checkpointing,
#   micro_batch_scenes, pin_memory, prefetch_factor, tag, resume, drive_*
# `batch_size` is per RANK, so 16 on two GPUs and 32 on one are the same
# effective batch -- which is why it must stay caller-controlled.


# A checkpoint written before `rotation_target` existed was trained on the
# absolute target. Read through this, never through the current default.
LEGACY_ROTATION_TARGET = "absolute"

# Monitorable values that mean the same thing whatever the rotation target.
_PROTOCOL_MONITORS = ("anchor_deg", "absolute_deg", "tilt", "twist")


def _fragment_limit_changed(stored_cfg: Dict, cfg: Config) -> bool:
    """
    Whether validation is a different set of scenes from the one the
    checkpoint's best was measured on, because `max_fragments` differs. A
    checkpoint from before the setting existed used every break pattern.
    """
    return int(stored_cfg.get("max_fragments") or 0) != cfg.max_fragments


def _monitor_changed_meaning(monitor: str, stored_target: str, target: str,
                             checkpoint_knows_targets: bool) -> bool:
    """
    Whether the best-so-far in a checkpoint was measured differently from how
    `monitor` is measured now, although its name is the same: a loss term
    after the rotation target changed, or a protocol figure from a checkpoint
    that predates the protocol (tilt and twist were then the absolute ones,
    and anchor_deg / absolute_deg did not exist).
    """
    if monitor in _PROTOCOL_MONITORS:
        return not checkpoint_knows_targets
    return stored_target != target


def adopt_checkpoint_config(cfg: Config, resume_path, is_main: bool) -> None:
    """
    Make a checkpoint SELF-SUFFICIENT, so resuming needs no flags.

    `--resume auto --checkpoint_dir X` continues a run exactly as it was --
    same schedule, same horizon, same loss weights, same data -- on Kaggle, on
    Colab, or moving between them. Anything typed on the command line still
    wins, and is logged when it differs.

    Called BEFORE the model is built, so the architecture check below fails with
    a readable message instead of a shape error deep inside load_state_dict.
    """
    if resume_path is None or not resume_path.is_file():
        return
    stored = (torch.load(resume_path, map_location="cpu", weights_only=False)
              .get("config") or {})
    if not stored:
        return
    if "head_dim" not in stored:
        # Every v5 checkpoint records `head_dim`; the previous version's never
        # did. Its mesh layer has different parameters, so its weights cannot
        # load -- say so, rather than fail on the first missing key.
        raise SystemExit(
            f"{resume_path} was written by the previous VN-GAT (Thesis 1), whose "
            f"mesh-edge layer and edge features differ from this version's. Its "
            f"weights cannot be loaded here. Start fresh with --resume none, or "
            f"point --checkpoint_dir at a new directory."
        )

    explicit = getattr(cfg, "_explicit", frozenset())
    if "rotation_target" not in stored:
        stored = dict(stored, rotation_target=LEGACY_ROTATION_TARGET)
        if is_main:
            write(f"  [ckpt] this checkpoint predates --rotation_target: it was trained on "
                  f"the '{LEGACY_ROTATION_TARGET}' target"
                  + (f", and the command line switches it to '{cfg.rotation_target}'"
                     if "rotation_target" in explicit else
                     ", which this run continues (pass --rotation_target anchor to switch)"))
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
        head_dim=cfg.head_dim,
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
    A minimal 1-fragment, 2-vertex, 1-edge scene, used ONLY to fire DDP's
    gradient all-reduce from a step that cannot fail (`_sync_gradients`).

    Its loss is multiplied by zero, so it contributes nothing to the gradient;
    what matters is that its backward reaches every parameter, which it does:
    one edge exercises the mesh layer's edge terms, one fragment the virtual
    nodes and the head, and the embedding losses stay graph-connected when
    there is nothing to cluster.
    """
    graph = SceneBatch(
        node_vec=torch.zeros(2, 2, 3, device=device),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long, device=device),
        edge_attr=torch.zeros(1, 3, 3, device=device),
        node_frag=torch.zeros(2, dtype=torch.long, device=device),
        frag_scene=torch.zeros(1, dtype=torch.long, device=device),
        frag_centroid=torch.zeros(1, 3, device=device),
        frag_unit=torch.ones(1, device=device),
        frag_log_scale=torch.zeros(1, 1, device=device),
        vertex_cluster_id=torch.full((2,), -1, dtype=torch.long, device=device),
        edge_cluster_id=torch.full((1,), -1, dtype=torch.long, device=device),
        num_fragments=1,
        num_scenes=1,
    )
    # A tiny non-degenerate geometry so normalisations do not divide by zero.
    graph.node_vec[0, 0, 0] = -0.5
    graph.node_vec[1, 0, 0] = 0.5
    graph.node_vec[:, 1, 2] = 1.0
    graph.edge_attr[0, 0, 2] = 1.0
    graph.edge_attr[0, 1, 2] = 1.0
    graph.edge_attr[0, 2, 0] = -1.0
    rot = torch.eye(3, device=device).unsqueeze(0)
    return {
        "clean_target": graph,
        "diffused_target": graph,
        "diffused_input": graph,
        "rot": rot,
        "categories": [""],
        "scene_dirs": [""],
    }


def _forward_loss(model, loss_fn, scene: Dict, device: torch.device, amp: bool):
    """
    One forward and the loss breakdown.

    Under `loss_fn.rotation_target == "anchor"` the rotation every geometric
    term compares is the prediction with its scene's largest fragment set to
    its true pose (`vngat.evaluation.anchor`), and the anchors are left out of
    those terms. The anchor-protocol metrics ride along under underscored keys
    whatever the target; they are computed without gradient and never enter
    `total`.
    """
    with make_autocast(device, amp):
        outputs = model(**build_model_inputs(scene["diffused_input"]))
        targets = build_targets(scene["clean_target"], scene["rot"], scene["diffused_input"])
        R_raw = outputs["R_pred"]
        aligned, scored = anchor_alignment(R_raw, targets["R_gt"], scene["clean_target"])
        R_used = R_raw
        if getattr(loss_fn, "rotation_target", "absolute") == "anchor":
            R_used = aligned
            targets["fragment_keep"] = scored
        predictions = build_predictions(scene["diffused_target"], R_used)
        merged = dict(
            R_pred=R_used,
            vertex_embedding=outputs["vertex_embedding"],
            edge_embedding=outputs["edge_embedding"],
            **predictions,
        )
        losses = dict(loss_fn(merged, targets))
        losses["head_cos"] = outputs["head_cos"]
    with torch.no_grad():
        degrees = 180.0 / math.pi
        R_gt = targets["R_gt"].to(aligned.dtype)
        losses["_anchor_deg"] = geodesic_rotation_loss(aligned[scored].detach(), R_gt[scored]) * degrees
        losses["_absolute_deg"] = geodesic_rotation_loss(
            R_raw.detach().to(aligned.dtype), R_gt) * degrees
        tilt, twist = swing_twist_error(aligned[scored].detach(), R_gt[scored],
                                        axis=getattr(loss_fn, "symmetry_axis", "z"))
        losses["_anchor_tilt"], losses["_anchor_twist"] = tilt, twist
        losses["_anchor_scene"] = scene["clean_target"].frag_scene[scored]
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


# ---------------------------------------------------------------------------
# One micro-batch
# ---------------------------------------------------------------------------
class MicroResult(NamedTuple):
    losses: Optional[Dict[str, torch.Tensor]]
    status: str        # "ok" | "fp32" (recomputed in float32) | "nonfinite" | "oom"
    recovered: bool    # needed the gradient-checkpointing retry


_CONTRIBUTED = ("ok", "fp32")


def _finite(losses: Dict[str, torch.Tensor]) -> bool:
    return bool(torch.isfinite(losses["total"]))


def _take_grads(params: List[torch.Tensor]) -> List[Optional[torch.Tensor]]:
    """Move the gradients accumulated so far aside; the next backward starts from None."""
    stash = [p.grad for p in params]
    for p in params:
        p.grad = None
    return stash


def _put_back(params: List[torch.Tensor], stash: List[Optional[torch.Tensor]]) -> None:
    """Throw away whatever this micro-step wrote; restore the accumulation."""
    for p, g in zip(params, stash):
        p.grad = g


def _merge_grads(params: List[torch.Tensor], stash: List[Optional[torch.Tensor]]) -> None:
    """Keep this micro-step's gradient: add the accumulation back onto it."""
    for p, g in zip(params, stash):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)


def _grads_finite(params: List[torch.Tensor]) -> bool:
    norms = [torch.linalg.vector_norm(p.grad.detach()) for p in params if p.grad is not None]
    return (not norms) or bool(torch.isfinite(torch.stack(norms)).all())


def _run_micro_step(net, raw_model, loss_fn, micro, device, cfg, train, scaler, make_sync,
                    scale) -> MicroResult:
    """
    Forward (and, when training, backward) for one micro-batch.

    WHY THE GRADIENT IS ISOLATED PER MICRO-STEP
    -------------------------------------------
    The accumulated gradient is moved aside before the backward and merged back
    only if the micro-step succeeds. Two failures need that:

      * an OUT-OF-MEMORY error part way through a backward has already added
        part of this scene's gradient into `.grad`. The retry then added the
        whole gradient again, so some parameters received this scene twice.
      * a FINITE loss can still produce a NON-FINITE gradient (a norm at zero,
        an overflow inside the backward). Merged into the accumulation, one
        such scene poisons the whole optimiser step.

    Both are now discarded cleanly, and only this micro-batch is lost.

    The OOM ladder: retry the scene once with gradient checkpointing; if it
    still does not fit, SKIP it. (The previous version substituted a
    placeholder scene and back-propagated its real loss, which added a
    meaningless gradient. Rank symmetry never needed it: every real micro-step
    runs under `no_sync()`, so skipping one changes no collective -- the single
    all-reduce per step is fired by `_sync_gradients`.)

    `make_sync` is a FACTORY, not a context manager: `DistributedDataParallel.
    no_sync()` returns a generator-based context manager that cannot be
    entered twice, and this function may enter it twice.
    """
    checkpoint_was = raw_model.grad_checkpointing
    params = [p for p in raw_model.parameters() if p.requires_grad] if train else []
    stash = _take_grads(params) if train else None
    recovered = False
    try:
        for attempt in range(2):
            try:
                scene = prepare_scene(micro, device)
                if not train:
                    with torch.no_grad():
                        losses = _forward_loss(net, loss_fn, scene, device, cfg.amp)
                        if not _finite(losses) and cfg.amp:
                            losses = _forward_loss(net, loss_fn, scene, device, amp=False)
                    if not _finite(losses):
                        write(f"  [nan] non-finite validation loss on "
                              f"{micro.get('scene_dirs')} -- excluded from the average")
                        return MicroResult(losses, "nonfinite", recovered)
                    return MicroResult(losses, "ok", recovered)

                if attempt == 1:
                    raw_model.grad_checkpointing = True
                status = "ok"
                with make_sync():
                    losses = _forward_loss(net, loss_fn, scene, device, cfg.amp)
                    # Check BEFORE backward: a non-finite loss makes non-finite
                    # gradients, and letting those near the optimiser at all
                    # risks poisoning the weights -- after which every later
                    # forward is NaN and the run silently burns hours.
                    if not _finite(losses):
                        # Retry in FULL PRECISION before giving up on the scene:
                        # the failure is usually a half-precision activation
                        # overflow, and recomputing in float32 keeps the data.
                        if cfg.amp:
                            losses = _forward_loss(net, loss_fn, scene, device, amp=False)
                            status = "fp32"
                        if not _finite(losses):
                            bad = [k for k, v in losses.items()
                                   if torch.is_tensor(v) and not bool(torch.isfinite(v).all())]
                            write(f"  [nan] non-finite loss {bad} -- skipping. "
                                  f"{micro['target'].num_nodes} nodes, "
                                  f"{micro['target'].num_fragments} fragments, "
                                  f"scenes {micro.get('scene_dirs')}")
                            return MicroResult(losses, "nonfinite", recovered)
                    scaler.scale(losses["total"] * scale).backward()
                # Under AMP the GradScaler owns this check -- an inf there means
                # "lower the scale", and hiding it would stop the scale adapting.
                if not scaler.is_enabled() and not _grads_finite(params):
                    write(f"  [nan] finite loss but non-finite gradient on "
                          f"{micro.get('scene_dirs')} -- this micro-batch is dropped")
                    return MicroResult(losses, "nonfinite", recovered)
                _merge_grads(params, stash)
                stash = None
                return MicroResult(losses, status, recovered)
            except RuntimeError as err:
                if not _is_oom(err):
                    _report_batch_failure(micro, device, err)
                    raise
                for p in params:           # discard the partial gradient
                    p.grad = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not train or attempt == 1:
                    write(f"  [oom] scene with {micro['target'].num_nodes} nodes / "
                          f"{micro['target'].num_edges} edges does not fit -- skipped")
                    return MicroResult(None, "oom", recovered)
                recovered = True
                write(f"  [oom] scene with {micro['target'].num_nodes} nodes / "
                      f"{micro['target'].num_edges} edges did not fit; retrying with "
                      f"gradient checkpointing")
        raise RuntimeError("unreachable")
    finally:
        raw_model.grad_checkpointing = checkpoint_was
        if stash is not None:
            _put_back(params, stash)


def _make_micro_batches(batch: Dict, micro_scenes: int) -> List[Dict]:
    if micro_scenes >= batch["num_scenes"]:
        return [batch]
    scenes = split_batch_by_scene(batch)
    return [merge_micro_batches(scenes[start:start + micro_scenes])
            for start in range(0, len(scenes), micro_scenes)]


# ---------------------------------------------------------------------------
# One optimiser step
# ---------------------------------------------------------------------------
def _sync_gradients(model, loss_fn, device, cfg, scaler) -> None:
    """
    Trigger DDP's gradient all-reduce exactly once per optimiser step, from a
    step that cannot fail.

    Runs a forward/backward on the two-vertex placeholder scene with the loss
    multiplied by ZERO. Backward still traverses the whole graph, so every
    parameter's DDP hook fires and every bucket is reduced -- carrying the
    gradients accumulated during the preceding `no_sync()` micro-steps, because
    DDP reduces the contents of `param.grad`, which accumulation has already
    filled in. Multiplying by zero means this step contributes nothing itself.

    Two properties follow, and both matter for a multi-day unattended run:
      * the number and order of collectives per step is FIXED, independent of
        scene sizes, OOM retries or how many fragments a rank happened to draw;
      * the reduction runs on a graph small enough that it cannot itself OOM,
        so a recoverable failure never leaves the collectives half-finished.
    """
    scene = _dummy_scene(device)
    losses = _forward_loss(model, loss_fn, scene, device, cfg.amp)
    scaler.scale(losses["total"] * 0.0).backward()


def _finish_step(model, loss_fn, optimizer, scaler, device, cfg,
                 rank_fragments: float, contributed: float) -> Tuple[str, float]:
    """
    Turn the accumulated micro-batch gradients into one optimiser step whose
    gradient is the mean over every FRAGMENT that contributed, on every rank.

    Each micro-batch's loss is a mean over its fragments, and it was scaled by
    its share of this rank's fragments. The step then rescales once, by
    `rank_fragments * world / contributed_everywhere`, BEFORE the all-reduce:

      * micro-batches are weighted by fragment count, not equally -- a
        2-fragment and an 8-fragment scene no longer count the same;
      * ranks are weighted by what they contributed, not equally -- DDP
        averages ranks, and the factor turns that average into the exact
        global fragment mean;
      * a dropped micro-batch (non-finite or out of memory) shrinks the
        denominator instead of silently shrinking the step.

    The rescale must happen BEFORE `_sync_gradients`, because that is where DDP
    all-reduces. After it, every rank holds the same averaged gradient, and a
    rank-local rescale would make the replicas step differently and drift
    apart with nothing to say so.

    One scalar all-reduce per step, entered by every rank at the same point.
    Returns (outcome, gradient norm before clipping).
    """
    distributed = isinstance(model, DDP)
    world = D.world_size() if distributed else 1
    everywhere = D.all_reduce_sum(contributed, device) if distributed else float(contributed)
    if everywhere <= 0.0:
        # Nothing contributed on ANY rank (every rank sees the same number, so
        # every rank skips together). AdamW would still move the weights on
        # momentum and decay alone; a step with no data should not.
        optimizer.zero_grad(set_to_none=True)
        return "empty", float("nan")
    factor = rank_fragments * world / everywhere
    if contributed > 0 and abs(factor - 1.0) > 1e-12:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(factor)
    if distributed:
        _sync_gradients(model, loss_fn, device, cfg, scaler)
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
        # Identical on every rank after the all-reduce, so this is a shared
        # decision, not a divergence.
        optimizer.zero_grad(set_to_none=True)
        return "nonfinite", float("nan")
    scaler.step(optimizer)
    scaler.update()
    return "stepped", float(norm)


# ---------------------------------------------------------------------------
# One epoch phase
# ---------------------------------------------------------------------------
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
    categories: Tuple[str, ...] = (),
) -> Tuple[Dict[str, float], Dict]:
    """
    One train or validation pass. Returns (metrics, diagnostics).

    `metrics` are FRAGMENT-weighted means over every rank, reduced exactly (as
    sums and counts), so every rank returns the same numbers and a scheduler
    stepping on them stays in lockstep. `diagnostics["by_category"]` holds the
    mean rotation error per category when `categories` is given.
    """
    model.train(train)
    raw_model = model.module if isinstance(model, DDP) else model
    # Validation never goes through the DDP wrapper. It needs no gradient sync,
    # and a wrapper with buffers (norm='batch') broadcasts them on EVERY forward
    # -- which deadlocks as soon as one rank's validation shard is a batch
    # shorter than another's.
    net = model if train else raw_model
    phase = "train" if train else "val"

    sums = {k: 0.0 for k in _LOSS_KEYS}
    weight = 0.0
    protocol = {k: 0.0 for k in _PROTOCOL_SUMS}
    anchored = absolute = 0.0          # fragments behind each protocol mean
    target = getattr(loss_fn, "rotation_target", "absolute")
    cat_sum = {c: 0.0 for c in categories}
    cat_n = {c: 0.0 for c in categories}
    counts = {k: 0.0 for k in ("micro", "fp32", "nonfinite", "oom", "recovered",
                               "steps", "empty_steps", "nonfinite_steps")}
    norms: List[float] = []
    data_seconds = compute_seconds = 0.0
    max_nodes = max_edges = 0
    nan_scenes: Dict[str, int] = {}
    data_start = time.perf_counter()

    for batch in loader:
        data_seconds += time.perf_counter() - data_start
        compute_start = time.perf_counter()
        max_nodes = max(max_nodes, batch["target"].num_nodes)
        max_edges = max(max_edges, batch["target"].num_edges)

        micro_batches = _make_micro_batches(batch, cfg.micro_batch_scenes)
        # The fragments each micro-batch's loss averages over -- all but one per
        # scene under the anchor target -- so the step is the exact mean over
        # every scored fragment on every rank.
        fragments = [loss_fragments(m["target"], target) for m in micro_batches]
        rank_fragments = float(sum(fragments))
        contributed = 0.0
        if train:
            optimizer.zero_grad(set_to_none=True)
        make_sync = model.no_sync if (train and isinstance(model, DDP)) else nullcontext

        for micro, nf in zip(micro_batches, fragments):
            result = _run_micro_step(
                net, raw_model, loss_fn, micro, device, cfg, train, scaler, make_sync,
                scale=nf / max(rank_fragments, 1.0),
            )
            counts["micro"] += 1
            counts["recovered"] += float(result.recovered)
            if result.status not in _CONTRIBUTED:
                counts[result.status] += 1
                if result.status == "nonfinite":
                    for name in micro.get("scene_dirs", []):
                        nan_scenes[name] = nan_scenes.get(name, 0) + 1
                continue
            counts["fp32"] += float(result.status == "fp32")
            contributed += nf
            weight += nf
            for key in _LOSS_KEYS:
                sums[key] += float(result.losses[key].detach()) * nf
            anchor_deg = result.losses["_anchor_deg"].float()
            absolute_deg = result.losses["_absolute_deg"].float()
            protocol["anchor_deg"] += float(anchor_deg.sum())
            protocol["absolute_deg"] += float(absolute_deg.sum())
            protocol["tilt"] += float(result.losses["_anchor_tilt"].float().sum())
            protocol["twist"] += float(result.losses["_anchor_twist"].float().sum())
            anchored += float(anchor_deg.numel())
            absolute += float(absolute_deg.numel())
            if categories:
                # The anchor protocol's per-fragment errors, one label each.
                degrees = anchor_deg.cpu().tolist()
                scene_of = result.losses["_anchor_scene"].cpu().tolist()
                names = micro.get("categories") or []
                for deg, s in zip(degrees, scene_of):
                    name = names[s] if s < len(names) else ""
                    if name in cat_sum:
                        cat_sum[name] += deg
                        cat_n[name] += 1

        if train:
            outcome, norm = _finish_step(model, loss_fn, optimizer, scaler, device, cfg,
                                         rank_fragments, contributed)
            counts["steps"] += 1
            if outcome == "empty":
                counts["empty_steps"] += 1
            elif outcome == "nonfinite":
                counts["nonfinite_steps"] += 1
            elif math.isfinite(norm):
                norms.append(norm)

        compute_seconds += time.perf_counter() - compute_start
        bar.update(batch["num_scenes"])
        if weight:
            bar.set_postfix_str(f"{phase} tot={sums['total'] / weight:.3f} "
                                f"rot={sums['rot_deg'] / weight:.1f}deg")
        data_start = time.perf_counter()

    # --- exact reduction over ranks: sums and counts, divided afterwards ----
    packed = {f"sum:{k}": v for k, v in sums.items()}
    packed["weight"] = weight
    for name, value in protocol.items():
        packed[f"protocol:{name}"] = value
    packed["protocol_n:anchor"] = anchored
    packed["protocol_n:absolute"] = absolute
    for name in categories:
        packed[f"cat:{name}"] = cat_sum[name]
        packed[f"catn:{name}"] = cat_n[name]
    for name, value in counts.items():
        packed[f"n:{name}"] = value
    packed["norm_sum"] = float(sum(norms))
    packed["norm_n"] = float(len(norms))
    packed = D.reduce_sums(packed, device)

    total_weight = packed["weight"]
    if total_weight <= 0:
        write(f"!! {phase} epoch {epoch} produced no usable micro-batch on any rank -- "
              f"check steps_per_epoch/val_steps and the [nan]/[oom] lines above.")
        metrics = {k: float("nan") for k in _LOSS_KEYS}
    else:
        metrics = {k: packed[f"sum:{k}"] / total_weight for k in _LOSS_KEYS}
    # The anchor protocol, exact over every rank: these replace the loss-side
    # tilt/twist, so the figures read against chance mean the same thing
    # whichever target the run trains on.
    n_anchor, n_absolute = packed["protocol_n:anchor"], packed["protocol_n:absolute"]
    for name in ("anchor_deg", "tilt", "twist"):
        metrics[name] = (packed[f"protocol:{name}"] / n_anchor) if n_anchor > 0 else float("nan")
    metrics["absolute_deg"] = ((packed["protocol:absolute_deg"] / n_absolute)
                               if n_absolute > 0 else float("nan"))
    by_category = {
        name: (packed[f"cat:{name}"] / packed[f"catn:{name}"]
               if packed[f"catn:{name}"] > 0 else float("nan"))
        for name in categories
    }
    diagnostics = {
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "max_nodes": float(max_nodes),
        "max_edges": float(max_edges),
        "fragments": total_weight,
        "anchor_fragments": n_anchor,
        "grad_norm": (packed["norm_sum"] / packed["norm_n"]) if packed["norm_n"] else float("nan"),
        "nan_scenes": nan_scenes,
        "by_category": by_category,
        "category_counts": {name: packed[f"catn:{name}"] for name in categories},
        **{name: packed[f"n:{name}"] for name in counts},
    }
    return metrics, diagnostics


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
def _wrap_ddp(model, device: torch.device, rank: int, sync_buffers: bool):
    """
    DistributedDataParallel, across torch versions.

    `find_unused_parameters` is OFF on purpose: the loss keeps every parameter
    connected, so no graph traversal is needed and every rank builds identical
    buckets. Buffer syncing is off unless there are buffers to sync
    (VNLayerNorm has none; norm='batch' does). torch 2.13 renamed the switch
    from `broadcast_buffers` to `forward_sync_buffers` and warns on the old
    name; Kaggle's torch predates the new one.
    """
    kwargs = dict(device_ids=[rank] if device.type == "cuda" else None,
                  find_unused_parameters=False)
    try:
        import inspect

        if "forward_sync_buffers" in inspect.signature(DDP.__init__).parameters:
            return DDP(model, forward_sync_buffers=sync_buffers, **kwargs)
    except (TypeError, ValueError):
        pass
    return DDP(model, broadcast_buffers=sync_buffers, **kwargs)


def _fragment_lines(cfg: Config, train_set: BreakingBadDataset,
                    val_set: BreakingBadDataset) -> List[str]:
    """The pieces-per-scene range, and what the limit removed from each split."""
    if cfg.max_fragments <= 0:
        return [f"  fragments     {cfg.min_fragments}+ pieces per scene, no upper limit "
                f"(--max_fragments 20 is the benchmark's 2-20)"]
    train, val = train_set.fragment_limit, val_set.fragment_limit
    return [
        f"  fragments     {cfg.min_fragments}-{cfg.max_fragments} pieces per scene "
        f"(counted in {train.seconds + val.seconds:.1f} s)",
        f"                train keeps {train.describe()}",
        f"                val   keeps {val.describe()}",
    ]


def _data_banner(cfg: Config, train_set: BreakingBadDataset, val_set: BreakingBadDataset,
                 world_size: int) -> List[str]:
    """What the run is trained and judged on, as numbers rather than flags."""
    lines = [
        f"data: split_source={cfg.split_source} split_by={cfg.split_by}"
        + (f" (pool={cfg.fracture_pool})" if cfg.split_by == "fracture" else "")
        + f" | subsets={cfg.data_subsets or 'ALL'} | normalize={cfg.normalize_mode} "
        f"| input={cfg.input_source}",
    ]
    lines += describe(train_set.objects, train_set.weights, cfg.balance, cfg.balance_temperature)
    lines += _fragment_lines(cfg, train_set, val_set)
    if cfg.val_fixed:
        lines.append(f"  val           FIXED: {len(val_set)} scenes from {val_set.num_objects} "
                     f"objects, same rotations every epoch, sharded over {world_size} rank(s)")
    else:
        lines.append(f"  val           random: {cfg.val_steps * cfg.batch_size} fresh scenes per "
                     f"rank per epoch, from {val_set.num_objects} objects")
    if cfg.split_by == "fracture":
        report = fracture_split_report(train_set.objects, cfg.val_frac, cfg.test_frac,
                                       cfg.split_seed)
        lines.append("  !! split_by=fracture: validation holds out BREAK PATTERNS of objects "
                     "the model trains on. An easier question than the benchmark's; never "
                     "report it as the object-split number.")
        if report["objects_train_only"]:
            lines.append(f"     {report['objects_train_only']} object(s) have fewer than 3 "
                         f"patterns and are train-only.")
        if cfg.fracture_pool == "all":
            lines.append("     fracture_pool=all: the held-out SHAPES are consumed too -- "
                         "diagnostic only.")
    return lines


def _verdict(tilt: float, twist: float) -> str:
    if tilt < 25 and twist > 60:
        return ("  <- axis learned, azimuth not yet (the axis-only landmark -- not a hard "
                "floor: a fragment's fracture boundary is unique even on a surface of "
                "revolution)")
    if tilt > 60:
        return "  <- axis not learned either: an optimisation problem, not a symmetry one"
    return ""


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

    # Peek at the checkpoint FIRST. Its stored config decides the architecture,
    # the schedule AND THE DATA. Nothing may be constructed before this:
    # building the dataloaders first once meant they used the un-adopted config
    # while everything downstream used the adopted one.
    _probe = CheckpointManager(cfg.checkpoint_dir, cfg.save_every, None, cfg.tag)
    _resume_path = (_probe.locate(cfg.resume)
                    if cfg.resume not in ("", "none", "None") else None)
    adopt_checkpoint_config(cfg, _resume_path, is_main)
    if is_main:
        # Written HERE, after adoption: this is the configuration the run
        # actually uses. Written before it (as the entry point used to), a
        # resumed run's config.yaml showed the command line's defaults --
        # split_source=hash, say -- while training ran on the official split.
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        cfg.save_yaml(str(Path(cfg.checkpoint_dir) / "config.yaml"))

    train_loader, val_loader, train_set, val_set = build_dataloaders(cfg, rank, world_size)
    categories = tuple(val_set.categories)
    model = build_model(cfg, device)

    if is_main:
        write("chance level: geodesic 126.47 deg. A matched-budget scaling run of the "
              "previous version reached 30.9 deg on 8 objects, so there is no known "
              "structural floor above that.")
        write(f"VN-GAT v5 | {model.num_parameters():,} parameters | hidden={cfg.hidden_channels} "
              f"layers={cfg.num_layers} heads={cfg.heads}x{cfg.head_dim} slots={cfg.num_vn_slots} "
              f"norm={cfg.norm}")
        for line in _data_banner(cfg, train_set, val_set, world_size):
            write(line)
        write(f"world_size={world_size} | batch={cfg.batch_size} per rank "
              f"(micro={cfg.micro_batch_scenes}) | amp={cfg.amp} | best.pt on val "
              f"{cfg.checkpoint_monitor}")
        write(f"rotation target: {cfg.rotation_target} -- "
              + ("each fragment relative to its scene's largest, which is set to its "
                 "true pose" if cfg.rotation_target == "anchor" else
                 "each fragment in its object's stored frame")
              + ". Metrics use the anchor protocol either way (anchor_deg), with "
                "the stored-frame error beside it (absolute_deg).")
        if cfg.amp:
            write("!! amp=True: a controlled comparison found half precision both unstable "
                  "(non-finite losses on 6 of 8 objects from epoch 66) and WORSE than fp32 "
                  "(54.3 vs 43.2 deg). Use --amp false unless you have a specific reason.")

    if distributed:
        model = _wrap_ddp(model, device, rank, sync_buffers=(cfg.norm == "batch"))

    loss_fn = CompositeLoss(
        w_rot=cfg.w_rot, w_pos=cfg.w_pos, w_node=cfg.w_node,
        w_face=cfg.w_face, w_emb_v=cfg.w_emb_v, w_emb_e=cfg.w_emb_e,
        emb_pull_margin=cfg.emb_pull_margin, emb_push_margin=cfg.emb_push_margin,
        symmetry_axis=cfg.symmetry_axis, rotation_target=cfg.rotation_target,
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
        stored_cfg = state.get("config") or {}
        stored_monitor = stored_cfg.get("checkpoint_monitor", cfg.checkpoint_monitor)
        stored_target = stored_cfg.get("rotation_target", LEGACY_ROTATION_TARGET)
        if stored_monitor != cfg.checkpoint_monitor:
            # best_val was measured in a different quantity; comparing the new
            # one against it would keep or discard best.pt for no reason.
            manager.best_val = float("inf")
            if is_main:
                write(f"  [ckpt] checkpoint_monitor {stored_monitor} -> {cfg.checkpoint_monitor}: "
                      f"the best-so-far is reset (the old value is a different quantity)")
        elif _monitor_changed_meaning(cfg.checkpoint_monitor, stored_target,
                                      cfg.rotation_target, "rotation_target" in stored_cfg):
            # Same name, different quantity: the loss terms follow the target,
            # and a checkpoint from before the anchor protocol measured tilt and
            # twist without it.
            manager.best_val = float("inf")
            if is_main:
                write(f"  [ckpt] val {cfg.checkpoint_monitor} is not measured the way the "
                      f"checkpoint's best was (rotation target {stored_target} -> "
                      f"{cfg.rotation_target}); the best-so-far is reset")
        if _fragment_limit_changed(stored_cfg, cfg):
            # A different limit is a different validation set: the best-so-far
            # was measured on other scenes, so comparing against it keeps or
            # discards best.pt for no reason.
            manager.best_val = float("inf")
            if is_main:
                was = int(stored_cfg.get("max_fragments") or 0)
                write(f"  [ckpt] max_fragments {was or 'none'} -> {cfg.max_fragments or 'none'}: "
                      f"validation is a different set of scenes, so the best-so-far is reset")
        history = History.from_dict(state.get("history") or {})
        history.truncate_to(start_epoch)
        # The checkpoint's RNG state is RANK 0's, and every rank just restored
        # it -- after which both ranks (and their data workers, whose seeds
        # derive from it) drew IDENTICAL scenes for the rest of the run, halving
        # the effective batch without a trace. Re-derive distinct per-rank
        # streams, deterministic in (seed, resume epoch).
        seed_everything(cfg.seed + 7919 * start_epoch, rank)
        if is_main:
            write(f"  [ckpt] resumed from {resume_path} at epoch {start_epoch} "
                  f"(best val {cfg.checkpoint_monitor} {manager.best_val:.4f})")
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
            # scheduler's first step.
            group["lr"] = cfg.lr
            group["initial_lr"] = cfg.lr
        scheduler, scheduler_needs_metric = build_scheduler(cfg, optimizer, span=remaining)
        if is_main:
            actual = optimizer.param_groups[0]["lr"]
            write(f"  [lr  ] schedule restarted at {actual:.2e}, annealing over the "
                  f"{remaining} remaining epoch(s) to {cfg.lr_min:.2e}")
            if abs(actual - cfg.lr) > 1e-12:
                write(f"!! requested --lr {cfg.lr:.2e} but the optimiser holds "
                      f"{actual:.2e}; the schedule is NOT what you asked for")

    if is_main:
        write(table_header())

    val_scenes = len(val_loader.sampler) if cfg.val_fixed else cfg.val_steps * cfg.batch_size
    total_steps = cfg.steps_per_epoch * cfg.batch_size + val_scenes
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
            train=False, epoch=epoch, bar=bar, categories=categories,
        )

        # Both metric dicts are already reduced over ranks (identical on every
        # rank), so the plateau scheduler steps identically everywhere.
        if scheduler is not None and epoch >= cfg.lr_warmup_epochs:
            if scheduler_needs_metric:
                scheduler.step(val_metrics.get(cfg.lr_monitor, val_metrics["total"]))
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            write(table_row(epoch, "train", train_metrics,
                            train_diag["data_seconds"], train_diag["compute_seconds"], current_lr))
            write(table_row(epoch, "val", val_metrics,
                            val_diag["data_seconds"], val_diag["compute_seconds"], current_lr))
            tilt_v, twist_v = val_metrics.get("tilt"), val_metrics.get("twist")
            anchor_v = val_metrics.get("anchor_deg", float("nan"))
            if math.isfinite(anchor_v):
                write(f"  [val ] anchor {anchor_v:6.2f} deg  absolute "
                      f"{val_metrics.get('absolute_deg', float('nan')):6.2f} deg  "
                      f"(chance 126.47; anchor = each scene's largest fragment "
                      f"set to its true pose)")
            if tilt_v is not None and math.isfinite(tilt_v):
                write(f"  [val ] tilt {tilt_v:6.2f}  twist {twist_v:6.2f}"
                      f"  (chance 90/90){_verdict(tilt_v, twist_v)}")
            by_cat = {k: v for k, v in val_diag["by_category"].items() if math.isfinite(v)}
            if len(by_cat) > 1:
                ordered = sorted(by_cat.items(), key=lambda kv: kv[1])
                macro = sum(by_cat.values()) / len(by_cat)
                write("  [val ] by category (deg): "
                      + "  ".join(f"{k} {v:.1f}" for k, v in ordered)
                      + f"  | category mean {macro:.1f}")
            if train_diag["fp32"]:
                write(f"  [amp] {int(train_diag['fp32'])} micro-step(s) recomputed in "
                      f"float32 after a half-precision overflow (data kept)")
            if train_diag["oom"] or train_diag["recovered"]:
                write(f"  [oom] {int(train_diag['recovered'])} scene(s) needed gradient "
                      f"checkpointing; {int(train_diag['oom'])} did not fit and were skipped")
            if train_diag["empty_steps"] or train_diag["nonfinite_steps"]:
                write(f"  [step] {int(train_diag['empty_steps'])} step(s) had no usable data; "
                      f"{int(train_diag['nonfinite_steps'])} had a non-finite gradient -- "
                      f"both skipped on every rank")
            if train_diag["nonfinite"]:
                skipped = int(train_diag["nonfinite"])
                attempted = max(int(train_diag["micro"]), 1)
                write(f"  [nan] skipped {skipped} of {attempted} micro-step(s) this epoch "
                      f"({100.0 * skipped / attempted:.1f}%) with non-finite losses or gradients")
                # Name the repeat offenders. Dropping a step is safe for the
                # optimiser but NOT neutral for the data: if the same scenes
                # fail every epoch they are effectively excluded from training.
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
            if categories:
                history.append("val_category", val_diag["by_category"])
            history.append_meta(
                lr=current_lr,
                grad_norm=train_diag["grad_norm"],
                train_data_seconds=train_diag["data_seconds"],
                train_compute_seconds=train_diag["compute_seconds"],
                max_nodes=train_diag["max_nodes"],
                oom_skips=train_diag["oom"],
                oom_recoveries=train_diag["recovered"],
                nan_skips=train_diag["nonfinite"],
                val_fragments=val_diag["fragments"],
                epoch_seconds=time.perf_counter() - wall_start,
            )

        # Weights themselves, not just the reported loss. Once a parameter is
        # NaN every later forward is NaN and the run produces nothing.
        corrupted = _first_non_finite_parameter(model)
        if corrupted is not None and is_main:
            write(f"!! [nan] parameter '{corrupted}' is non-finite -- the model is unrecoverable.")
            write(f"   Stopping at epoch {epoch}. The last good checkpoint is still in "
                  f"{cfg.checkpoint_dir}; resume from it with a lower --lr.")

        elapsed = time.perf_counter() - wall_start
        out_of_time = elapsed > budget_seconds
        last_epoch = epoch + 1 >= cfg.epochs or corrupted is not None
        # Vote BEFORE the checkpoint/exit decision so every rank leaves the
        # loop together. A rank that decided alone would strand the others.
        stop_now = not D.all_ranks_agree(not (out_of_time or last_epoch), device)

        if is_main:
            monitored_train = train_metrics.get(cfg.checkpoint_monitor, float("nan"))
            monitored_val = val_metrics.get(cfg.checkpoint_monitor, float("nan"))
            rolling = manager.should_save(epoch, cfg.epochs) or stop_now
            # best.pt is judged EVERY epoch, not only on save epochs -- with
            # save_every=10 the best epoch was otherwise usually never saved.
            improved = math.isfinite(monitored_val) and monitored_val < manager.best_val
            if (rolling or improved) and corrupted is None:
                wrote = manager.save(
                    model, optimizer, scheduler, scaler, epoch,
                    monitored_train, monitored_val,
                    history.to_dict(), cfg.to_dict(),
                    force=stop_now, rolling=rolling,
                )
                tags = [k for k, v in wrote.items() if v]
                write(f"  [ckpt] epoch {epoch}: wrote {', '.join(tags)}"
                      + ("" if wrote["rolling"] or not rolling
                         else "  (checkpoint.pt held: older better on both)"))
            else:
                manager.write_history(history.to_dict())

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
