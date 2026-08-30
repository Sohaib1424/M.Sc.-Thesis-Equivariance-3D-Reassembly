"""
Training: config, dataset, loops, metrics, checkpoints, schedule.

One module rather than five. The pipeline is already a lot of moving parts, and
splitting an engine across files mostly buys import diagrams.

    from reassembly.training import Config, train
    train(Config(root="/kaggle/input/breaking-bad", epochs=40))

or from the command line::

    python -m scripts.train --root /kaggle/input/breaking-bad --epochs 40

What this module refuses to do quietly
--------------------------------------
A wrong model still runs. Every guard here exists because the corresponding
failure produces finite numbers and a descending curve rather than an error:

* **Chance is printed in the banner**, so an epoch sitting at 126.5 deg is
  recognisable as "learned nothing" rather than "needs more epochs".
* **Loss at initialisation is checked** against its reference values before the
  first step, and the run refuses to start if a term is far off -- that means
  the inputs are not what the loss assumes.
* **Skipped samples are counted and named**, because a sample that fails every
  epoch is silently removed from the dataset.
* **The checkpoint that is always current is separate from the best-so-far.**
  A policy-gated file can freeze while training continues, and a resume then
  discards the gap without saying so.
* **The learning rate is logged every epoch.** A schedule decayed to nothing
  looks exactly like a model that stopped learning.
* **The schedule is a pure function of the step count**, not a
  ``CosineAnnealingLR`` object. That scheduler is periodic -- past ``T_max`` the
  rate climbs back up -- and its ``T_max`` lives in its own state dict, so
  changing an ``--epochs`` flag on resume may not change the schedule at all.

Precision
---------
fp32 by default. AMP is available and off, deliberately: attention scores are
inner products of unnormalised vector features, and in fp16 a product overflows
to ``inf`` once both operands exceed ~256, after which the max-subtracting
softmax computes ``inf - inf = NaN``. The stability trick manufactures the NaN.
This model is also scatter/gather-bound rather than matmul-bound, so tensor
cores have little to offer. Measure before trusting it.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ==========================================================================
# Config
# ==========================================================================


@dataclass
class Config:
    """
    Everything that decides a run. Serialised into every checkpoint, so a
    result can always be traced back to the settings that produced it.
    """

    # -- data -------------------------------------------------------------
    root: str = "data"
    subsets: Optional[Sequence[str]] = None
    official_subset: str = "everyday"
    modes_per_scene: Optional[int] = 8
    """
    Fracture modes sampled per object per epoch. Breaking Bad ships ~80 modes
    for each of 407 training objects; taking all of them is a 32k-sample epoch
    in which the same shape appears 80 times. 8 keeps an epoch to ~3k samples
    and still shows every object under many break patterns over a run.
    """
    label_method: str = "dihedral"
    """
    ``dihedral`` or ``coincidence``. Dihedral over-labels by ~3.9x but is what
    exists at inference, so it keeps train and test consistent. Coincidence is
    exact and training-time only -- use it to *measure* the gap, not to report
    a number as if it were achievable.
    """
    sharp_threshold: float = 0.9
    min_fracture_faces: int = 1
    tokens_per_scene: int = 2048
    token_mode: str = "sample"
    token_metric: str = "geodesic"
    normalize_mode: str = "scene"
    supervise_embedding: bool = True
    """Compute coincidence clusters for the embedding loss. Costs a KD-tree
    pass per sample; without it that loss term is absent."""
    # No translation setting, and that is deliberate rather than an omission.
    # `build_scene` applies rotation only, then centres every fragment at its
    # own centroid -- which removes any translation that had been applied. A
    # `translation_std` here would be a knob that changes nothing, and stage one
    # predicts rotation alone. Translation is solved geometrically in stage two,
    # in the world frame, from the centroids `Normalized` keeps.

    # -- model ------------------------------------------------------------
    channels: int = 64
    heads: int = 4
    head_dim: int = 8
    embedding_dim: int = 32
    negative_slope: float = 0.2
    checkpoint_cross: bool = True
    """
    Recompute the cross-attention pair gathers in the backward pass instead of
    storing them. Measured as bytes per pair: 1109 -> 86, which on 2048 tokens
    over 6 fragments (3.5M pairs) is 3.6 GB -> 0.3 GB per layer, for one extra
    forward of a cheap indexing op. On by default because the trade is that
    lopsided; outputs and gradients are bitwise identical either way.
    """
    checkpoint_intra: bool = False
    """
    The same for the intra-fragment layers. Off by default: here recomputation
    re-runs the projections themselves, which is real work, so it is worth it
    only when memory is the binding constraint. Turn it on if preflight cannot
    fit `batch_size=1`.
    """
    schedule: Sequence[str] = ("intra", "intra", "cross", "intra", "cross", "intra")
    """Four intra-fragment layers with two cross layers interleaved. A cross
    layer updates only token vertices, so each is followed by propagation --
    without that the rest of the fragment never hears about its neighbours."""

    # -- loss weights (all 1.0 and untuned, on purpose) --------------------
    w_rotation: float = 1.0
    w_position: float = 1.0
    w_normal: float = 1.0
    w_face: float = 1.0
    w_embedding: float = 1.0

    # -- optimisation -----------------------------------------------------
    epochs: int = 40
    batch_size: int = 2
    """
    Scenes per step *per device*. A scene is a whole graph, so this is not
    comparable to an image batch size.

    2 rather than 4 because that is what was **measured** to fit on the target
    card: on a 15.6 GB T4, 4 copies of the largest of twelve sampled Breaking
    Bad scenes ran out of memory and 2 peaked at 11.96 GB (76%). With
    `accumulate=2` the effective batch is unchanged, so this costs a little
    speed and nothing else.
    """
    lr: float = 1e-3
    min_lr_fraction: float = 0.02
    warmup_fraction: float = 0.03
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    accumulate: int = 1
    """
    Optimizer steps every `accumulate` batches.

    This was briefly 2, to "keep the effective batch the same as the old
    `batch_size=4` default". That reasoning was wrong twice over. Accumulation
    exists to *recover* an effective batch when memory forces `batch_size` down
    from a justified value -- and 4 was never justified, it was a guess, so
    there was nothing to recover. It also halves the number of optimizer steps
    for identical wall-clock, which for a 635k-parameter model on a fixed
    session budget is a real cost: 40 epochs is ~65k steps at 1 and ~32k at 2.

    At 1 an optimizer step still sees `batch_size * devices` = 4 scenes, and the
    losses average over *fragments*, so that is around 24 fragments per step,
    not 4 samples. Raise this if the loss curve turns out to be gradient-noise
    limited -- which is something to measure, not assume. The LR schedule is
    expressed in forward passes, so changing it does not shift warmup.
    """
    amp: bool = False

    # -- runtime ----------------------------------------------------------
    out_dir: str = "runs/vgat"
    workers: int = 2
    seed: int = 0
    devices: int = -1
    """GPUs to use. ``-1`` means all visible (2 on Kaggle), ``1`` forces a
    single device, ``0`` forces CPU."""
    max_hours: float = 11.0
    """Kaggle cuts a session at 12 hours. Stop cleanly before that with a
    checkpoint written, rather than losing the epoch in progress."""
    resume: bool = True
    resume_from: Optional[str] = None
    """
    Where to *load* a checkpoint from, when that differs from where this run
    *saves*. This is the normal case on Kaggle: a session writes to
    ``/kaggle/working``, and the next session mounts that output read-only under
    ``/kaggle/input/<name>``. Point ``resume_from`` at the input copy and
    ``out_dir`` at working, and the chain continues across sessions.
    """
    checkpoint_every_minutes: float = 30.0
    """
    Mid-epoch checkpoint interval. An epoch over 746 objects can outlast a whole
    session, and a checkpoint written only at epoch boundaries would then never
    be written at all -- twelve hours of compute for nothing. A mid-epoch save
    records the epoch as *incomplete*, so resuming restarts it rather than
    skipping the part that never ran.
    """
    strict_resume: bool = True
    """Refuse to resume onto a config whose architecture differs. Off, a shape
    mismatch surfaces as a cryptic ``load_state_dict`` error, and a *data*
    change surfaces as nothing at all."""
    check_init: bool = True
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None
    """
    Use only this many samples per split -- for smoke tests and small-scale
    experiments. Selected by striding across the sorted item list, not by taking
    the first N: scenes are sorted by path, so a prefix would be the first few
    objects of one category and neither a useful smoke test nor a representative
    subset.
    """

    def __post_init__(self) -> None:
        if self.channels % self.heads:
            raise ValueError(
                f"channels={self.channels} must be divisible by heads={self.heads}"
            )
        if self.label_method not in ("dihedral", "coincidence"):
            raise ValueError(f"label_method must be dihedral or coincidence")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if not self.supervise_embedding and self.w_embedding:
            # Not an error -- it is a legitimate ablation -- but it must not be
            # silent. The embedding-consistency loss is the only supervision the
            # per-vertex embedding gets, so without it that head trains not at
            # all while the rotation loss descends exactly as before. Stage two
            # matches interface points in embedding space, so the result is a
            # rotation model that works and a translation solver with nothing
            # to use.
            print("[config] supervise_embedding=False: the embedding head will "
                  "receive no gradient at all, and stage two's correspondence "
                  "search depends on it. Set w_embedding=0 to acknowledge this.")


# Reference values every metric is read against. Measured by Monte Carlo in
# tests/test_losses.py, not asserted from memory.
CHANCE = {
    "geodesic_deg": 126.48,      # pi/2 + 2/pi
    "euler_rmse_deg": 86.29,     # random prediction
    "euler_identity_deg": 83.14,  # ALWAYS-IDENTITY beats random on this metric
    "axis_only_deg": 89.9,       # axis recovered, rotation about it not
    "normal": 1.0,
    "face": 2.0,
}


# ==========================================================================
# Dataset
# ==========================================================================


class BreakingBadScenes:
    """
    One item is one (object, fracture mode) pair.

    Returns the ``SceneSample`` that :func:`~reassembly.data.features.collate`
    consumes, so all the geometry runs in ``DataLoader`` workers and the main
    process only concatenates.

    ``SceneReader`` caching is the single largest saving on a pass: an object
    has one mesh and ~80 modes, and parsing the ``.obj`` plus the sparse cell
    matrix once instead of once per mode removes most of the I/O. A small
    per-worker cache keeps that benefit even under a shuffled sampler.
    """

    def __init__(self, config: Config, split: str, epoch_seed: int = 0,
                 cache_size: int = 4):
        from .data.paths import filter_by_split, find_scenes, load_official_split

        self.config = config
        self.split = split
        self.epoch_seed = epoch_seed
        self.cache_size = cache_size

        scenes = find_scenes(config.root, config.subsets)
        if not scenes:
            raise FileNotFoundError(
                f"no Breaking Bad scenes under {config.root!r}. Expected "
                "directories holding compressed_mesh.obj and compressed_data.npz."
            )
        official = load_official_split(config.root, split, config.official_subset)
        self.scenes = filter_by_split(scenes, split, official=official)
        if split in ("train", "val") and official is not None:
            _assert_splits_disjoint(scenes, config)
        if not self.scenes:
            raise FileNotFoundError(f"no scenes left in split {split!r}")
        self.official = official is not None

        # (scene index, mode name) pairs, deterministic in order.
        self.items: List[Tuple[int, str]] = []
        for index, scene in enumerate(self.scenes):
            modes = [d.name for d in scene.mode_dirs()]
            if not modes:
                continue
            if config.modes_per_scene is not None and len(modes) > config.modes_per_scene:
                # Deterministic per (object, epoch): every epoch shows a
                # different slice of an object's break patterns, but two runs
                # with the same seed see the same ones.
                picker = random.Random(f"{scene.object_key}/{epoch_seed}")
                modes = sorted(picker.sample(modes, config.modes_per_scene))
            self.items.extend((index, mode) for mode in modes)

        limit = config.limit_train if split == "train" else config.limit_val
        if limit is not None and limit < len(self.items):
            # Evenly spaced, not the first `limit`. Scenes are sorted by path,
            # so a prefix would be the first few objects of the first category
            # alphabetically -- 40 items at 8 modes each is five Bottles. That
            # is a poor smoke test (it exercises one kind of geometry) and a
            # worse small-scale experiment (the subset is not representative,
            # and nothing says so). Striding keeps it deterministic while
            # spanning the whole split.
            keep = np.linspace(0, len(self.items) - 1, limit).astype(int)
            self.items = [self.items[i] for i in dict.fromkeys(keep.tolist())]

        self._readers: Dict[int, object] = {}
        self._order: List[int] = []
        self.failures: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _reader(self, index: int):
        from .data.scene import SceneReader

        if index not in self._readers:
            if len(self._order) >= self.cache_size:
                self._readers.pop(self._order.pop(0), None)
            self._readers[index] = SceneReader(self.scenes[index].path)
            self._order.append(index)
        return self._readers[index]

    def __getitem__(self, i: int):
        from .data.features import build_scene
        from .mesh.correspondence import compute_scene_correspondence
        from .mesh.fracture import fracture_face_mask, fracture_vertex_masks

        config = self.config
        scene_index, mode = self.items[i]
        key = f"{self.scenes[scene_index].object_key}/{mode}"

        result = self._reader(scene_index).load_mode(mode)
        meshes = result.fragments
        if len(meshes) < 2:
            # A single-fragment mode has no cross-fragment structure and no
            # relative pose to learn from. Returned as a named Skipped rather
            # than None, so the loop can tally which items keep failing.
            return Skipped(key, f"{len(meshes)} fragment(s)")

        vertices = [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
        faces = [np.asarray(m.faces) for m in meshes]

        if config.label_method == "coincidence":
            masks = fracture_vertex_masks(meshes)
        else:
            masks = []
            for v, f in zip(vertices, faces):
                labelled = fracture_face_mask(
                    v, f, sharp_threshold=config.sharp_threshold,
                    min_faces=config.min_fracture_faces,
                ).face_mask
                mask = np.zeros(len(v), bool)
                if labelled.any():
                    mask[np.unique(f[labelled])] = True
                masks.append(mask)

        cluster = None
        if config.supervise_embedding:
            # Coincidence clusters are defined in the ASSEMBLED frame. Computed
            # here, before any perturbation -- afterwards nothing coincides and
            # the labels come back empty while looking like a result.
            per_fragment, _ = compute_scene_correspondence(meshes)
            cluster = np.concatenate(per_fragment) if per_fragment else None

        seed = (hash((key, self.epoch_seed)) ^ config.seed) & 0xFFFFFFFF
        return build_scene(
            vertices, faces, masks,
            rng=np.random.default_rng(seed),
            normalize_mode=config.normalize_mode,
            token_mode=config.token_mode,
            token_metric=config.token_metric,
            tokens_per_scene=config.tokens_per_scene,
            max_tokens_per_fragment=None,
            cluster=cluster,
        )


class Skipped(NamedTuple):
    """A sample the dataset could not build, carrying why."""
    key: str
    reason: str


def _assert_splits_disjoint(scenes, config: Config) -> None:
    """
    Refuse to build a dataset whose train and val splits overlap.

    An object in both makes validation partly a memorisation test, and the
    resulting number is *better* than the honest one -- so it never looks like
    an error, it looks like success. Checked here rather than only in preflight
    because a preflight is easy to skip and this must not be skippable.
    """
    from .data.paths import filter_by_split, load_official_split

    keys = {}
    for name in ("train", "val", "test"):
        official = load_official_split(config.root, name, config.official_subset)
        if official is None:
            return
        keys[name] = {s.object_key for s in filter_by_split(scenes, name,
                                                            official=official)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = keys[a] & keys[b]
        if shared:
            listed = ", ".join(sorted(shared)[:5])
            raise ValueError(
                f"{len(shared)} object(s) appear in both the {a} and {b} splits "
                f"-- validation would be measuring memorisation. First few: "
                f"{listed}. This usually means the split files list "
                f"<category>/<object> while matching used the object name "
                f"alone; check reassembly.data.paths.split_key."
            )


def _collate_samples(samples):
    """
    Separate the unusable samples from the real ones, and pass the *names* of
    what was dropped through to the loop.

    Dropping a bad sample is not neutral. If the same items fail every epoch
    they are effectively removed from the dataset, and a per-batch counter
    hides that entirely: a batch of four where one item is unusable still
    collates fine and reports nothing. So the reasons travel with the batch and
    the loop tallies them by name.
    """
    from .data.features import collate

    kept = [s for s in samples if not isinstance(s, Skipped)]
    dropped = [s for s in samples if isinstance(s, Skipped)]
    return (collate(kept) if kept else None), dropped


# ==========================================================================
# Schedule
# ==========================================================================


def learning_rate(step: int, total: int, config: Config) -> float:
    """
    Linear warmup then cosine decay -- a pure function of the step.

    Deliberately not ``CosineAnnealingLR``. That object is *periodic*: past
    ``T_max`` the rate climbs back toward its base, which turns a mis-set epoch
    count into a silent late-training rate increase. Its ``T_max`` also lives
    inside its own state dict, so changing ``--epochs`` on resume may not change
    the schedule. A function of ``(step, total)`` has neither problem, and
    resuming is exact because the step count is what is checkpointed.
    """
    total = max(int(total), 1)
    warmup = max(int(total * config.warmup_fraction), 1)
    floor = config.lr * config.min_lr_fraction
    if step < warmup:
        return config.lr * (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor + 0.5 * (config.lr - floor) * (1.0 + math.cos(math.pi * progress))


# ==========================================================================
# Progress bar
# ==========================================================================


class Progress:
    """
    A dependency-free progress bar that behaves in a notebook.

    ``tqdm`` is not assumed: Kaggle has it, a bare container may not, and a
    training run should not fail on a cosmetic import. When stdout is not a
    terminal -- which is the case for a captured notebook cell -- carriage
    returns produce one enormous line, so it prints a fresh line periodically
    instead.
    """

    def __init__(self, total: int, prefix: str = "", width: int = 28,
                 enabled: bool = True):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = width
        self.enabled = enabled
        self.tty = sys.stdout.isatty()
        self.start = time.time()
        self.n = 0
        self._last = 0.0

    def update(self, n: int = 1, suffix: str = "") -> None:
        self.n += n
        if not self.enabled:
            return
        now = time.time()
        done = self.n >= self.total
        if not done and now - self._last < 0.25:
            return
        self._last = now
        frac = min(self.n / self.total, 1.0)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        line = (f"{self.prefix} [{bar}] {self.n}/{self.total} "
                f"{100 * frac:5.1f}%  {rate:5.2f} it/s  eta {_hms(eta)}  {suffix}")
        if self.tty:
            sys.stdout.write("\r" + line[:200].ljust(120))
        else:
            # One line every 5% in a notebook, so the cell stays readable.
            if done or int(frac * 20) > int((self.n - n) / self.total * 20):
                sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        if self.enabled and self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ==========================================================================
# One pass
# ==========================================================================


def _to_device(batch, device):
    import torch

    moved = {}
    for name, value in batch._asdict().items():
        moved[name] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return type(batch)(**moved)


def _forward(model, batch, criterion, config):
    """
    One forward pass and the full loss breakdown.

    The predicted rotation is applied to the *centred, normalised* input and
    compared against the assembled target, which is the convention derived in
    ``nn/model.py``: ``v_perturbed @ R.T == v_assembled`` exactly when the
    prediction is right.
    """
    import torch

    from .data.features import clustered_vertices
    from .nn.model import apply_rotation

    prediction = model(
        batch.node_features, batch.edge_index, batch.edge_attr,
        batch.vertex_fragment, batch.num_fragments,
        log_scale=batch.log_scale, token_index=batch.token_index,
        token_query=batch.token_query, token_key=batch.token_key,
    )
    R = prediction.rotation
    fragment = batch.vertex_fragment
    edge_fragment = fragment[batch.edge_index[0]]

    positions = apply_rotation(batch.node_features[:, 0, :], R, fragment)
    normals = apply_rotation(batch.node_features[:, 1, :], R, fragment)
    # (E, C, 3) rotated per edge; the loss slices out the two normal channels.
    edge_normals = torch.einsum("eij,ekj->eki", R[edge_fragment], batch.edge_attr)

    embeddings = cluster = None
    clusters = 0
    if batch.num_clusters:
        keep, renumbered, clusters = clustered_vertices(batch.cluster)
        if clusters:
            embeddings, cluster = prediction.vertex_embedding[keep], renumbered

    total, report = criterion(
        R, batch.target_rotation,
        vertices=positions, target_vertices=batch.target_vertices,
        vertex_batch=fragment,
        normals=normals, target_normals=batch.target_normals,
        face_normals=edge_normals, target_face_normals=batch.target_edge_normals,
        edge_batch=edge_fragment,
        embeddings=embeddings, cluster=cluster, num_clusters=clusters,
    )
    return total, report, R


def _metrics(predicted, target) -> Dict[str, float]:
    """Geodesic (primary) and Euler RMSE (GARF comparability), plus accuracy."""
    import torch

    from .nn.losses import euler_rmse, geodesic_angle

    angle = torch.rad2deg(geodesic_angle(predicted, target))
    out = {
        "geodesic_deg": float(angle.mean()),
        "geodesic_median_deg": float(angle.median()),
        "euler_rmse_deg": float(euler_rmse(predicted, target)),
    }
    for threshold in (5.0, 10.0, 30.0):
        out[f"acc@{threshold:g}deg"] = float((angle < threshold).float().mean())
    return out


def run_epoch(model, loader, criterion, config, *, optimizer=None, scaler=None,
              device="cpu", step=0, total_steps=1, label="train",
              show_progress=True, deadline=None, stop_signal=None,
              on_checkpoint=None):
    """
    One pass over ``loader``. Training when ``optimizer`` is given, else eval.

    Returns ``(summary, step, stopped)``. ``summary`` carries every loss term
    separately as well as the total, because the terms have very different
    natural scales and the only way to learn the balance is to watch them move.
    """
    import torch

    training = optimizer is not None
    model.train(training)

    sums: Dict[str, float] = {}
    counts = 0
    fragments = 0
    skipped = 0
    oom = 0
    dropped: Dict[str, str] = {}
    predictions, targets = [], []
    bar = Progress(len(loader), prefix=f"  {label:5s}", enabled=show_progress)
    stopped = False

    interval = max(config.checkpoint_every_minutes, 0.0) * 60.0
    next_save = time.time() + interval if (training and on_checkpoint and interval) else None

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, (batch, unusable) in enumerate(loader):
            for item in unusable:
                dropped[item.key] = item.reason
            if batch is None or batch.num_fragments == 0:
                skipped += 1
                bar.update(1)
                continue
            batch = _to_device(batch, device)

            if training:
                rate = learning_rate(step, total_steps, config)
                for group in optimizer.param_groups:
                    group["lr"] = rate

            try:
                with torch.autocast(device_type=torch.device(device).type,
                                    enabled=bool(scaler)):
                    loss, report, R = _forward(model, batch, criterion, config)
            except torch.cuda.OutOfMemoryError:
                # One pathological scene must not end an eleven-hour session.
                # Breaking Bad's largest fragment is 83,039 vertices, and a
                # preflight that samples a dozen scenes will not have seen it,
                # so a batch that fits everything measured can still be handed
                # something several times larger at hour six.
                #
                # Skipped, counted and named -- not swallowed. If this fires
                # more than a handful of times the batch size is wrong, and a
                # silent skip would hide that while quietly removing the
                # largest objects from training.
                oom += 1
                dropped[f"OOM:batch{index}"] = (
                    f"{batch.node_features.shape[0]:,} vertices, "
                    f"{batch.num_fragments} fragments"
                )
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"\n  [oom] batch {index} "
                      f"({batch.node_features.shape[0]:,} vertices) did not fit; "
                      f"skipped")
                bar.update(1)
                continue

            if not torch.isfinite(loss):
                # Not skipped silently: a non-finite loss means the inputs or
                # the precision are wrong, and hiding it wastes a whole run.
                skipped += 1
                print(f"\n  [warn] non-finite loss at {label} batch {index}; skipped")
                bar.update(1)
                continue

            if training:
                scaled = loss / config.accumulate
                if scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (index + 1) % config.accumulate == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
            else:
                predictions.append(R.detach().float().cpu())
                targets.append(batch.target_rotation.detach().float().cpu())

            weight = int(batch.num_fragments)
            for name, value in report.items():
                sums[name] = sums.get(name, 0.0) + value * weight
            fragments += weight
            counts += 1
            bar.update(1, f"loss {report['total']:.4f}")

            now = time.time()
            if next_save is not None and now > next_save and step % config.accumulate == 0:
                # Mid-epoch insurance. An epoch over the full training set can
                # outlast a session, and a checkpoint written only at epoch
                # boundaries would then never be written at all.
                on_checkpoint(step, completed=False)
                next_save = now + interval
            if stop_signal is not None and stop_signal.triggered:
                stopped = True
                break
            if deadline is not None and now > deadline:
                stopped = True
                print(f"\n  [time] budget reached during {label}; stopping cleanly")
                break
    bar.close()

    summary = {k: v / max(fragments, 1) for k, v in sums.items()}
    summary["batches"] = counts
    summary["fragments"] = fragments
    summary["skipped"] = skipped
    summary["dropped"] = len(dropped)
    summary["oom"] = oom
    summary["dropped_names"] = sorted(dropped)[:20]
    if predictions:
        summary.update(_metrics(torch.cat(predictions), torch.cat(targets)))
    return summary, step, stopped


# ==========================================================================
# Reporting
# ==========================================================================

_TERMS = ("rotation", "position", "normal", "face", "embedding")


def format_losses(summary: Dict[str, float]) -> str:
    """Every term, then the total -- one line, aligned."""
    parts = [f"{name}={summary[name]:.4f}" for name in _TERMS if name in summary]
    parts.append(f"TOTAL={summary.get('total', float('nan')):.4f}")
    return "  ".join(parts)


def format_metrics(summary: Dict[str, float]) -> str:
    if "geodesic_deg" not in summary:
        return ""
    return (f"geo {summary['geodesic_deg']:6.2f}deg "
            f"(median {summary['geodesic_median_deg']:6.2f}, chance "
            f"{CHANCE['geodesic_deg']:.1f})  euler {summary['euler_rmse_deg']:6.2f}deg"
            f"  acc@10 {summary['acc@10deg']:.3f}")


def check_initial_losses(summary: Dict[str, float]) -> List[str]:
    """
    Compare an untrained epoch against the values each term must read at
    chance. A term far from its reference is measuring something other than
    what its name says, and that is worth stopping for -- it will not show up
    later as anything but slow convergence.
    """
    complaints = []
    rotation_deg = summary.get("rotation_degrees")
    if rotation_deg is not None and not 100.0 < rotation_deg < 155.0:
        complaints.append(
            f"rotation starts at {rotation_deg:.1f} deg, expected ~{CHANCE['geodesic_deg']:.0f}"
        )
    if "normal" in summary and not 0.75 < summary["normal"] < 1.3:
        complaints.append(f"normal starts at {summary['normal']:.3f}, expected ~1.0")
    if "face" in summary and not 1.5 < summary["face"] < 2.6:
        complaints.append(f"face starts at {summary['face']:.3f}, expected ~2.0")
    return complaints


# ==========================================================================
# Checkpoints and history
# ==========================================================================


def save_checkpoint(path: Path, model, optimizer, config: Config, epoch: int,
                    step: int, history: List[dict], best: float, scaler=None,
                    completed: bool = True, elapsed: float = 0.0) -> None:
    """
    Everything needed to continue, not just weights.

    Written to a temporary file and moved into place, so a session killed
    mid-write leaves the previous checkpoint intact rather than a truncated one
    -- which matters more than usual here, because Kaggle kills the process.

    ``completed`` says whether ``epoch`` finished. A mid-epoch save records
    ``False`` and resume restarts that epoch instead of skipping to the next;
    skipping would silently drop the part of the epoch that never ran.
    ``elapsed`` is cumulative training seconds across *all* sessions, so a run
    spanning four Kaggle sessions still knows how long it has actually trained.
    """
    import torch

    core = model.module if hasattr(model, "module") else model
    payload = {
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": asdict(config),
        "epoch": epoch,
        "completed": completed,
        "step": step,
        "history": history,
        "best": best,
        "elapsed": elapsed,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "format": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(out_dir: Path, history: List[dict]) -> None:
    """
    The full history, rewritten every epoch.

    Every epoch, not at the end: a 12-hour session cap that fires during epoch
    30 must not cost the first 29 epochs of curve.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    if not history:
        return
    columns: List[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [",".join(columns)]
    for row in history:
        lines.append(",".join("" if row.get(c) is None else str(row.get(c))
                              for c in columns))
    (out_dir / "history.csv").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Resuming across sessions
# --------------------------------------------------------------------------

# Architecture fields: a mismatch makes `load_state_dict` fail, but with a shape
# error that names a tensor rather than the setting that caused it.
_ARCHITECTURE = ("channels", "heads", "head_dim", "embedding_dim", "schedule")
# Data fields: a mismatch produces no error at all. The run simply continues on
# a different problem than the weights were trained for, and the loss curve has
# a step in it that looks like noise.
_DATA = ("label_method", "sharp_threshold", "tokens_per_scene", "token_mode",
         "token_metric", "normalize_mode", "supervise_embedding", "subsets")


def _resume_path(config: Config, default: Path) -> Optional[Path]:
    """
    Where to load from: ``resume_from`` if given, else this run's own ``last.pt``.

    ``resume_from`` may be a directory or a file, because on Kaggle the previous
    session's output is mounted as ``/kaggle/input/<name>/`` and it is easy to
    point at either.
    """
    if config.resume_from:
        candidate = Path(config.resume_from)
        if candidate.is_dir():
            for name in ("last.pt", "best.pt"):
                if (candidate / name).exists():
                    return candidate / name
            return None
        return candidate if candidate.exists() else None
    return default if default.exists() else None


def _check_resume_compatible(config: Config, saved: Dict, announce: bool) -> None:
    """
    Refuse an architecture change, warn loudly about a data change.

    The two fail differently and both fail badly. A different ``channels`` makes
    `load_state_dict` raise a shape error naming a tensor, which takes a while to
    trace back to the flag that caused it. A different ``label_method`` or
    ``tokens_per_scene`` raises *nothing*: training continues on a different
    problem than the weights were trained for.
    """
    if not saved:
        return
    current = asdict(config)
    architecture = [(k, saved.get(k), current.get(k)) for k in _ARCHITECTURE
                    if k in saved and _differs(saved[k], current[k])]
    if architecture:
        detail = "; ".join(f"{k}: checkpoint {a!r} vs config {b!r}"
                           for k, a, b in architecture)
        if config.strict_resume:
            raise ValueError(
                f"cannot resume: the checkpoint was trained with a different "
                f"architecture ({detail}). Either match the config, point "
                f"--out-dir somewhere fresh, or pass --no-strict-resume to "
                f"attempt it anyway."
            )
        if announce:
            print(f"[resume] architecture differs ({detail}) -- load will "
                  f"probably fail with a shape error")

    changed = [(k, saved.get(k), current.get(k)) for k in _DATA
               if k in saved and _differs(saved[k], current[k])]
    if changed and announce:
        print("[resume] the DATA settings changed since this checkpoint. "
              "Training will continue on a different problem than the weights "
              "were trained for, and nothing else will warn you:")
        for name, was, now in changed:
            print(f"           {name}: {was!r} -> {now!r}")

    if saved.get("epochs") not in (None, config.epochs) and announce:
        # The schedule is a function of (step, total_steps), and total_steps
        # depends on `epochs`. Changing it mid-run rescales the whole curve.
        print(f"[resume] epochs changed {saved['epochs']} -> {config.epochs}: "
              f"the learning-rate curve is recomputed, so the rate will jump "
              f"at this step rather than continuing smoothly.")


def _differs(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return tuple(a or ()) != tuple(b or ())
    return a != b


def _restore_rng(state: Dict, rank: int) -> None:
    """
    Put the generators back where they were.

    Saved but never restored is the same as not saved: a resumed run draws a
    different perturbation stream from the one it would have had, so "resume"
    and "run straight through" diverge and neither is reproducible. Per-rank
    offsetting is preserved so two ranks do not draw the same stream.
    """
    import torch

    try:
        if state.get("torch_rng") is not None and rank == 0:
            torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        if state.get("numpy_rng") is not None and rank == 0:
            np.random.set_state(state["numpy_rng"])
        if state.get("python_rng") is not None and rank == 0:
            random.setstate(state["python_rng"])
    except Exception as error:                                # noqa: BLE001
        # Deliberately broad. A checkpoint written by another torch or numpy
        # build can fail here in several unrelated ways -- wrong dtype, wrong
        # length, a type that has no `.cpu()` -- and none of them is worth
        # losing a resume over: the weights and the optimiser are what matter,
        # and the run is still valid with a fresh RNG. It is reported rather
        # than swallowed, because the sample stream then genuinely differs.
        print(f"[resume] could not restore RNG state ({type(error).__name__}: "
              f"{error}); the sample stream will differ from an uninterrupted run")


class StopSignal:
    """
    Catches the signal Kaggle sends when it reclaims the session.

    A SIGTERM arriving mid-epoch otherwise kills the process where it stands and
    the epoch's work is gone. The handler only sets a flag -- saving from inside
    a signal handler risks a half-written file, and under DDP it would desync
    the ranks -- and the batch loop checks it at the next boundary.
    """

    def __init__(self) -> None:
        self.triggered = False
        self.name = ""
        self._previous: Dict = {}

    def install(self) -> "StopSignal":
        import signal

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass          # not the main thread, or unsupported platform
        return self

    def _handle(self, signum, frame) -> None:
        import signal

        self.triggered = True
        self.name = signal.Signals(signum).name
        print(f"\n[signal] {self.name} received -- finishing this batch, then "
              f"checkpointing and stopping cleanly")

    def restore(self) -> None:
        import signal

        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _time_report(history: List[dict], config: Config, elapsed: float,
                 start_epoch: int) -> str:
    """
    How much longer, and how many more sessions.

    The number that actually decides whether a plan is workable, and it cannot
    be guessed before the first epoch is timed.
    """
    times = [h.get("seconds", 0.0) for h in history if h.get("seconds")]
    if not times:
        return ""
    recent = times[-3:]
    per_epoch = sum(recent) / len(recent)
    remaining = max(config.epochs - (history[-1]["epoch"] + 1), 0)
    left = per_epoch * remaining
    sessions = math.ceil(left / (config.max_hours * 3600)) if left > 0 else 0
    return (f"  ~{_hms(per_epoch)}/epoch, {remaining} epoch(s) left "
            f"= ~{_hms(left)}  ({sessions} more session(s) at "
            f"{config.max_hours:g}h)   total trained so far {_hms(elapsed)}")


# ==========================================================================
# Entry point
# ==========================================================================


def build_model(config: Config):
    from .nn.model import ReassemblyNet

    return ReassemblyNet(
        channels=config.channels, heads=config.heads, head_dim=config.head_dim,
        embedding_dim=config.embedding_dim, schedule=tuple(config.schedule),
        negative_slope=config.negative_slope,
        checkpoint_intra=config.checkpoint_intra,
        checkpoint_cross=config.checkpoint_cross,
    )


def build_criterion(config: Config):
    from .nn.losses import ReassemblyLoss

    return ReassemblyLoss(
        rotation=config.w_rotation, position=config.w_position,
        normal=config.w_normal, face=config.w_face, embedding=config.w_embedding,
    )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, config: Config, shuffle: bool, rank: int, world: int,
            epoch: int):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    sampler = None
    if world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=shuffle, drop_last=False)
        sampler.set_epoch(epoch)
    return DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=config.workers, collate_fn=_collate_samples,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=config.workers > 0,
    )


def train(config: Config) -> List[dict]:
    """
    Run training. Spawns one process per GPU when more than one is available.

    Returns the history. On multiple GPUs only rank 0's history is returned;
    the others train and synchronise gradients but do not report.
    """
    import torch

    visible = torch.cuda.device_count()
    requested = visible if config.devices < 0 else min(config.devices, visible)
    if config.devices > 1 and visible == 0:
        # Explicitly asked for several devices on a machine with none. Honoured
        # over gloo rather than silently downgraded -- a run that quietly used
        # one device when told to use two is a benchmark nobody can interpret.
        requested = config.devices
    if requested > 1:
        import __main__
        import torch.multiprocessing as mp

        if not hasattr(__main__, "__file__"):
            # `mp.spawn` re-imports __main__ in each child, and a notebook cell
            # or a stdin heredoc has no file to import -- the child dies with a
            # FileNotFoundError on `<stdin>` that says nothing about the cause.
            # Caught here so the message is actionable instead of cryptic.
            raise RuntimeError(
                "multi-GPU training cannot be launched from a notebook cell or "
                "stdin: torch's process spawner re-imports __main__, which does "
                "not exist there.\n"
                "  Write a small file and run it instead:\n"
                "      %%writefile train_run.py\n"
                "      from reassembly.training import Config, train\n"
                "      if __name__ == '__main__':\n"
                "          train(Config(root=..., devices=2))\n"
                "      # then, in the next cell:  !python train_run.py\n"
                "  Or set devices=1 to train on one GPU from the notebook."
            )
        mp.spawn(_worker, args=(requested, config), nprocs=requested, join=True)
        path = Path(config.out_dir) / "history.json"
        return json.loads(path.read_text()) if path.exists() else []
    return _worker(0, max(requested, 1) if visible else 0, config)


def _worker(rank: int, world: int, config: Config) -> List[dict]:
    import torch

    from torch.nn.parallel import DistributedDataParallel

    cuda = torch.cuda.is_available()
    distributed = world > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29513")
        # gloo when there is no CUDA, so the multi-process path can be
        # exercised on a CPU machine. A DDP bug that only appears with two
        # ranks is otherwise untestable until it wastes a GPU session.
        torch.distributed.init_process_group("nccl" if cuda else "gloo",
                                             rank=rank, world_size=world)
        if cuda:
            torch.cuda.set_device(rank)
        device = f"cuda:{rank}" if cuda else "cpu"
    elif world >= 1 and cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    main = rank == 0
    _seed_everything(config.seed + rank)
    out_dir = Path(config.out_dir)
    if main:
        _check_writable(out_dir)

    model = build_model(config).to(device)
    criterion = build_criterion(config)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.split(":")[0]) if config.amp else None

    if distributed:
        # find_unused_parameters because a batch whose scenes all lack a
        # fracture surface skips the cross-fragment layers entirely, leaving
        # their parameters without gradient. Rare, and a hang six hours in.
        model = DistributedDataParallel(
            model, device_ids=[rank] if cuda else None,
            find_unused_parameters=True,
        )

    history: List[dict] = []
    start_epoch, step, best, elapsed = 0, 0, float("inf"), 0.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    load_path = _resume_path(config, last_path)

    if config.resume and load_path is not None:
        state = torch.load(load_path, map_location=device, weights_only=False)
        _check_resume_compatible(config, state.get("config", {}), main)
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        # An incomplete epoch is restarted, not skipped past.
        start_epoch = state["epoch"] + (1 if state.get("completed", True) else 0)
        step = state["step"]
        history = state.get("history", [])
        best = state.get("best", float("inf"))
        elapsed = float(state.get("elapsed", 0.0))
        _restore_rng(state, rank)
        if main:
            partial = "" if state.get("completed", True) else " (epoch was partial, restarting it)"
            print(f"resumed from {load_path}")
            print(f"  epoch {start_epoch}, step {step}, "
                  f"{_hms(elapsed)} trained so far{partial}")
            if history:
                print(f"  best val geodesic so far: {best:.3f} deg")
        if load_path.resolve() != last_path.resolve():
            # Copy the resume point into *this* run's out_dir straight away.
            # Two failures otherwise break the chain: a session killed during
            # its first epoch leaves out_dir empty, and a session that finds
            # nothing left to do writes nothing at all -- so the next session,
            # resuming from this one's output, starts from scratch and silently
            # discards every hour spent so far.
            save_checkpoint(last_path, model, optimizer, config,
                            state["epoch"], step, history, best, scaler,
                            completed=state.get("completed", True),
                            elapsed=elapsed)
            if main:
                print(f"  carried the resume point into {last_path}")
    elif main and config.resume:
        # Announced, because "starting from scratch" is one line in a long log
        # and costs a whole session when missed.
        looked = load_path or config.resume_from or last_path
        print(f"no checkpoint at {looked} -- starting from scratch")

    train_set = BreakingBadScenes(config, "train", epoch_seed=0)
    val_set = BreakingBadScenes(config, "val", epoch_seed=0)
    per_epoch = max(math.ceil(len(train_set) / (config.batch_size * max(world, 1))), 1)
    total_steps = per_epoch * config.epochs

    if main:
        print(_banner(config, train_set, val_set, parameters, device, world,
                      per_epoch))

    deadline = time.time() + config.max_hours * 3600
    stopped = False
    signal_watch = StopSignal().install() if main else StopSignal()
    session_started = time.time()

    def checkpoint(current_step: int, epoch_index: int, completed: bool):
        """Only rank 0 writes; the others' weights are identical under DDP."""
        if not main:
            return
        save_checkpoint(last_path, model, optimizer, config, epoch_index,
                        current_step, history, best, scaler, completed=completed,
                        elapsed=elapsed + (time.time() - session_started))

    for epoch in range(start_epoch, config.epochs):
        if stopped:
            break
        train_set = BreakingBadScenes(config, "train", epoch_seed=epoch)
        train_loader = _loader(train_set, config, True, rank, world, epoch)
        val_loader = _loader(val_set, config, False, rank, world, epoch)

        if main:
            print(f"\nepoch {epoch + 1}/{config.epochs}"
                  f"   lr {learning_rate(step, total_steps, config):.2e}")

        began = time.time()
        train_summary, step, stopped = run_epoch(
            model, train_loader, criterion, config, optimizer=optimizer,
            scaler=scaler, device=device, step=step, total_steps=total_steps,
            label="train", show_progress=main, deadline=deadline,
            stop_signal=signal_watch,
            on_checkpoint=lambda s_, completed: checkpoint(s_, epoch, completed),
        )
        # Validation gets its own margin rather than `None`. A stop that fires
        # at the end of training must not then spend an unbounded amount of the
        # remaining time validating and get killed before it can checkpoint.
        val_summary, _, _ = run_epoch(
            model, val_loader, criterion, config, device=device,
            label="val", show_progress=main,
            deadline=time.time() + max(config.max_hours * 3600 * 0.05, 300),
        )

        if epoch == start_epoch and main and config.check_init and not history:
            for complaint in check_initial_losses(train_summary):
                print(f"  [check] {complaint}")

        if not main:
            continue

        print(f"  train  {format_losses(train_summary)}")
        print(f"  val    {format_losses(val_summary)}")
        print(f"  val    {format_metrics(val_summary)}")
        _report_dropped(train_summary, val_summary)

        train_summary.pop("dropped_names", None)
        val_summary.pop("dropped_names", None)
        row = {"epoch": epoch, "step": step,
               # An epoch cut short by the time budget is recorded, because its
               # metrics are real measurements -- but flagged, because the next
               # session re-runs that epoch and the same number then appears
               # twice. Plot `partial == 0` for a clean curve.
               "partial": int(stopped),
               "lr": learning_rate(step, total_steps, config),
               "seconds": round(time.time() - began, 1)}
        row.update({f"train_{k}": v for k, v in train_summary.items()})
        row.update({f"val_{k}": v for k, v in val_summary.items()})
        history.append(row)
        write_history(out_dir, history)

        # The always-current checkpoint is written unconditionally and the
        # best-so-far separately. A policy-gated file alone can freeze while
        # training continues, and a resume then silently discards the gap.
        checkpoint(step, epoch, completed=not stopped)
        score = val_summary.get("geodesic_deg", val_summary["total"])
        if score < best:
            best = score
            save_checkpoint(best_path, model, optimizer, config, epoch, step,
                            history, best, scaler, completed=not stopped,
                            elapsed=elapsed + (time.time() - session_started))
            print(f"  new best: {best:.3f} deg -> {best_path.name}")
        report = _time_report(history, config, elapsed + (time.time() - session_started),
                              start_epoch)
        if report:
            print(report)

    if main:
        signal_watch.restore()
        if signal_watch.triggered:
            print(f"\n[signal] stopped on {signal_watch.name}. Progress is in "
                  f"{last_path} -- rerun with the same --out-dir to continue.")
        elif stopped:
            print(f"\n[time] session budget reached. Progress is in {last_path}.")
            print(_resume_recipe(config, out_dir))
    if main and history:
        _final_report(history)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return history


def _banner(config, train_set, val_set, parameters, device, world, per_epoch) -> str:
    lines = [
        "=" * 74,
        "V-GAT reassembly training",
        "=" * 74,
        f"  device        {device}" + (f"  x{world} (DDP)" if world > 1 else ""),
        f"  parameters    {parameters:,}",
        f"  channels      {config.channels}  heads {config.heads}  "
        f"schedule {'+'.join(config.schedule)}",
        f"  objects       {len(train_set.scenes)} train / {len(val_set.scenes)} val"
        + ("  (official split)" if train_set.official else "  (hashed split)"),
        f"  samples       {len(train_set)} train / {len(val_set)} val"
        f"   {per_epoch} steps/epoch",
        f"  labels        {config.label_method}"
        + (f" @ {config.sharp_threshold}" if config.label_method == "dihedral" else ""),
        f"  tokens        {config.tokens_per_scene}/scene, {config.token_mode}"
        f" by {config.token_metric} distance",
        f"  precision     {'AMP' if config.amp else 'fp32'}",
        f"  lr            {config.lr:.2e} -> {config.lr * config.min_lr_fraction:.2e}"
        f"  (warmup {config.warmup_fraction:.0%}, cosine)",
        "-" * 74,
        "  read every number against chance, not against zero:",
        f"    geodesic      {CHANCE['geodesic_deg']:.2f} deg   <- a model here has learned nothing",
        f"    euler RMSE    {CHANCE['euler_rmse_deg']:.2f} deg random, "
        f"{CHANCE['euler_identity_deg']:.2f} deg if it collapses to identity",
        f"    axis only     {CHANCE['axis_only_deg']:.1f} deg   <- axis right, rotation about it not",
        f"    normal/face   {CHANCE['normal']:.1f} / {CHANCE['face']:.1f} at initialisation",
        "=" * 74,
    ]
    return "\n".join(lines)


def _final_report(history: List[dict]) -> None:
    """
    Refuse to imply a conclusion the run did not earn.

    Two preconditions are checked because both invalidate a result while
    leaving it looking perfectly reportable: a run still descending at its
    cutoff gives a lower bound on progress rather than a ceiling, and a
    validation error at chance is not a small number that needs more epochs.
    """
    last = history[-1]
    best = min(h.get("val_geodesic_deg", float("inf")) for h in history)
    print("\n" + "=" * 74)
    print(f"finished {len(history)} epoch(s).  best val geodesic {best:.3f} deg"
          f"  (chance {CHANCE['geodesic_deg']:.1f})")
    if best > CHANCE["geodesic_deg"] - 5:
        print("  [verdict] at chance -- the model has not learned rotation. Do not")
        print("            report this as a small error; check the data and the")
        print("            loss conventions before tuning anything.")
    elif abs(best - CHANCE["axis_only_deg"]) < 5:
        print("  [verdict] parked at the axis-only floor. That is the signature of")
        print("            a model recovering a fragment's axis but not its rotation")
        print("            about it -- a structural result, not a tuning failure.")
    tail = [h.get("val_geodesic_deg") for h in history[-max(len(history) // 4, 2):]]
    tail = [t for t in tail if t is not None]
    if len(tail) >= 2 and tail[0] - tail[-1] > 0.5:
        print("  [verdict] still descending at the last epoch -- this is a lower")
        print("            bound on progress, not a converged result.")
    print(f"  last epoch: {last.get('seconds', 0):.0f}s"
          f"   lr {last.get('lr', 0):.2e}")
    print("=" * 74)


def evaluate(config: Config, checkpoint: str = "best.pt",
             split: str = "test") -> Dict[str, float]:
    """
    Score a saved checkpoint on a held-out split.

    Separate from training and single-device on purpose: an evaluation that
    shards across GPUs has to gather predictions to be correct, and getting
    that subtly wrong produces a plausible number.
    """
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = Path(config.out_dir) / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])

    dataset = BreakingBadScenes(config, split, epoch_seed=0)
    loader = _loader(dataset, config, False, 0, 1, 0)
    summary, _, _ = run_epoch(model, loader, build_criterion(config), config,
                              device=device, label=split)

    print(f"\n{split} ({len(dataset)} samples, checkpoint from epoch "
          f"{state['epoch'] + 1})")
    print(f"  {format_losses(summary)}")
    print(f"  {format_metrics(summary)}")
    print(f"  geodesic is the primary number. Euler RMSE is for comparability "
          f"with GARF's tables\n  and rewards collapsing to the identity "
          f"({CHANCE['euler_identity_deg']:.1f} deg), so never quote it alone.")
    out = Path(config.out_dir) / f"{split}_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


def _report_dropped(train_summary: Dict, val_summary: Dict) -> None:
    """
    Name what was dropped, not just how much.

    A sample that fails every epoch has been removed from the dataset. Counting
    batches hides it -- a batch of four with one unusable item still collates
    and reports nothing -- and even a count hides *which* items, which is what
    turns "3 dropped" into a diagnosable fact.
    """
    for label, summary in (("train", train_summary), ("val", val_summary)):
        names = summary.get("dropped_names") or []
        if summary.get("dropped"):
            shown = ", ".join(names[:3])
            more = f" (+{summary['dropped'] - len(names[:3])} more)" if summary["dropped"] > 3 else ""
            print(f"  {label}: {summary['dropped']} sample(s) unusable: {shown}{more}")
        if summary.get("skipped"):
            print(f"  {label}: {summary['skipped']} batch(es) skipped entirely")
        if summary.get("oom"):
            print(f"  {label}: {summary['oom']} batch(es) hit OOM and were "
                  f"skipped. A handful is survivable; more than that means "
                  f"--batch-size is too high and the largest objects are being "
                  f"dropped from training.")


def _resume_recipe(config: Config, out_dir: Path) -> str:
    """
    The exact next step, printed at the moment it becomes relevant.

    On Kaggle the checkpoint does not stay where it was written: the session's
    ``/kaggle/working`` becomes a *dataset*, mounted read-only under
    ``/kaggle/input`` in the next session. So "just rerun it" is wrong advice
    there, and getting it wrong costs a whole session to discover.
    """
    kaggle = str(out_dir).startswith("/kaggle")
    if not kaggle:
        return (f"  To continue: rerun the same command. It will load "
                f"{out_dir / 'last.pt'} and pick up where it stopped.")
    return (
        "  To continue in the next Kaggle session:\n"
        "    1. Save this notebook's version so /kaggle/working becomes an output.\n"
        "    2. In the new session, add that output as an input dataset.\n"
        "    3. Point the run at it:\n"
        "         Config(..., resume_from='/kaggle/input/<the-output-name>',\n"
        "                     out_dir='/kaggle/working/vgat')\n"
        "  out_dir must stay under /kaggle/working -- /kaggle/input is read-only,\n"
        "  and a run that cannot write its checkpoint discovers that 11 hours in."
    )


def _check_writable(out_dir: Path) -> None:
    """
    Prove the checkpoint directory is writable *now*, not after eleven hours.

    On Kaggle the obvious mistake is pointing ``out_dir`` at ``/kaggle/input``,
    which is mounted read-only. Nothing complains until the first checkpoint.

    The probe writes a real file rather than checking permission bits, because
    Kaggle notebooks run as root and root ignores the bits -- what actually
    stops the write there is the read-only *mount*, which returns EROFS to
    everyone.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as error:
        raise RuntimeError(
            f"cannot write checkpoints to {out_dir} ({error}). On Kaggle, "
            f"out_dir must be under /kaggle/working -- /kaggle/input is "
            f"read-only. Use resume_from to *load* from an input dataset."
        ) from error


# ==========================================================================
# Preflight
# ==========================================================================


def preflight(config: Config, samples: int = 12, timed_batches: int = 6) -> bool:
    """
    Everything that can go wrong on a new machine, checked in a few minutes.

    Run this before committing a session. The local test suite proves the maths
    and the plumbing, but it runs on synthetic meshes on CPU, and none of the
    failures that actually cost a Kaggle session are visible from there: a
    dataset path that finds nothing, an official split file that is absent so
    the run silently uses a hashed one, a 83,039-vertex fragment that fits in
    the tests' imagination but not in 16 GB, an epoch that turns out to take
    four hours.

    Returns True if it is safe to train. Prints what is wrong and returns False
    if not -- and prints the *measured* epoch time and session count either way,
    because that is the number that decides whether the plan is workable and it
    cannot be guessed in advance.
    """
    import torch

    print("=" * 74)
    print("PREFLIGHT")
    print("=" * 74)
    problems: List[str] = []
    warnings: List[str] = []

    # -- 1. environment ----------------------------------------------------
    print("\n[1/7] environment")
    print(f"  torch {torch.__version__}   cuda available: {torch.cuda.is_available()}")
    devices = torch.cuda.device_count()
    for i in range(devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"  gpu {i}: {properties.name}  "
              f"{properties.total_memory / 1e9:.1f} GB")
    if devices == 0:
        warnings.append("no GPU visible -- training will run on CPU and be "
                        "far too slow for a real dataset")
    requested = devices if config.devices < 0 else config.devices
    if requested > devices:
        problems.append(f"devices={config.devices} but only {devices} GPU(s) visible")
    # Under DDP the worker count is per *rank*, so the machine actually runs
    # `workers * devices` of them plus one main process each. Comparing the
    # per-rank number to the CPU count understates it by the world size, which
    # on a 2-GPU 4-CPU Kaggle box is the difference between "fine" and
    # oversubscribed -- and this pipeline builds scenes in the workers, so
    # contention there shows up directly as epoch time.
    world = max(requested, 1)
    cpus = os.cpu_count() or 2
    total_workers = config.workers * world
    print(f"  cpus: {cpus}   dataloader workers: {config.workers}"
          + (f" x {world} ranks = {total_workers}" if world > 1 else ""))
    if total_workers + world > cpus:
        warnings.append(
            f"{total_workers} dataloader workers plus {world} training "
            f"process(es) on {cpus} cpus is oversubscribed. Scene building is "
            f"CPU-bound here, so this comes straight off epoch time; try "
            f"--workers {max((cpus - world) // world, 1)}."
        )

    # -- 2. output location ------------------------------------------------
    print("\n[2/7] checkpoint directory")
    try:
        _check_writable(Path(config.out_dir))
        print(f"  {config.out_dir}  writable")
    except RuntimeError as error:
        problems.append(str(error))
        print(f"  {config.out_dir}  NOT WRITABLE")
    resume = _resume_path(config, Path(config.out_dir) / "last.pt")
    print(f"  resume source: {resume if resume else 'none -- this starts from scratch'}")

    # -- 3. dataset --------------------------------------------------------
    print("\n[3/7] dataset")
    try:
        train_set = BreakingBadScenes(config, "train", epoch_seed=0)
        val_set = BreakingBadScenes(config, "val", epoch_seed=0)
    except FileNotFoundError as error:
        print(f"  FAILED: {error}")
        # Two different failures reach here and they need opposite advice. If no
        # scenes were found at all, --root is wrong. If scenes were found but a
        # split came out empty, --root is right and the split is the problem --
        # telling someone to fix a correct path sends them the wrong way.
        if "no scenes left in split" in str(error):
            print("\n  The dataset was found, so --root is right. Either the")
            print("  official split lists no object present here, or there are")
            print("  too few objects for the hashed fallback to fill every split.")
        else:
            print("\n  --root must point at the directory *containing* the subset")
            print("  folders, e.g. /kaggle/input/breaking-bad, not at one object.")
        return False
    print(f"  objects: {len(train_set.scenes)} train / {len(val_set.scenes)} val")
    print(f"  samples: {len(train_set)} train / {len(val_set)} val "
          f"({config.modes_per_scene} modes per object)")
    if train_set.official:
        print("  split:   official Breaking Bad lists")
    else:
        warnings.append("no official split file found -- falling back to a "
                        "hashed split. Results will not be comparable to "
                        "published numbers. Check that data_split/ is present.")
        print("  split:   HASHED (no official list found)")
    overlap = {s.object_key for s in train_set.scenes} & {s.object_key for s in val_set.scenes}
    if overlap:
        problems.append(f"{len(overlap)} object(s) appear in BOTH train and val "
                        f"-- validation would be measuring memorisation")

    # -- 4. build real samples --------------------------------------------
    print(f"\n[4/7] building {samples} real scenes")
    built, unusable, sizes = [], [], []
    began = time.time()
    for index in np.linspace(0, max(len(train_set) - 1, 0), samples).astype(int):
        item = train_set[int(index)]
        if isinstance(item, Skipped):
            unusable.append(item)
            continue
        built.append(item)
        vertices = sum(len(f.vertices) for f in item.fragments)
        tokens = sum(len(f.token_vertices) for f in item.fragments)
        sizes.append((vertices, tokens, len(item.fragments)))
    build_time = (time.time() - began) / max(samples, 1)
    if not built:
        print("  FAILED: not one sample could be built")
        return False
    vertices = np.array([s[0] for s in sizes])
    tokens = np.array([s[1] for s in sizes])
    fragments = np.array([s[2] for s in sizes])
    print(f"  built {len(built)}/{samples} in {build_time * 1000:.0f} ms each")
    print(f"  fragments/scene  min {fragments.min()}  median "
          f"{int(np.median(fragments))}  max {fragments.max()}")
    print(f"  vertices/scene   min {vertices.min():,}  median "
          f"{int(np.median(vertices)):,}  max {vertices.max():,}")
    print(f"  tokens/scene     min {tokens.min()}  median "
          f"{int(np.median(tokens))}  max {tokens.max()}  "
          f"(budget {config.tokens_per_scene})")
    if unusable:
        print(f"  {len(unusable)} unusable: "
              f"{', '.join(u.key for u in unusable[:3])}")
    if tokens.max() == 0:
        problems.append("every scene produced ZERO cross-fragment tokens -- the "
                        "fracture mask is empty, so the cross layers do nothing. "
                        "Check --label-method and --sharp-threshold.")
    elif np.median(tokens) < 16:
        warnings.append(f"median {int(np.median(tokens))} tokens/scene is very "
                        f"low; the cross-fragment layers have little to attend over")

    # -- 5. forward and backward on the biggest ---------------------------
    print("\n[5/7] forward + backward on the largest sampled scene")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    criterion = build_criterion(config)
    biggest = built[int(np.argmax(vertices))]
    total_memory = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if device.startswith("cuda") else 0.0)

    # Try the configured batch size, then halve until something fits. Reporting
    # "out of memory" alone leaves the useful question unanswered; the number
    # that is actually wanted is the largest batch this card can take, and it
    # costs seconds to measure rather than a session to discover.
    fitted, peak, report = 0, 0.0, {}
    batch = loss = None
    for candidate in _halvings(config.batch_size):
        batch, _ = _collate_samples([biggest] * candidate)
        batch = _to_device(batch, device)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            loss, report, _ = _forward(model, batch, criterion, config)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            # `loss` must go too, not just `batch`. If the OOM landed in
            # `backward` rather than the forward, `loss` is bound and holds the
            # entire autograd graph of the candidate that just failed -- which
            # would then still be resident while the *next*, smaller candidate
            # is measured, inflating its peak and possibly failing it too.
            model.zero_grad(set_to_none=True)
            batch = loss = report = None
            torch.cuda.empty_cache()
            print(f"  batch_size={candidate}: out of memory")
            continue
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.startswith("cuda") else 0.0)
        fitted = candidate
        print(f"  batch_size={candidate}: fits"
              + (f", peak {peak:.2f} GB of {total_memory:.1f} GB "
                 f"({100 * peak / total_memory:.0f}%)" if peak else ""))
        break

    if fitted == 0:
        problems.append(
            f"out of memory even at batch_size=1 on a {total_memory:.0f} GB card. "
            f"Try, in this order: --checkpoint-intra (recomputes the "
            f"intra-fragment projections in the backward pass, the largest "
            f"remaining saving), then --channels 32 (roughly halves every edge "
            f"activation), then --tokens-per-scene 1024 (the cross-attention "
            f"cost is quadratic in this)."
        )
        _summarise(problems, warnings)
        return False
    if fitted < config.batch_size:
        effective = max(config.batch_size // fitted, 1)
        problems.append(
            f"batch_size={config.batch_size} does not fit; {fitted} does. Use "
            f"--batch-size {fitted} --accumulate {effective}, which keeps the "
            f"effective batch at {fitted * effective} and costs only a little "
            f"speed."
        )
    if peak and peak > 0.65 * total_memory:
        # Keep the effective batch identical when suggesting a smaller one.
        # `--accumulate 2` alongside a halved batch would quietly halve the
        # effective batch as well, which changes the optimisation rather than
        # just the memory -- and a suggestion that silently retunes the run is
        # worse than no suggestion.
        safer = max(fitted // 2, 1)
        keep = max(config.accumulate * max(fitted // safer, 1), 1)
        warnings.append(
            f"peak memory is {100 * peak / total_memory:.0f}% of the card on the "
            f"largest of {samples} sampled scenes -- and the dataset's largest "
            f"single fragment is 83,039 vertices, several times anything sampled "
            f"here. Training catches an OOM and skips the batch and reports the "
            f"count; if that count is more than a percent or so of an epoch, the "
            f"largest objects are being dropped from training, which biases the "
            f"result. Then use --batch-size {safer} --accumulate {keep}, which "
            f"holds the effective batch at "
            f"{safer * keep * max(requested, 1)}."
        )

    parameters = sum(p.numel() for p in model.parameters())
    print(f"  parameters {parameters:,}   worst case tested: "
          f"{fitted} x the largest sampled scene")
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    # Release step 5's batch and autograd graph before step 6 allocates its own.
    # Without this the checks compete for the card and step 6 dies on an OOM
    # that says nothing about the model -- a preflight that fails for its own
    # reasons is worse than no preflight.
    del batch, loss
    model.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if dead:
        warnings.append(f"{len(dead)} parameter(s) got no gradient: "
                        f"{', '.join(dead[:4])}")
        print(f"  {len(dead)} parameter(s) without gradient")
    else:
        print("  every parameter received gradient")

    # -- 6. loss at initialisation ----------------------------------------
    print("\n[6/7] loss at initialisation, against chance")
    # Never larger than what step 5 proved fits. An earlier version used
    # `fitted * 2`, reasoning that `no_grad` stores no autograd graph and so
    # costs a fraction -- true, but a guess, and preflight is the one place that
    # must not guess about memory. It duly died on a card where `fitted` was 2.
    #
    # The `try` is the belt to that brace: this step measures the loss, not the
    # memory, so if it runs out anyway it must shrink and carry on rather than
    # take the whole report down with it.
    # Averaged over *every* scene built in step 4, not just one batch.
    #
    # This is what makes the check worth having. The rotation angle of a random
    # rotation has a standard deviation of 37 deg, so the mean over one batch of
    # two scenes -- about a dozen fragments -- carries a standard error of
    # 10.7 deg. A check with that much noise cannot tell chance from 20 deg off
    # chance, which is most of what it is for. Over twelve scenes the standard
    # error is ~4.4 deg, and the scenes are already built and paid for.
    report, fragments = {}, 0
    chunk = max(fitted, 1)
    for start in range(0, len(built), chunk):
        plain = None
        try:
            plain, _ = _collate_samples(built[start:start + chunk])
            with torch.no_grad():
                _, piece, _ = _forward(model, _to_device(plain, device),
                                       criterion, config)
            # Fragment-weighted, the same way `run_epoch` aggregates, so the
            # average does not over-count scenes that happen to be small.
            weight = int(plain.num_fragments)
            for name, value in piece.items():
                report[name] = report.get(name, 0.0) + value * weight
            fragments += weight
        except torch.cuda.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        finally:
            plain = None
    if fragments:
        report = {name: total / fragments for name, total in report.items()}
        print(f"  averaged over {len(built)} scenes / {fragments} fragments "
              f"(+-{37.0 / math.sqrt(fragments):.1f} deg noise on rotation)")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not report:
        problems.append("could not evaluate the loss at initialisation even on "
                        "a single scene -- something is wrong beyond batch size.")
        _summarise(problems, warnings)
        return False
    print(f"  {format_losses(report)}")
    print(f"  rotation {report['rotation_degrees']:.1f} deg "
          f"(chance {CHANCE['geodesic_deg']:.1f})   "
          f"normal {report.get('normal', float('nan')):.3f} (1.0)   "
          f"face {report.get('face', float('nan')):.3f} (2.0)")
    # Worth stating, because the obvious reading of this line is wrong. Because
    # the model is equivariant, the perturbation cancels out of the error at
    # initialisation exactly: R_pred R_label^T = frame(assembled)^T. So this
    # number is the mean angle of the untrained frames on *assembled* fragments,
    # not a draw from the chance distribution -- it equals 126.5 deg only if
    # those frames happen to be uniformly spread. A few degrees either way is a
    # property of the initialisation, and there is nothing there to fix.
    print("  (equivariance cancels the perturbation here, so this is the "
          "untrained frame's own angle, not a sample from chance)")
    for complaint in check_initial_losses(report):
        problems.append(f"loss at init: {complaint}")
    if "embedding" not in report:
        warnings.append("no embedding term -- these scenes produced no "
                        "coincidence clusters, so the embedding head will not "
                        "train and stage two has nothing to match on")

    # -- 7. speed ----------------------------------------------------------
    # Time at the size step 5 *proved* fits, never at the configured one. Timing
    # at a size already shown to OOM makes this step die for preflight's own
    # reasons and throws away the whole report -- which is exactly the failure
    # step 6 had, in a second place.
    timing_batch = max(min(config.batch_size, fitted), 1)
    print(f"\n[7/7] timing {timed_batches} training steps at "
          f"batch_size={timing_batch}"
          + (f" (not {config.batch_size} -- that did not fit)"
             if timing_batch != config.batch_size else ""))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=timing_batch, shuffle=True,
        collate_fn=_collate_samples, num_workers=config.workers,
    )
    model.train()
    began, done, skipped = time.time(), 0, 0
    for index, (batch, _) in enumerate(loader):
        if batch is None:
            continue
        try:
            batch = _to_device(batch, device)
            loss, _, _ = _forward(model, batch, criterion, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            # Step 5 measured the largest *sampled* scene; the loader draws from
            # the whole set, where the largest single fragment is 83k vertices.
            # Training skips such a batch rather than dying, so preflight must
            # too -- otherwise it reports a failure the real run would survive.
            skipped += 1
            batch = loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        finally:
            batch = loss = None
        optimizer.zero_grad(set_to_none=True)
        done += 1
        if done >= timed_batches:
            break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if done == 0:
        problems.append(
            f"every one of the first {skipped} batches ran out of memory at "
            f"batch_size={timing_batch}, even though a batch of {fitted} copies "
            f"of the largest sampled scene fits. Real scenes are bigger than the "
            f"sample; lower --batch-size or --tokens-per-scene."
        )
        _summarise(problems, warnings)
        return False
    if skipped:
        warnings.append(
            f"{skipped} of the first {done + skipped} batches ran out of memory "
            f"and were skipped. Training survives this by design, but that rate "
            f"means the largest objects are being dropped from training -- lower "
            f"--batch-size and raise --accumulate to keep the effective batch."
        )
    per_step = (time.time() - began) / max(done, 1)
    world = max(requested, 1)
    steps = math.ceil(len(train_set) / (timing_batch * world))
    epoch_seconds = per_step * steps
    total_seconds = epoch_seconds * config.epochs
    sessions = math.ceil(total_seconds / (config.max_hours * 3600))
    print(f"  {per_step:.2f} s/step (includes warm-up, so pessimistic)")
    print(f"  {steps} steps/epoch at batch_size={timing_batch} on {world} "
          f"device(s) -> ~{_hms(epoch_seconds)}/epoch")
    if timing_batch != config.batch_size:
        print(f"  (--accumulate does not change this: it changes how often the "
              f"optimizer steps, not how many forward/backward passes run)")
    print(f"  {config.epochs} epochs -> ~{_hms(total_seconds)} "
          f"= {sessions} session(s) at {config.max_hours:g}h")
    if epoch_seconds > config.max_hours * 3600:
        warnings.append(
            f"one epoch (~{_hms(epoch_seconds)}) is longer than a session. That "
            f"works -- mid-epoch checkpoints every "
            f"{config.checkpoint_every_minutes:g} min cover it -- but no epoch "
            f"will ever complete, so val metrics never update. Consider fewer "
            f"--modes-per-scene."
        )

    return _summarise(problems, warnings)


def _halvings(start: int) -> List[int]:
    """``4 -> [4, 2, 1]``: the batch sizes to try, largest first."""
    sizes, value = [], max(int(start), 1)
    while value >= 1:
        sizes.append(value)
        if value == 1:
            break
        value //= 2
    return sizes


def _summarise(problems: List[str], warnings: List[str]) -> bool:
    print("\n" + "=" * 74)
    for warning in warnings:
        print(f"  [warn]  {warning}")
    for problem in problems:
        print(f"  [STOP]  {problem}")
    if problems:
        print("\n  NOT ready to train -- fix the [STOP] items above.")
        print("=" * 74)
        return False
    print("\n  ready to train." + ("  Read the warnings first." if warnings else ""))
    print("=" * 74)
    return True
