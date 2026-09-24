"""
Training: config, dataset, loops, metrics, checkpoints, schedule.

One module rather than five. The pipeline is already a lot of moving parts, and
splitting an engine across files mostly buys import diagrams.

    from reassembly.training import Config, train
    train(Config(root="/kaggle/input/breaking-bad", epochs=40))

or from the command line::

    python -m scripts.train --root_dir /kaggle/input/breaking-bad --epochs 40

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

import dataclasses
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
    mode_filter: Optional[str] = "fractured_"
    """
    ``--fracture_pattern``. Keep only fracture-mode directories whose name
    starts with this: ``"fractured_"``, the default as in Thesis 1, keeps the
    standard break patterns and excludes the ``mode_*`` variants, so results
    stay comparable with the benchmark. ``None`` (``--fracture_pattern ""``)
    keeps every mode.
    """
    split_by: str = "object"
    """
    What is held out: whole shapes (``"object"``) or break patterns
    (``"fracture"``). See :func:`reassembly.data.catalog.split_catalog`.

    ``"object"`` is the default and the only setting whose numbers are
    comparable with published results. ``"fracture"`` answers a different and
    strictly easier question -- can the model orient a fragment of a shape it
    HAS seen, broken in a way it has not -- and exists so the two can be run
    against each other. A model at chance on ``object`` and well under it on
    ``fracture`` has learned per-shape canonical poses rather than a
    transferable rule; at chance on both, generalisation is not the problem.
    """
    fracture_pool: str = "train"
    """
    Which objects the fracture split draws from: ``"train"`` (the official
    training shapes only, leaving the official val/test shapes untouched for
    the benchmark) or ``"all"``. ``"all"`` is fine for a diagnostic and
    disqualifying for a reported number.
    """
    val_frac: float = 0.1
    test_frac: float = 0.1
    """
    Held-out proportions. Under ``split_by="object"`` they are used only when
    no official list is found; under ``split_by="fracture"`` they are always
    used, as the proportion of each object's modes held out.
    """
    split_seed: int = 0
    split_source: str = "auto"
    """
    Where the object split comes from: ``"official"`` -- Breaking Bad's own
    train/val lists, required for numbers comparable with published results,
    and an error if they are missing; ``"hash"`` -- a deterministic split of
    whatever is on disk, by ``val_frac``/``test_frac``/``split_seed``; ``"auto"``
    -- the official lists when present, the hash otherwise (printed either way).
    """
    balance: str = "none"
    """
    Correct the category imbalance when drawing training samples: ``"none"``,
    ``"category"`` (uniform over categories, then objects, then modes) or
    ``"object"`` (uniform over objects only). Everyday holds 17 distinct
    bottles against 5 cups, so natural sampling shows the model roughly three
    bottles per cup and a shape prior is cheaper to fit than an orientation
    rule.

    Training only. Validation is never reweighted -- a val number that moved
    with the sampler would not be comparable across settings -- so the
    per-category validation breakdown is the honest way to see whether
    balancing helped.
    """
    balance_temperature: float = 1.0
    """
    ``0`` natural, ``1`` full balance, ``0.5`` square-root softening.
    Full balance is not obviously right: it also means each cup is drawn 3.4x
    as often as each bottle, which is a different skew rather than none.
    """
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
    symmetry_axis: str = "z"
    """
    The dataset's canonical up-axis, used *only* for the reported tilt/twist
    split. Verify it rather than assume it -- run evaluation with x, y and z
    and see which shows the signature. A wrong choice makes the diagnostic
    meaningless; it cannot make the training wrong, because nothing optimises
    it.
    """
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
    grad_checkpointing: bool = False
    """
    Recompute each layer in the backward pass instead of storing its insides,
    as in Thesis 1 (``--grad_checkpointing True``). Roughly one extra forward
    pass of time for a large cut in memory: measured on CPU at 128 channels,
    see ``docs/PROJECT-STATE.md`` §11. Outputs and gradients are identical.
    Off by default; a batch that runs out of memory is retried with it on.

    (The cross layers recompute their pair gathers in the backward pass
    whatever this says -- 1109 -> 86 bytes per pair for nearly no time -- so
    there is nothing to switch there.)
    """
    schedule: Sequence[str] = ("intra",) * 5 + ("cross",) * 3 + ("intra",)
    """
    Five intra-fragment layers, then three cross-fragment layers: describe each
    fragment first, then let the fragments talk.

    The trade this makes is real and worth knowing before reading a curve. A
    cross layer updates only *token* vertices, and the head pools a mean over
    *every* vertex -- so with nothing after the last cross layer, about 78% of
    the pooled signal (2,048 tokens against a median 9,149 vertices) comes from
    vertices that never heard from another fragment. The network can compensate
    by scaling token features up, but it does not start there. See
    `nn/model.py` for what to try if it parks at the axis-only landmark.
    """

    # -- loss weights (all 1.0 and untuned, on purpose) --------------------
    w_rotation: float = 1.0
    w_position: float = 1.0
    w_normal: float = 1.0
    w_face: float = 1.0
    w_embedding: float = 1.0

    # -- optimisation -----------------------------------------------------
    epochs: int = 40
    steps_per_epoch: int = 1600
    """
    Forward passes (micro-batches) per GPU per epoch; ``0`` is one full pass
    over the training split. On the command line ``--steps_per_epoch`` counts
    *optimizer* steps, as in Thesis 1, and is multiplied by ``accumulate``
    here: the default 1600 is the command line's 800 steps x 2 scenes.

    An epoch is the unit the validation curve, the checkpoints, the time
    projection and the learning-rate schedule are counted in, so it should be a
    fixed amount of training. A pass is not one: it moves with
    ``modes_per_scene``, the split, ``limit_train``, the number of GPUs and
    balancing. 800 steps x 2 scenes on each of two GPUs is 3,200 scenes, about
    one pass over the Everyday training split. The draws are seeded and every
    GPU gets exactly the same number of them, which it must -- every GPU takes
    part in every step. The startup banner says how many passes an epoch is.
    """
    batch_size: int = 1
    """
    Scenes per forward pass per GPU -- ``--micro_batch_scenes`` on the command
    line, 1 by default as in Thesis 1. Peak memory grows with this and with
    nothing else in the batch settings. A scene is a whole graph, not an image.
    """
    lr: float = 1e-3
    min_lr_fraction: float = 0.02
    """``--lr_min`` on the command line, as an absolute rate."""
    warmup_fraction: float = 0.03
    """``--lr_warmup_epochs`` on the command line, in epochs."""
    lr_schedule: str = "cosine"
    """
    ``"cosine"``: after the warmup, decay once to the floor and stay there.
    ``"constant"``: stay at ``lr`` after the warmup. (Thesis 1's ``"plateau"``
    is not implemented here.)
    """
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_vertices_per_batch: int = 0
    """
    Skip any batch with more vertices than this, before attempting it. 0 is off.

    Optional. A batch that runs out of memory is retried once with gradient
    checkpointing and skipped only if that fails too, on one GPU or several:
    each GPU's batches are its own business until the optimizer step, so a GPU
    that skips cannot leave another waiting. (This setting used to be sold as
    the one safe way to skip under DDP, "identical on every rank". It was not
    identical -- each rank holds different scenes -- and the skip it made was
    one of the ways the ranks fell out of step.) The limit only saves the two
    doomed attempts on a batch known not to fit.

    Preflight reports the value to use: it is the largest batch it could
    actually fit, with a margin. Batches skipped this way are counted and named
    exactly like an out-of-memory skip, because they are the same event.
    """
    accumulate: int = 2
    """
    Forward passes per optimizer step. The command line does not ask for it:
    it is ``--batch_size`` (scenes per step per GPU, as in Thesis 1) divided by
    ``--micro_batch_scenes``. The gradients of the passes in a step are summed
    and divided by their fragment count, so a step over 8 passes of 1 scene is
    the same step as one pass over all 8 -- only the peak memory differs.
    """
    amp: bool = False
    perturb_on_device: bool = True
    """
    Ship only the assembled copy of each scene through the data loader and
    derive the perturbed copy on the GPU, by rotation (see
    :func:`reassembly.data.features.perturb_on_device`). The result is the same
    tensors to float32 round-off; what changes is where the work happens.

    Measured, per batch of two scenes, CPU side: 3% less loader time and 4%
    fewer bytes on a median 10k-vertex scene, 23% and 18% on an 82k-vertex one.
    Modest, because the perturbed copy was already made by rotating rather than
    recomputing. The same measurement showed where the loader's time actually
    went -- the cross-fragment pair lists, 89% of a batch's bytes -- and on a
    GPU those are now built on the device too (``_collate_samples``); with
    both, the loader does 18-40% of the work it did and ships 6-27% of the
    bytes. The GPU's side of the trade (three batched matmuls and one index
    build per step) could not be timed without a GPU. ``docs/PROJECT-STATE.md``
    has the table.
    """

    # -- runtime ----------------------------------------------------------
    out_dir: str = "runs/vgat"
    workers: int = 2
    seed: int = 0
    devices: int = -1
    """
    GPUs to use: ``-1`` every visible GPU (2 on Kaggle), ``1`` one, ``N`` the
    first N, ``0`` the CPU. More than one runs one process per GPU; each draws
    its own share of every epoch, and the gradients are averaged over all of
    them once per optimizer step (:mod:`reassembly.distributed`). With no GPU
    at all, ``devices >= 2`` still runs that many processes on the CPU over
    gloo -- slow, and meant for testing the multi-process path.
    """
    max_hours: float = 11.0
    """
    ``--time_budget_hours``. Read between epochs, as in Thesis 1: once it has
    run out, the epoch in progress is finished, validated and saved, and the
    run stops. So leave one epoch's time below any hard session limit
    (Kaggle's is 12 h; the log prints the time per epoch). A kill signal still
    stops at once, with a checkpoint, and the 30-minute mid-epoch checkpoints
    cover a session that is cut off without one.
    """
    resume: bool = True
    resume_from: Optional[str] = None
    """
    Where to *load* a checkpoint from, when that differs from where this run
    *saves*. This is the normal case on Kaggle: a session writes to
    ``/kaggle/working``, and the next session mounts that output read-only under
    ``/kaggle/input/<name>``. Point ``resume_from`` at the input copy and
    ``out_dir`` at working, and the chain continues across sessions.
    """
    save_every: int = 1
    """
    Write ``last.pt`` every this many epochs, and always on the last one and on
    a stop. ``best.pt`` is written whenever validation improves, regardless.
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
    max_objects: Optional[int] = None
    """
    Train on only this many distinct OBJECTS (every break pattern of each),
    evenly strided across the sorted training catalogue so every category can
    appear. For the object-count scaling experiment
    (``python -m scripts.scaling_sweep``), whose question is how error moves
    with the number of *shapes* -- which ``limit_train`` cannot ask, since it
    thins (object, pattern) pairs and so keeps nearly every shape. Validation
    is unaffected.
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
        if self.steps_per_epoch < 0:
            raise ValueError("steps_per_epoch must be >= 0 (0 = one full pass)")
        if self.max_objects is not None and self.max_objects < 1:
            raise ValueError("max_objects must be >= 1, or None for every object")
        if self.accumulate < 1:
            raise ValueError("accumulate must be >= 1")
        if self.split_source not in ("auto", "official", "hash"):
            raise ValueError(f"split_source must be 'auto', 'official' or 'hash', "
                             f"got {self.split_source!r}")
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"lr_schedule must be 'cosine' or 'constant', got "
                             f"{self.lr_schedule!r}" + (" -- Thesis 1's 'plateau' is not "
                             "implemented here" if self.lr_schedule == "plateau" else ""))
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1")
        from .data.catalog import BALANCE_SCHEMES, SPLIT_MODES

        if self.split_by not in SPLIT_MODES:
            raise ValueError(
                f"split_by must be one of {SPLIT_MODES}, got {self.split_by!r}")
        if self.fracture_pool not in ("train", "all"):
            raise ValueError(
                f"fracture_pool must be 'train' or 'all', got {self.fracture_pool!r}")
        if self.balance not in BALANCE_SCHEMES:
            raise ValueError(
                f"balance must be one of {BALANCE_SCHEMES}, got {self.balance!r}")
        if not 0.0 <= self.balance_temperature <= 1.0:
            raise ValueError("balance_temperature must be in [0, 1]")
        if not 0.0 <= self.val_frac < 1.0 or not 0.0 <= self.test_frac < 1.0:
            raise ValueError("val_frac and test_frac must be in [0, 1)")
        if self.val_frac + self.test_frac >= 1.0:
            raise ValueError("val_frac + test_frac must leave something for training")
        if self.split_by == "fracture":
            # Not an error, and it is the point of the mode -- but a result
            # from it is not a result on the benchmark, and that has to be said
            # once where it cannot be missed rather than inferred from a config
            # key three months later.
            print("[config] split_by=fracture: train and val share every SHAPE "
                  "and differ only in break pattern. This measures "
                  "generalisation to unseen fractures of known objects, which "
                  "is a strictly easier question than the benchmark's. Do not "
                  "report a number from it as an object-split result.")
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


# The command-line name of each Config field that Thesis 1 also has, under
# Thesis 1's name, so the two projects are driven by the same flags. Every
# other field's flag is its own name. `scripts/config_flags.py` builds the
# parser from this; the messages here use it to name the flag the user types.
# (`--batch_size`, `--steps_per_epoch`, `--lr_min`, `--lr_warmup_epochs`,
# `--val_steps` and `--resume` carry Thesis 1's *meaning* too and are converted
# there.)
THESIS1_NAMES = {
    "root": "root_dir",
    "out_dir": "checkpoint_dir",
    "subsets": "data_subsets",
    "mode_filter": "fracture_pattern",
    "channels": "hidden_channels",
    "embedding_dim": "embed_dim",
    "w_rotation": "w_rot",
    "w_position": "w_pos",
    "workers": "num_workers",
    "devices": "num_gpus",
    "max_hours": "time_budget_hours",
    "batch_size": "micro_batch_scenes",
}


def flag(field: str) -> str:
    """The command-line flag that sets a Config field."""
    return "--" + THESIS1_NAMES.get(field, field)


# Reference values every metric is read against. Measured by Monte Carlo in
# tests/test_losses.py, not asserted from memory.
#
# `euler_identity_deg` used to sit 3 deg BELOW `euler_rmse_deg`, and that gap
# was treated here, in the banner and in the design document as a property of
# the metric worth warning about. It was a property of our own convention:
# subtracting two Euler charts componentwise. `euler_rmse` now reports the
# angles of the residual rotation, and the gap closes to Monte-Carlo noise --
# so collapsing to the identity buys nothing and there is no trap left to warn
# about. Both numbers are kept because a reference table with one row reads as
# though the other were never checked.
CHANCE = {
    "geodesic_deg": 126.48,      # pi/2 + 2/pi
    "euler_rmse_deg": 83.25,     # random prediction, residual convention
    "euler_identity_deg": 83.18,  # always-identity: the same, within noise
    "axis_only_deg": 90.0,       # axis recovered, rotation about it uniform
    "normal": 1.0,
    "face": 2.0,
}

# `axis_only_deg` is a LANDMARK, not a floor. A model parked at ~90 deg has
# found the object's symmetry axis and not the rotation about it -- which the
# tilt/twist split in `reassembly.evaluation.metrics` distinguishes directly
# (tilt ~ 0 with twist ~ 90). It was previously described as a floor that a
# per-fragment canonicaliser could not beat on surfaces of revolution. That is
# false, and measurably so: a fragment of a symmetric object is not itself
# symmetric, because its fracture boundary is jagged and unique. The earlier
# VN-GAT design reached 30.9 deg training error on eight Everyday objects --
# bottles, bowls and mugs -- which is well under it.


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
        from .data.catalog import Catalog, build_catalog, split_catalog
        from .data.paths import find_scenes, load_official_split

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
        # Group first, filter second. Doing it the other way round counts a
        # shape once per variant directory, which is how the object count came
        # out at twice the official one -- and, worse, lets the SAME shape land
        # in train under `everyday_compressed` and in val under
        # `volume_constrained-everyday_compressed`, so an object split is not
        # one. Grouping keeps every break pattern and counts every shape once.
        full = build_catalog(scenes, mode_filter=config.mode_filter)
        official = {} if config.split_source == "hash" else {
            name: load_official_split(config.root, name, config.official_subset)
            for name in ("train", "val", "test")
        }
        official = {k: v for k, v in official.items() if v}
        if config.split_source == "official" and not official:
            raise FileNotFoundError(
                f"--split_source official, but no official split lists for "
                f"{config.official_subset!r} were found under {config.root!r}. "
                f"Use --split_source hash (or auto) to split what is on disk.")
        self.official = bool(official)

        self.catalog: Catalog = split_catalog(
            full, split,
            split_by=config.split_by,
            official=official or None,
            val_frac=config.val_frac, test_frac=config.test_frac,
            seed=config.split_seed, fracture_pool=config.fracture_pool,
        )
        if config.split_by == "object" and split in ("train", "val") and official:
            _assert_splits_disjoint(full, config, official)
        if split == "train" and config.max_objects:
            objects = list(self.catalog.objects)
            if config.max_objects < len(objects):
                keep = np.linspace(0, len(objects) - 1, config.max_objects).round().astype(int)
                self.catalog = dataclasses.replace(
                    self.catalog, objects=tuple(objects[i] for i in dict.fromkeys(keep.tolist())))
        if not self.catalog.objects:
            raise FileNotFoundError(
                f"no objects left in split {split!r} "
                f"(split_by={config.split_by!r}, {len(full)} objects on disk). "
                + ("Breaking Bad ships train and val lists only, so there is no "
                   "official test split -- use split_by='fracture' or the hash "
                   "fallback if you need a third partition."
                   if split == "test" and official else
                   "Check --root_dir and --official_subset.")
            )

        # (object index, mode name) pairs, deterministic in order. A mode's
        # directory is carried alongside rather than derived from the object,
        # because after grouping one shape's patterns can span more than one
        # directory.
        self.scenes = list(self.catalog.objects)
        self._directories: Dict[Tuple[int, str], Path] = {}
        self.items: List[Tuple[int, str]] = []
        for index, entry in enumerate(self.catalog.objects):
            modes = list(entry.modes)
            if config.modes_per_scene is not None and len(modes) > config.modes_per_scene:
                # Deterministic per (object, epoch): every epoch shows a
                # different slice of an object's break patterns, but two runs
                # with the same seed see the same ones.
                picker = random.Random(f"{entry.key}/{epoch_seed}")
                modes = sorted(picker.sample(modes, config.modes_per_scene))
            self.items.extend((index, mode) for _directory, mode in modes)
            self._directories.update(
                {(index, mode): directory for directory, mode in modes})

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

        # Keyed by DIRECTORY, not by object: a reader parses one
        # compressed_mesh.obj plus its cell matrix, and a grouped object may own
        # several. Keying by object index would hand a reader built from one
        # directory a mode name that only exists in another.
        self._readers: Dict[Path, object] = {}
        self._order: List[Path] = []
        self.failures: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.items)

    def categories(self) -> List[str]:
        """Category per item, for the per-category metric breakdown."""
        return [self.catalog.objects[index].category or "(uncategorised)"
                for index, _mode in self.items]

    def sampling_weights(self) -> Optional[List[float]]:
        """
        Per-item weights for the balanced sampler, aligned with ``self.items``.

        Computed from the item list rather than from the catalogue, because
        ``modes_per_scene`` and ``limit_*`` both trim it -- and a weight vector
        that is merely the right *length* while being misaligned would reweight
        the wrong samples and look like it worked.
        """
        from .data.catalog import item_weights

        object_ids = [index for index, _mode in self.items]
        categories = [self.catalog.objects[index].category for index in object_ids]
        return item_weights(categories, object_ids, self.config.balance,
                            self.config.balance_temperature)

    def _reader(self, directory: Path):
        from .data.scene import SceneReader

        if directory not in self._readers:
            if len(self._order) >= self.cache_size:
                self._readers.pop(self._order.pop(0), None)
            self._readers[directory] = SceneReader(directory)
            self._order.append(directory)
        return self._readers[directory]

    def key(self, i: int) -> str:
        """``<object key>/<mode>`` -- the name every log, tally and script uses."""
        scene_index, mode = self.items[i]
        return f"{self.catalog.objects[scene_index].key}/{mode}"

    def index_of(self, key: str) -> Optional[int]:
        """The item whose :meth:`key` is ``key``, or ``None``."""
        for i in range(len(self.items)):
            if self.key(i) == key:
                return i
        return None

    def meshes(self, i: int) -> list:
        """Item ``i``'s fragments as loaded, in the ASSEMBLED frame (trimesh)."""
        scene_index, mode = self.items[i]
        return self._reader(self._directories[(scene_index, mode)]).load_mode(mode).fragments

    def labels(self, meshes) -> list:
        """Per-fragment boolean fracture masks, by ``config.label_method``."""
        from .mesh.fracture import fracture_face_mask, fracture_vertex_masks

        config = self.config
        if config.label_method == "coincidence":
            return fracture_vertex_masks(meshes)
        masks = []
        for mesh in meshes:
            v, f = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
            labelled = fracture_face_mask(
                v, f, sharp_threshold=config.sharp_threshold,
                min_faces=config.min_fracture_faces,
            ).face_mask
            mask = np.zeros(len(v), bool)
            if labelled.any():
                mask[np.unique(f[labelled])] = True
            masks.append(mask)
        return masks

    def clusters(self, meshes) -> Optional[np.ndarray]:
        """Coincidence clusters for the embedding loss, or ``None`` when off."""
        from .mesh.correspondence import compute_scene_correspondence

        if not self.config.supervise_embedding:
            return None
        # Coincidence clusters are defined in the ASSEMBLED frame. Computed
        # here, before any perturbation -- afterwards nothing coincides and the
        # labels come back empty while looking like a result.
        per_fragment, _ = compute_scene_correspondence(meshes)
        return np.concatenate(per_fragment) if per_fragment else None

    def build(self, i: int, *, rotations=None, seed=None, perturb=None):
        """
        Item ``i`` as the loop sees it, with the perturbation overridable:
        ``rotations`` (per fragment) or ``seed`` replace the item's own stable
        draw, ``perturb`` overrides ``config.perturb_on_device``. The scripts
        use this to rebuild exactly what training saw, or a chosen variant.
        """
        from .data.features import build_scene

        config = self.config
        scene_index, _mode = self.items[i]
        key = self.key(i)
        meshes = self.meshes(i)
        if len(meshes) < 2:
            # A single-fragment mode has no cross-fragment structure and no
            # relative pose to learn from. Returned as a named Skipped rather
            # than None, so the loop can tally which items keep failing.
            return Skipped(key, f"{len(meshes)} fragment(s)")

        vertices = [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
        faces = [np.asarray(m.faces) for m in meshes]
        if not all(np.isfinite(v).all() for v in vertices):
            # A non-finite coordinate cannot be repaired the way a degenerate
            # normal can -- there is no right value to put there -- and it would
            # reach every feature through the centroid and the scale. Named, so
            # a scene that does this every epoch shows up as the same name.
            return Skipped(key, "non-finite vertex coordinates")

        draw = stable_seed(key, self.epoch_seed, config.seed) if seed is None else seed
        return build_scene(
            vertices, faces, self.labels(meshes),
            category=self.catalog.objects[scene_index].category or "(uncategorised)",
            rotations=rotations,
            rng=np.random.default_rng(draw),
            normalize_mode=config.normalize_mode,
            token_mode=config.token_mode,
            token_metric=config.token_metric,
            tokens_per_scene=config.tokens_per_scene,
            max_tokens_per_fragment=None,
            cluster=self.clusters(meshes),
            key=key,
            perturb=(not config.perturb_on_device) if perturb is None else perturb,
        )

    def __getitem__(self, i: int):
        return self.build(i)


def stable_seed(*parts) -> int:
    """
    A seed that is the same in every process, every session and every Python.

    The scene seed used to be ``hash((key, epoch))``. Python salts ``str``
    hashes per *interpreter* (``PYTHONHASHSEED``), so the same scene drew a
    different perturbation and different tokens in every session and in every
    spawned GPU process -- including the validation set, whose rotations are
    meant to stay fixed so that the numbers before and after a resume measure
    the same thing. Within one session it looked stable, which is why it went
    unnoticed. A cryptographic digest has no salt.
    """
    import hashlib

    text = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")


class Skipped(NamedTuple):
    """A sample the dataset could not build, carrying why."""
    key: str
    reason: str


def _assert_splits_disjoint(catalog, config: Config, official: Dict) -> None:
    """
    Refuse to build a dataset whose train and val splits share a shape.

    An object in both makes validation partly a memorisation test, and the
    resulting number is *better* than the honest one -- so it never looks like
    an error, it looks like success. Checked here rather than only in preflight
    because a preflight is easy to skip and this must not be skippable.

    Applies to ``split_by="object"`` only. Under ``split_by="fracture"`` the
    splits share every shape *by construction*, and that is the question being
    asked -- so the equivalent guarantee there is that no break PATTERN is
    shared, which :func:`~reassembly.data.catalog.partition_modes` gives by
    partitioning a shuffled list rather than by hashing each mode.
    """
    from .data.catalog import split_catalog

    keys = {
        name: {entry.key for entry in split_catalog(
            catalog, name, split_by="object", official=official,
            val_frac=config.val_frac, test_frac=config.test_frac,
            seed=config.split_seed).objects}
        for name in ("train", "val", "test")
    }
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


def _collate_samples(samples, pairs: bool = False):
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
    # No pair lists by default: they are ~90% of a batch's bytes and are built
    # on the device in `_forward` instead (`data.features.complete_batch`).
    # `pairs=True` builds them here -- right when training on the CPU, where
    # the loader's worker processes are the parallel part.
    return (collate(kept, pairs=pairs) if kept else None), dropped


# ==========================================================================
# Schedule
# ==========================================================================


def learning_rate(step: int, total: int, config: Config) -> float:
    """
    Linear warmup, then cosine decay to the floor (``lr_schedule="cosine"``) or
    a constant rate (``"constant"``) -- a pure function of the step.

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
    if config.lr_schedule == "constant":
        return config.lr
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


def _forward(model, batch, criterion, config, keep=None):
    """
    One forward pass and the full loss breakdown.

    ``keep``, a dict, receives the completed batch and the whole prediction --
    for evaluation, which assembles from the embeddings as well as the
    rotations.

    The predicted rotation is applied to the *centred, normalised* input and
    compared against the assembled target, which is the convention derived in
    ``nn/model.py``: ``v_perturbed @ R.T == v_assembled`` exactly when the
    prediction is right.
    """
    import torch

    from .data.features import clustered_vertices, complete_batch
    from .nn.model import apply_rotation

    batch = complete_batch(batch)
    prediction = model(
        batch.node_features, batch.edge_index, batch.edge_attr,
        batch.vertex_fragment, batch.num_fragments,
        log_scale=batch.log_scale, token_index=batch.token_index,
        token_query=batch.token_query, token_key=batch.token_key,
    )
    R = prediction.rotation
    if keep is not None:
        keep["batch"], keep["prediction"] = batch, prediction
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
    # Diagnostics ride along in `report`, which the epoch loop already averages
    # weighted by fragment count and writes into the history. They are computed
    # under no_grad and are not part of `total`, so they cannot influence what
    # is optimised -- `tests/test_training.py` pins that the terms still sum to
    # the total.
    with torch.no_grad():
        from .evaluation.metrics import head_collinearity, swing_twist_error

        tilt, twist = swing_twist_error(R.detach(), batch.target_rotation,
                                        axis=config.symmetry_axis)
        report["tilt_deg"] = float(tilt.mean())
        report["twist_deg"] = float(twist.mean())
        if prediction.head_axes is not None:
            report["head_cos"] = float(head_collinearity(prediction.head_axes.detach()))
    return total, report, R


def _metrics(predicted, target,
             categories: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """
    Geodesic (primary) and Euler RMSE (GARF comparability), plus accuracy.

    ``categories`` adds a per-category breakdown under the ``by_category`` key.
    It is the honest counterpart to the balanced sampler: balancing changes
    what the model is *shown*, and this shows what it then *does*, on a
    validation set that is never reweighted. Without it, "validation improved"
    cannot be told apart from "validation is now dominated by different
    categories", and the second is what a sampler change produces for free.
    """
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

    if categories and len(categories) == angle.numel():
        from .evaluation.metrics import group_means

        out["by_category"] = {
            name: {"geodesic_deg": value, "fragments": count}
            for name, (value, count) in group_means(
                angle.tolist(), list(categories)).items()
        }
    elif categories:
        # Length mismatch means the labels and the angles came from different
        # things, and a breakdown built on that is worse than none: it looks
        # right and attributes every number to the wrong category.
        out["by_category"] = {}
    return out


def run_epoch(model, loader, criterion, config, *, optimizer=None, scaler=None,
              device="cpu", step=0, total_steps=1, label="train",
              show_progress=True, deadline=None, stop_signal=None,
              on_checkpoint=None, distributed=False, on_prediction=None):
    """
    One pass over ``loader``. Training when ``optimizer`` is given, else eval.

    Returns ``(summary, step, stopped)``. ``summary`` carries every loss term
    separately as well as the total, because the terms have very different
    natural scales and the only way to learn the balance is to watch them move.

    How a step is built -- the same on one GPU or several
    ------------------------------------------------------
    Every batch is one *micro-batch*. Its gradient is computed on its own, with
    the gradient accumulated so far set aside (:func:`_train_micro_batch`), and
    merged in only if it came out usable. So anything can happen to one batch
    -- too many vertices, a non-finite loss or gradient, an out-of-memory error
    even after a retry with gradient checkpointing -- and it is simply absent
    from the step, while every other batch's contribution stands.

    Every ``accumulate`` batches, by *position* rather than by success, the step
    is finished (:func:`_finish_step`): the fragments that contributed are
    counted across every GPU, the summed gradient is all-reduced, divided by
    that count, clipped and applied. Because the step boundary depends only on
    the batch position, every GPU reaches every boundary -- whatever each one
    skipped -- and that is the only place the GPUs wait for each other. A
    trailing group cut short by the end of the loader is stepped rather than
    carried into the next epoch.

    Stopping (the time budget, or a SIGTERM) is voted at the same boundaries,
    so the GPUs stop together at one step instead of one leaving the others
    blocked in a collective.

    ``distributed=True`` requires a running process group; the summary is then
    gathered from every GPU, so it describes the whole epoch rather than one
    GPU's share of it.

    ``on_prediction(batch, prediction)`` is called in eval mode for every batch
    that was scored -- :func:`evaluate` uses it to assemble.
    """
    import torch

    from . import distributed as dist

    training = optimizer is not None
    model.train(training)
    parameters = [p for p in model.parameters() if p.requires_grad]
    world_rank = dist.rank() if distributed else 0
    prefix = f"[gpu{world_rank}] " if distributed else ""

    tally = _new_tally()
    # Fragments this GPU has contributed since the last optimizer step, and
    # how many batch positions that group has used.
    pending = 0.0
    in_group = 0
    bar = Progress(len(loader), prefix=f"  {label:5s}", enabled=show_progress)
    stopped = False

    interval = max(config.checkpoint_every_minutes, 0.0) * 60.0
    next_save = time.time() + interval if (training and on_checkpoint and interval) else None

    def want_stop() -> bool:
        if stop_signal is not None and stop_signal.triggered:
            return True
        return deadline is not None and time.time() > deadline

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, (batch, unusable) in enumerate(loader):
            tally["attempted"] += 1
            for item in unusable:
                tally["dropped"][item.key] = item.reason
                _note_failure(tally, item.key, item.reason)

            if training:
                # Position-based, like everything that decides a step: the
                # rate must be the same on every GPU, whatever each skipped.
                rate = learning_rate(step, total_steps, config)
                for group in optimizer.param_groups:
                    group["lr"] = rate

            report = None
            if batch is None or batch.num_fragments == 0:
                tally["skipped"] += 1
            else:
                _note_repairs(tally, batch)
                batch = _to_device(batch, device)
                vertices = int(batch.vertex_fragment.numel())
                names = _batch_name(batch, index)
                if config.max_vertices_per_batch and vertices > config.max_vertices_per_batch:
                    # Decided before the forward. It no longer has to be decided
                    # identically on every GPU -- nothing here waits on another
                    # GPU -- but attempting a batch known not to fit costs a
                    # forward and an allocator flush for nothing.
                    reason = (f"{vertices:,} vertices over the "
                              f"{config.max_vertices_per_batch:,} limit, "
                              f"{batch.num_fragments} fragments")
                    tally["oom"] += 1
                    tally["skipped"] += 1
                    tally["dropped"][f"toobig:{names}"] = reason
                    _note_batch_failure(tally, batch, "too many vertices")
                elif training:
                    contributed, report, outcome, detail = _train_micro_batch(
                        model, batch, criterion, config, scaler=scaler,
                        parameters=parameters, device=device)
                    pending += contributed
                    _count_outcome(tally, outcome, batch, names, vertices, detail,
                                   prefix, label, index)
                else:
                    holder = {} if on_prediction is not None else None
                    report, R, outcome, detail = _eval_batch(
                        model, batch, criterion, config, device=device, keep=holder)
                    _count_outcome(tally, outcome, batch, names, vertices, detail,
                                   prefix, label, index)
                    if R is not None and holder:
                        on_prediction(holder["batch"], holder["prediction"])
                    if R is not None:
                        tally["predictions"].append(R.detach().float().cpu())
                        tally["targets"].append(batch.target_rotation.detach().float().cpu())
                        # One category label per FRAGMENT, not per scene, so it
                        # lines up with the per-fragment angles `_metrics`
                        # produces. Taken from the batch rather than from the
                        # loader's position, because samples get dropped and
                        # position stops indexing the item list from the first
                        # drop onward.
                        if batch.categories:
                            scene_of = batch.fragment_scene.detach().cpu().tolist()
                            tally["categories"].extend(batch.categories[s] for s in scene_of)

            if report is not None:
                weight = int(batch.num_fragments)
                for name, value in report.items():
                    tally["sums"][name] = tally["sums"].get(name, 0.0) + value * weight
                tally["fragments"] += weight
                tally["batches"] += 1

            if training:
                step += 1
                in_group += 1
                if in_group == config.accumulate:
                    stop_vote = _finish_step(
                        optimizer, scaler, config, parameters, pending, tally,
                        distributed=distributed, device=device, want_stop=want_stop())
                    pending, in_group = 0.0, 0
                    now = time.time()
                    if next_save is not None and now > next_save and not stop_vote:
                        # Mid-epoch insurance, right after a step so no half
                        # accumulated gradient is in flight. Only rank 0
                        # writes; the callback knows which rank it is on.
                        on_checkpoint(step, completed=False)
                        next_save = now + interval
                    if stop_vote:
                        stopped = True
                        if show_progress:
                            reason = ("signal" if stop_signal is not None and
                                      stop_signal.triggered else "time budget")
                            print(f"\n  [stop] {reason} -- every GPU stops at this "
                                  f"step (batch {index + 1} of {len(loader)})")
                        bar.update(1)
                        break
            elif want_stop():
                # Validation holds no collective inside the loop, so a GPU may
                # stop early on its own; the results are gathered afterwards.
                stopped = True
                if show_progress:
                    reason = ("signal" if stop_signal is not None and
                              stop_signal.triggered else "time budget")
                    print(f"\n  [stop] {reason} during {label}; stopping cleanly")
                bar.update(1)
                break
            bar.update(1, f"loss {report['total']:.4f}" if report else "")

        if training and in_group:
            # Every GPU has the same number of batches, so every GPU is here
            # with the same partial group. Stepping it is what keeps its
            # gradient out of the next epoch's first step.
            _finish_step(optimizer, scaler, config, parameters, pending, tally,
                         distributed=distributed, device=device, want_stop=False)
    bar.close()

    if distributed:
        tally = _merge_tallies(dist.gather(tally))
    return _summarise_tally(tally), step, stopped


# --------------------------------------------------------------------------
# One micro-batch, one step
# --------------------------------------------------------------------------

def _is_oom(error: BaseException) -> bool:
    """A CUDA out-of-memory error, including the plain-RuntimeError forms
    cuBLAS and cuDNN raise when their workspace allocation fails."""
    import torch

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _release_cache(device) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


class _checkpointing:
    """
    Switch whole-layer gradient checkpointing on, for one retry.

    Outputs and gradients are identical -- only the peak memory and the time
    change -- so a scene that fits this way contributes exactly what it would
    have if it had fitted the first time.
    """

    def __init__(self, model, forced: bool):
        self.model, self.forced, self.saved = model, forced, None

    def __enter__(self):
        if self.forced:
            self.saved = self.model.grad_checkpointing
            self.model.grad_checkpointing = True
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.model.grad_checkpointing = self.saved
        self.saved = None
        return False


def _take_gradients(parameters) -> list:
    """Set the accumulated gradient aside, leaving ``.grad`` empty."""
    taken = [p.grad for p in parameters]
    for p in parameters:
        p.grad = None
    return taken


def _clear_gradients(parameters) -> None:
    for p in parameters:
        p.grad = None


def _restore_gradients(parameters, stash) -> None:
    for p, g in zip(parameters, stash):
        p.grad = g


def _merge_gradients(parameters, stash) -> None:
    """``.grad`` (this micro-batch) plus the stash (everything before it)."""
    for p, g in zip(parameters, stash):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)


def _gradients_finite(parameters) -> bool:
    import torch

    flags = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return bool(torch.stack(flags).all()) if flags else True


def _attempt(model, batch, criterion, config, scaler, weight, forced):
    """One forward + backward. Raises on out-of-memory; returns its outcome."""
    import torch

    with _checkpointing(model, forced):
        with torch.autocast(device_type=batch.vertex_fragment.device.type,
                            enabled=bool(scaler)):
            loss, report, _R = _forward(model, batch, criterion, config)
        value = float(loss.detach())
        # BEFORE the backward, not after it. A NaN that reaches `backward`
        # writes NaN into every gradient, and once Adam's moments hold NaN they
        # never recover: the run continues for hours producing nothing.
        if not math.isfinite(value):
            return "nonfinite-loss", report, value
        # Weighted by the micro-batch's fragment count, not divided by
        # `accumulate`. Every loss term is a *mean over fragments*, so
        # multiplying by the count turns it back into a sum; the step divides
        # the sum by the fragments that actually contributed, on every GPU.
        # That makes 2 passes of 1 scene the same gradient as one pass of
        # 2 scenes, which `loss / accumulate` is not -- it weights each
        # *scene* equally, and a scene holds 2 to 35 fragments. Measured on a
        # 2- and an 8-fragment scene, the two had cosine similarity 0.80.
        scaled = loss * weight
        (scaler.scale(scaled) if scaler else scaled).backward()
    return "ok", report, value


def _train_micro_batch(model, batch, criterion, config, *, scaler, parameters,
                       device):
    """
    Add one micro-batch's gradient to the step, or nothing at all.

    Returns ``(fragments contributed, report, outcome, detail)``. The gradient
    accumulated so far is taken out of ``.grad`` first and merged back at the
    end, so a failure part-way through a backward -- which leaves a partial
    gradient behind -- is discarded without touching what earlier micro-batches
    contributed.

    An out-of-memory error is retried once with gradient checkpointing forced
    on every layer, which trades time for memory and changes nothing else.
    Only if that also fails is the batch dropped -- and named, because the
    batches that run out of memory are the largest scenes, and dropping them
    silently biases training toward small objects.

    The retry is outside the ``except`` block on purpose: the exception's
    traceback pins every activation of the failed attempt, and retrying while
    it is alive would measure the retry against a card that is still full.
    """
    weight = float(batch.num_fragments)
    stash = _take_gradients(parameters)
    outcome, report, value = "oom", None, float("nan")
    for attempt in (0, 1):
        if attempt:
            _release_cache(device)
        try:
            outcome, report, value = _attempt(model, batch, criterion, config,
                                              scaler, weight, forced=attempt > 0)
        except Exception as error:                       # noqa: BLE001
            if not _is_oom(error):
                _restore_gradients(parameters, stash)
                raise
            outcome, report = "oom", None
            _clear_gradients(parameters)
            continue
        if attempt and outcome == "ok":
            outcome = "recovered"
        break

    if outcome in ("ok", "recovered") and scaler is None and not _gradients_finite(parameters):
        # A finite loss can still produce a non-finite gradient -- a norm of a
        # zero vector, a Gram-Schmidt step on parallel axes. Under AMP this is
        # the GradScaler's job (it skips the step and lowers the scale), so the
        # check is fp32 only.
        outcome = "nonfinite-grad"
        _clear_gradients(parameters)
    if outcome in ("ok", "recovered"):
        _merge_gradients(parameters, stash)
        return weight, report, outcome, value
    _restore_gradients(parameters, stash)
    # No report either: the epoch's loss average describes what was trained
    # on, and a dropped micro-batch was not.
    return 0.0, None, outcome, value


def _eval_batch(model, batch, criterion, config, *, device, keep=None):
    """
    One validation forward. An out-of-memory error skips the batch: under
    ``no_grad`` checkpointing saves nothing, so there is nothing to retry with.
    """
    try:
        if keep is None:
            loss, report, R = _forward(model, batch, criterion, config)
        else:
            loss, report, R = _forward(model, batch, criterion, config, keep=keep)
    except Exception as error:                           # noqa: BLE001
        if not _is_oom(error):
            raise
        outcome = "oom"
    else:
        value = float(loss.detach())
        if math.isfinite(value):
            return report, R, "ok", value
        return None, None, "nonfinite-loss", value
    _release_cache(device)
    return None, None, outcome, float("nan")


def _finish_step(optimizer, scaler, config, parameters, pending, tally, *,
                 distributed, device, want_stop):
    """
    Close one optimizer step on every GPU. Returns whether any GPU asked to stop.

    1. One all-reduce of two numbers: the fragments contributed on each GPU,
       and each GPU's stop vote.
    2. If anything contributed anywhere, one all-reduce of the summed gradient,
       then a division by the total fragment count -- so the step is the exact
       mean over every fragment that contributed on any GPU, and a GPU that
       dropped a batch shrinks the denominator rather than the step. The
       division happens after the reduction, on identical numbers, so every
       replica applies the identical update and they never drift apart.
    3. Clip, then step -- unless the reduced gradient is non-finite, which
       every GPU sees identically and so skips identically.

    A step with nothing in it anywhere does not call the optimizer: AdamW would
    still move the weights, on momentum and weight decay alone.
    """
    import torch

    from . import distributed as dist

    total, votes = (dist.sum_scalars([pending, 1.0 if want_stop else 0.0], device)
                    if distributed else (pending, 1.0 if want_stop else 0.0))
    if total > 0:
        if distributed:
            dist.sum_gradients(parameters)
        inverse = 1.0 / total
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(inverse)
        if scaler:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        norm = float(norm)
        if scaler:
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                tally["nonfinite_steps"] += 1      # the scaler skipped it
            else:
                tally["steps"] += 1
                tally["grad_norm"] += norm
        elif math.isfinite(norm):
            optimizer.step()
            tally["steps"] += 1
            tally["grad_norm"] += norm
        else:
            tally["nonfinite_steps"] += 1
    else:
        tally["empty_steps"] += 1
    optimizer.zero_grad(set_to_none=True)
    return votes > 0


# --------------------------------------------------------------------------
# What one pass measured, in a form that adds across GPUs
# --------------------------------------------------------------------------

def _new_tally() -> Dict:
    return {
        "sums": {}, "fragments": 0, "batches": 0, "attempted": 0,
        "skipped": 0, "oom": 0, "oom_recovered": 0, "nonfinite": 0,
        "steps": 0, "empty_steps": 0, "nonfinite_steps": 0, "grad_norm": 0.0,
        "repaired_faces": 0, "zero_normals": 0, "repaired": {},
        "dropped": {}, "failures": {},
        "predictions": [], "targets": [], "categories": [],
    }


# Counted once per optimizer step on EVERY GPU -- the steps are shared events,
# so summing them across GPUs would count each one `world` times.
_PER_STEP = ("steps", "empty_steps", "nonfinite_steps", "grad_norm")


def _merge_tallies(tallies: Sequence[Dict]) -> Dict:
    merged = _new_tally()
    for key in _PER_STEP:
        merged[key] = tallies[0][key] if tallies else merged[key]
    for tally in tallies:
        for key, value in tally.items():
            if key in _PER_STEP:
                continue
            if key == "sums":
                for name, total in value.items():
                    merged["sums"][name] = merged["sums"].get(name, 0.0) + total
            elif key in ("dropped", "repaired"):
                merged[key].update(value)
            elif key == "failures":
                for name, (count, reason) in value.items():
                    previous = merged["failures"].get(name, (0, reason))
                    merged["failures"][name] = (previous[0] + count, reason)
            elif isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key] += value
    return merged


def _summarise_tally(tally: Dict) -> Dict:
    import torch

    fragments = tally["fragments"]
    summary = {name: value / max(fragments, 1) for name, value in tally["sums"].items()}
    summary["batches"] = tally["batches"]
    summary["attempted"] = tally["attempted"]
    summary["fragments"] = fragments
    summary["skipped"] = tally["skipped"]
    summary["dropped"] = len(tally["dropped"])
    summary["oom"] = tally["oom"]
    summary["oom_recovered"] = tally["oom_recovered"]
    summary["nonfinite"] = tally["nonfinite"]
    summary["repaired_faces"] = tally["repaired_faces"]
    summary["zero_normals"] = tally["zero_normals"]
    if tally["steps"] or tally["empty_steps"] or tally["nonfinite_steps"]:
        summary["steps"] = tally["steps"]
        summary["empty_steps"] = tally["empty_steps"]
        summary["nonfinite_steps"] = tally["nonfinite_steps"]
        if tally["steps"]:
            summary["grad_norm"] = tally["grad_norm"] / tally["steps"]
    summary["dropped_names"] = sorted(tally["dropped"])[:20]
    summary["most_repaired"] = sorted(tally["repaired"].items(),
                                      key=lambda kv: (-kv[1], kv[0]))[:3]
    summary["failures"] = dict(tally["failures"])
    if tally["predictions"]:
        summary.update(_metrics(torch.cat(tally["predictions"]),
                                torch.cat(tally["targets"]),
                                categories=tally["categories"]))
    return summary


def _batch_name(batch, index: int) -> str:
    keys = [k for k in (batch.scene_keys or ()) if k]
    return "+".join(keys) if keys else f"batch{index}"


def _note_failure(tally: Dict, key: str, reason: str) -> None:
    count, _ = tally["failures"].get(key, (0, reason))
    tally["failures"][key] = (count + 1, reason)


def _note_batch_failure(tally: Dict, batch, reason: str) -> None:
    """
    Charge a failed batch to every scene in it.

    A batch of two that fails cannot say which scene did it -- but the culprit
    is in every batch it lands in, while its partners change from epoch to
    epoch, so across epochs the counts single it out.
    """
    for key in batch.scene_keys or ():
        if key:
            _note_failure(tally, key, reason)


def _note_repairs(tally: Dict, batch) -> None:
    for key, repairs in zip(batch.scene_keys or (), batch.repairs or ()):
        faces, normals = int(repairs[0]), int(repairs[1])
        tally["repaired_faces"] += faces
        tally["zero_normals"] += normals
        if faces or normals:
            tally["repaired"][key or "?"] = faces + normals


def _count_outcome(tally, outcome, batch, names, vertices, detail, prefix,
                   label, index) -> None:
    if outcome == "recovered":
        tally["oom_recovered"] += 1
        print(f"\n  {prefix}[oom] {label} batch {index} ({vertices:,} vertices) did "
              f"not fit; retried with gradient checkpointing and it did")
    elif outcome == "oom":
        tally["oom"] += 1
        tally["skipped"] += 1
        tally["dropped"][f"OOM:{names}"] = (
            f"{vertices:,} vertices, {batch.num_fragments} fragments")
        _note_batch_failure(tally, batch, "out of memory")
        print(f"\n  {prefix}[oom] {label} batch {index} ({vertices:,} vertices) did "
              f"not fit even with gradient checkpointing; skipped. "
              f"--max_vertices_per_batch {int(vertices * 0.9)} skips these up front.")
    elif outcome in ("nonfinite-loss", "nonfinite-grad"):
        what = "loss" if outcome == "nonfinite-loss" else "gradient"
        tally["nonfinite"] += 1
        tally["skipped"] += 1
        tally["dropped"][f"nonfinite:{names}"] = (
            f"{vertices:,} vertices, {batch.num_fragments} fragments, "
            + (f"loss={detail:.4g}" if what == "loss" else "non-finite gradient"))
        _note_batch_failure(tally, batch, f"non-finite {what}")
        print(f"\n  {prefix}[warn] non-finite {what} at {label} batch {index} "
              f"({names}); dropped from the step")


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
    line = (f"geo {summary['geodesic_deg']:6.2f}deg "
            f"(median {summary['geodesic_median_deg']:6.2f}, chance "
            f"{CHANCE['geodesic_deg']:.1f})  euler {summary['euler_rmse_deg']:6.2f}deg"
            f"  acc@10 {summary['acc@10deg']:.3f}")
    if "match@1" in summary:
        # The embedding head's only honest number. Its loss needs a per-batch
        # reference to interpret, and the term it replaced could be driven to
        # zero by a collapsed embedding; this cannot -- it is stage two's own
        # retrieval, from ~0 at chance to 1.0.
        line += f"  match@1 {summary['match@1']:.3f}"
    if "tilt_deg" in summary:
        # The split that makes ~90 deg readable. Printed every epoch rather
        # than only at the end, because which of the two regimes a run is in
        # can change during training and the final number cannot show that.
        line += (f"\n        tilt {summary['tilt_deg']:6.2f}  "
                 f"twist {summary['twist_deg']:6.2f}")
        if "head_cos" in summary:
            line += f"  head|cos| {summary['head_cos']:.3f}"
    return line


def check_initial_losses(summary: Dict[str, float]) -> List[str]:
    """
    Compare an untrained epoch against the values each term must read at
    chance. A term far from its reference is measuring something other than
    what its name says, and that is worth stopping for -- it will not show up
    later as anything but slow convergence.
    """
    complaints = []
    rotation_deg = summary.get("rotation_degrees")
    # The band is wide on purpose, and the reason is the equivariance property.
    # An untrained equivariant model does not sample the chance distribution: the
    # perturbation cancels, so this number *is* the angle of the untrained frame
    # on assembled fragments, which depends on the architecture. Measured over 12
    # seeds on a 24-fragment batch: 132 +- 12 for the old six-layer schedule,
    # 122 +- 15 for intra x5 + cross x3, 134 +- 14 with a trailing intra layer,
    # spanning 92-154 overall. A band tight around 126.5 would therefore flag
    # every architecture change as a defect.
    #
    # What this still catches is gross breakage -- a model that starts near 0 or
    # near 180 cannot be measuring the angle it claims to. It was never what
    # caught a transposed label: chance is chance in either direction, which is
    # why `test_model.py` asserts the round trip directly instead.
    if rotation_deg is not None and not 85.0 < rotation_deg < 170.0:
        complaints.append(
            f"rotation starts at {rotation_deg:.1f} deg, which is too far from "
            f"~{CHANCE['geodesic_deg']:.0f} to be an untrained frame"
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
                    completed: bool = True, elapsed: float = 0.0,
                    epoch_step: Optional[int] = None,
                    offenders: Optional[Dict] = None) -> None:
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

    ``epoch_step`` is the step count when ``epoch`` began. A restarted epoch
    goes back to it, so the learning-rate schedule of an interrupted run is the
    schedule of an uninterrupted one instead of running ahead by the part of the
    epoch that is about to be repeated. ``offenders`` is the run-long tally of
    scenes that failed, so a scene that fails in every session is still named
    as a repeat offender after a resume.
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
        "epoch_step": step if epoch_step is None else epoch_step,
        "offenders": offenders or {},
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
         "token_metric", "normalize_mode", "supervise_embedding", "subsets",
         # The split definition belongs here for the same reason and more
         # sharply: changing `split_by` mid-run swaps the validation set for a
         # different question, and changing `val_frac`, `split_seed` or
         # `mode_filter` reshuffles it. Nothing raises, the curve just steps,
         # and every number after the resume is about a set the earlier epochs
         # were partly trained on. `balance` is here too -- it changes what the
         # model is shown, which is a change of problem even though the data on
         # disk is identical.
         "split_by", "fracture_pool", "val_frac", "test_frac", "split_seed",
         "mode_filter", "official_subset", "balance", "balance_temperature",
         "max_objects")


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
                f"--checkpoint_dir somewhere fresh, or pass --strict_resume False to "
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
    a signal handler risks a half-written file, and with several GPUs it would
    desync them -- and the loop checks it at the next optimizer step, where the
    GPUs vote, so they all stop at the same step.
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
                 start_epoch: int, steps_per_epoch: int = 1) -> str:
    """
    How much longer, and how many more sessions.

    The number that actually decides whether a plan is workable, and it cannot
    be guessed before the first epoch is timed.
    """
    # Seconds *per step*, not per epoch. An epoch's duration is only comparable
    # to another epoch of the same size, and they are not always the same size:
    # a `--limit_train` calibration run resumed at the full dataset leaves three
    # 236-second epochs sitting in the history in front of three 3,555-second
    # ones. Averaging those gave "~22:09/epoch" for an epoch that took 59
    # minutes -- a 2.7x under-estimate, in the one number the session plan is
    # built on. Per-step is invariant to that, and to `--limit_train` changing
    # again on the next resume.
    rates = [h["seconds"] / h["steps"] for h in history
             if h.get("seconds") and h.get("steps")]
    if rates:
        recent = rates[-3:]
        per_epoch = (sum(recent) / len(recent)) * max(steps_per_epoch, 1)
    else:
        # A checkpoint written before `steps` was recorded. Fall back to the
        # most recent epoch alone rather than an average: one epoch of the
        # right size beats three of unknown sizes, which is the whole point.
        times = [h["seconds"] for h in history if h.get("seconds")]
        if not times:
            return ""
        per_epoch = times[-1]
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
        grad_checkpointing=config.grad_checkpointing,
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


def _steps_per_epoch(config: Config, items: int, world: int) -> int:
    """Batches per GPU per epoch: the fixed count, or one full pass."""
    if config.steps_per_epoch > 0:
        return int(config.steps_per_epoch)
    return max(math.ceil(items / (config.batch_size * max(world, 1))), 1)


def _loader(dataset, config: Config, shuffle: bool, rank: int, world: int,
            epoch: int, pairs_in_worker: bool = False):
    """
    Training (``shuffle=True``) or validation loader for one GPU.

    Training draws ``steps_per_epoch * batch_size`` scenes per GPU, or one full
    pass when it is 0; either way every GPU gets the same number, because every
    GPU must reach every optimizer step. Validation gives each GPU a disjoint,
    unpadded shard, so every scene is scored exactly once.

    ``pairs_in_worker`` builds the cross-fragment pair lists in the loader's
    worker processes instead of on the device -- the right call on a CPU run,
    the wrong one on a GPU (see ``_collate_samples``).
    """
    import functools

    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    from .data.sampling import EpochSampler, ShardSampler, WeightedDistributedSampler

    world = max(int(world), 1)
    rank = rank if world > 1 else 0
    sampler = None
    if shuffle:
        # Balancing applies to TRAINING only. A reweighted validation set would
        # move with the sampler setting, so two runs could not be compared on
        # it, and the quantity reported would no longer be "error on the val
        # split" but "error on the val split as this run happened to weight it".
        weights = dataset.sampling_weights()
        if config.steps_per_epoch > 0:
            sampler = EpochSampler(
                len(dataset), config.steps_per_epoch * config.batch_size,
                num_replicas=world, rank=rank, seed=config.seed, weights=weights,
            )
        elif weights is not None:
            sampler = WeightedDistributedSampler(
                weights, num_replicas=world, rank=rank, seed=config.seed)
        elif world > 1:
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                         shuffle=True, drop_last=False,
                                         seed=config.seed)
        if sampler is not None:
            sampler.set_epoch(epoch)
    elif world > 1:
        sampler = ShardSampler(len(dataset), num_replicas=world, rank=rank)
    return DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=config.workers,
        collate_fn=functools.partial(_collate_samples, pairs=pairs_in_worker),
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=config.workers > 0,
    )


def _launch_size(config: Config) -> int:
    """How many processes ``train`` starts: 0 = CPU inline, 1 = one GPU inline."""
    import torch

    visible = torch.cuda.device_count()
    if config.devices == 0:
        return 0
    if visible == 0:
        # Explicitly asked for several devices on a machine with none. Honoured
        # over gloo rather than silently downgraded -- a run that quietly used
        # one device when told to use two is a benchmark nobody can interpret.
        return config.devices if config.devices > 1 else 0
    return visible if config.devices < 0 else min(config.devices, visible)


def train(config: Config) -> List[dict]:
    """
    Run training: inline on one GPU (or the CPU), one process per GPU on more.

    Returns the history -- read back from ``out_dir`` when several processes
    ran, since only rank 0 records it.
    """
    requested = _launch_size(config)
    if 0 < requested < config.devices:
        # Said out loud: a run told to use two GPUs that quietly used one would
        # read as a two-GPU result.
        print(f"[setup] --num_gpus {config.devices} requested but only {requested} "
              f"GPU(s) visible -- using {requested}.")
    if requested > 1:
        import __main__
        import torch.multiprocessing as mp

        from .distributed import free_port

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
        # A fresh port per launch unless one is pinned, so two runs on one
        # machine -- or a run and the test suite -- cannot collide on it.
        os.environ.setdefault("MASTER_PORT", str(free_port()))
        context = mp.spawn(_worker, args=(requested, config), nprocs=requested,
                           join=False)
        with _ForwardTerminate(context.processes):
            while not context.join():
                pass
        path = Path(config.out_dir) / "history.json"
        return json.loads(path.read_text()) if path.exists() else []
    return _worker(0, requested, config)


class _ForwardTerminate:
    """
    Pass a SIGTERM sent to the launching process on to every GPU process.

    Under ``mp.spawn`` the launcher only waits; the GPU processes are its
    children. A SIGTERM aimed at the launcher -- which is what ``kill`` and most
    schedulers send -- would otherwise never reach them, and the launcher would
    die leaving them training into the void. Each GPU process catches it
    (:class:`StopSignal`), they vote to stop at the next step, rank 0
    checkpoints, and all of them exit. SIGINT is not forwarded: a terminal
    delivers Ctrl+C to the whole process group already.
    """

    def __init__(self, processes):
        self.processes = processes
        self.previous = None

    def __enter__(self):
        import signal

        try:
            self.previous = signal.signal(signal.SIGTERM, self._forward)
        except (ValueError, OSError):
            self.previous = None       # not the main thread: nothing to forward
        return self

    def _forward(self, signum, frame) -> None:
        for process in self.processes:
            if process.is_alive() and process.pid:
                try:
                    os.kill(process.pid, signum)
                except OSError:
                    pass

    def __exit__(self, *exc):
        import signal

        if self.previous is not None:
            signal.signal(signal.SIGTERM, self.previous)
        return False


def _worker(rank: int, world: int, config: Config) -> List[dict]:
    """
    One GPU's whole run. ``world`` processes run this at once; ``world <= 1``
    means this is the only one (``0``: on the CPU).

    Every rank does everything except the printing and the writing: it trains
    on its share of each epoch, validates its shard, and takes part in the
    per-step and per-epoch reductions, so every rank ends each epoch holding
    the same weights and the same summary.
    """
    import torch

    from . import distributed as dist

    cuda = torch.cuda.is_available()
    distributed = world > 1
    if distributed:
        device = dist.setup(rank, world, cuda)
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

    history: List[dict] = []
    start_epoch, step, best, elapsed = 0, 0, float("inf"), 0.0
    offenders: Dict[str, Dict] = {}
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    load_path = _resume_path(config, last_path)

    if config.resume and load_path is not None:
        state = torch.load(load_path, map_location=device, weights_only=False)
        _check_resume_compatible(config, state.get("config", {}), main)
        model.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        completed = state.get("completed", True)
        # An incomplete epoch is restarted, not skipped past -- and from the
        # step it began at, so the schedule does not run ahead by the part of
        # it that is about to be repeated.
        start_epoch = state["epoch"] + (1 if completed else 0)
        step = state["step"] if completed else state.get("epoch_step", state["step"])
        history = state.get("history", [])
        best = state.get("best", float("inf"))
        elapsed = float(state.get("elapsed", 0.0))
        offenders = dict(state.get("offenders") or {})
        _restore_rng(state, rank)
        if main:
            partial = "" if completed else " (epoch was partial, restarting it)"
            print(f"resumed from {load_path}")
            print(f"  epoch {start_epoch}, step {step}, "
                  f"{_hms(elapsed)} trained so far{partial}")
            if history:
                print(f"  best val geodesic so far: {best:.3f} deg")
        if main and load_path.resolve() != last_path.resolve():
            # Copy the resume point into *this* run's out_dir straight away.
            # Two failures otherwise break the chain: a session killed during
            # its first epoch leaves out_dir empty, and a session that finds
            # nothing left to do writes nothing at all -- so the next session,
            # resuming from this one's output, starts from scratch and silently
            # discards every hour spent so far. Rank 0 only: every rank used to
            # write the same file at once.
            save_checkpoint(last_path, model, optimizer, config,
                            state["epoch"], state["step"], history, best, scaler,
                            completed=completed, elapsed=elapsed,
                            epoch_step=state.get("epoch_step"),
                            offenders=offenders)
            print(f"  carried the resume point into {last_path}")
    elif main and config.resume:
        # Announced, because "starting from scratch" is one line in a long log
        # and costs a whole session when missed.
        looked = load_path or config.resume_from or last_path
        print(f"no checkpoint at {looked} -- starting from scratch")

    # Each rank initialised from its own seed. From here on the replicas stay
    # identical by construction; this is where they become identical.
    dist.broadcast_parameters(model)

    train_set = BreakingBadScenes(config, "train", epoch_seed=0)
    val_set = BreakingBadScenes(config, "val", epoch_seed=0)
    per_epoch = _steps_per_epoch(config, len(train_set), world)
    total_steps = per_epoch * config.epochs

    if main:
        print(_banner(config, train_set, val_set, parameters, device, world,
                      per_epoch))

    deadline = time.time() + config.max_hours * 3600
    stopped = val_stopped = out_of_time = False
    # On EVERY rank. A signal that reaches one GPU process and not another
    # used to kill the one without a handler, leaving its peer blocked in the
    # next all-reduce; now each rank only sets a flag and they vote.
    signal_watch = StopSignal().install()
    session_started = time.time()
    epoch_step = step

    def checkpoint(current_step: int, epoch_index: int, completed: bool) -> None:
        """Only rank 0 writes; every rank holds the same weights."""
        if not main:
            return
        save_checkpoint(last_path, model, optimizer, config, epoch_index,
                        current_step, history, best, scaler, completed=completed,
                        elapsed=elapsed + (time.time() - session_started),
                        epoch_step=epoch_step, offenders=offenders)

    for epoch in range(start_epoch, config.epochs):
        if stopped:
            break
        epoch_step = step
        train_set = BreakingBadScenes(config, "train", epoch_seed=epoch)
        on_cpu = not str(device).startswith("cuda")
        train_loader = _loader(train_set, config, True, rank, world, epoch,
                               pairs_in_worker=on_cpu)
        val_loader = _loader(val_set, config, False, rank, world, epoch,
                             pairs_in_worker=on_cpu)

        if main:
            print(f"\nepoch {epoch + 1}/{config.epochs}"
                  f"   lr {learning_rate(step, total_steps, config):.2e}")

        began = time.time()
        # No deadline inside an epoch: the time budget is read between epochs
        # (below), so an epoch that has started is trained to its last step and
        # validated in full. Only a signal -- the session being killed -- stops
        # one part-way, and then with a checkpoint.
        train_summary, step, stopped = run_epoch(
            model, train_loader, criterion, config, optimizer=optimizer,
            scaler=scaler, device=device, step=step, total_steps=total_steps,
            label="train", show_progress=main, stop_signal=signal_watch,
            on_checkpoint=lambda s_, completed: checkpoint(s_, epoch, completed),
            distributed=distributed,
        )
        val_summary, _, val_stopped = run_epoch(
            model, val_loader, criterion, config, device=device,
            label="val", show_progress=main,
            stop_signal=signal_watch,
            distributed=distributed,
        )
        # The budget, as Thesis 1 reads it: once it has run out, the epoch just
        # finished is the last of this session. Voted with any signal that cut
        # validation short, so every GPU leaves the loop at the same epoch.
        votes = dist.sum_scalars([float(val_stopped), float(time.time() >= deadline)],
                                 device)
        val_stopped, out_of_time = votes[0] > 0, votes[1] > 0
        new_failures = _record_offenders(offenders, epoch, train_summary, val_summary)
        repaired = {"train": train_summary.pop("most_repaired", []),
                    "val": val_summary.pop("most_repaired", [])}

        if epoch == start_epoch and main and config.check_init and not history:
            for complaint in check_initial_losses(train_summary):
                print(f"  [check] {complaint}")

        if not main:
            if stopped or val_stopped or out_of_time:
                break
            continue

        print(f"  train  {format_losses(train_summary)}")
        print(f"  val    {format_losses(val_summary)}")
        print(f"  val    {format_metrics(val_summary)}")
        _report_category_gap(val_summary, config)
        _report_dropped(train_summary, val_summary, repaired)
        _report_offenders(offenders, new_failures)

        train_summary.pop("dropped_names", None)
        val_summary.pop("dropped_names", None)
        row = {"epoch": epoch, "step": step,
               # An epoch cut short by a signal is recorded, because its
               # metrics are real measurements -- but flagged: in training the
               # next session re-runs that epoch and the same number appears
               # twice, in validation the numbers cover part of the split.
               # Plot `partial == 0` for a clean curve.
               "partial": int(stopped or val_stopped),
               "lr": learning_rate(step, total_steps, config),
               # Recorded so the time projection can work in seconds *per step*
               # and stay right when --limit_train changes between sessions.
               "steps": max(step - epoch_step, 1),
               "gpus": max(world, 1),
               "seconds": round(time.time() - began, 1)}
        row.update({f"train_{k}": v for k, v in train_summary.items()})
        row.update({f"val_{k}": v for k, v in val_summary.items()})
        history.append(row)
        write_history(out_dir, history)
        _write_offenders(out_dir, offenders)

        # The always-current checkpoint is written unconditionally and the
        # best-so-far separately. A policy-gated file alone can freeze while
        # training continues, and a resume then silently discards the gap.
        #
        # `best` is updated BEFORE last.pt is written. It used to be updated
        # after, so last.pt carried the previous best; a run resumed from it
        # then took the next epoch as a "new best" even when it was worse, and
        # overwrote best.pt with it.
        score = val_summary.get("geodesic_deg", val_summary["total"])
        # A validation cut short scored part of the split: not a best.
        improved = score < best and not val_stopped
        if improved:
            best = score
        if (stopped or val_stopped or out_of_time or (epoch + 1) % config.save_every == 0
                or epoch + 1 == config.epochs):
            checkpoint(step, epoch, completed=not stopped)
        if improved:
            save_checkpoint(best_path, model, optimizer, config, epoch, step,
                            history, best, scaler, completed=not stopped,
                            elapsed=elapsed + (time.time() - session_started),
                            epoch_step=epoch_step, offenders=offenders)
            print(f"  new best: {best:.3f} deg -> {best_path.name}")
        report = _time_report(history, config, elapsed + (time.time() - session_started),
                              start_epoch, steps_per_epoch=per_epoch)
        if report:
            print(report)
        if stopped or val_stopped or out_of_time:
            break

    signal_watch.restore()
    if main:
        finished = bool(history) and history[-1]["epoch"] + 1 >= config.epochs
        if signal_watch.triggered:
            print(f"\n[signal] stopped on {signal_watch.name}. Progress is in "
                  f"{last_path}.")
            print(_resume_recipe(config, out_dir))
        elif out_of_time and not finished:
            print(f"\n[time] the {config.max_hours:g} h budget ran out during epoch "
                  f"{history[-1]['epoch'] + 1}, which was finished, validated and "
                  f"saved. Progress is in {last_path}.")
            print(_resume_recipe(config, out_dir))
    if main and history:
        _final_report(history)
    dist.teardown()
    return history


def _record_offenders(offenders: Dict[str, Dict], epoch: int, *summaries) -> int:
    """
    Fold this epoch's failed scenes into the run-long tally; return how many
    scenes failed this epoch.

    One failure is noise -- a batch that happened to be too large together.
    The same scene failing in epoch after epoch is a scene that has been removed
    from training without anyone deciding to, and only a tally that outlives
    the epoch, and the session, can see that.
    """
    failed = 0
    for summary in summaries:
        for key, (count, reason) in (summary.pop("failures", None) or {}).items():
            entry = offenders.setdefault(key, {"count": 0, "epochs": [], "reason": reason})
            entry["count"] += int(count)
            entry["reason"] = reason
            if epoch not in entry["epochs"]:
                entry["epochs"].append(epoch)
            failed += 1
    return failed


def _repeat_offenders(offenders: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """Scenes that failed in more than one epoch, worst first."""
    repeat = [(key, entry) for key, entry in offenders.items()
              if len(entry.get("epochs", ())) > 1]
    return sorted(repeat, key=lambda kv: (-len(kv[1]["epochs"]), -kv[1]["count"], kv[0]))


def _report_offenders(offenders: Dict[str, Dict], new_failures: int) -> None:
    if not new_failures:
        return
    repeat = _repeat_offenders(offenders)
    if not repeat:
        return
    print(f"  [offenders] {len(repeat)} scene(s) have failed in more than one "
          f"epoch -- each is effectively excluded from training:")
    for key, entry in repeat[:5]:
        print(f"     {len(entry['epochs'])} epochs, {entry['count']}x  {key}  "
              f"({entry['reason']})")
    if len(repeat) > 5:
        print(f"     (+{len(repeat) - 5} more in offenders.json)")
    print("     diagnose one with: python -m scripts.check_scene --scene <key> --locate")


def _write_offenders(out_dir: Path, offenders: Dict[str, Dict]) -> None:
    if not offenders:
        return
    ordered = dict(sorted(offenders.items(),
                          key=lambda kv: (-len(kv[1]["epochs"]), -kv[1]["count"], kv[0])))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offenders.json").write_text(json.dumps(ordered, indent=2))


def _split_banner(config, train_set, val_set) -> List[str]:
    """
    What is actually held out, in words, at the top of every run.

    "held out: whole shapes" and "held out: break patterns" are two different
    experiments producing two numbers with the same name, and a config key
    three scrolls up is not enough to tell them apart six weeks later when the
    log is all that survives.
    """
    train_objects = {entry.key for entry in train_set.scenes}
    val_objects = {entry.key for entry in val_set.scenes}
    if config.split_by == "object":
        return [f"  held out      whole SHAPES -- {len(val_objects)} val objects "
                f"never seen in training"]
    shared = len(train_objects & val_objects)
    return [
        f"  held out      BREAK PATTERNS -- {shared}/{len(val_objects)} val objects "
        f"are also in train",
        f"                measures unseen fractures of KNOWN shapes; easier than "
        f"the benchmark,",
        f"                so do not report it as an object-split result",
    ]


def _balance_banner(config, train_set) -> List[str]:
    """
    The projected sampling shares, and what balancing costs in variance.

    A balancing setting that is silently a no-op -- or that over-corrects --
    would otherwise be invisible for a whole run, and the only symptom would be
    a validation number that moved for a reason nobody recorded.
    """
    from .data.catalog import effective_sample_size
    from .evaluation.metrics import group_means

    if config.balance == "none":
        return []
    weights = train_set.sampling_weights()
    if weights is None:
        return ["  balance       requested but inactive (temperature 0)"]

    categories = train_set.categories()
    total = sum(weights)
    shares = group_means([w / total for w in weights], categories)
    summed = {name: value * count for name, (value, count) in shares.items()}
    ordered = sorted(summed.items(), key=lambda kv: -kv[1])
    head = "  ".join(f"{name}:{100 * value:.1f}%" for name, value in ordered[:5])
    more = f"  (+{len(ordered) - 5} more)" if len(ordered) > 5 else ""
    ess = effective_sample_size(weights)
    return [
        f"  balance       {config.balance} @ T={config.balance_temperature:g}"
        f"   ({len(ordered)} categories)",
        f"                {head}{more}",
        f"                spread {ordered[0][1] / max(ordered[-1][1], 1e-12):.1f}x"
        f"   effective samples {ess:,.0f} of {len(weights):,}"
        f"  ({100 * ess / max(len(weights), 1):.0f}%)",
        f"                training only -- validation is never reweighted",
    ]


def _epoch_lines(config, items: int, world: int, per_epoch: int) -> List[str]:
    """What one epoch is: optimizer steps, scenes, and passes over the split."""
    gpus = max(world, 1)
    scenes = per_epoch * config.batch_size * gpus
    passes = scenes / max(items, 1)
    steps = math.ceil(per_epoch / max(config.accumulate, 1))
    kind = "fixed length" if config.steps_per_epoch > 0 else "one full pass"
    lines = [f"  epoch         {steps} steps/GPU x batch "
             f"{config.batch_size * config.accumulate} x {gpus} GPU(s) = {scenes:,} "
             f"scenes = {passes:.2f} passes ({kind})",
             f"                {config.batch_size} scene(s) per forward pass "
             f"(--micro_batch_scenes), gradient checkpointing "
             f"{'on' if config.grad_checkpointing else 'off'}"]
    if config.steps_per_epoch > 0 and passes > 4:
        lines.append(f"                every scene is drawn ~{passes:.0f}x per epoch -- "
                     f"for a smoke test use --steps_per_epoch 0 (one pass)")
    return lines


def _banner(config, train_set, val_set, parameters, device, world, per_epoch) -> str:
    lines = [
        "=" * 74,
        "V-GAT reassembly training",
        "=" * 74,
        f"  device        {device}" + (f"  x{world} (one process per GPU)" if world > 1 else ""),
        f"  parameters    {parameters:,}",
        f"  channels      {config.channels}  heads {config.heads}  "
        f"schedule {'+'.join(config.schedule)}",
        f"  objects       {len(train_set.scenes)} train / {len(val_set.scenes)} val"
        + ("  (official split)" if train_set.official else "  (hashed split)"),
        f"  samples       {len(train_set)} train / {len(val_set)} val",
        *_epoch_lines(config, len(train_set), world, per_epoch),
        *_split_banner(config, train_set, val_set),
        *_balance_banner(config, train_set),
        f"  labels        {config.label_method}"
        + (f" @ {config.sharp_threshold}" if config.label_method == "dihedral" else ""),
        f"  tokens        {config.tokens_per_scene}/scene, {config.token_mode}"
        f" by {config.token_metric} distance",
        f"  precision     {'AMP' if config.amp else 'fp32'}",
        (f"  lr            {config.lr:.2e} -> {config.lr * config.min_lr_fraction:.2e}"
         if config.lr_schedule == "cosine" else f"  lr            {config.lr:.2e}")
        + f"  ({config.lr_schedule}, warmup {config.warmup_fraction * config.epochs:g}"
          f" epoch(s))",
        "-" * 74,
        "  read every number against chance, not against zero:",
        f"    geodesic      {CHANCE['geodesic_deg']:.2f} deg   <- a model here has learned nothing",
        f"    euler RMSE    {CHANCE['euler_rmse_deg']:.2f} deg   <- residual convention; "
        f"collapsing to identity scores the same",
        f"    axis only     {CHANCE['axis_only_deg']:.1f} deg   <- axis right, rotation about it not"
        f"  (tilt/twist tells them apart)",
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
        tilt = last.get("val_tilt_deg")
        twist = last.get("val_twist_deg")
        print("  [verdict] parked near the axis-only landmark. Read tilt/twist"
              " before concluding anything:")
        if tilt is not None and twist is not None:
            print(f"            tilt {tilt:.1f} deg, twist {twist:.1f} deg.")
            if tilt < 20 and twist > 70:
                print("            The axis IS being recovered and the rotation about it")
                print("            is not. That is a real finding about this geometry --")
                print("            but it is not a hard floor: a fragment's fracture")
                print("            boundary is unique even when the whole object is a")
                print("            surface of revolution, so the information is there.")
            elif tilt > 70:
                print("            The axis is not being recovered either, so this is an")
                print("            optimisation problem rather than a symmetry one.")
        else:
            print("            (tilt/twist not recorded for this run)")
    tail = [h.get("val_geodesic_deg") for h in history[-max(len(history) // 4, 2):]]
    tail = [t for t in tail if t is not None]
    if len(tail) >= 2 and tail[0] - tail[-1] > 0.5:
        print("  [verdict] still descending at the last epoch -- this is a lower")
        print("            bound on progress, not a converged result.")
    print(f"  last epoch: {last.get('seconds', 0):.0f}s"
          f"   lr {last.get('lr', 0):.2e}")
    print("=" * 74)


def evaluate(config: Config, checkpoint: str = "best.pt",
             split: str = "test", assemble: bool = True,
             collision: bool = False,
             data_from_checkpoint: bool = True) -> Dict[str, float]:
    """
    Score a saved checkpoint on a held-out split: rotation, and -- with
    ``assemble`` -- the full assembly the benchmark scores.

    ``assemble`` runs the translation solver on the predicted rotations
    (:mod:`reassembly.assembly`) and reports translation RMSE, Chamfer distance
    and part accuracy in world units, averaged per scene then over scenes as
    the benchmark does. Without it, only rotation is reported -- and said to be,
    rather than the other numbers being silently absent.

    The model, and by default the DATA definition, come from the checkpoint's
    own config: a model is scored on the split, labels, tokens and
    normalisation it was trained with, whatever the flags say, and every
    setting that differed is printed. ``data_from_checkpoint=False`` keeps the
    flags' data settings (to score a fracture-split model on the object split,
    say); the architecture is always the checkpoint's.

    Single-device on purpose: an evaluation that shards across GPUs has to
    gather predictions to be correct, and getting that subtly wrong produces a
    plausible number.
    """
    import torch

    from .assembly import mean_over_scenes, score_batch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, config, state, path = load_checkpoint(
        checkpoint, config, device, data_from_checkpoint=data_from_checkpoint)

    dataset = BreakingBadScenes(config, split, epoch_seed=0)
    loader = _loader(dataset, config, False, 0, 1, 0,
                     pairs_in_worker=not device.startswith("cuda"))
    scenes: List[Dict] = []

    def collect(batch, prediction) -> None:
        scores = score_batch(batch, prediction, collision=collision)
        for key, category, score in zip(batch.scene_keys, batch.categories, scores):
            score["_scene"], score["_category"] = key, category
            scenes.append(score)

    summary, _, _ = run_epoch(model, loader, build_criterion(config), config,
                              device=device, label=split,
                              on_prediction=collect if assemble else None)

    print(f"\n{split} ({len(dataset)} samples, checkpoint {path.name} from epoch "
          f"{state['epoch'] + 1})")
    print(f"  {format_losses(summary)}")
    print(f"  {format_metrics(summary)}")
    breakdown = summary.get("by_category")
    if breakdown:
        from .evaluation.metrics import format_group_table

        print(format_group_table(
            {name: (values["geodesic_deg"], values["fragments"])
             for name, values in breakdown.items()},
            "geodesic error by category"))
        print("  A large spread means the headline mean is partly a statement "
              "about which\n  categories are numerous. Symmetric categories "
              "scoring worse than asymmetric\n  ones is evidence for the "
              "azimuth-ambiguity reading; check tilt/twist to confirm.")
    print(f"  geodesic is the primary number; Euler RMSE (residual convention, "
          f"chance {CHANCE['euler_rmse_deg']:.1f} deg)\n  is for comparability "
          f"with GARF's tables. Compare against the VANILLA Everyday "
          f"supplementary\n  table -- SE(3)-Equiv 79.30 deg, GARF-mini 10.41 deg "
          f"-- not the headline row.")

    if assemble:
        public = [{k: v for k, v in scene.items() if not k.startswith("_")}
                  for scene in scenes]
        assembly = mean_over_scenes(public)
        by_category: Dict[str, Dict[str, float]] = {}
        for category in sorted({scene["_category"] for scene in scenes}):
            rows = [p for p, scene in zip(public, scenes) if scene["_category"] == category]
            by_category[category] = dict(mean_over_scenes(rows), scenes=len(rows))
        summary["assembly"] = assembly
        summary["assembly_by_category"] = by_category
        summary["assembly_scenes"] = [dict(scene) for scene in scenes]
        _print_assembly(assembly, by_category, len(scenes))
    else:
        print("  (rotation only: --no_assemble skipped the translation solver, so "
              "there is no RMSE(T), Chamfer or part accuracy)")

    summary.pop("failures", None)
    summary.pop("most_repaired", None)
    out = Path(config.out_dir) / f"{split}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def load_checkpoint(checkpoint, config: Optional[Config] = None, device="cpu",
                    data_from_checkpoint: bool = True):
    """
    ``(model, config, state, path)`` for a saved checkpoint, the model in eval
    mode on ``device``.

    ``checkpoint`` is a path, or a file name inside ``config.out_dir``. The
    architecture is always the checkpoint's; the data definition too unless
    ``data_from_checkpoint=False``. Every setting that differs from ``config``
    is printed, because a model scored or inspected on data it was not trained
    for gives a number that looks fine and means something else.
    """
    import torch

    config = config if config is not None else Config()
    path = Path(checkpoint)
    if not path.exists():
        path = Path(config.out_dir) / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint} or {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    config = _adopt_checkpoint_settings(config, state.get("config") or {},
                                        data=data_from_checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config, state, path


def find_scene(config: Config, key: str, splits=("train", "val", "test")):
    """
    ``(dataset, index, split)`` for the scene named ``key`` --
    ``<object key>/<mode>``, as training logs, ``offenders.json`` and
    ``dump_prediction --list`` print it. Every break pattern is searched, not
    only the ones an epoch samples. Raises ``KeyError`` with the nearest names
    when there is no such scene.
    """
    searched = dataclasses.replace(config, modes_per_scene=None, limit_train=None,
                                   limit_val=None, max_objects=None)
    names: List[str] = []
    for split in splits:
        try:
            dataset = BreakingBadScenes(searched, split, epoch_seed=0)
        except FileNotFoundError:
            continue
        index = dataset.index_of(key)
        if index is not None:
            return dataset, index, split
        names += [dataset.key(i) for i in range(len(dataset))]
    import difflib

    close = difflib.get_close_matches(key, names, n=3, cutoff=0.3)
    raise KeyError(f"no scene {key!r} under {config.root}"
                   + (f"; closest: {', '.join(close)}" if close else ""))


def _adopt_checkpoint_settings(config: Config, stored: Dict, data: bool) -> Config:
    """
    The checkpoint's architecture (always) and data definition (by default),
    with every change announced.
    """
    import dataclasses

    names = {field.name for field in dataclasses.fields(Config)}
    keys = _ARCHITECTURE + (_DATA if data else ())
    changes = {key: stored[key] for key in keys
               if key in stored and key in names and _differs(stored[key], getattr(config, key))}
    for key, value in changes.items():
        print(f"[checkpoint] using the checkpoint's {flag(key)} {value!r} "
              f"(the flags said {getattr(config, key)!r})")
    if "schedule" in changes:
        changes["schedule"] = tuple(changes["schedule"])
    return dataclasses.replace(config, **changes) if changes else config


def _print_assembly(assembly: Dict[str, float], by_category: Dict[str, Dict],
                    scenes: int) -> None:
    if not assembly:
        print("  assembly: no scene could be scored")
        return
    print(f"\n  assembly ({scenes} scenes, translation solver on the predicted "
          f"rotations, world units, per-scene means):")
    print(f"    RMSE(T)        {assembly.get('rmse_t', float('nan')):.4f}")
    print(f"    Chamfer (CD)   {assembly.get('chamfer', float('nan')):.5f}   "
          f"(whole shape; per part {assembly.get('part_chamfer', float('nan')):.5f})")
    print(f"    part accuracy  {assembly.get('part_accuracy', float('nan')):.3f}   "
          f"(Chamfer < 0.01 per fragment)")
    print(f"    geodesic       {assembly.get('geodesic_deg', float('nan')):.2f} deg   "
          f"Euler RMSE {assembly.get('euler_rmse_deg', float('nan')):.2f} deg   "
          f"(per scene, as the benchmark averages)")
    print(f"    matches/scene  {assembly.get('matches', float('nan')):.0f}")
    if len(by_category) > 1:
        print(f"    {'category':<20}{'scenes':>7}{'PA':>8}{'RMSE(T)':>10}{'CD':>10}")
        for name, row in sorted(by_category.items(),
                                key=lambda kv: -kv[1].get("part_accuracy", 0.0)):
            print(f"    {name:<20}{row['scenes']:>7}{row.get('part_accuracy', float('nan')):>8.3f}"
                  f"{row.get('rmse_t', float('nan')):>10.4f}{row.get('chamfer', float('nan')):>10.5f}")


def _report_category_gap(summary: Dict, config: Config, spread: float = 15.0) -> None:
    """
    One line, only when the categories actually disagree.

    A full table every epoch is noise; silence when the spread is real is
    worse. The threshold is on the gap between the worst and best category, in
    degrees, because that is the quantity that decides whether the mean is
    telling the truth about the model or about the category mix. When
    balancing is off and the gap is large, the mean is partly a statement
    about which categories happen to be numerous.
    """
    breakdown = summary.get("by_category")
    if not breakdown or len(breakdown) < 2:
        return
    ordered = sorted(breakdown.items(), key=lambda kv: -kv[1]["geodesic_deg"])
    worst_name, worst = ordered[0]
    best_name, best = ordered[-1]
    gap = worst["geodesic_deg"] - best["geodesic_deg"]
    if gap < spread:
        return
    print(f"  val    by category: worst {worst_name} {worst['geodesic_deg']:.1f}deg "
          f"(n={worst['fragments']}), best {best_name} {best['geodesic_deg']:.1f}deg "
          f"(n={best['fragments']}), spread {gap:.1f}deg")
    if config.balance == "none":
        print(f"         balance=none, so the mean above is weighted by how many "
              f"shapes each category happens to have.")


def _report_dropped(train_summary: Dict, val_summary: Dict,
                    repaired: Optional[Dict[str, list]] = None) -> None:
    """
    Name what was dropped, not just how much.

    A sample that fails every epoch has been removed from the dataset. Counting
    batches hides it -- a batch of four with one unusable item still collates
    and reports nothing -- and even a count hides *which* items, which is what
    turns "3 dropped" into a diagnosable fact.
    """
    repaired = repaired or {}
    for label, summary in (("train", train_summary), ("val", val_summary)):
        names = summary.get("dropped_names") or []
        if summary.get("dropped"):
            shown = ", ".join(names[:3])
            more = f" (+{summary['dropped'] - len(names[:3])} more)" if summary["dropped"] > 3 else ""
            print(f"  {label}: {summary['dropped']} sample(s)/batch(es) unusable: {shown}{more}")
        if summary.get("skipped"):
            print(f"  {label}: {summary['skipped']} of {summary.get('attempted', '?')} "
                  f"batch(es) contributed nothing")
        if summary.get("oom_recovered"):
            print(f"  {label}: {summary['oom_recovered']} batch(es) ran out of memory "
                  f"and fitted on the retry with gradient checkpointing -- kept, "
                  f"at the cost of a second forward.")
        if summary.get("oom"):
            print(f"  {label}: {summary['oom']} batch(es) did not fit even with "
                  f"gradient checkpointing and were skipped. A handful is "
                  f"survivable; more than that means the largest objects are being "
                  f"dropped from training: lower --micro_batch_scenes (1 is the "
                  f"floor), or --hidden_channels.")
        if summary.get("nonfinite"):
            print(f"  {label}: {summary['nonfinite']} batch(es) produced a "
                  f"non-finite loss or gradient and were dropped from the step. "
                  f"This is a defect, not a capacity limit -- run "
                  f"`python -m scripts.check_scene --scene <name> --locate` on "
                  f"the named samples. Repeat offenders are effectively excluded "
                  f"from training.")
        if summary.get("nonfinite_steps"):
            print(f"  {label}: {summary['nonfinite_steps']} optimizer step(s) had a "
                  f"non-finite gradient after averaging and were not applied.")
        if summary.get("repaired_faces") or summary.get("zero_normals"):
            worst = ", ".join(f"{key} ({count})" for key, count in repaired.get(label, [])[:3])
            print(f"  {label}: repaired {summary.get('repaired_faces', 0):,} zero-area "
                  f"face(s) and {summary.get('zero_normals', 0):,} zero-length vertex "
                  f"normal(s) -- set to 0, not NaN" + (f"; most in {worst}" if worst else ""))


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
        return (f"  To continue: rerun the same command with --resume auto (the "
                f"default) -- not --resume none, which starts over. It loads "
                f"{out_dir / 'last.pt'} and picks up where it stopped.")
    return (
        "  To continue in the next Kaggle session:\n"
        "    1. Save this notebook's version so /kaggle/working becomes an output.\n"
        "    2. In the new session, add that output as an input dataset.\n"
        "    3. Rerun the same command, pointed at it:\n"
        "         --resume /kaggle/input/<the-output-name> "
        "--checkpoint_dir /kaggle/working/vgat\n"
        "  --checkpoint_dir must stay under /kaggle/working -- /kaggle/input is\n"
        "  read-only, and a run that cannot write its checkpoint discovers that\n"
        "  hours in."
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
    # With several GPUs the worker count is per *process*, so the machine runs
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
            f"--num_workers {max((cpus - world) // world, 1)}."
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
            print("\n  The dataset was found, so --root_dir is right. Either the")
            print("  official split lists no object present here, or there are")
            print("  too few objects for the hashed fallback to fill every split.")
        else:
            print("\n  --root_dir must point at the directory *containing* the subset")
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

    train_objects = {entry.key for entry in train_set.scenes}
    val_objects = {entry.key for entry in val_set.scenes}
    overlap = train_objects & val_objects
    if config.split_by == "object":
        print(f"  held out: whole SHAPES ({len(val_objects)} val objects unseen "
              f"in training)")
        if overlap:
            problems.append(f"{len(overlap)} object(s) appear in BOTH train and "
                            f"val -- validation would be measuring memorisation")
    else:
        # Here the overlap is the design, so the thing to verify is the other
        # half: that no BREAK PATTERN is shared. Objects in val but not in
        # train would mean the pool was built wrong.
        shared_modes = set(train_set.items) & set(val_set.items)
        print(f"  held out: BREAK PATTERNS ({len(overlap)} of {len(val_objects)} "
              f"val objects also in train, by design)")
        print(f"            this measures unseen fractures of KNOWN shapes -- "
              f"an easier question\n            than the benchmark's; do not "
              f"report it as an object-split result.")
        if shared_modes:
            problems.append(
                f"{len(shared_modes)} (object, mode) pair(s) appear in both "
                f"train and val -- the fracture split is not disjoint")
        if val_objects - train_objects:
            warnings.append(
                f"{len(val_objects - train_objects)} val object(s) are absent "
                f"from train, so part of this validation set is still an "
                f"unseen-shape test and the two effects are mixed")

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
        vertices = sum(len(f.target_vertices) for f in item.fragments)
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

    # How far cross-fragment information can travel before the head pools.
    #
    # A cross layer writes only to token vertices, and the head takes a mean
    # over *every* vertex -- so a vertex the tokens never reach contributes to
    # the rotation without having heard from another fragment. Each intra layer
    # *after* the last cross layer buys one hop along mesh edges. This measures
    # the reach on the real meshes rather than assuming it, because the answer
    # depends on how the fracture surface is shaped and cannot be guessed.
    trailing = 0
    for kind in reversed(list(config.schedule)):
        if kind != "intra":
            break
        trailing += 1
    reached = _token_reach(built, max(trailing, 1) + 1)
    token_share = 100 * tokens.sum() / max(vertices.sum(), 1)
    spread = "  ".join(f"+{h}: {100 * r:.0f}%" for h, r in enumerate(reached, start=1))
    print(f"  cross-fragment reach   tokens alone: {token_share:.0f}%   {spread}")
    if trailing:
        print(f"    {trailing} intra layer(s) follow the last cross layer, so "
              f"{100 * reached[trailing - 1]:.0f}% of vertices reach the head "
              f"having heard from another fragment")
    else:
        print("    no intra layer follows the last cross layer")
    if trailing == 0:
        warnings.append(
            f"no intra layer follows the last cross layer, so only the "
            f"{100 * tokens.sum() / vertices.sum():.0f}% of vertices that are "
            f"tokens carry cross-fragment information into the pooled rotation. "
            f"Append 'intra' to --schedule to propagate it."
        )
    elif reached[trailing - 1] < 0.5:
        warnings.append(
            f"after the last cross layer only {100 * reached[trailing - 1]:.0f}% "
            f"of vertices are reached, and the head pools over all of them. One "
            f"more trailing intra layer would reach "
            f"{100 * reached[min(trailing, len(reached) - 1)]:.0f}%."
        )
    if unusable:
        print(f"  {len(unusable)} unusable: "
              f"{', '.join(u.key for u in unusable[:3])}")
    if tokens.max() == 0:
        problems.append("every scene produced ZERO cross-fragment tokens -- the "
                        "fracture mask is empty, so the cross layers do nothing. "
                        "Check --label_method and --sharp_threshold.")
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
            print(f"  micro_batch_scenes={candidate}: out of memory")
            continue
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.startswith("cuda") else 0.0)
        fitted = candidate
        print(f"  micro_batch_scenes={candidate}: fits"
              + (f", peak {peak:.2f} GB of {total_memory:.1f} GB "
                 f"({100 * peak / total_memory:.0f}%)" if peak else ""))
        break

    if fitted == 0:
        problems.append(
            f"out of memory even at micro_batch_scenes=1 on a "
            f"{total_memory:.0f} GB card. Try, in this order: "
            + ("" if config.grad_checkpointing else
               "--grad_checkpointing True (recomputes every layer in the backward "
               "pass, the largest saving), then ")
            + f"a smaller --hidden_channels (every edge activation scales with "
            f"it), then --tokens_per_scene 1024 (the cross-attention cost is "
            f"quadratic in this)."
        )
        _summarise(problems, warnings)
        return False
    if fitted < config.batch_size:
        problems.append(
            f"micro_batch_scenes={config.batch_size} does not fit; {fitted} does. "
            f"Use --micro_batch_scenes {fitted}: --batch_size stays the scenes per "
            f"optimizer step, and the step is identical on all four geometric "
            f"terms -- passes are weighted by fragment count -- at a little "
            f"speed. The contrastive term differs by construction: its negatives "
            f"come from the pass, so smaller passes give it within-scene "
            f"negatives only, which is the confusion set matching actually faces."
        )
    if peak and peak > 0.65 * total_memory:
        # Only the pass size is suggested: --batch_size, the scenes per
        # optimizer step, stays, so the suggestion changes the memory and not
        # the optimisation.
        safer = max(fitted // 2, 1)
        warnings.append(
            f"peak memory is {100 * peak / total_memory:.0f}% of the card on the "
            f"largest of {samples} sampled scenes -- and the dataset's largest "
            f"single fragment is 83,039 vertices, several times anything sampled "
            f"here. Training catches an OOM and skips the batch and reports the "
            f"count; if that count is more than a percent or so of an epoch, the "
            f"largest objects are being dropped from training, which biases the "
            f"result. Then use "
            + (f"--micro_batch_scenes {safer}" if safer < fitted else "")
            + (" or " if safer < fitted and not config.grad_checkpointing else "")
            + ("--grad_checkpointing True" if not config.grad_checkpointing else "")
            + (f"; the optimizer step keeps its "
               f"{config.batch_size * config.accumulate} scenes per GPU."
               if safer < fitted else ".")
        )

    parameters = sum(p.numel() for p in model.parameters())
    print(f"  parameters {parameters:,}   worst case tested: "
          f"{fitted} x the largest sampled scene")
    # The vertex count that was *proved* to fit, which is the number the
    # training loop needs to skip the batches that will not. Reported rather
    # than inferred, because the alternative is discovering it as an
    # out-of-memory error hours into a session -- which is how this got here.
    fitted_vertices = int(vertices.max()) * fitted
    suggested = int(fitted_vertices * 0.95)
    print(f"  largest batch proved to fit: {fitted_vertices:,} vertices"
          + (f"   -- pass --max_vertices_per_batch {suggested}"
             if not config.max_vertices_per_batch else ""))
    if not config.max_vertices_per_batch:
        # A note, no longer a warning. A batch that does not fit is retried
        # with gradient checkpointing and skipped only if that fails too, on
        # one GPU or several -- the run survives it either way. The limit just
        # saves the two doomed attempts.
        print(f"  (a larger batch that runs out of memory is retried with "
              f"gradient checkpointing, then skipped; --max_vertices_per_batch "
              f"{suggested} skips such batches before attempting them)")
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
          f"micro_batch_scenes={timing_batch}"
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
            f"micro_batch_scenes={timing_batch}, even though {fitted} copies of "
            f"the largest sampled scene fit. Real scenes are bigger than the "
            f"sample; lower --micro_batch_scenes or --tokens_per_scene, or turn "
            f"on --grad_checkpointing True."
        )
        _summarise(problems, warnings)
        return False
    if skipped:
        warnings.append(
            f"{skipped} of the first {done + skipped} batches ran out of memory "
            f"and were skipped. Training survives this by design, but that rate "
            f"means the largest objects are being dropped from training -- lower "
            f"--micro_batch_scenes (the step keeps its --batch_size scenes) or "
            f"turn on --grad_checkpointing True."
        )
    per_step = (time.time() - began) / max(done, 1)

    # An epoch is train *and* validation, and leaving the second one out was a
    # 15% under-estimate on the real run -- 362 val batches at 0.71 it/s is
    # 8.5 minutes on top of every epoch. Measured rather than assumed, because
    # validation runs under `no_grad` and is not simply "the same but cheaper".
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=timing_batch, shuffle=False,
        collate_fn=_collate_samples, num_workers=config.workers,
    )
    model.eval()
    val_began, val_done = time.time(), 0
    with torch.no_grad():
        for batch, _ in val_loader:
            if batch is None:
                continue
            _forward(model, _to_device(batch, device), criterion, config)
            val_done += 1
            if val_done >= max(timed_batches // 2, 2):
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    val_per_step = (time.time() - val_began) / max(val_done, 1)

    world = max(requested, 1)
    steps = (config.steps_per_epoch if config.steps_per_epoch > 0
             else math.ceil(len(train_set) / (timing_batch * world)))
    val_steps = math.ceil(len(val_set) / (timing_batch * world))
    epoch_seconds = per_step * steps + val_per_step * val_steps
    total_seconds = epoch_seconds * config.epochs
    sessions = math.ceil(total_seconds / (config.max_hours * 3600))
    print(f"  {per_step:.2f} s/step train, {val_per_step:.2f} s/step val")
    print(f"  {steps} train + {val_steps} val steps/epoch per GPU at "
          f"micro_batch_scenes={timing_batch} on {world} device(s) "
          f"-> ~{_hms(epoch_seconds)}/epoch"
          + ("  (--steps_per_epoch)" if config.steps_per_epoch > 0 else "  (one pass)"))
    if timing_batch != config.batch_size:
        print(f"  (--batch_size does not change this: it sets how often the "
              f"optimizer steps, not how many forward/backward passes run)")
    print(f"  {config.epochs} epochs -> ~{_hms(total_seconds)} "
          f"= {sessions} session(s) at {config.max_hours:g}h")
    if world > 1:
        # Measured in one process on one GPU. With several, every optimizer
        # step also all-reduces the gradient (a few milliseconds at this model
        # size) and the processes contend for the same CPUs to build scenes.
        # On the earlier two-GPU run the total gap was about 45%.
        print(f"  measured on 1 GPU -- with {world} processes expect roughly "
              f"{_hms(epoch_seconds * 1.45)}/epoch "
              f"({_hms(total_seconds * 1.45)} total)")
    if epoch_seconds > config.max_hours * 3600:
        warnings.append(
            f"one epoch (~{_hms(epoch_seconds)}) is longer than a session. That "
            f"works -- mid-epoch checkpoints every "
            f"{config.checkpoint_every_minutes:g} min cover it -- but no epoch "
            f"will ever complete, so val metrics never update. Consider fewer "
            f"--modes_per_scene."
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


def _token_reach(samples, hops: int) -> List[float]:
    """
    Fraction of vertices within 1..`hops` mesh edges of a cross-fragment token.

    The number that says whether the schedule's trailing intra layers are
    enough. A cross layer writes only to tokens; each intra layer after it
    spreads that one hop further; and the rotation head means over *every*
    vertex, so whatever is never reached dilutes the prediction with features
    that know nothing about the other fragments.

    Averaged over the sampled scenes, weighted by size, so one large scene does
    not get the same say as one small one.
    """
    import torch

    totals = [0] * max(hops, 1)
    vertices = 0
    for sample in samples:
        batch, _ = _collate_samples([sample])
        n = int(batch.vertex_fragment.numel())
        if n == 0 or batch.token_index is None:
            continue
        src, dst = batch.edge_index[0], batch.edge_index[1]
        reach = torch.zeros(n, dtype=torch.bool)
        reach[batch.token_index] = True
        for hop in range(max(hops, 1)):
            # One round of messages along real mesh edges, matching what an
            # intra layer moves.
            reach = reach | torch.zeros(n, dtype=torch.bool).index_put_(
                (dst[reach[src]],), torch.ones(1, dtype=torch.bool), accumulate=False)
            totals[hop] += int(reach.sum())
        vertices += n
    if not vertices:
        return [0.0] * max(hops, 1)
    return [t / vertices for t in totals]
