"""
Typed configuration.

One flat dataclass drives both the YAML files and the argparse interface, so
there is exactly one place where a hyper-parameter is declared, defaulted and
documented. Precedence: CLI flag > YAML file > dataclass default.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Config:
    # ---------------- data ----------------
    root_dir: str = "data"
    """Directory holding everyday_compressed/ and/or artifact_compressed/."""
    data_subsets: str = "everyday_compressed,artifact_compressed"
    """Comma-separated top-level directories under `root_dir` to draw scenes
    from. Empty string means EVERY scene found anywhere under root_dir --
    including the `volume_constrained-*` trees, which are a different fracture
    generation mode and are excluded by default so results stay comparable to
    the standard benchmark. Run `python -m scripts.inspect_data` to see what
    your copy actually contains."""
    input_source: str = "full"
    """'full' = train on complete fragment meshes; 'frac' = on the pruned
    fracture surfaces only. Losses and metrics are always evaluated on the full
    mesh either way, so the two are directly comparable."""
    fracture_pattern: str = "fractured_"
    """Restrict which fracture sub-directories are drawn (e.g. 'fractured_' to
    exclude the mode_* variants). Empty string = all of them."""
    split_source: str = "hash"
    """'official' uses the benchmark's own data_split/*.txt lists -- required if
    results are to be compared against published numbers, since the leaderboard
    is computed on that exact partition. It provides TRAIN and VAL only (no
    test split). 'hash' is a deterministic three-way split of whatever is on
    disk, useful when the official lists are absent."""
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 0
    max_scenes: int = 0
    """0 = the whole split. >0 = a deterministic bounded subset, for making a
    long run fit a fixed session budget without changing the split logic."""
    correspondence: bool = True
    correspondence_tol: float = 1e-5
    min_fragments: int = 2

    # ---------------- model ----------------
    hidden_channels: int = 64
    num_layers: int = 4
    num_vn_slots: int = 8
    heads: int = 4
    embed_dim: int = 32
    gram_bottleneck: int = 16
    norm: str = "layer"
    """'layer' (default, batch-statistics free), 'batch', or 'none'."""
    grad_checkpointing: bool = False
    """Recompute layer activations in backward. ~35% slower, roughly halves
    activation memory. Left off by default; the loop turns it on automatically
    for a single scene that would otherwise OOM."""

    # ---------------- loss weights ----------------
    w_rot: float = 1.0
    w_pos: float = 1.0
    w_node: float = 1.0
    w_mid: float = 1.0
    w_face: float = 1.0
    w_emb_v: float = 1.0
    w_emb_e: float = 1.0
    symmetry_axis: str = "z"
    """Canonical up-axis of the dataset's meshes, used only for the reported
    tilt/twist split. Verify it -- run evaluation with x, y and z and see which
    shows the signature. Wrong choice makes the diagnostic meaningless, not the
    training wrong."""
    emb_pull_margin: float = 0.1
    """Cluster members closer than this to their centroid are already good
    enough. Embeddings are L2-normalised, so distances lie in [0, 2]."""
    emb_push_margin: float = 0.5
    """Centroids further apart than 2x this are left alone. Must be below 1.0,
    since normalised embeddings cannot be more than 2 apart. Set to 0 to
    disable repulsion -- which reproduces the collapse it exists to prevent."""

    # ---------------- optimisation ----------------
    epochs: int = 200
    batch_size: int = 4
    """Scenes per optimiser step PER RANK. Peak memory does not scale with
    this: the loop processes `micro_batch_scenes` scenes at a time and
    accumulates gradients."""
    micro_batch_scenes: int = 1
    steps_per_epoch: int = 50
    val_steps: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = False
    """Mixed precision. OFF by default -- MEASURED, not assumed.

    A controlled comparison (same seed, same config, only precision differing)
    on 8 objects found half precision both unstable and WORSE:

        epoch    AMP     fp32
           62   79.11    74.06
           70   85.24    62.82
           85   65.84    43.18
        best  54.26 in 160 epochs   43.18 in 90 epochs

    AMP also produced non-finite losses from epoch 66 onward, on six of the
    eight training objects, which excluded them from training. fp32 ran 90
    epochs with none.

    The gradient corruption preceded the visible NaN: fp32 was already 5-19 deg
    ahead at epochs 62-65, before any overflow was reported.

    Cost is ~10-20% wall clock, not the ~2x one might expect, because this
    model is dominated by scatter/gather and many small matmuls rather than the
    large GEMMs tensor cores accelerate. Peak memory roughly doubles, from
    ~1.8 GB to ~4 GB, which a 16 GB T4 absorbs.

    Set True only with a reason, and watch for `[amp]` fp32-retry lines."""
    lr_schedule: str = "plateau"
    """'plateau' (ReduceLROnPlateau), 'cosine' (CosineAnnealingLR over `epochs`),
    or 'constant'."""
    lr_monitor: str = "rot"
    """Which validation term the plateau scheduler watches. 'rot' is the primary
    objective. 'total' is dominated by terms that barely move -- face + norm +
    rot were 4.95 of a 5.40 total in a real run, so the remaining signal was
    noise, every epoch looked like a plateau, and the learning rate was halved
    nine times down to 2e-6."""
    lr_patience: int = 20
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    lr_warmup_epochs: int = 5
    """Linearly ramp the learning rate from lr/10 over this many epochs.

    On by default because initialisation, not object count, is the dominant
    source of variance in this model: three seeds of an identical 8-object run
    finished at 30.9, 54.3 and 104.7 degrees, with the worst reaching a plateau
    by epoch 20 and never leaving it in the following 140. A warmup does not
    guarantee escape, but it is the cheapest measure that reduces how much the
    early, largest updates depend on the draw. 0 disables."""
    """Floor for every schedule. Below this nothing moves, so a mis-triggered
    decay wastes the session instead of merely slowing it."""
    seed: int = 0

    # ---------------- dataloading ----------------
    num_workers: int = 2
    """PER RANK. Total worker processes = num_gpus * num_workers; Kaggle's
    2xT4 image has ~4 vCPUs, so 2 is usually the ceiling before the workers
    start fighting each other."""
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # ---------------- runtime ----------------
    num_gpus: int = 2
    device: str = "cuda"
    master_port: int = 12355
    time_budget_hours: float = 11.0
    """Stop cleanly (after checkpointing) once this much wall clock has been
    used, so a Kaggle 12h session ends with a saved, resumable state instead
    of being killed mid-epoch."""

    # ---------------- checkpointing ----------------
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    resume: str = "auto"
    """'auto' = resume from the newest available checkpoint (remote, then
    local); 'none' = start fresh; or an explicit path."""
    drive_folder_id: str = ""
    """Google Drive folder id to mirror checkpoints into. Empty = local only."""
    drive_credentials: str = ""
    """Path to a service-account JSON, or the name of a Kaggle Secret holding
    one. Empty = try KAGGLE_SECRET / GOOGLE_APPLICATION_CREDENTIALS."""

    # ---------------- misc ----------------
    log_every: int = 1
    tag: str = "vngat"

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml

        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save_yaml(self, path: str) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True)

    def validate(self) -> None:
        if self.hidden_channels % self.heads:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible by heads "
                f"({self.heads}): the hidden width is split across heads, not replicated."
            )
        if self.input_source not in ("full", "frac"):
            raise ValueError(f"input_source must be 'full' or 'frac', got {self.input_source!r}")
        if self.norm not in ("layer", "batch", "none"):
            raise ValueError(f"norm must be 'layer', 'batch' or 'none', got {self.norm!r}")
        if self.micro_batch_scenes < 1:
            raise ValueError("micro_batch_scenes must be >= 1")
        if self.batch_size % self.micro_batch_scenes:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be a multiple of micro_batch_scenes "
                f"({self.micro_batch_scenes}) so every rank performs the same number of "
                f"backward passes -- unequal counts deadlock DDP's gradient all-reduce."
            )


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VN-GAT: SO(3)-equivariant 3D fracture reassembly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file.")
    defaults = Config()
    for f in fields(Config):
        flag = f"--{f.name}"
        current = getattr(defaults, f.name)
        if f.type is bool or isinstance(current, bool):
            parser.add_argument(flag, type=_str2bool, default=None, metavar="BOOL")
        else:
            parser.add_argument(flag, type=type(current), default=None)
    return parser


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_config(argv: Optional[List[str]] = None) -> Config:
    """
    Build a Config from CLI args (and optionally a YAML file).

    `argv=None` reads `sys.argv`, i.e. normal command-line use. From a
    notebook, pass an explicit list -- `sys.argv` there holds the Jupyter
    KERNEL's own launch flags, which argparse would try to parse and reject
    with a bare `SystemExit: 2`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()
    for f in fields(Config):
        value = getattr(args, f.name, None)
        if value is not None:
            setattr(cfg, f.name, value)
    cfg.validate()
    return cfg
