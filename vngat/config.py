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
    input_source: str = "full"
    """'full' = train on complete fragment meshes; 'frac' = on the pruned
    fracture surfaces only. Losses and metrics are always evaluated on the full
    mesh either way, so the two are directly comparable."""
    fracture_pattern: str = "fractured_"
    """Restrict which fracture sub-directories are drawn (e.g. 'fractured_' to
    exclude the mode_* variants). Empty string = all of them."""
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
    emb_pull_margin: float = 0.5
    """Cluster members closer than this to their centroid are already good
    enough; stops the pull term demanding infinite precision."""
    emb_push_margin: float = 1.5
    """Centroids further apart than 2x this are left alone; keeps gradient on
    the pairs that are actually confusable. Set to 0 to disable repulsion --
    which reproduces the embedding collapse it exists to prevent."""

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
    amp: bool = True
    lr_patience: int = 8
    lr_factor: float = 0.5
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
