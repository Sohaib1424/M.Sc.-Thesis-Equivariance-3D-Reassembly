"""
One typed configuration object for the whole project.

Everything that used to be a loose ``argparse`` flag lives here, so a run is
described by a single serializable object that gets written next to the
checkpoint. That matters for a thesis specifically: "which settings produced
this number" should not have to be reconstructed from shell history.

Precedence: dataclass defaults < YAML file < explicit CLI flags.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    root_dir: str = "data"
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 0
    #: 'full' = whole fragment mesh; 'frac' = pruned fracture-surface mesh.
    #: Losses are always evaluated against the full mesh either way.
    input_source: str = "full"
    #: Total vertex budget per scene, enforced by voxel decimation.
    #: None disables it. This is the primary VRAM control.
    decimate_to: Optional[int] = 20_000
    #: Hard reject (and redraw) scenes still above this after decimation.
    max_vertices: Optional[int] = 60_000
    min_vertices_per_fragment: int = 32
    # Restrict which fracture subdirectories are eligible, e.g. 'fractured_*'.
    # None uses every subdirectory; see scene_io.load_scene for why that is
    # a decision worth making explicitly.
    fracture_pattern: Optional[str] = None
    # Disk cache for preprocessed scenes. Measured on a real run, preprocessing
    # was 49% of epoch wall-clock with four workers, and it is a pure function
    # of (scene, fracture, decimation settings) -- so it only has to be paid
    # once. None disables. On Kaggle use /kaggle/working/cache (writable and
    # persisted between sessions of the same notebook).
    cache_dir: Optional[str] = None
    #: Cache size cap in GiB. Kaggle's /kaggle/working quota is 20 GB TOTAL and
    #: also holds your checkpoints -- an unbounded cache fills it and makes
    #: checkpoint writes fail, losing the run. Entries average ~437 KB at
    #: decimate_to=6000, so 8 GiB is roughly 19,000 of them.
    cache_max_gib: float = 8.0
    correspondence_tol: float = 1e-5
    num_workers: int = 2
    seed: int = 0


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_layers: int = 3
    num_vn_slots: int = 8
    heads: int = 4
    #: None -> hidden_channels // heads (standard multi-head splitting).
    #: Set equal to hidden_channels to reproduce the old full-width-per-head
    #: behaviour, which costs `heads` times the message memory.
    head_dim: Optional[int] = None
    embed_dim: int = 32
    norm: str = "layer"          # layer | batch | none
    gradient_checkpointing: bool = True
    angular: bool = False        # optional DimeNet-style triplet features
    max_triplets_per_node: int = 24


@dataclass
class LossConfig:
    rot: float = 1.0
    pos: float = 1.0
    node: float = 1.0
    mid: float = 1.0
    face: float = 1.0
    embv: float = 1.0
    embe: float = 1.0
    rot_loss: str = "chordal"    # chordal | geodesic | hybrid
    auto_balance: bool = False


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    scheduler: str = "plateau"   # plateau | cosine | none
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    warmup_steps: int = 0


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 2               # scenes per step, per rank
    accum_steps: int = 1              # gradient accumulation -> larger effective batch
    steps_per_epoch: int = 100        # per rank
    val_steps: int = 20               # per rank
    amp: bool = True
    amp_dtype: str = "fp16"           # fp16 (T4/Turing) | bf16 (Ampere+)
    num_gpus: int = 1
    device: str = "cuda"
    master_port: int = 12355
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None
    resume_history: bool = False
    max_consecutive_oom: int = 5
    log_every: int = 1
    save_every: int = 1
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        sections = {
            "data": DataConfig, "model": ModelConfig, "loss": LossConfig,
            "optim": OptimConfig, "train": TrainConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            values = dict(d.get(name, {}) or {})
            valid = {f.name for f in dataclasses.fields(klass)}
            unknown = set(values) - valid
            if unknown:
                raise ValueError(f"unknown key(s) in config section {name!r}: {sorted(unknown)}")
            kwargs[name] = klass(**values)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def save(self, path: str) -> None:
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def apply_overrides(self, overrides: Dict[str, Any]) -> "Config":
        """Apply ``{'model.hidden_channels': 64}``-style overrides in place."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            if not key:
                raise ValueError(f"override {dotted!r} must be of the form section.key")
            target = getattr(self, section, None)
            if target is None or not hasattr(target, key):
                raise ValueError(f"unknown config key {dotted!r}")
            setattr(target, key, value)
        return self

    def describe(self) -> str:
        lines = []
        for section in ("data", "model", "loss", "optim", "train"):
            obj = getattr(self, section)
            body = ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(obj).items())
            lines.append(f"  [{section}] {body}")
        return "\n".join(lines)
