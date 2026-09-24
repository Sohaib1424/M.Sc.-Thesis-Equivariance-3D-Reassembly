"""
The command-line flags of every script that builds a training ``Config`` --
``train``, ``benchmark_data``, ``check_scene``, ``dump_prediction``,
``scaling_sweep`` -- so a setting is spelled the same way everywhere.

Named as in Thesis 1 wherever the two projects share a setting, so one set of
habits drives both: ``--root_dir``, ``--checkpoint_dir``, ``--num_gpus``,
``--hidden_channels``, ``--grad_checkpointing True`` and so on
(``reassembly.training.THESIS1_NAMES``). Seven of them carry Thesis 1's
*meaning* as well, which differs from the Config field's, and are converted
here:

    --batch_size B          scenes per optimizer step per GPU  -> accumulate = B / M
    --micro_batch_scenes M  scenes per forward pass            -> batch_size = M
    --steps_per_epoch S     optimizer steps per epoch          -> steps_per_epoch = S x accumulate
    --lr_min X              the schedule's floor, absolute     -> min_lr_fraction = X / lr
    --lr_warmup_epochs E    warmup length, in epochs           -> warmup_fraction = E / epochs
    --val_steps V           V x B validation scenes            -> limit_val = V x B
    --resume auto|none|PATH                                    -> resume, resume_from

Settings only this project has keep their own names, in the same style
(``--tokens_per_scene``, ``--modes_per_scene``, ``--schedule``, ...).
Booleans take a value, as in Thesis 1 (``--amp False``); a bare flag means
True. A Thesis 1 flag with no counterpart here is refused by name, saying what
to use instead. Defaults are the dataclass's own, so there is no second copy of
them to drift.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.training import THESIS1_NAMES, Config, flag  # noqa: E402

CHOICES = {
    "split_by": ("object", "fracture"),
    "split_source": ("auto", "official", "hash"),
    "fracture_pool": ("train", "all"),
    "balance": ("none", "category", "object"),
    "symmetry_axis": ("x", "y", "z"),
    "label_method": ("dihedral", "coincidence"),
    "normalize_mode": ("scene", "fragment"),
    "lr_schedule": ("cosine", "constant"),
}

HELP = {
    "split_by": "what is held out: whole shapes (object, the benchmark) or "
                "break patterns (fracture, an easier diagnostic question)",
    "split_source": "official: Breaking Bad's own lists (an error if missing); "
                    "hash: split what is on disk; auto: official when present",
    "fracture_pool": "which objects the fracture split draws from: the official "
                     "training shapes (train) or every shape (all)",
    "balance": "correct the category imbalance when sampling training data: "
               "uniform over categories then objects then modes (category), "
               "over objects only (object), or not at all (none)",
    "symmetry_axis": "the dataset's canonical up-axis, for the reported "
                     "tilt/twist split only",
    "lr_schedule": "after the warmup: decay once to --lr_min (cosine) or stay "
                   "at --lr (constant)",
    "mode_filter": "keep only break patterns whose directory starts with this; "
                   "'' keeps every one",
    "subsets": "subset directories under --root_dir, comma- or space-separated",
    "grad_checkpointing": "recompute every layer in the backward pass: much "
                          "less memory, about one extra forward of time",
    "devices": "GPUs: -1 every visible one, 0 the CPU",
    "max_objects": "train on this many objects only, strided across the "
                   "catalogue (validation is untouched); 0 = all",
    "limit_train": "use only this many training scenes, strided across the "
                   "split; 0 = all",
    "modes_per_scene": "break patterns drawn per object per epoch; 0 = all",
}

# Config fields that only the converted flags below set.
_CONVERTED = {"batch_size", "accumulate", "steps_per_epoch", "min_lr_fraction",
              "warmup_fraction", "limit_val", "resume", "resume_from"}

_NO_PLATEAU = "the plateau schedule is not implemented here; --lr_schedule is cosine or constant"
# Thesis 1 flags with no counterpart here, and what to use instead.
THESIS1_ONLY = {
    "config": "there are no YAML configs here; pass the flags themselves",
    "tag": "checkpoints are always last.pt and best.pt inside --checkpoint_dir",
    "num_layers": "the depth here is --schedule, e.g. the default "
                  "--schedule intra intra intra intra intra cross cross cross intra",
    "num_vn_slots": "there are no virtual nodes here; fragments exchange "
                    "information through --tokens_per_scene fracture-surface tokens",
    "max_scenes": "use --max_objects, which limits the TRAINING objects only "
                  "(evenly strided across the catalogue) and leaves validation whole",
    "input_source": "the fragments are always the full meshes here",
    "lr_monitor": _NO_PLATEAU, "lr_patience": _NO_PLATEAU, "lr_factor": _NO_PLATEAU,
    "restart_schedule": "the schedule is a function of the step, so a resume "
                        "continues it exactly; change --epochs or --lr to reshape it",
    "master_port": "a free port is chosen automatically (set MASTER_PORT to pin one)",
    "drive_folder_id": "point --checkpoint_dir at a mounted Drive folder instead",
    "drive_credentials": "point --checkpoint_dir at a mounted Drive folder instead",
    "prefetch_factor": "not configurable here", "pin_memory": "not configurable here",
    "persistent_workers": "not configurable here", "log_every": "every epoch is logged",
    "w_node": "the loss weights here are --w_rot --w_pos --w_normal --w_face --w_embedding",
    "w_mid": "the loss weights here are --w_rot --w_pos --w_normal --w_face --w_embedding",
    "w_emb_v": "the loss weights here are --w_rot --w_pos --w_normal --w_face --w_embedding",
    "w_emb_e": "the loss weights here are --w_rot --w_pos --w_normal --w_face --w_embedding",
}


def _bool(value) -> bool:
    """``True``/``False`` as Thesis 1 spells them, plus the usual variants."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1", "on"):
        return True
    if lowered in ("false", "f", "no", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")


class _Refused(argparse.Action):
    """A Thesis 1 flag this project does not have: stop, and say what to use."""

    def __init__(self, *args, reason: str = "", **kwargs):
        self.reason = reason
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is a Thesis 1 setting with no counterpart "
                     f"here: {self.reason}.")


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the Config flags to ``parser``, under their Thesis 1 names."""
    default = Config()
    for field in dataclasses.fields(Config):
        name = field.name
        if name in _CONVERTED:
            continue
        current = getattr(default, name)
        option = flag(name)
        metavar = option[2:].upper()             # the flag's name, not the field's
        text = HELP.get(name)
        if isinstance(current, bool):
            parser.add_argument(option, dest=name, type=_bool, nargs="?", const=True,
                                default=None, metavar="BOOL",
                                help=(text + "; " if text else "") + f"default {current}")
        elif name == "schedule":
            parser.add_argument(option, dest=name, nargs="+", default=None, metavar="KIND",
                                help="layer kinds in order, e.g. intra intra cross intra")
        elif name == "subsets":
            parser.add_argument(option, dest=name, nargs="+", default=None,
                                metavar=metavar, help=text)
        elif name in CHOICES:
            parser.add_argument(option, dest=name, choices=CHOICES[name], default=None,
                                help=(text + "; " if text else "") + f"default {current}")
        elif name in ("modes_per_scene", "limit_train", "max_objects") or isinstance(current, int):
            parser.add_argument(option, dest=name, type=int, default=None, metavar=metavar,
                                help=(text + "; " if text else "") + f"default {current}")
        elif isinstance(current, float):
            parser.add_argument(option, dest=name, type=float, default=None,
                                metavar=metavar, help=f"default {current}")
        else:
            parser.add_argument(option, dest=name, type=str, default=None, metavar=metavar,
                                help=(text + "; " if text else "") + f"default {current!r}")

    steps = default.steps_per_epoch // default.accumulate
    parser.add_argument("--batch_size", type=int, default=None,
                        help="scenes per optimizer step per GPU, as in Thesis 1; "
                             f"default {default.batch_size * default.accumulate}")
    parser.add_argument("--micro_batch_scenes", type=int, default=None,
                        help="scenes per forward pass; peak memory grows with "
                             f"this, not with --batch_size; default {default.batch_size}")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help=f"optimizer steps per GPU per epoch, 0 = one full pass; "
                             f"default {steps}")
    parser.add_argument("--lr_min", type=float, default=None,
                        help=f"the cosine schedule's floor; default "
                             f"{default.min_lr_fraction:g} x --lr")
    parser.add_argument("--lr_warmup_epochs", type=float, default=None,
                        help=f"linear warmup length in epochs; default "
                             f"{default.warmup_fraction:g} x --epochs")
    parser.add_argument("--val_steps", type=int, default=None,
                        help="validate on val_steps x batch_size scenes, a fixed "
                             "evenly spread subset scored once each; 0 (the "
                             "default) = the whole validation split")
    parser.add_argument("--resume", default=None, metavar="auto|none|PATH",
                        help="auto (default): continue from --checkpoint_dir if a "
                             "checkpoint is there; none: start fresh; or a checkpoint "
                             "file or directory to load from")
    for name, reason in THESIS1_ONLY.items():
        parser.add_argument(f"--{name}", action=_Refused, reason=reason, nargs="?",
                            help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace, **forced) -> Config:
    """The Config the flags describe, plus ``forced`` overrides."""
    default = Config()
    names = {f.name for f in dataclasses.fields(Config)} - _CONVERTED
    values = vars(args)
    overrides = {k: v for k, v in values.items() if k in names and v is not None}

    if "subsets" in overrides:
        parts = [p.strip() for item in overrides["subsets"] for p in item.split(",")]
        overrides["subsets"] = [p for p in parts if p] or None
    if overrides.get("mode_filter") == "":
        overrides["mode_filter"] = None
    for name in ("modes_per_scene", "limit_train", "max_objects"):
        if overrides.get(name) == 0:
            overrides[name] = None                 # 0 = no limit

    micro = values.get("micro_batch_scenes") or default.batch_size
    batch = values.get("batch_size") or default.batch_size * default.accumulate
    if micro < 1 or batch < 1:
        raise SystemExit("--batch_size and --micro_batch_scenes must be >= 1")
    if batch % micro:
        raise SystemExit(f"--batch_size {batch} must be a multiple of "
                         f"--micro_batch_scenes {micro}: a step is a whole number "
                         f"of forward passes.")
    accumulate = batch // micro
    overrides["batch_size"], overrides["accumulate"] = micro, accumulate

    steps = values.get("steps_per_epoch")
    steps = default.steps_per_epoch // default.accumulate if steps is None else steps
    overrides["steps_per_epoch"] = steps * accumulate

    lr = overrides.get("lr", default.lr)
    if values.get("lr_min") is not None:
        overrides["min_lr_fraction"] = values["lr_min"] / lr
    epochs = overrides.get("epochs", default.epochs)
    if values.get("lr_warmup_epochs") is not None:
        overrides["warmup_fraction"] = values["lr_warmup_epochs"] / max(epochs, 1)
    if values.get("val_steps"):
        overrides["limit_val"] = values["val_steps"] * batch

    resume = values.get("resume")
    if resume is not None:
        lowered = resume.strip().lower()
        if lowered in ("auto", "true", "yes", "1"):
            overrides["resume"] = True
        elif lowered in ("none", "false", "no", "0", "fresh"):
            overrides["resume"] = False
        else:
            overrides["resume"], overrides["resume_from"] = True, resume

    overrides.update(forced)
    return Config(**overrides)


def config_flags(config: Config, fields=None) -> list:
    """The command line that reproduces ``config`` (fields that differ from the
    default, restricted to ``fields`` when given)."""
    default = Config()
    wanted = ({f.name for f in dataclasses.fields(Config)} if fields is None
              else set(fields))
    changed = {f.name for f in dataclasses.fields(Config)
               if getattr(config, f.name) != getattr(default, f.name)} & wanted
    out = []
    batch = config.batch_size * config.accumulate
    if changed & {"batch_size", "accumulate"}:
        out += ["--batch_size", str(batch), "--micro_batch_scenes", str(config.batch_size)]
    if "steps_per_epoch" in changed:
        out += ["--steps_per_epoch", str(math.ceil(config.steps_per_epoch / config.accumulate))]
    if "min_lr_fraction" in changed:
        out += ["--lr_min", repr(config.lr * config.min_lr_fraction)]
    if "warmup_fraction" in changed:
        out += ["--lr_warmup_epochs", repr(config.warmup_fraction * config.epochs)]
    if "limit_val" in changed and config.limit_val:
        out += ["--val_steps", str(math.ceil(config.limit_val / batch))]
    if changed & {"resume", "resume_from"}:
        out += ["--resume", config.resume_from if config.resume and config.resume_from
                else ("auto" if config.resume else "none")]
    for field in dataclasses.fields(Config):
        name = field.name
        if name in _CONVERTED or name not in changed:
            continue
        value = getattr(config, name)
        if value is None:
            out += [flag(name), ""] if name == "mode_filter" else []
        elif isinstance(value, bool):
            out += [flag(name), str(value)]
        elif name == "subsets":
            out += [flag(name), ",".join(value)]
        elif isinstance(value, (list, tuple)):
            out += [flag(name), *map(str, value)]
        else:
            out += [flag(name), str(value)]
    return out


__all__ = ["THESIS1_NAMES", "THESIS1_ONLY", "add_config_arguments", "config_from_args",
           "config_flags"]
