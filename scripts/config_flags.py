"""
One command-line flag per :class:`reassembly.training.Config` field.

Shared by every script that builds a ``Config`` -- ``train``,
``benchmark_data``, ``check_scene``, ``scaling_sweep`` -- so a setting is
spelled the same way everywhere (``--steps-per-epoch``, ``--split-by``), its
default is the dataclass's own, and there is no second copy of either to drift.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.training import Config  # noqa: E402

CHOICES = {
    "split_by": ("object", "fracture"),
    "fracture_pool": ("train", "all"),
    "balance": ("none", "category", "object"),
    "symmetry_axis": ("x", "y", "z"),
    "label_method": ("dihedral", "coincidence"),
    "normalize_mode": ("scene", "fragment"),
}

CHOICE_HELP = {
    "split_by": "what is held out: whole shapes (object, the benchmark) or "
                "break patterns (fracture, an easier diagnostic question)",
    "fracture_pool": "which objects the fracture split draws from: the official "
                     "training shapes (train) or every shape (all)",
    "balance": "correct the category imbalance when sampling training data: "
               "uniform over categories then objects then modes (category), "
               "over objects only (object), or not at all (none)",
    "symmetry_axis": "the dataset's canonical up-axis, for the reported "
                     "tilt/twist split only",
}

_OPTIONAL_INTS = ("modes_per_scene", "limit_train", "limit_val", "max_objects")


def add_config_arguments(parser: argparse.ArgumentParser,
                         only: tuple = ()) -> argparse.ArgumentParser:
    """
    Add the Config flags to ``parser`` -- all of them, or ``only`` these fields.

    Defaults are ``None`` rather than copies of the dataclass values, so an
    unset flag leaves the dataclass default in charge.
    """
    config = Config()
    for field in dataclasses.fields(Config):
        if only and field.name not in only:
            continue
        flag = "--" + field.name.replace("_", "-")
        current = getattr(config, field.name)
        if isinstance(current, bool):
            # Both directions, so a default-true flag can actually be turned off.
            parser.add_argument(flag, dest=field.name, action="store_true",
                                default=None)
            parser.add_argument("--no-" + field.name.replace("_", "-"),
                                dest=field.name, action="store_false")
        elif field.name == "schedule":
            parser.add_argument(flag, nargs="+", default=None,
                                help="layer kinds in order, e.g. intra intra cross intra")
        elif field.name == "subsets":
            parser.add_argument(flag, nargs="+", default=None)
        elif field.name in CHOICES:
            # Enumerated fields get their options on the flag itself, so a typo
            # is caught by argparse with the alternatives printed rather than
            # by a Config assertion after the dataset has been scanned.
            parser.add_argument(flag, choices=CHOICES[field.name], default=None,
                                help=CHOICE_HELP.get(field.name))
        elif isinstance(current, int) or (current is None and field.name in _OPTIONAL_INTS):
            parser.add_argument(flag, type=int, default=None)
        elif isinstance(current, float):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)
    return parser


def config_from_args(args: argparse.Namespace, **forced) -> Config:
    """The Config the flags describe, plus ``forced`` overrides."""
    names = {f.name for f in dataclasses.fields(Config)}
    overrides = {k: v for k, v in vars(args).items() if k in names and v is not None}
    overrides.update(forced)
    return Config(**overrides)


def config_flags(config: Config, fields=None) -> list:
    """The command line that reproduces ``config`` (non-default fields only)."""
    default = Config()
    out = []
    for field in dataclasses.fields(Config):
        if fields is not None and field.name not in fields:
            continue
        value = getattr(config, field.name)
        if value == getattr(default, field.name) or value is None:
            continue
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            out.append(flag if value else "--no-" + field.name.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            out += [flag, *map(str, value)]
        else:
            out += [flag, str(value)]
    return out
