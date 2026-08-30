"""
Train, resume, or evaluate the V-GAT reassembly model.

    python -m scripts.train --root D:\\path\\to\\breaking_bad --preflight
    python -m scripts.train --root D:\\path\\to\\breaking_bad --epochs 40
    python -m scripts.train --root ... --evaluate            # score best.pt
    python -m scripts.train --root ... --devices 1           # one GPU

Interrupted runs continue by themselves: rerun the same command and it loads
``<out-dir>/last.pt``. To continue from a checkpoint written *somewhere else* --
the Kaggle case, where the previous session's output is mounted read-only --
add ``--resume-from /path/to/previous`` and keep ``--out-dir`` writable.

Every flag maps to a field of :class:`reassembly.training.Config`, so the
default value shown by ``--help`` is the default the code actually uses -- there
is no second copy to drift.

On Kaggle
---------
For **one** GPU, a notebook cell is fine::

    from reassembly.training import Config, train, evaluate
    config = Config(root="/kaggle/input/breaking-bad-dataset", epochs=40, devices=1)
    train(config); evaluate(config)

For **both** T4s, it has to be a file. Torch spawns one process per device and
each child re-imports ``__main__``, which does not exist in a notebook cell --
the children die with a ``FileNotFoundError`` on ``<stdin>`` that says nothing
about the cause. So::

    %%writefile train_run.py
    from reassembly.training import Config, train
    if __name__ == "__main__":
        train(Config(root="/kaggle/input/breaking-bad-dataset", devices=2))

then in the next cell::

    !python train_run.py

``train`` raises with these instructions if it is called the other way, rather
than letting the spawn fail obscurely.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Same bootstrap as every other script here, so a clone runs without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.training import Config, evaluate, preflight, train


def build_parser() -> argparse.ArgumentParser:
    """One flag per config field, typed from the dataclass itself."""
    config = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preflight", action="store_true",
                        help="check the dataset, memory, losses and speed on "
                             "this machine, then exit without training")
    parser.add_argument("--evaluate", action="store_true",
                        help="score a checkpoint on the test split and exit")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])

    for field in dataclasses.fields(Config):
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
        elif isinstance(current, int) or current is None and field.name in (
            "modes_per_scene", "limit_train", "limit_val"
        ):
            parser.add_argument(flag, type=int, default=None)
        elif isinstance(current, float):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    names = {f.name for f in dataclasses.fields(Config)}
    overrides = {k: v for k, v in vars(args).items() if k in names and v is not None}
    return Config(**overrides)


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    if args.preflight:
        raise SystemExit(0 if preflight(config) else 1)
    if args.evaluate:
        evaluate(config, checkpoint=args.checkpoint, split=args.split)
    else:
        train(config)


if __name__ == "__main__":
    main()
