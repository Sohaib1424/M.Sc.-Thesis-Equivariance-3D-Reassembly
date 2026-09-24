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

Two questions, two splits
-------------------------
"Validation is not improving" has at least two causes and they call for
different work, so there are two ways to hold data out::

    --split-by object       # the default and the benchmark: val objects are
                            # SHAPES the model has never seen
    --split-by fracture     # val objects are shapes it HAS seen, broken in
                            # ways it has not

Run both from the same checkpoint budget and read the pair, not either alone:

* **object at chance, fracture well under it** -- the model has learned
  per-shape canonical orientations and no transferable rule. More data or more
  epochs will not fix that; the input representation or the loss has to change.
* **both at chance** -- generalisation is not the problem yet. Look upstream:
  labels, conventions, the head (watch ``head|cos|``).
* **both improving together** -- it is learning the intended thing.

``--split-by fracture`` draws only from the official *training* shapes, so the
official validation shapes stay untouched and the benchmark number remains
available from the same dataset. It is a diagnostic, not a result: a number
from it must never be reported as an object-split result.

The category imbalance
----------------------
Everyday's categories hold very different numbers of distinct shapes -- 17
bottles against 5 cups -- so uniform sampling shows the model roughly three
bottles per cup, and a shape prior is cheaper to fit than an orientation rule::

    --balance category                       # uniform over categories
    --balance category --balance-temperature 0.5   # square-root softening
    --balance object                         # equalise objects, not categories

Off by default. Training only -- validation is never reweighted, so the two
settings stay comparable -- and the per-category breakdown printed with the
validation metrics is how to see whether it helped. Balancing is not free: the
startup banner reports the effective sample size, which is how much of an epoch
survives drawing from skewed weights.

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


_CHOICES = {
    "split_by": ("object", "fracture"),
    "fracture_pool": ("train", "all"),
    "balance": ("none", "category", "object"),
    "symmetry_axis": ("x", "y", "z"),
    "label_method": ("dihedral", "coincidence"),
    "normalize_mode": ("scene", "fragment"),
}

_CHOICE_HELP = {
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
                        help="score a checkpoint on a held-out split and exit")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="which split to score. Defaults to val, NOT test: Breaking Bad "
             "ships train and val lists only, so under --split-by object there "
             "is no official test partition and asking for one falls back to a "
             "hashed split that is comparable with nothing. --split-by fracture "
             "does define a test partition, of held-out break patterns.")

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
        elif field.name in _CHOICES:
            # Enumerated fields get their options on the flag itself, so a typo
            # is caught by argparse with the alternatives printed rather than
            # by a Config assertion after the dataset has been scanned.
            parser.add_argument(flag, choices=_CHOICES[field.name], default=None,
                                help=_CHOICE_HELP.get(field.name))
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
