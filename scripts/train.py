"""
Train, resume, or evaluate the V-GAT reassembly model.

    python -m scripts.train --root_dir D:\\path\\to\\breaking_bad --preflight
    python -m scripts.train --root_dir D:\\path\\to\\breaking_bad --epochs 40
    python -m scripts.train --root_dir ... --evaluate          # score best.pt, assembled
    python -m scripts.train --root_dir ... --evaluate --placement global   # v6's placement
    python -m scripts.train --root_dir ... --evaluate --jitter 0.01 --drop 0.5   # no shared break vertices
    python -m scripts.train --root_dir ... --evaluate --data_subsets artifact_compressed --split all \\
        --modes_per_scene 0 --predictions predictions   # a whole unseen subset, every prediction zipped
    python -m scripts.train --root_dir ... --num_gpus 2        # both GPUs (the default: all)
    python -m scripts.train --root_dir ... --num_gpus 1        # one GPU
    python -m scripts.train --root_dir ... --max_fragments 20  # the benchmark's 2-20 pieces

The flags are Thesis 1's wherever the two projects share a setting, with
Thesis 1's meaning (``scripts/config_flags.py`` lists the conversions):
``--batch_size`` is scenes per optimizer step per GPU, processed
``--micro_batch_scenes`` at a time; ``--steps_per_epoch`` counts optimizer
steps; ``--grad_checkpointing True`` recomputes every layer in the backward
pass. A Thesis 1 flag with no counterpart here stops with what to use instead.

Interrupted runs continue by themselves (``--resume auto``, the default): rerun
the same command and it loads ``<checkpoint_dir>/last.pt``. To continue from a
checkpoint written *somewhere else* -- the Kaggle case, where the previous
session's output is mounted read-only -- pass ``--resume /path/to/previous``
and keep ``--checkpoint_dir`` writable. ``--resume none`` starts fresh.

The default shown by ``--help`` is the default the code actually uses -- there
is no second copy to drift.

Two questions, two splits
-------------------------
"Validation is not improving" has at least two causes and they call for
different work, so there are two ways to hold data out::

    --split_by object       # the default and the benchmark: val objects are
                            # SHAPES the model has never seen
    --split_by fracture     # val objects are shapes it HAS seen, broken in
                            # ways it has not

Run both from the same checkpoint budget and read the pair, not either alone:

* **object at chance, fracture well under it** -- the model has learned
  per-shape canonical orientations and no transferable rule. More data or more
  epochs will not fix that; the input representation or the loss has to change.
* **both at chance** -- generalisation is not the problem yet. Look upstream:
  labels, conventions, the embedding (watch ``match@1``).
* **both improving together** -- it is learning the intended thing.

``--split_by fracture`` draws only from the official *training* shapes, so the
official validation shapes stay untouched and the benchmark number remains
available from the same dataset. It is a diagnostic, not a result: a number
from it must never be reported as an object-split result.

The category imbalance
----------------------
Everyday's categories hold very different numbers of distinct shapes -- 17
bottles against 5 cups -- so uniform sampling shows the model roughly three
bottles per cup, and a shape prior is cheaper to fit than an orientation rule::

    --balance category                       # uniform over categories
    --balance category --balance_temperature 0.5   # square-root softening
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
import sys
from pathlib import Path

# Same bootstrap as every other script here, so a clone runs without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reassembly.hub import HubSync  # noqa: E402
from reassembly.training import evaluate, preflight, train  # noqa: E402
from scripts.config_flags import add_config_arguments, config_from_args  # noqa: E402,F401


def build_parser() -> argparse.ArgumentParser:
    """One flag per config field (``scripts.config_flags``), plus the modes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preflight", action="store_true",
                        help="check the dataset, memory, losses and speed on "
                             "this machine, then exit without training")
    parser.add_argument("--evaluate", action="store_true",
                        help="score a checkpoint on a held-out split and exit")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="a path, or a file name inside --checkpoint_dir")
    parser.add_argument("--no_assemble", dest="assemble", action="store_false",
                        help="with --evaluate: rotation metrics only, no "
                             "translation solver (so no RMSE(T), Chamfer or "
                             "part accuracy)")
    parser.add_argument("--collision", action="store_true",
                        help="with --evaluate: push apart overlapping fragments "
                             "after solving (off: it trades metric accuracy for "
                             "looks)")
    parser.add_argument("--data_from_flags", action="store_true",
                        help="with --evaluate: use the flags' data settings "
                             "instead of the checkpoint's (split, labels, "
                             "tokens, normalisation)")
    parser.add_argument("--rotations", default="matched", choices=["matched"],
                        help="kept so earlier commands still run: since v7 every rotation "
                             "is fitted from the embedding matches and chained from the "
                             "anchor (reassembly/assembly/rotation.py) -- there is no "
                             "rotation head to take it from")
    parser.add_argument("--placement", default=None, choices=["checked", "global"],
                        help="with --evaluate: how the turned fragments are placed. "
                             "checked (the default): from the pair fits the chain agrees "
                             "with, the anchor held (reassembly/assembly/placement.py). "
                             "global (v6's only placement): one least-squares solve over "
                             "every embedding match; adds _global to the output files")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="with --evaluate: Gaussian noise on every input vertex before "
                             "the network sees it, in units of the scene's largest "
                             "fragment's radius, so the two sides of a break no longer "
                             "coincide (0, the default: off). The assembly is scored on the "
                             "clean fragments. As probe_val.py --jitter, for the whole "
                             "evaluation; adds _jitter<S> to the output files")
    parser.add_argument("--drop", type=float, default=0.0,
                        help="with --evaluate: leave this share of the break vertices out "
                             "of the matching, at random (0, the default: off; below 1). "
                             "As probe_val.py --drop; adds _drop<P> to the output files")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test", "all"],
        help="which split to score. Defaults to val, NOT test: Breaking Bad "
             "ships train and val lists only, so under --split_by object there "
             "is no official test partition and asking for one falls back to a "
             "hashed split that is comparable with nothing. --split_by fracture "
             "does define a test partition, of held-out break patterns. all: "
             "every object of --data_subsets whatever its split, for a subset "
             "the checkpoint was not trained on; the report counts the objects "
             "that are shapes it was trained on. With --evaluate an explicit "
             "--data_subsets (and --max_fragments) wins over the checkpoint's; "
             "--modes_per_scene 0 scores every break pattern.")
    parser.add_argument("--predictions", default="", metavar="FOLDER",
                        help="with --evaluate: also write every scored scene's prediction "
                             "(the dump_prediction format) into one zip in FOLDER: "
                             "<FOLDER>/<subsets>-<split>.zip, holding "
                             "<FOLDER name>/<subset>/<category>/<object>/<mode>.npz -- the "
                             "data directory's layout -- and a CSV of every scene")
    hub = parser.add_argument_group(
        "Hugging Face Hub mirror", "optional; on only when all three are given")
    hub.add_argument("--hf_repo_id", default="",
                     help="repository to push last.pt, best.pt, history.json, "
                          "history.csv and offenders.json to after every epoch")
    hub.add_argument("--hf_local_dir", default="",
                     help="the folder mirrored: must be --checkpoint_dir")
    hub.add_argument("--hf_token", default="", help="Hugging Face access token")
    return add_config_arguments(parser)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (args.jitter or args.drop) and not args.evaluate:
        parser.error("--jitter and --drop apply to --evaluate only")
    if args.predictions and not args.evaluate:
        parser.error("--predictions applies to --evaluate only")
    if args.split == "all" and not args.evaluate:
        parser.error("--split all applies to --evaluate only")
    config = config_from_args(args)
    if args.preflight:
        raise SystemExit(0 if preflight(config) else 1)
    if args.evaluate:
        from reassembly.evaluation.noise import check_noise

        if args.predictions and not args.assemble:
            parser.error("--predictions writes placed scenes: not with --no_assemble")
        try:
            check_noise(args.jitter, args.drop)
        except ValueError as error:
            parser.error(str(error))
        # The rest of the data definition stays the checkpoint's unless
        # --data_from_flags, but an explicit --max_fragments wins -- how a model
        # does on the benchmark's 2-20 pieces is a fair question whatever range
        # it was trained on -- and so does an explicit --data_subsets: scoring
        # on another subset keeps the labels, tokens and normalisation the
        # model was trained with.
        override = tuple(name for name in ("max_fragments", "subsets")
                         if getattr(args, name) is not None)
        evaluate(config, checkpoint=args.checkpoint, split=args.split,
                 assemble=args.assemble, collision=args.collision,
                 data_from_checkpoint=not args.data_from_flags, override=override,
                 placement=args.placement, jitter=args.jitter, drop=args.drop,
                 predictions=args.predictions or None)
    else:
        # Not Config fields: the token must not reach a checkpoint, and the
        # checkpoints are what gets uploaded.
        train(config, hub=HubSync(args.hf_repo_id, args.hf_local_dir, args.hf_token))


if __name__ == "__main__":
    main()
