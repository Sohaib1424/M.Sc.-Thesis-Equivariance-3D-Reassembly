#!/usr/bin/env python
"""
Find, inspect, and import checkpoints — for carrying a run across Kaggle
sessions.

    # what checkpoints exist anywhere, and what state are they in?
    python scripts/checkpoints.py list

    # copy the newest one found under /kaggle/input into the working dir,
    # so `resume: auto` picks it up
    python scripts/checkpoints.py import

    # what is inside a specific file?
    python scripts/checkpoints.py show /kaggle/working/checkpoints/last.pt

WHY THIS EXISTS
---------------
An interactive Kaggle session does **not** reliably persist `/kaggle/working`.
The durable route is: **Save Version** at the end of a session, then in the
next session **Add Data → Notebook Output** and select that notebook. Kaggle
mounts it read-only under `/kaggle/input/<something>/`.

`resume: auto` already searches there, so in the common case nothing needs
doing. This tool is for when you want to see what survived, or to stage a
specific checkpoint by hand.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from reassembly.utils.console import quiet_third_party_warnings

quiet_third_party_warnings()

SEARCH_ROOTS = [
    "/kaggle/working",
    "/kaggle/input",
    "checkpoints",
    ".",
]
PATTERNS = ["**/last.pt", "**/best.pt", "**/final.pt", "**/snapshot_e*.pt"]


def discover(roots=None):
    """Every checkpoint-looking file under the search roots, newest last."""
    found = {}
    for root in roots or SEARCH_ROOTS:
        base = Path(root)
        if not base.exists():
            continue
        for pattern in PATTERNS:
            try:
                for path in base.glob(pattern):
                    if path.is_file():
                        found[path.resolve()] = path
            except (OSError, ValueError):
                continue
    return sorted(found.values(), key=lambda p: (p.stat().st_mtime, str(p)))


def summarize(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                                 # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    history = state.get("history") or {}
    epochs = len(history.get("train", {}).get("total", []))
    return {
        "epoch": state.get("epoch"),
        "best_val": state.get("best_val"),
        "history_epochs": epochs,
        "has_optimizer": bool(state.get("optimizer")),
        "has_scheduler": bool(state.get("scheduler")),
        "has_scaler": bool(state.get("scaler")),
        "has_rng": bool(state.get("rng")),
        "version": state.get("version", 1),
        "size_mb": path.stat().st_size / 1e6,
    }


def cmd_list(args):
    paths = discover(args.roots or None)
    if not paths:
        print("No checkpoints found under:", ", ".join(args.roots or SEARCH_ROOTS))
        print("\nIf a previous session saved some, attach that notebook's output:")
        print("  Add Data -> Notebook Output -> select your notebook")
        return

    print(f"{'checkpoint':<58} {'epoch':>7} {'best val':>11} {'hist':>6} {'MB':>7}  state")
    print("-" * 108)
    for path in paths:
        info = summarize(path)
        if "error" in info:
            print(f"{str(path)[-58:]:<58} {info['error']}")
            continue
        flags = "".join(
            k for k, present in (
                ("O", info["has_optimizer"]), ("S", info["has_scheduler"]),
                ("A", info["has_scaler"]), ("R", info["has_rng"]),
            ) if present
        ) or "-"
        best = info["best_val"]
        best_s = f"{best:.6f}" if isinstance(best, float) and best == best else "-"
        print(f"{str(path)[-58:]:<58} {str(info['epoch']):>7} {best_s:>11} "
              f"{info['history_epochs']:>6} {info['size_mb']:>7.1f}  {flags}")
    print("\nstate flags: O=optimizer S=scheduler A=amp-scaler R=rng")
    print("A checkpoint missing O will restart the optimizer from zero momentum,")
    print("which shows up as a loss bump at the resume point.")


def cmd_show(args):
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    info = summarize(path)
    if "error" in info:
        raise SystemExit(info["error"])
    for key, value in info.items():
        print(f"  {key:<16} {value}")

    state = torch.load(path, map_location="cpu", weights_only=False)
    history = state.get("history") or {}
    series = history.get("val", {}).get("total", [])
    if series:
        print(f"\n  val total: first={series[0]:.4f} last={series[-1]:.4f} "
              f"min={min(series):.4f} over {len(series)} epochs")
    cfg = state.get("config") or {}
    if cfg:
        data = cfg.get("data", {})
        train = cfg.get("train", {})
        print(f"\n  trained with: decimate_to={data.get('decimate_to')} "
              f"batch_size={train.get('batch_size')} "
              f"num_gpus={train.get('num_gpus')} epochs={train.get('epochs')}")


def cmd_import(args):
    """Stage the newest checkpoint found outside the working dir into it."""
    target_dir = Path(args.checkpoint_dir)
    candidates = [p for p in discover(args.roots or None)
                  if target_dir.resolve() not in p.resolve().parents]
    if not candidates:
        print("Nothing to import -- no checkpoints found outside "
              f"{target_dir}")
        return

    source = candidates[-1]
    info = summarize(source)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "last.pt"

    if target.exists() and not args.force:
        existing = summarize(target)
        if (existing.get("epoch") or -1) >= (info.get("epoch") or -1):
            print(f"{target} is already at epoch {existing.get('epoch')}, "
                  f"newer than or equal to {source} (epoch {info.get('epoch')}).")
            print("Nothing done. Pass --force to overwrite anyway.")
            return

    shutil.copy2(source, target)
    print(f"copied {source}\n    -> {target}")
    print(f"    epoch={info.get('epoch')} best_val={info.get('best_val')} "
          f"history={info.get('history_epochs')} epochs")
    print("\nresume: auto will now pick this up. Continue with your usual command.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="*", default=None,
                        help=f"Where to search (default: {' '.join(SEARCH_ROOTS)})")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="show every checkpoint found")

    show = sub.add_parser("show", help="inspect one checkpoint")
    show.add_argument("path")

    imp = sub.add_parser("import", help="stage the newest external checkpoint")
    imp.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    imp.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "show":
        cmd_show(args)
    elif args.command == "import":
        cmd_import(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
