#!/usr/bin/env python
"""
Inspect the dataset tree WITHOUT assuming any particular layout.

    python -m scripts.inspect_data --root data

Walks everything, reports what is actually present, and compares that against
what `vngat.data.splits.list_scene_directories` currently finds. Written after
the scanner's hardcoded per-source nesting depths turned out to be a guess:
missing paths were skipped silently, so a wrong guess produced a plausible-
looking but too-small scene count rather than an error.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MESH = "compressed_mesh.obj"
DATA = "compressed_data.npz"


def walk_scene_dirs(root: Path):
    """
    Every directory holding both compressed files, found by a plain walk.

    Prunes as soon as a scene directory is identified: each scene contains ~100
    fracture sub-directories, so descending into them would turn a ~10k
    directory walk into a ~1M one.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        names = set(filenames)
        if MESH in names and DATA in names:
            found.append(Path(dirpath))
            dirnames[:] = []          # do not descend into fracture folders
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--show", type=int, default=3, help="Example paths per subset.")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"{root} does not exist")
        return 1

    print(f"=== top level of {root} ===")
    for entry in sorted(root.iterdir()):
        print(f"  {'dir ' if entry.is_dir() else 'file'}  {entry.name}")

    print("\n=== scene directories found by an assumption-free walk ===")
    scenes = walk_scene_dirs(root)
    print(f"  {len(scenes):,} directories contain both {MESH} and {DATA}\n")

    by_subset = defaultdict(list)
    for s in scenes:
        rel = s.relative_to(root)
        by_subset[rel.parts[0] if len(rel.parts) > 1 else "<root>"].append(rel)

    for subset in sorted(by_subset):
        paths = by_subset[subset]
        depths = Counter(len(pp.parts) for pp in paths)
        print(f"  {subset:<44} {len(paths):>7,} scenes   nesting depth {dict(depths)}")
        for example in paths[:args.show]:
            print(f"      {example}")

    # Fracture patterns available inside a sample scene.
    if scenes:
        sample = scenes[0]
        subdirs = sorted(d.name for d in sample.iterdir() if d.is_dir())
        kinds = Counter(n.split("_")[0] for n in subdirs)
        print(f"\n=== fracture sub-directories in {sample.relative_to(root)} ===")
        print(f"  {len(subdirs)} total, prefixes: {dict(kinds)}")
        print(f"  examples: {subdirs[:6]}")

    # The official split files, if the release shipped them.
    split_dir = root / "data_split"
    if split_dir.is_dir():
        print(f"\n=== {split_dir.relative_to(root)} (official splits) ===")
        for f in sorted(split_dir.rglob("*")):
            if f.is_file():
                try:
                    lines = f.read_text().splitlines()
                    head = lines[0] if lines else "<empty>"
                    print(f"  {f.relative_to(split_dir)!s:<40} {len(lines):>7,} lines | first: {head}")
                except (UnicodeDecodeError, OSError):
                    print(f"  {f.relative_to(split_dir)!s:<40} (binary)")

    # What the project's own scanner sees, for comparison.
    print("\n=== what vngat.data.splits.list_scene_directories currently finds ===")
    try:
        from vngat.data.splits import list_scene_directories

        current = list_scene_directories(str(root))
        print(f"  {len(current):,} scenes")
        missed = len(scenes) - len(current)
        if missed > 0:
            print(f"  MISSING {missed:,} scenes that the walk found.")
            seen = {c.resolve() for c in current}
            for s in scenes:
                if s.resolve() not in seen:
                    print(f"    first missed: {s.relative_to(root)}")
                    break
        elif missed < 0:
            print(f"  finds {-missed:,} MORE than the walk -- unexpected, inspect the layouts")
        else:
            print("  matches the walk")
    except Exception as exc:  # noqa: BLE001
        print(f"  scanner raised {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
