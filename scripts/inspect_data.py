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


def _debug_split(root: Path) -> None:
    """
    Show the matcher's inputs side by side.

    Also prints which `splits.py` is actually imported and its hash: a stale
    file that still LOOKS current is indistinguishable from a logic bug from
    the outside, and unzip restores archived timestamps, which can leave a
    cached .pyc in play.
    """
    import hashlib

    from vngat.data import splits as splits_mod

    module_path = Path(splits_mod.__file__)
    digest = hashlib.md5(module_path.read_bytes()).hexdigest()[:12]
    print("\n=== official-split matcher debug ===")
    print(f"  module        : {module_path}")
    print(f"  md5           : {digest}")
    print(f"  suffix matcher: {'present' if 'by_suffix' in module_path.read_text() else 'ABSENT (stale file)'}")

    for subset in ("everyday_compressed", "artifact_compressed"):
        if not (root / subset).is_dir():
            continue
        scenes = splits_mod.list_scene_directories(str(root), [subset])
        print(f"\n  --- {subset}: {len(scenes)} scenes under this subset ---")
        if not scenes:
            print("    filter returned NOTHING -- the subset name does not match the "
                  "top-level directory")
            continue
        keys = set()
        for scene in scenes:
            parts = scene.relative_to(root).parts
            for k in (1, 2, 3):
                if len(parts) >= k:
                    keys.add("/".join(parts[-k:]))
        for scene in scenes[:2]:
            parts = scene.relative_to(root).parts
            print(f"    disk : {'/'.join(parts)}")
            print(f"           keys -> {[('/'.join(parts[-k:])) for k in (1, 2) if len(parts) >= k]}")

        stem = subset.replace("_compressed", "")
        listing = root / "data_split" / f"{stem}.train.txt"
        if not listing.is_file():
            print(f"    no {listing.name}")
            continue
        for line in listing.read_text(encoding="utf-8-sig").splitlines()[:3]:
            entry = line.strip().strip("/")
            parts = Path(entry).parts
            tried = [("/".join(parts[-k:]), "/".join(parts[-k:]) in keys)
                     for k in (3, 2, 1) if len(parts) >= k]
            print(f"    entry: {entry!r}")
            print(f"           tried -> {tried}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--show", type=int, default=3, help="Example paths per subset.")
    p.add_argument("--debug_split", action="store_true",
                   help="Dump the exact keys the official-split matcher builds and looks up.")
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
                    lines = f.read_text(encoding="utf-8-sig").splitlines()
                    head = lines[0] if lines else "<empty>"
                    print(f"  {f.relative_to(split_dir)!s:<40} {len(lines):>7,} lines | first: {head}")
                    # repr() so hidden characters -- BOM, CR, stray whitespace,
                    # unexpected leading components -- are actually visible.
                    for raw in lines[:2]:
                        print(f"      raw: {raw!r}")
                except (UnicodeDecodeError, OSError):
                    print(f"  {f.relative_to(split_dir)!s:<40} (binary)")

    # Whether the official split entries actually resolve to what is on disk.
    if split_dir.is_dir():
        try:
            from vngat.data.splits import load_official_split

            for subset in ("everyday_compressed", "artifact_compressed"):
                if not (root / subset).is_dir():
                    continue
                for which in ("train", "val"):
                    try:
                        got = load_official_split(str(root), which, [subset])
                        print(f"  resolves: {subset} {which} -> {len(got):,} scenes")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {subset} {which} FAILED:")
                        for detail in str(exc).splitlines():
                            print(f"      {detail}")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not test official splits: {type(exc).__name__}: {exc}")

    if args.debug_split and split_dir.is_dir():
        _debug_split(root)

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
