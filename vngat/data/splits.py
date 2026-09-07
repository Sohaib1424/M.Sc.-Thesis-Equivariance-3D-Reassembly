"""
Scene-directory discovery and a deterministic train/val/test split.

The split is keyed on each scene directory's *name*, hashed with a salt, so:
  - the same shape always lands in the same split, run to run and call to call;
  - the three splits are disjoint by construction;
  - nothing depends on directory iteration order or on how many samples have
    already been drawn.

This matters because `BreakingBadDataset.__getitem__` ignores its index and
draws a fresh random scene every call -- index-range slicing
(`dataset[:8000]` vs `dataset[8000:]`) would therefore give two views of the
*same* pool and leak the validation set into training without any visible
symptom.
"""
from __future__ import annotations

import hashlib
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

_SPLITS = ("train", "val", "test")


MESH_FILE = "compressed_mesh.obj"
DATA_FILE = "compressed_data.npz"


def list_scene_directories(
    root: str = "data",
    subsets: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    Every scene directory under `root`, found by walking rather than guessing.

    A scene directory is any directory containing both `compressed_mesh.obj`
    and `compressed_data.npz`. The walk PRUNES as soon as one is identified:
    each scene holds ~100 fracture sub-directories, so descending into them
    would turn a ~10k-directory walk into a ~1M-directory one.

    This replaces a hardcoded table of (path, nesting depth) pairs, which was
    wrong in two ways at once: it enumerated only `everyday_compressed` and
    `artifact_compressed`, so the `volume_constrained-*` trees a full release
    ships were never looked at; and it assumed an exact nesting depth per
    source, silently finding nothing when the guess missed. Both failures were
    SILENT -- a wrong guess produced a plausible but too-small scene count with
    no warning, which is precisely the kind of error that survives into a
    thesis's methodology section.

    `subsets` restricts to named top-level directories (e.g.
    ("everyday_compressed", "artifact_compressed") for the standard benchmark).
    None or empty means every scene found anywhere under `root`.
    """
    base = Path(root)
    if not base.exists():
        return []

    wanted = {s.strip() for s in subsets if s and s.strip()} if subsets else None

    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        names = set(filenames)
        if MESH_FILE in names and DATA_FILE in names:
            path = Path(dirpath)
            dirnames[:] = []                       # never descend into fractures
            if wanted is not None:
                relative = path.relative_to(base)
                if not relative.parts or relative.parts[0] not in wanted:
                    continue
            found.append(path)
        else:
            dirnames.sort()                        # deterministic traversal order
    return sorted(found)


def _is_scene_dir(path: Path) -> bool:
    """A scene directory holds the compressed base mesh + piece mapping."""
    return (path / MESH_FILE).is_file() and (path / DATA_FILE).is_file()


# --------------------------------------------------------------------------
# Official Breaking Bad splits
# --------------------------------------------------------------------------
SPLIT_DIR = "data_split"


def load_official_split(
    root: str,
    split: str,
    subsets: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    Scene directories listed in the benchmark's own `data_split/*.txt` files.

    Use these rather than the hash split whenever results are to be compared
    against published numbers: the leaderboard everyone quotes is computed on
    this exact partition, and a different one makes the comparison invalid no
    matter how carefully the model is trained.

    Each line looks like `everyday/BeerBottle/<hash>` or `artifact/<id>`, i.e.
    a subset token followed by the path tail. Objects are matched by that tail,
    so the same list selects either the vanilla or the volume-constrained copy
    depending on which one `subsets` admits.

    The benchmark ships TRAIN and VAL only -- there is no official test split.
    Asking for one raises rather than silently inventing a partition.
    """
    if split == "test":
        raise ValueError(
            "The Breaking Bad release provides no official test split (only "
            "*.train.txt and *.val.txt). Evaluate on --split val to match the "
            "published tables, or use split_source='hash' for a three-way split."
        )
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    base = Path(root)
    split_dir = base / SPLIT_DIR
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. The official splits ship with the dataset; "
            f"use split_source='hash' if your copy lacks them."
        )

    # Index the scenes on disk by every path suffix of 1..3 components, and
    # match split entries the same way, most specific first. Matching by
    # SUFFIX rather than by a fixed prefix convention means the lookup does not
    # care how a subset nests, nor what token the split file happens to lead
    # with -- `everyday/Cat/<hash>`, `artifact/<id>` and any variant of either
    # all resolve, because object ids are unique.
    scenes = list_scene_directories(root, subsets)
    by_suffix: dict = {}
    for scene in scenes:
        parts = scene.relative_to(base).parts
        for k in (1, 2, 3):
            if len(parts) >= k:
                by_suffix.setdefault("/".join(parts[-k:]), []).append(scene)

    selected: List[Path] = []
    unmatched: List[str] = []
    seen: set = set()
    # rglob, not glob. The release nests the lists one level deeper --
    # data/data_split/data_split/everyday.train.txt -- exactly as it does with
    # everyday_compressed/everyday_compressed/. A non-recursive glob found
    # nothing there and raised "matched anything on disk" with zero entries
    # parsed, which reads like a format mismatch rather than a lookup that
    # never opened a file. Searching recursively removes the assumption, the
    # same way `list_scene_directories` does for the scenes themselves.
    listings = sorted(split_dir.rglob(f"*.{split}.txt"))
    total_entries = 0

    for listing in listings:
        for line in listing.read_text(encoding="utf-8-sig").splitlines():
            entry = line.strip().strip("/")
            if not entry:
                continue
            total_entries += 1
            entry_parts = Path(entry).parts
            resolved = None
            for k in (3, 2, 1):                      # most specific first
                if len(entry_parts) < k:
                    continue
                matches = by_suffix.get("/".join(entry_parts[-k:]), [])
                if len(matches) == 1:
                    resolved = matches[0]
                    break
                if len(matches) > 1:
                    raise ValueError(
                        f"'{entry}' matches {len(matches)} directories on disk "
                        f"({[str(m) for m in matches[:3]]}). Restrict `data_subsets` to a "
                        f"single variant -- the vanilla and volume-constrained copies "
                        f"share object ids, so both match the same split entry."
                    )
            if resolved is None:
                unmatched.append(entry)
                continue
            if resolved not in seen:
                seen.add(resolved)
                selected.append(resolved)

    if not listings:
        present = sorted(p.name for p in split_dir.rglob("*.txt"))[:6]
        raise ValueError(
            f"No *.{split}.txt files found anywhere under {split_dir}.\n"
            f"  .txt files present: {present or 'none'}\n"
            f"Expected names like 'everyday.{split}.txt'. If the directory is empty, "
            f"the split lists were not downloaded -- use split_source='hash'."
        )

    if not selected:
        disk_examples = sorted(k for k in by_suffix if "/" in k)[:3] or sorted(by_suffix)[:3]
        raise ValueError(
            f"No scenes from {split_dir}/*.{split}.txt matched anything on disk.\n"
            f"  split files read : {[f.name for f in listings]}\n"
            f"  entries parsed   : {total_entries}\n"
            f"  scenes on disk   : {len(scenes)} under subsets "
            f"{list(subsets) if subsets else 'ALL'}\n"
            f"  example entries  : {unmatched[:3]}\n"
            f"  example on disk  : {disk_examples}\n"
            f"Compare the two: the object ids should be identical. If they are not, the "
            f"split lists describe a different copy of the dataset -- use "
            f"split_source='hash' instead."
        )
    if unmatched:
        # Expected when a subset was not downloaded (e.g. `other`, 4,050 objects).
        from ..utils.progress import write

        write(f"  [split] {len(unmatched)} of {total_entries} official {split} entries are "
              f"not on disk (subsets not downloaded); using the {len(selected)} that are.")
    return sorted(selected)


def assign_split(name: str, val_frac: float, test_frac: float, seed: int) -> str:
    """Stable hash-bucket assignment for one directory name."""
    h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


@lru_cache(maxsize=32)
def _scene_pool_cached(
    root: str,
    split: Optional[str],
    val_frac: float,
    test_frac: float,
    split_seed: int,
    max_scenes: int,
    subsets: tuple = (),
    split_source: str = "hash",
) -> tuple:
    """
    The list of eligible scene directories for one split.

    Cached because it is a pure function of its arguments and involves walking
    a directory tree of tens of thousands of entries -- doing that per
    `__getitem__` would dominate loading time. This caches *paths only*: no
    mesh, feature, or tensor data is ever retained.
    """
    if split_source == "official":
        if split is None:
            raise ValueError("split_source='official' requires an explicit split")
        pool = load_official_split(root, split, subsets or None)
        if max_scenes and max_scenes > 0:
            pool = sorted(pool, key=lambda d: hashlib.md5(f"sub:{d.name}".encode()).hexdigest())[:max_scenes]
        return tuple(pool)

    all_dirs = list_scene_directories(root, subsets or None)
    if not all_dirs:
        raise FileNotFoundError(
            f"No Breaking Bad scene directories found under '{root}'"
            + (f" restricted to subsets {list(subsets)}" if subsets else "")
            + f". A scene directory is one containing both '{MESH_FILE}' and '{DATA_FILE}'. "
            f"Run `python -m scripts.inspect_data --root {root}` to see what is actually there."
        )
    if split is None:
        pool = all_dirs
    else:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS} or None, got {split!r}")
        pool = [d for d in all_dirs if assign_split(d.name, val_frac, test_frac, split_seed) == split]
        if not pool:
            raise ValueError(
                f"No directories assigned to split={split!r} with val_frac={val_frac}, "
                f"test_frac={test_frac}, split_seed={split_seed}."
            )
    if max_scenes and max_scenes > 0:
        # Deterministic bounded subset: sort by hash so the subset is stable
        # across runs/ranks and is not biased by alphabetical category order.
        pool = sorted(pool, key=lambda d: hashlib.md5(f"sub:{d.name}".encode()).hexdigest())[:max_scenes]
    return tuple(pool)


def scene_pool(
    root: str = "data",
    split: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
    max_scenes: int = 0,
    subsets: Optional[Sequence[str]] = None,
    split_source: str = "hash",
) -> tuple:
    return _scene_pool_cached(
        root, split, val_frac, test_frac, split_seed, max_scenes,
        tuple(subsets or ()), split_source,
    )


def get_random_directory(
    root: str = "data",
    split: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
    max_scenes: int = 0,
    subsets: Optional[Sequence[str]] = None,
    split_source: str = "hash",
) -> Path:
    """Uniformly random scene directory from the requested split."""
    return random.choice(
        scene_pool(root, split, val_frac, test_frac, split_seed, max_scenes,
                   subsets, split_source)
    )
