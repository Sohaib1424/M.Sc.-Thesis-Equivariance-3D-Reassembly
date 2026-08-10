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
) -> tuple:
    """
    The list of eligible scene directories for one split.

    Cached because it is a pure function of its arguments and involves walking
    a directory tree of tens of thousands of entries -- doing that per
    `__getitem__` would dominate loading time. This caches *paths only*: no
    mesh, feature, or tensor data is ever retained.
    """
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
) -> tuple:
    return _scene_pool_cached(
        root, split, val_frac, test_frac, split_seed, max_scenes, tuple(subsets or ()),
    )


def get_random_directory(
    root: str = "data",
    split: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
    max_scenes: int = 0,
    subsets: Optional[Sequence[str]] = None,
) -> Path:
    """Uniformly random scene directory from the requested split."""
    return random.choice(
        scene_pool(root, split, val_frac, test_frac, split_seed, max_scenes, subsets)
    )
