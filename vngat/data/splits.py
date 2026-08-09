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
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

_SPLITS = ("train", "val", "test")


def list_scene_directories(root: str = "data", layouts: Optional[Sequence[dict]] = None) -> List[Path]:
    """
    All leaf scene directories across both Breaking Bad sources.

    `everyday_compressed` nests one level deeper (category/shape) than
    `artifact_compressed` (shape), hence the per-source `depth`. Missing
    sources are skipped silently so a partial download still works.
    """
    if layouts is None:
        layouts = [
            {"path": f"{root}/everyday_compressed/everyday_compressed", "depth": 1},
            {"path": f"{root}/artifact_compressed/artifact_compressed", "depth": 0},
            # Tolerate the un-nested layouts people end up with after manual
            # extraction, so a valid dataset is never reported as "empty".
            {"path": f"{root}/everyday_compressed", "depth": 1},
            {"path": f"{root}/artifact_compressed", "depth": 0},
            {"path": f"{root}", "depth": 0},
        ]

    seen: set = set()
    all_dirs: List[Path] = []
    for entry in layouts:
        base = Path(entry["path"])
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            candidates = [s for s in sorted(d.iterdir()) if s.is_dir()] if entry["depth"] == 1 else [d]
            for cand in candidates:
                if _is_scene_dir(cand) and cand.resolve() not in seen:
                    seen.add(cand.resolve())
                    all_dirs.append(cand)
    return all_dirs


def _is_scene_dir(path: Path) -> bool:
    """A scene directory holds the compressed base mesh + piece mapping."""
    return (path / "compressed_mesh.obj").is_file() and (path / "compressed_data.npz").is_file()


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
) -> tuple:
    """
    The list of eligible scene directories for one split.

    Cached because it is a pure function of its arguments and involves walking
    a directory tree of tens of thousands of entries -- doing that per
    `__getitem__` would dominate loading time. This caches *paths only*: no
    mesh, feature, or tensor data is ever retained.
    """
    all_dirs = list_scene_directories(root)
    if not all_dirs:
        raise FileNotFoundError(
            f"No Breaking Bad scene directories found under '{root}'. A scene directory is one "
            f"containing both 'compressed_mesh.obj' and 'compressed_data.npz'."
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
) -> tuple:
    return _scene_pool_cached(root, split, val_frac, test_frac, split_seed, max_scenes)


def get_random_directory(
    root: str = "data",
    split: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
    max_scenes: int = 0,
) -> Path:
    """Uniformly random scene directory from the requested split."""
    return random.choice(scene_pool(root, split, val_frac, test_frac, split_seed, max_scenes))
