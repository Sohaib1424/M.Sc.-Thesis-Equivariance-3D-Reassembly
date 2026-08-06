"""
Scene discovery and deterministic train/val/test splitting.

Why a hash-based split rather than an index range: ``BreakingBadDataset``
draws a *fresh random scene* on every ``__getitem__`` call and ignores its
index entirely, so slicing the dataset object (``ds[:8000]`` vs ``ds[8000:]``)
partitions nothing -- both halves sample from the identical pool, with
replacement. Any benchmark number produced that way is contaminated.

Here each scene *directory name* is hashed (salted with a split seed) into a
stable fraction in [0, 1) and bucketed. Consequences:

* the same shape always lands in the same split, run to run, machine to
  machine, regardless of iteration order;
* the three splits are disjoint by construction (one name -> one bucket);
* no state needs to be persisted alongside a checkpoint to reproduce a split,
  only ``split_seed`` / ``val_frac`` / ``test_frac``.

``hashlib.md5`` is used rather than Python's built-in ``hash()`` because the
latter is salted per-process for strings (PYTHONHASHSEED), which would make
the split silently differ between runs and, worse, between DDP ranks.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SPLITS = ("train", "val", "test")

# Breaking Bad ships two subsets with different nesting depths on disk.
# depth=1 -> <root>/<category>/<shape>/   depth=0 -> <root>/<shape>/
_DEFAULT_SOURCES: Sequence[Dict] = (
    {"path": "everyday_compressed/everyday_compressed", "depth": 1},
    {"path": "artifact_compressed/artifact_compressed", "depth": 0},
)


def list_scene_directories(
    root: str = "data",
    sources: Optional[Sequence[Dict]] = None,
) -> List[Path]:
    """Every leaf scene directory under ``root``, across all dataset subsets.

    Returns a *sorted* list so that iteration order is deterministic; the OS
    does not guarantee ``Path.iterdir()`` ordering, and non-determinism here
    would leak into anything that samples by position.
    """
    sources = _DEFAULT_SOURCES if sources is None else sources
    root_path = Path(root)
    found: List[Path] = []

    for entry in sources:
        base = root_path / entry["path"]
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if entry["depth"] == 1:
                found.extend(sub for sub in child.iterdir() if sub.is_dir())
            else:
                found.append(child)

    return sorted(found)


def assign_split(name: str, val_frac: float, test_frac: float, seed: int) -> str:
    """Map a scene directory *name* to 'train' / 'val' / 'test', deterministically."""
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError(
            f"val_frac + test_frac must be in [0, 1); got {val_frac} + {test_frac}"
        )
    digest = hashlib.md5(f"{seed}:{name}".encode("utf-8")).hexdigest()
    frac = (int(digest, 16) % 10_000) / 10_000.0
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def split_scene_directories(
    root: str = "data",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    sources: Optional[Sequence[Dict]] = None,
) -> Dict[str, List[Path]]:
    """All scene directories, bucketed into the three splits."""
    all_dirs = list_scene_directories(root, sources=sources)
    buckets: Dict[str, List[Path]] = {s: [] for s in SPLITS}
    for d in all_dirs:
        buckets[assign_split(d.name, val_frac, test_frac, seed)].append(d)
    return buckets


class SceneIndex:
    """Caches the (expensive) directory walk and the split assignment.

    ``BreakingBadDataset.__getitem__`` is called tens of thousands of times per
    run. Re-walking the dataset tree on every single call -- which the original
    code did -- costs a full recursive ``iterdir()`` per sample, on top of the
    real mesh work. On a network/FUSE-mounted Kaggle input directory that is
    not a rounding error. Walking once and reusing the list is a strict win
    with no behavioural change (the directory tree does not mutate mid-run).
    """

    def __init__(
        self,
        root: str = "data",
        split: Optional[str] = None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 0,
        sources: Optional[Sequence[Dict]] = None,
    ):
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")

        self.root = root
        self.split = split
        all_dirs = list_scene_directories(root, sources=sources)
        if not all_dirs:
            raise FileNotFoundError(
                f"No scene directories found under {root!r}. Expected a layout like "
                f"{root}/everyday_compressed/everyday_compressed/<category>/<shape>/ or "
                f"{root}/artifact_compressed/artifact_compressed/<shape>/."
            )

        if split is None:
            self.directories = all_dirs
        else:
            self.directories = [
                d for d in all_dirs
                if assign_split(d.name, val_frac, test_frac, seed) == split
            ]
            if not self.directories:
                raise ValueError(
                    f"No directories landed in split={split!r} "
                    f"(val_frac={val_frac}, test_frac={test_frac}, seed={seed}) "
                    f"out of {len(all_dirs)} total. Either the fractions are too small "
                    f"for this many shapes, or --root_dir points somewhere unexpected."
                )

    def __len__(self) -> int:
        return len(self.directories)

    def sample(self, rng: Optional[random.Random] = None) -> Path:
        """Draw one scene directory uniformly at random from this split."""
        chooser = rng.choice if rng is not None else random.choice
        return chooser(self.directories)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SceneIndex(root={self.root!r}, split={self.split!r}, "
                f"n_scenes={len(self.directories)})")


def get_random_directory(
    root: str = "data",
    split: Optional[str] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 0,
) -> Path:
    """Backwards-compatible one-shot helper (walks the tree every call).

    Prefer :class:`SceneIndex` inside anything that samples repeatedly.
    """
    return SceneIndex(
        root=root, split=split, val_frac=val_frac, test_frac=test_frac, seed=split_seed
    ).sample()
