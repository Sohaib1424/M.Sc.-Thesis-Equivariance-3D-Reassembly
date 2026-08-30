"""
Dataset layout: finding scenes, enumerating fracture modes, assigning splits.

On-disk Breaking Bad looks like this (the doubled directory name is real, it
is how the archives extract)::

    <root>/everyday_compressed/everyday_compressed/<category>/<object>/
    <root>/artifact_compressed/artifact_compressed/<object>/
        compressed_mesh.obj                  the intact fine mesh
        compressed_data.npz                  sparse cell -> fine-vertex matrix
        fractured_<k>/compressed_fracture.npy    cell -> piece label
        mode_<k>/compressed_fracture.npy

Two things the original ``list_scene_directories`` assumed that are worth not
assuming:

* a fixed nesting depth per subset (1 for everyday, 0 for artifact). Real
  extractions vary -- ``data_split/data_split/`` turned up nested one level
  deeper than expected earlier in this project. :func:`find_scenes` instead
  walks until it finds directories that actually contain ``compressed_mesh.obj``,
  so it is right regardless of depth.
* that only those two subsets exist. ``volume_constrained-*`` variants hold
  the *same objects* under a different fracture mode, so counting scene
  directories overstates the number of distinct shapes. :func:`find_scenes`
  takes an explicit subset filter and :func:`object_key` gives the identity
  that deduplicates them.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

MESH_FILE = "compressed_mesh.obj"
CELL_MATRIX_FILE = "compressed_data.npz"
FRACTURE_FILE = "compressed_fracture.npy"

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Scene:
    """One object directory, with the fracture modes available for it."""
    path: Path
    subset: str          # e.g. "everyday_compressed"
    category: str        # "" when the subset has no category level
    name: str            # object directory name

    @property
    def object_key(self) -> str:
        """Identity that is stable across fracture-mode variants of one shape."""
        return f"{base_subset(self.subset)}/{self.category}/{self.name}".strip("/")

    def mode_dirs(self) -> List[Path]:
        """Fracture-mode subdirectories, sorted, each holding a fracture file."""
        return sorted(
            d for d in self.path.iterdir()
            if d.is_dir() and (d / FRACTURE_FILE).is_file()
        )


# Fracture-mode prefixes that wrap a subset name. A directory called
# `volume_constrained-everyday_compressed` holds the *same objects* as
# `everyday_compressed` under a different fracture mode, so the two must
# collapse to one object identity or every object is counted twice.
_MODE_PREFIXES = ("volume_constrained",)


def base_subset(subset: str) -> str:
    """
    ``volume_constrained-everyday_compressed`` -> ``everyday_compressed``.

    Both separators are accepted. The archives are not consistent about it, and
    an unrecognised separator fails silently: ``count_objects`` simply reports
    every mode variant as its own object, which is exactly the 1,442-vs-746
    discrepancy this project has already tripped over once. Splitting on ``_``
    in general is not safe -- it would turn ``everyday_compressed`` into
    ``compressed`` -- so only the known prefixes are stripped.
    """
    for prefix in _MODE_PREFIXES:
        for separator in ("-", "_"):
            token = f"{prefix}{separator}"
            if subset.startswith(token):
                return subset[len(token):]
    return subset.split("-")[-1]


def is_scene_dir(path: Path) -> bool:
    return (path / MESH_FILE).is_file() and (path / CELL_MATRIX_FILE).is_file()


def find_scenes(
    root: str | os.PathLike = "data",
    subsets: Optional[Sequence[str]] = None,
    max_depth: int = 4,
) -> List[Scene]:
    """
    Every scene directory under ``root``, sorted and deduplicated.

    Depth-agnostic: descends until it finds directories holding the mesh and
    cell-matrix files, so it does not care how many levels the archive was
    extracted with. ``subsets`` filters on the top-level directory name (for
    instance ``["everyday_compressed"]``); ``None`` accepts all of them.

    The sort is by path, so iteration order is deterministic across machines
    and runs -- a precondition for resumable exhaustive passes.
    """
    root = Path(root)
    if not root.exists():
        return []

    found: List[Scene] = []
    for top in sorted(p for p in root.iterdir() if p.is_dir()):
        if subsets is not None and top.name not in subsets:
            continue
        for scene_dir in _walk_for_scenes(top, max_depth):
            rel = scene_dir.relative_to(top).parts
            # strip the repeated "<subset>/<subset>" wrapper if present
            parts = [p for p in rel if p != top.name]
            category = "/".join(parts[:-1]) if len(parts) > 1 else ""
            found.append(Scene(scene_dir, top.name, category, scene_dir.name))
    return sorted(found, key=lambda s: s.path)


def _walk_for_scenes(base: Path, max_depth: int) -> Iterable[Path]:
    stack = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (PermissionError, FileNotFoundError):
            continue
        if is_scene_dir(current):
            yield current
            continue                      # scene dirs do not nest
        if depth >= max_depth:
            continue
        stack.extend((c, depth + 1) for c in children)


def find_scene_dirs(root: str | os.PathLike = "data",
                    subsets: Optional[Sequence[str]] = None) -> List[Path]:
    """Paths only, for callers that do not need the metadata."""
    return [s.path for s in find_scenes(root, subsets)]


def count_objects(scenes: Sequence[Scene]) -> int:
    """Distinct shapes, collapsing fracture-mode variants of the same object."""
    return len({s.object_key for s in scenes})


# ---------------------------------------------------------------- splits --

def assign_split(name: str, val_frac: float = 0.1, test_frac: float = 0.1,
                 seed: int = 0) -> str:
    """
    Deterministically bucket a name into train/val/test by stable hash.

    Keyed on the name rather than on position, so a shape lands in the same
    split regardless of directory ordering, machine, or how many scenes were
    scanned. The three buckets are disjoint by construction.

    This is a *fallback*. Breaking Bad ships official split lists, and
    published numbers are only comparable against those -- prefer
    :func:`load_official_split` and keep this for ad-hoc subsets.
    """
    digest = hashlib.md5(f"{seed}:{name}".encode()).hexdigest()
    frac = (int(digest, 16) % 10_000) / 10_000
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def load_official_split(root: str | os.PathLike, split: str,
                        subset: str = "everyday") -> Optional[set]:
    """
    Read Breaking Bad's shipped split lists, if present.

    Looks for ``data_split/**/{subset}.{split}.txt`` anywhere under ``root``
    -- ``rglob`` rather than ``glob`` because the directory is often nested
    one level deeper than expected. Returns a set of object names, or ``None``
    when no list is found so the caller can fall back to :func:`assign_split`.
    """
    root = Path(root)
    patterns = [f"{subset}.{split}.txt", f"{subset}_{split}.txt", f"{split}.txt"]
    for pattern in patterns:
        for candidate in sorted(root.rglob(pattern)):
            lines = [ln.strip() for ln in candidate.read_text().splitlines() if ln.strip()]
            if lines:
                return {Path(ln).name if "/" in ln else ln for ln in lines}
    return None


def filter_by_split(scenes: Sequence[Scene], split: Optional[str],
                    val_frac: float = 0.1, test_frac: float = 0.1,
                    seed: int = 0, official: Optional[set] = None) -> List[Scene]:
    """Restrict scenes to one split, by official list if given, else by hash."""
    if split is None:
        return list(scenes)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
    if official is not None:
        return [s for s in scenes if s.name in official]
    return [s for s in scenes
            if assign_split(s.name, val_frac, test_frac, seed) == split]


def choose_scene(scenes: Sequence[Scene], rng: Optional[random.Random] = None) -> Scene:
    """Uniformly random scene, from an explicit RNG so runs stay reproducible."""
    if not scenes:
        raise FileNotFoundError("no scene directories to choose from")
    return (rng or random).choice(list(scenes))
