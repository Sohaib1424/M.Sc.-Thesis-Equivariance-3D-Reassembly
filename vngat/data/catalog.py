"""
What a run draws from: distinct objects, their break patterns, the split, and
how often each object is drawn.

`vngat.data.splits` answers "which scene directories are on disk". This module
answers the three questions that decide what a number MEANS:

* WHICH DIRECTORIES ARE ONE OBJECT. Breaking Bad ships the same shapes under
  more than one fracture-generation mode, in sibling top-level directories
  (`everyday_compressed` and `volume_constrained-everyday_compressed`).
  Counted as separate scenes they double the object count and -- worse -- can
  put one shape in train under one directory and in validation under the
  other. `build_catalog` groups them into one `ObjectEntry` whose break
  patterns are the union, so nothing is discarded and every shape is counted
  once. (The default `data_subsets` excludes the variants; this matters as
  soon as someone includes them.)

* WHAT IS HELD OUT. Two different questions, two split modes -- see
  `split_objects`.

* HOW OFTEN EACH OBJECT IS DRAWN. The previous version drew a scene directory
  uniformly, i.e. every OBJECT equally often. The categories are not balanced
  -- 17 bottles against 5 cups in Everyday -- so a bottle-shaped prior is the
  cheapest thing to learn. `object_weights` can correct for that, and is off by
  default so the old behaviour is one setting away.

* HOW MANY PIECES A SCENE MAY HAVE. Breaking Bad's break patterns run from 2
  to ~100 pieces; the benchmark uses those of 2 to 20. `limit_fragments` keeps
  only the patterns in a given range, and is off by default.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .splits import assign_split, list_scene_directories, read_official_entries

SPLIT_BY = ("object", "fracture")
FRACTURE_POOLS = ("train", "all")
BALANCE_SCHEMES = ("none", "category")

# Prefixes that wrap a subset name without changing which objects it holds.
_VARIANT_PREFIXES = ("volume_constrained",)


def base_subset(subset: str) -> str:
    """`volume_constrained-everyday_compressed` -> `everyday_compressed`."""
    for prefix in _VARIANT_PREFIXES:
        for separator in ("-", "_"):
            token = f"{prefix}{separator}"
            if subset.startswith(token):
                return subset[len(token):]
    return subset


@dataclass(frozen=True)
class ObjectEntry:
    """One distinct shape, with every break pattern available for it."""

    key: str                    # "everyday_compressed/Bottle/<name>", variant prefix stripped
    subset: str                 # base subset, e.g. "everyday_compressed"
    category: str               # "Bottle"; the subset's short name ("artifact") if it has no categories
    category_dir: str           # the category as it appears on disk, "" if none
    name: str                   # object directory name
    modes: Tuple[Tuple[str, str], ...]   # (scene directory, break-pattern directory name)

    def __len__(self) -> int:
        return len(self.modes)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _mode_dirs(scene_dir: str, fracture_pattern: Optional[str]) -> List[str]:
    """
    Break-pattern sub-directories of one scene, sorted.

    Same selection as the loader always used: those starting with
    `fracture_pattern`, or ALL of them when none match -- a scene that happens
    to lack the prefix is still usable rather than silently dropped.
    """
    with os.scandir(scene_dir) as it:
        names = sorted(e.name for e in it if e.is_dir())
    if fracture_pattern:
        chosen = [n for n in names if n.startswith(fracture_pattern)]
        if chosen:
            return chosen
    return names


@lru_cache(maxsize=8)
def _build_catalog_cached(root: str, subsets: Tuple[str, ...],
                          fracture_pattern: str) -> Tuple[ObjectEntry, ...]:
    base = Path(root)
    grouped: Dict[str, list] = defaultdict(list)
    for scene in list_scene_directories(root, list(subsets) or None):
        parts = scene.relative_to(base).parts
        if not parts:
            continue
        top = parts[0]
        subset = base_subset(top)
        short = subset[:-len("_compressed")] if subset.endswith("_compressed") else subset
        # Strip the wrapper directories the archives extract with -- the
        # doubled "<subset>/<subset>", and whatever a variant tree repeats
        # inside itself ("everyday_compressed", "everyday") -- so a variant's
        # object gets the SAME key as the vanilla copy of that shape.
        inner = list(parts[1:-1])
        while inner and inner[0] in (top, subset, short):
            inner.pop(0)
        category_dir = "/".join(inner)
        key = "/".join(p for p in (subset, category_dir, parts[-1]) if p)
        grouped[key].append((scene, subset, category_dir, parts[-1]))

    entries: List[ObjectEntry] = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda m: str(m[0]))
        modes = tuple(
            (str(scene), mode)
            for scene, *_ in members
            for mode in _mode_dirs(str(scene), fracture_pattern or None)
        )
        if not modes:
            continue
        _, subset, category_dir, name = members[0]
        short = subset[:-len("_compressed")] if subset.endswith("_compressed") else subset
        entries.append(ObjectEntry(
            key=key, subset=subset, category=category_dir or short,
            category_dir=category_dir, name=name, modes=modes,
        ))
    return tuple(entries)


def build_catalog(root: str, subsets: Optional[Sequence[str]] = None,
                  fracture_pattern: Optional[str] = None) -> Tuple[ObjectEntry, ...]:
    """
    Every distinct object under `root`, with its break patterns.

    Cached per process: it walks the dataset once (~1k scene directories and
    their pattern sub-directories), and both the train and the validation set
    need it. Only paths are cached -- never a mesh or a tensor.
    """
    wanted = tuple(sorted(s.strip() for s in (subsets or ()) if s and s.strip()))
    objects = _build_catalog_cached(str(root), wanted, fracture_pattern or "")
    if not objects:
        raise FileNotFoundError(
            f"No Breaking Bad scenes found under '{root}'"
            + (f" restricted to subsets {list(wanted)}" if wanted else "")
            + ". A scene directory is one containing both 'compressed_mesh.obj' and "
            "'compressed_data.npz'. Run `python -m scripts.inspect_data --root "
            f"{root}` to see what is actually there."
        )
    return objects


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def official_assignment(objects: Sequence[ObjectEntry], root: str) -> Tuple[Dict[int, str], int]:
    """
    Which official split each object is listed in, by index into `objects`.

    Matching is on (category, object name) first and the object name alone as
    a fallback, against the entries of Breaking Bad's `data_split/*.txt`. It is
    done per OBJECT, not per directory, so the vanilla and volume-constrained
    copies of a shape are one match rather than an ambiguity. Returns
    `(assignment, unmatched_entries)`; entries for subsets that were not
    downloaded are the usual reason for the second number.
    """
    by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    by_name: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(objects):
        if entry.category_dir:
            by_pair[(entry.category_dir.split("/")[-1], entry.name)].append(index)
        by_name[entry.name].append(index)

    assignment: Dict[int, str] = {}
    unmatched = 0
    for split, entries in read_official_entries(root).items():
        for parts in entries:
            candidates = by_pair.get((parts[-2], parts[-1])) if len(parts) >= 3 else None
            if not candidates:
                candidates = by_name.get(parts[-1], [])
            if not candidates:
                unmatched += 1
                continue
            if len(set(candidates)) > 1:
                raise ValueError(
                    f"official entry '{'/'.join(parts)}' matches {len(set(candidates))} "
                    f"different objects ({[objects[i].key for i in candidates[:3]]})."
                )
            index = candidates[0]
            previous = assignment.get(index)
            if previous is not None and previous != split:
                raise ValueError(
                    f"object {objects[index].key} is listed in both the official "
                    f"{previous} and {split} splits -- the split lists are inconsistent."
                )
            assignment[index] = split
    return assignment, unmatched


def object_splits(objects: Sequence[ObjectEntry], root: str, split_source: str,
                  val_frac: float, test_frac: float, seed: int) -> List[str]:
    """
    The object-level split of every object: "train", "val", "test", or
    "unlisted" (official source only -- an object in neither list belongs to no
    split; sending it to train would quietly enlarge the training set past the
    published one while looking identical).

    The hash split keys on the object NAME, exactly as the previous version
    keyed on the scene directory's name, so the same objects land in the same
    split as before.
    """
    if split_source == "official":
        assignment, _ = official_assignment(objects, root)
        if not assignment:
            raise ValueError(
                f"split_source='official' but no object under '{root}' matched the "
                f"data_split/*.txt lists. Check they are present "
                f"(`python -m scripts.inspect_data --root {root}`) or use split_source='hash'."
            )
        return [assignment.get(i, "unlisted") for i in range(len(objects))]
    if split_source != "hash":
        raise ValueError(f"split_source must be 'official' or 'hash', got {split_source!r}")
    return [assign_split(e.name, val_frac, test_frac, seed) for e in objects]


def partition_modes(entry: ObjectEntry, val_frac: float = 0.1, test_frac: float = 0.1,
                    seed: int = 0) -> Dict[str, List[Tuple[str, str]]]:
    """
    Split one object's break patterns into train/val/test, disjointly.

    By POSITION after a deterministic per-object shuffle, not by hashing each
    pattern independently. Hashing gives a binomial count per object: with 0.1
    and eighty patterns, roughly one object in 3,500 gets no validation pattern
    at all and the rest get anywhere from two to fourteen. Positional
    assignment makes the count exact, which is the premise of this split mode.

    An object with fewer than three patterns cannot populate three disjoint
    splits; all of its patterns go to train rather than emitting an empty
    validation set that looks like a result.
    """
    modes = list(entry.modes)
    if len(modes) < 3:
        return {"train": modes, "val": [], "test": []}
    shuffled = modes[:]
    random.Random(f"{seed}:{entry.key}").shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    while n_test + n_val >= n and (n_test + n_val) > 0:     # never crowd out train
        if n_test >= n_val:
            n_test -= 1
        else:
            n_val -= 1
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def split_objects(
    objects: Sequence[ObjectEntry],
    split: Optional[str],
    *,
    root: str,
    split_by: str = "object",
    split_source: str = "hash",
    fracture_pool: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    max_objects: int = 0,
) -> Tuple[ObjectEntry, ...]:
    """
    Restrict the catalogue to one split, under one of two held-out definitions.

    `split_by="object"` (default; the one published numbers are comparable
    against) holds out whole SHAPES. Train and validation share no object, so
    validation asks: can the model orient a fragment of a shape it has never
    seen?

    `split_by="fracture"` holds out BREAK PATTERNS instead. Every object in the
    pool appears in every split; what differs is which of its break patterns.
    Validation then asks: can the model orient a fragment of a KNOWN shape,
    broken in a way it has not seen? That is a strictly easier question and a
    number from it must never be reported as if it came from the object split.
    It exists because the two together are diagnostic: at chance on the object
    split and good on the fracture split means the model learned per-shape
    orientations and nothing that transfers; at chance on both means the
    problem is upstream of generalisation.

    Under `split_by="fracture"` the pool is, by default, only the objects the
    OBJECT split puts in train (`fracture_pool="train"`), so the held-out
    validation and test shapes are never consumed by this experiment and stay
    clean for the benchmark number. `fracture_pool="all"` uses every object --
    fine for a diagnostic, disqualifying for a reported result.

    `max_objects > 0` keeps a deterministic bounded subset of the result, for
    fitting a run into a fixed session budget.
    """
    if split_by not in SPLIT_BY:
        raise ValueError(f"split_by must be one of {SPLIT_BY}, got {split_by!r}")
    if split is not None and split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', 'test' or None, got {split!r}")

    if split is None:
        chosen = list(objects)
    elif split_by == "object":
        if split_source == "official" and split == "test":
            raise ValueError(
                "The Breaking Bad release provides no official test split (only "
                "*.train.txt and *.val.txt). Use split='val' to match the published "
                "tables, split_source='hash' for a three-way split, or "
                "split_by='fracture' to hold out break patterns."
            )
        labels = object_splits(objects, root, split_source, val_frac, test_frac, seed)
        chosen = [e for e, label in zip(objects, labels) if label == split]
    else:
        if fracture_pool not in FRACTURE_POOLS:
            raise ValueError(f"fracture_pool must be one of {FRACTURE_POOLS}, got {fracture_pool!r}")
        pool = list(objects)
        if fracture_pool == "train":
            labels = object_splits(objects, root, split_source, val_frac, test_frac, seed)
            pool = [e for e, label in zip(objects, labels) if label == "train"]
        chosen = []
        for entry in pool:
            modes = partition_modes(entry, val_frac, test_frac, seed)[split]
            if modes:
                chosen.append(ObjectEntry(entry.key, entry.subset, entry.category,
                                          entry.category_dir, entry.name, tuple(modes)))

    chosen = bounded_subset(chosen, max_objects)
    if not chosen:
        raise ValueError(
            f"split={split!r} is empty (split_by={split_by!r}, split_source={split_source!r}, "
            f"val_frac={val_frac}, test_frac={test_frac}, split_seed={seed})."
        )
    return tuple(chosen)


def bounded_subset(objects: Sequence[ObjectEntry], max_objects: int = 0) -> Tuple[ObjectEntry, ...]:
    """
    A deterministic subset of `max_objects` objects; all of them when <= 0.
    Keyed on the name exactly as the previous `max_scenes` was, so the same
    objects are kept.
    """
    chosen = list(objects)
    if max_objects and max_objects > 0:
        chosen = sorted(chosen, key=lambda e: hashlib.md5(f"sub:{e.name}".encode()).hexdigest())
        chosen = sorted(chosen[:max_objects], key=lambda e: e.key)
    return tuple(chosen)


# ---------------------------------------------------------------------------
# Pieces per break pattern
# ---------------------------------------------------------------------------
FRACTURE_LABELS = "compressed_fracture.npy"


@lru_cache(maxsize=None)
def piece_count(scene_dir: str, mode: str) -> int:
    """
    Pieces in one break pattern, read from its label file alone.

    `compressed_fracture.npy` holds a piece label per cell, and `load_scene`
    builds one mesh per label 0..max -- so this is one small file read rather
    than a decompression, cheap enough to count a whole split at start-up. A
    label left with no triangles is dropped by the loader, so the loaded count
    can be lower than this, never higher. Cached, so the training set, the
    validation set and every rebuild read each file once.
    """
    labels = np.load(os.path.join(scene_dir, mode, FRACTURE_LABELS))
    return int(labels.max()) + 1 if labels.size else 0


@dataclass(frozen=True)
class FragmentLimit:
    """What `limit_fragments` kept of one split, for the start-up banner."""

    min_fragments: int
    max_fragments: int
    patterns: int              # break patterns before the limit
    patterns_kept: int
    objects: int
    objects_kept: int
    unreadable: int            # label files that could not be read -- kept
    seconds: float

    def describe(self) -> str:
        text = (f"{self.patterns_kept:,} of {self.patterns:,} break patterns "
                f"({self.objects_kept:,} of {self.objects:,} objects)")
        if self.unreadable:
            text += (f", {self.unreadable} with an unreadable label file "
                     f"kept for the loader to report")
        return text


def limit_fragments(objects: Sequence[ObjectEntry], min_fragments: int = 2,
                    max_fragments: int = 0, workers: int = 16,
                    ) -> Tuple[Tuple[ObjectEntry, ...], Optional[FragmentLimit]]:
    """
    `(objects, report)`: only the break patterns of `min_fragments` to
    `max_fragments` pieces, and only the objects left with at least one.

    `max_fragments <= 0` is no limit: the objects come back as they are and the
    report is None, so a run without the limit is exactly what it was.

    Why here rather than where `min_fragments` is checked, after a scene is
    loaded: a pattern rejected after loading costs a full decompression every
    time it is drawn (and the many-piece patterns are the slowest to load),
    and the fixed validation set replaces a rejected pattern with the object's
    next one, up to `max_retries` of them, before stopping the run. Filtered
    here, both only ever see patterns they can use.

    A pattern whose label file cannot be read is kept: the loader reads the
    same file, and reports the failure the way it always has.
    """
    objects = tuple(objects)
    if max_fragments <= 0:
        return objects, None
    started = time.perf_counter()
    pairs = [pair for entry in objects for pair in entry.modes]

    def count(pair: Tuple[str, str]) -> Optional[int]:
        try:
            return piece_count(*pair)
        except Exception:  # noqa: BLE001 - the loader reports it when drawn
            return None

    # One small file per pattern, thousands per split: bound by the
    # filesystem's latency rather than by CPU, so threads overlap the waits.
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as pool:
        counts = dict(zip(pairs, pool.map(count, pairs)))
    kept = []
    for entry in objects:
        modes = tuple(pair for pair in entry.modes
                      if counts[pair] is None
                      or min_fragments <= counts[pair] <= max_fragments)
        if modes:
            kept.append(replace(entry, modes=modes))
    report = FragmentLimit(
        min_fragments=min_fragments, max_fragments=max_fragments,
        patterns=len(pairs), patterns_kept=sum(len(entry) for entry in kept),
        objects=len(objects), objects_kept=len(kept),
        unreadable=sum(value is None for value in counts.values()),
        seconds=time.perf_counter() - started,
    )
    return tuple(kept), report


def fracture_split_report(objects: Sequence[ObjectEntry], val_frac: float = 0.1,
                          test_frac: float = 0.1, seed: int = 0) -> Dict[str, int]:
    """How the fracture split lands, for the startup banner."""
    counts = {"train": 0, "val": 0, "test": 0}
    for entry in objects:
        for name, modes in partition_modes(entry, val_frac, test_frac, seed).items():
            counts[name] += len(modes)
    counts["objects"] = len(objects)
    counts["objects_train_only"] = sum(1 for e in objects if len(e) < 3)
    return counts


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------
def object_weights(objects: Sequence[ObjectEntry], balance: str = "none",
                   temperature: float = 1.0) -> Optional[List[float]]:
    """
    Relative probability of drawing each object; None means uniform.

    `balance="none"` draws every object equally often -- what the previous
    version always did (a uniformly random scene directory, then a uniformly
    random break pattern inside it).

    `balance="category"` corrects for categories holding different numbers of
    objects. Each object's weight is (1 / objects in its category) ** T:

        T = 0    every object equally likely (same as "none")
        T = 1    every CATEGORY equally likely, and objects uniform within it
        T = 0.5  the usual square-root softening in between

    Full balance is not free. With 17 bottles and 5 cups it shows each cup 3.4x
    as often as each bottle -- a different skew rather than none -- and it
    lowers the number of distinct objects an epoch effectively sees (see
    `effective_sample_size`, printed at startup).
    """
    if balance not in BALANCE_SCHEMES:
        raise ValueError(f"balance must be one of {BALANCE_SCHEMES}, got {balance!r}")
    if balance == "none" or temperature <= 0.0:
        return None
    per_category = Counter(e.category for e in objects)
    return [(1.0 / per_category[e.category]) ** float(temperature) for e in objects]


def effective_sample_size(weights: Optional[Sequence[float]], n: int) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2; n for uniform weights."""
    if weights is None:
        return float(n)
    total = float(sum(weights))
    squares = float(sum(w * w for w in weights))
    return total * total / squares if squares > 0 else 0.0


def category_shares(objects: Sequence[ObjectEntry],
                    weights: Optional[Sequence[float]]) -> Dict[str, float]:
    """Probability mass each category receives under `weights`, largest first."""
    if weights is None:
        weights = [1.0] * len(objects)
    total = float(sum(weights)) or 1.0
    share: Dict[str, float] = defaultdict(float)
    for entry, w in zip(objects, weights):
        share[entry.category] += w / total
    return dict(sorted(share.items(), key=lambda kv: -kv[1]))


def describe(objects: Sequence[ObjectEntry], weights: Optional[Sequence[float]],
             balance: str, temperature: float, top: int = 6) -> List[str]:
    """Startup-banner lines: what the draw distribution actually does."""
    counts = Counter(e.category for e in objects)
    shares = category_shares(objects, weights)
    head = list(shares.items())[:top]
    lines = [
        f"  train objects {len(objects)} in {len(counts)} categories, "
        f"{sum(len(e) for e in objects)} break patterns",
        f"  balance       {balance}" + (f" @ T={temperature:g}" if balance != "none" else "")
        + f"   (effective objects per draw {effective_sample_size(weights, len(objects)):.0f}"
        f" of {len(objects)})",
    ]
    if len(counts) > 1:
        lines.append("                " + "  ".join(f"{k}:{100 * v:.1f}%" for k, v in head)
                     + (f"  (+{len(shares) - len(head)} more)" if len(shares) > len(head) else ""))
    return lines


# ---------------------------------------------------------------------------
# The fixed validation set
# ---------------------------------------------------------------------------
def fixed_items(objects: Sequence[ObjectEntry], count: int = 0,
                seed: int = 0) -> List[Tuple[int, str, str]]:
    """
    A deterministic list of `(object index, scene directory, pattern)`.

    Round-robin over objects, so every object appears once before any appears
    twice, and within each round the objects are INTERLEAVED BY CATEGORY
    (one Bottle, one Cup, one Mug, ..., then the next of each), so a set
    smaller than the number of objects still spans the categories instead of
    being the alphabetically first one. Within a category, and within an
    object's patterns, the order is a fixed shuffle. `count <= 0` means one
    pattern per object.

    Why a FIXED set: the previous validation drew fresh random scenes and fresh
    random rotations every epoch, so two epochs' validation numbers differed by
    sampling noise as well as by learning -- and `best.pt` was chosen on that
    noise. With the same scenes and the same rotations every epoch, a change in
    the number is a change in the model.
    """
    if not objects:
        return []
    count = len(objects) if count <= 0 else int(count)

    by_category: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(objects):
        by_category[entry.category].append(index)
    for name, members in by_category.items():
        random.Random(f"val:{seed}:{name}").shuffle(members)
    order: List[int] = []
    for k in range(max(len(m) for m in by_category.values())):
        for name in sorted(by_category):
            if k < len(by_category[name]):
                order.append(by_category[name][k])

    patterns = {}
    for index in order:
        modes = list(objects[index].modes)
        random.Random(f"val:{seed}:{objects[index].key}").shuffle(modes)
        patterns[index] = modes

    items: List[Tuple[int, str, str]] = []
    round_ = 0
    while len(items) < count:
        added = False
        for index in order:
            if round_ < len(patterns[index]):
                scene, mode = patterns[index][round_]
                items.append((index, scene, mode))
                added = True
                if len(items) >= count:
                    break
        if not added:          # every pattern of every object is already in
            break
        round_ += 1
    return items


__all__ = [
    "BALANCE_SCHEMES", "FRACTURE_LABELS", "FRACTURE_POOLS", "SPLIT_BY", "FragmentLimit",
    "ObjectEntry", "base_subset", "bounded_subset", "build_catalog", "category_shares",
    "describe", "effective_sample_size", "fixed_items", "fracture_split_report",
    "limit_fragments", "object_splits", "object_weights", "official_assignment",
    "partition_modes", "piece_count", "split_objects",
]
