"""
What a run is actually trained on: which shapes, which break patterns, and how
often each is drawn.

:mod:`reassembly.data.paths` answers "what is on disk". This module answers the
three questions that decide what a number means:

* **Which shapes count as one shape.** Breaking Bad ships the same objects
  under more than one fracture-generation mode, in sibling top-level
  directories (``everyday_compressed`` and
  ``volume_constrained-everyday_compressed``). Treated as separate scene
  directories they double the object count -- the preflight for this project
  reported 809 train / 181 val against an official partition of 407 / 91 --
  and, worse, they make "held out by object" untrue, because a shape can be in
  train under one directory and in val under the other. :func:`build_catalog`
  groups them into one :class:`ObjectEntry` whose mode list is the union, so no
  break pattern is discarded and every object is counted once.

* **What is held out.** Two different generalisation questions, two split
  modes. See :func:`split_catalog`.

* **How often each object is drawn.** The categories are not balanced --
  Everyday has 17 distinct bottles against 5 cups -- so under natural sampling
  a bottle-shaped prior is the cheapest thing the model can learn. See
  :func:`balance_weights`.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .paths import Scene, assign_split, scene_split_key

SPLIT_MODES = ("object", "fracture")
BALANCE_SCHEMES = ("none", "category", "object")


@dataclass(frozen=True)
class ObjectEntry:
    """
    One distinct shape, with every break pattern available for it.

    ``modes`` holds ``(scene directory, mode name)`` pairs rather than mode
    names alone, because after grouping, one shape's patterns can live in more
    than one directory.
    """

    key: str                 # "everyday_compressed/Bottle/2927d6c8..."
    subset: str              # base subset, variant prefixes stripped
    category: str            # "Bottle"; "" for subsets with no category level
    name: str
    modes: Tuple[Tuple[Path, str], ...]
    split_key: Tuple[str, ...] = ()
    """
    The identity used to match against Breaking Bad's split lists, taken from
    :func:`~reassembly.data.paths.scene_split_key` at build time rather than
    reconstructed here. Two functions deriving "the same" key independently is
    how an object ends up matching in one place and not the other, and the
    symptom -- a split quietly smaller than the published one -- looks like a
    dataset problem rather than a key problem.
    """

    def __len__(self) -> int:
        return len(self.modes)


@dataclass(frozen=True)
class Catalog:
    """An ordered, deduplicated set of objects and their break patterns."""

    objects: Tuple[ObjectEntry, ...]

    def __len__(self) -> int:
        return len(self.objects)

    @property
    def num_modes(self) -> int:
        return sum(len(entry) for entry in self.objects)

    def category_counts(self) -> Dict[str, int]:
        """Objects per category, most-populated first."""
        counts = Counter(entry.category or "(uncategorised)" for entry in self.objects)
        return dict(counts.most_common())

    def items(self) -> List[Tuple[int, Path, str]]:
        """``(object index, scene directory, mode)`` for every break pattern."""
        return [(index, directory, mode)
                for index, entry in enumerate(self.objects)
                for directory, mode in entry.modes]


# ------------------------------------------------------------------ build --

def build_catalog(scenes: Sequence[Scene],
                  mode_filter: Optional[str] = None) -> Catalog:
    """
    Group scene directories into distinct shapes.

    ``mode_filter`` keeps only mode directories whose name starts with it --
    ``"fractured_"`` restricts to the standard break patterns and excludes the
    ``mode_*`` variants. ``None`` keeps everything.

    Objects contributing no usable mode are dropped rather than carried as
    empty entries, so ``len(catalog)`` is the number of objects a run can
    actually draw from.
    """
    grouped: Dict[str, List[Scene]] = defaultdict(list)
    for scene in scenes:
        grouped[scene.object_key].append(scene)

    entries: List[ObjectEntry] = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda s: s.path)
        modes: List[Tuple[Path, str]] = []
        for scene in members:
            for directory in scene.mode_dirs():
                if mode_filter and not directory.name.startswith(mode_filter):
                    continue
                modes.append((scene.path, directory.name))
        if not modes:
            continue
        first = members[0]
        entries.append(ObjectEntry(
            key=key,
            # `object_key` is built from the base subset, so a group whose only
            # member is a variant directory still reports the base name.
            subset=key.split("/")[0],
            category=first.category,
            name=first.name,
            modes=tuple(modes),
            split_key=scene_split_key(first),
        ))
    return Catalog(tuple(entries))


# ------------------------------------------------------------------ split --

def split_catalog(
    catalog: Catalog,
    split: str,
    *,
    split_by: str = "object",
    official: Optional[Dict[str, set]] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    fracture_pool: str = "train",
) -> Catalog:
    """
    Restrict a catalogue to one split, under one of two held-out definitions.

    ``split_by="object"`` (the default, and the one published numbers are
    comparable against) holds out whole **shapes**. Train and val share no
    object, so validation measures whether the model can orient a fragment of a
    shape it has never seen.

    ``split_by="fracture"`` holds out **break patterns** instead. Every object
    appears in every split; what differs is which of its ~80 fracture modes.
    Validation then measures whether the model can orient a fragment of a
    *known* shape broken in a way it has not seen.

    The second is not a replacement for the first -- it is a strictly easier
    question, and a number from it must never be reported as if it came from
    the object split. It exists because the two together are diagnostic in a
    way neither is alone. A model that scores at chance on the object split and
    well on the fracture split has learned per-shape canonical orientations and
    no transferable notion of how a fracture surface constrains pose; one that
    scores at chance on both has not learned either, and the problem is
    upstream of generalisation.

    Under ``split_by="fracture"`` the pool defaults to the **official training
    objects only** (``fracture_pool="train"``), so the official validation and
    test shapes are never consumed by this experiment and stay available for
    the benchmark number. ``fracture_pool="all"`` uses every object, which is
    fine for a diagnostic and disqualifying for a reported result.

    ``official`` maps split name -> set of ``(category, object)`` keys, as
    :func:`~reassembly.data.paths.load_official_split` produces. When it is
    ``None`` the hash fallback is used for both modes.
    """
    if split_by not in SPLIT_MODES:
        raise ValueError(f"split_by must be one of {SPLIT_MODES}, got {split_by!r}")
    if split_by == "object":
        return Catalog(tuple(e for e in catalog.objects
                             if _object_split(e, official, val_frac, test_frac, seed) == split))

    # --- fracture mode -----------------------------------------------------
    if fracture_pool not in ("train", "all"):
        raise ValueError(f"fracture_pool must be 'train' or 'all', got {fracture_pool!r}")
    pool = catalog.objects
    if fracture_pool == "train" and official is not None:
        pool = tuple(e for e in catalog.objects
                     if _object_split(e, official, val_frac, test_frac, seed) == "train")

    kept: List[ObjectEntry] = []
    for entry in pool:
        modes = partition_modes(entry, val_frac, test_frac, seed)[split]
        if modes:
            kept.append(ObjectEntry(entry.key, entry.subset, entry.category,
                                    entry.name, tuple(modes)))
    return Catalog(tuple(kept))


def _object_split(entry: ObjectEntry, official: Optional[Dict[str, set]],
                  val_frac: float, test_frac: float, seed: int) -> str:
    """Which split one object belongs to, by official list or by hash."""
    if official is None:
        return assign_split(entry.key, val_frac, test_frac, seed)
    for name in ("test", "val", "train"):
        listed = official.get(name)
        if not listed:
            continue
        # Same two-tier match as `filter_by_split`: the (category, object) key,
        # with a per-entry fallback for split files that list bare names. The
        # fallback is per-entry so a list carrying categories is never weakened
        # by one that does not.
        bare = {k[-1] for k in listed if len(k) == 1}
        if entry.split_key in listed or entry.name in bare:
            return name
    # Listed nowhere. Breaking Bad ships train and val lists only, so under a
    # two-list release an unlisted object is genuinely unassigned; sending it
    # to train would quietly enlarge the training set past the published one
    # and make the result incomparable while looking identical.
    return "unlisted"


def partition_modes(entry: ObjectEntry, val_frac: float = 0.1,
                    test_frac: float = 0.1,
                    seed: int = 0) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Split one object's break patterns into train/val/test, disjointly.

    Assignment is by position after a deterministic per-object shuffle, not by
    hashing each mode name independently. Hashing is simpler but gives a
    *binomial* count per object: with 0.1 and eighty modes, roughly one object
    in 3,500 gets no validation mode at all, and the objects that do get one
    get between two and fourteen. Positional assignment makes the count exact,
    so every object contributes to every split in the intended proportion --
    which is the entire premise of this split mode.

    An object with fewer than three modes cannot populate three disjoint
    splits. Rather than emit an empty val set that looks like a result, all of
    its modes go to train and the caller is told how many objects that affected
    (see :func:`fracture_split_report`).
    """
    modes = list(entry.modes)
    if len(modes) < 3:
        return {"train": modes, "val": [], "test": []}

    order = random.Random(f"{seed}:{entry.key}")
    shuffled = modes[:]
    order.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    # Never let the held-out parts crowd out training.
    while n_test + n_val >= n and (n_test + n_val) > 0:
        if n_test >= n_val:
            n_test -= 1
        else:
            n_val -= 1
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def fracture_split_report(catalog: Catalog, val_frac: float = 0.1,
                          test_frac: float = 0.1, seed: int = 0) -> Dict[str, int]:
    """How the fracture split lands, for the startup banner."""
    too_few = sum(1 for e in catalog.objects if len(e) < 3)
    counts = {"train": 0, "val": 0, "test": 0}
    for entry in catalog.objects:
        parts = partition_modes(entry, val_frac, test_frac, seed)
        for name in counts:
            counts[name] += len(parts[name])
    counts["objects"] = len(catalog)
    counts["objects_train_only"] = too_few
    return counts


# ---------------------------------------------------------------- balance --

def balance_weights(catalog: Catalog, scheme: str = "none",
                    temperature: float = 1.0) -> Optional[List[float]]:
    """
    Per-item sampling weights, one per ``(object, mode)`` pair.

    Returns ``None`` for ``scheme="none"`` so the caller can take the plain
    shuffling path rather than a weighted sampler that happens to be uniform.

    THE IMBALANCE
    -------------
    Everyday's categories hold very different numbers of distinct shapes -- 17
    bottles, 13 plates, 5 cups -- and each shape ships a similar number of
    break patterns. Drawing items uniformly therefore shows the model roughly
    three bottles for every cup, and a shape prior is a cheaper thing to fit
    than an orientation rule. Nothing in the loss notices; the validation
    number just comes out weighted towards the categories that happened to be
    large, which is not the quantity anyone means to report.

    THE SCHEMES
    -----------
    ``"category"`` makes the three levels uniform in turn: category, then
    object within category, then mode within object. ``"object"`` equalises
    objects only, which corrects for objects having different numbers of break
    patterns while leaving the category proportions alone.

    ``temperature`` interpolates geometrically between natural frequency
    (``0.0``) and full balance (``1.0``); ``0.5`` is the usual square-root
    softening, which removes most of the imbalance without making a 5-shape
    category as influential per-shape as a 17-shape one. It is a real trade:
    full balance also means each cup is shown 3.4x as often as each bottle,
    which is a different kind of skew rather than none.

    Weights are relative -- the sampler normalises -- so no constant factor is
    carried here.
    """
    object_ids: List[int] = []
    categories: List[str] = []
    for index, entry in enumerate(catalog.objects):
        object_ids.extend([index] * len(entry.modes))
        categories.extend([entry.category] * len(entry.modes))
    return item_weights(categories, object_ids, scheme, temperature)


def item_weights(categories: Sequence[str], object_ids: Sequence[int],
                 scheme: str = "none",
                 temperature: float = 1.0) -> Optional[List[float]]:
    """
    The weight formula itself, over parallel per-item arrays.

    Separate from :func:`balance_weights` because the dataset's item list is
    not the catalogue's: ``modes_per_scene`` and ``limit_*`` both trim it, and
    the weights have to be computed over what will actually be *drawn*. Sharing
    one formula rather than writing it twice is the point -- a weight vector
    that is the right length but misaligned reweights the wrong samples and
    looks like it worked.
    """
    if scheme not in BALANCE_SCHEMES:
        raise ValueError(f"balance must be one of {BALANCE_SCHEMES}, got {scheme!r}")
    if scheme == "none" or temperature <= 0.0:
        return None
    if len(categories) != len(object_ids):
        raise ValueError("categories and object_ids must be the same length")

    modes_per_object: Counter = Counter(object_ids)
    # Objects per category, counted over DISTINCT objects -- counting items
    # would let an object with many break patterns stand in for several
    # objects, which is the imbalance this is meant to remove.
    seen: Dict[int, str] = {}
    for object_id, category in zip(object_ids, categories):
        seen.setdefault(object_id, category)
    objects_per_category: Counter = Counter(seen.values())

    weights: List[float] = []
    for object_id, category in zip(object_ids, categories):
        modes = max(modes_per_object[object_id], 1)
        if scheme == "object":
            target = 1.0 / modes
        else:
            target = 1.0 / (max(objects_per_category[category], 1) * modes)
        weights.append(target ** temperature)
    return weights


def balance_report(catalog: Catalog, scheme: str, temperature: float,
                   top: int = 6) -> List[str]:
    """
    What the weights actually do, as lines for the startup banner.

    Prints the *effective share* each category receives, because that is the
    quantity being controlled and it is not obvious from a temperature. A
    balancing setting that was silently a no-op -- or that over-corrected --
    would otherwise be invisible for a whole run.
    """
    counts = catalog.category_counts()
    if len(counts) <= 1:
        return ["  balance       single category; nothing to balance"]

    weights = balance_weights(catalog, scheme, temperature)
    items = catalog.items()
    if weights is None:
        weights = [1.0] * len(items)
    total = sum(weights) or 1.0

    share: Dict[str, float] = defaultdict(float)
    for (index, _directory, _mode), weight in zip(items, weights):
        share[catalog.objects[index].category or "(uncategorised)"] += weight / total

    ordered = sorted(share.items(), key=lambda kv: -kv[1])
    head = ordered[:top]
    lines = [
        f"  balance       {scheme}"
        + (f" @ T={temperature:g}" if scheme != "none" else "")
        + f"   ({len(counts)} categories, {len(catalog)} objects)",
        "                 " + "  ".join(
            f"{name}:{100 * value:.1f}%" for name, value in head)
        + (f"  (+{len(ordered) - len(head)} more)" if len(ordered) > len(head) else ""),
    ]
    spread = ordered[0][1] / max(ordered[-1][1], 1e-12)
    lines.append(f"                 largest/smallest category share {spread:.1f}x")
    return lines


def effective_sample_size(weights: Sequence[float]) -> float:
    """
    Kish's effective sample size, ``(sum w)^2 / sum w^2``.

    Balancing buys representativeness with variance: drawing with replacement
    from skewed weights means fewer distinct items per epoch than there are
    items. This says how many, so the cost is a number rather than a shrug. A
    value far below ``len(weights)`` means most of an epoch is a handful of
    repeatedly-drawn samples.
    """
    total = float(sum(weights))
    squares = float(sum(w * w for w in weights))
    if squares <= 0.0:
        return 0.0
    return total * total / squares


__all__ = [
    "BALANCE_SCHEMES",
    "Catalog",
    "ObjectEntry",
    "SPLIT_MODES",
    "balance_report",
    "balance_weights",
    "build_catalog",
    "effective_sample_size",
    "fracture_split_report",
    "item_weights",
    "partition_modes",
    "split_catalog",
]
