"""
What a run is trained on: which shapes, which break patterns, how often each.

Three separate things are checked here, because each of them has already gone
wrong once in this project or its predecessor and every one of them fails
*silently* -- the run trains, the loss descends, and the resulting number means
something other than what it is labelled.

1. A shape shipped under two fracture-generation modes is ONE shape. Counted as
   two it doubles the object count and, worse, can put the same shape in train
   under one directory and in val under the other.
2. "Held out" has two meanings and they measure different things.
3. The categories are not balanced, so uniform sampling is not neutral.
"""
from __future__ import annotations

from collections import Counter

import pytest

from reassembly.data.catalog import (
    balance_weights,
    build_catalog,
    effective_sample_size,
    fracture_split_report,
    item_weights,
    partition_modes,
    split_catalog,
)
from reassembly.data.paths import find_scenes

# --------------------------------------------------------------------------
# A synthetic dataset tree
# --------------------------------------------------------------------------

LAYOUT = {
    # category -> number of distinct shapes. Deliberately uneven, in the same
    # direction and roughly the same ratio as Everyday's real imbalance.
    "Bottle": 17,
    "Plate": 13,
    "Cup": 5,
    "Mug": 3,
}


def _make_tree(root, modes: int = 8, variants: bool = False) -> None:
    """A Breaking Bad-shaped directory tree, with the doubled subset wrapper."""
    subsets = ["everyday_compressed"]
    if variants:
        subsets.append("volume_constrained-everyday_compressed")
    for subset in subsets:
        for category, count in LAYOUT.items():
            for index in range(count):
                obj = root / subset / subset / category / f"{category.lower()}{index:03d}"
                obj.mkdir(parents=True)
                (obj / "compressed_mesh.obj").write_text("")
                (obj / "compressed_data.npz").write_text("")
                for mode in range(modes):
                    # A variant tree's modes are named differently, as the real
                    # release's are -- if they collided, a "merged" mode list
                    # would silently be the same list twice.
                    prefix = "vc_fractured" if "volume" in subset else "fractured"
                    directory = obj / f"{prefix}_{mode}"
                    directory.mkdir()
                    (directory / "compressed_fracture.npy").write_text("")


def _write_official(root, train_frac: float = 0.8) -> None:
    """Official-style split lists over the shapes, train and val only."""
    listing = {"train": [], "val": []}
    for category, count in LAYOUT.items():
        cut = int(round(count * train_frac))
        for index in range(count):
            name = f"everyday/{category}/{category.lower()}{index:03d}"
            listing["train" if index < cut else "val"].append(name)
    directory = root / "data_split" / "data_split"
    directory.mkdir(parents=True)
    for split, names in listing.items():
        (directory / f"everyday.{split}.txt").write_text("\n".join(names))


@pytest.fixture
def tree(tmp_path):
    _make_tree(tmp_path)
    _write_official(tmp_path)
    return tmp_path


@pytest.fixture
def tree_with_variants(tmp_path):
    _make_tree(tmp_path, variants=True)
    _write_official(tmp_path)
    return tmp_path


def _official(root):
    from reassembly.data.paths import load_official_split

    loaded = {name: load_official_split(root, name, "everyday")
              for name in ("train", "val", "test")}
    return {k: v for k, v in loaded.items() if v}


# --------------------------------------------------------------------------
# 1. One shape is one shape
# --------------------------------------------------------------------------

def test_variant_directories_collapse_to_one_object(tree_with_variants):
    """
    THE double-counting regression guard.

    ``volume_constrained-everyday_compressed`` holds the same shapes as
    ``everyday_compressed`` under a different fracture generator. Treated as
    separate scene directories they double the object count -- this project's
    preflight reported 809 train / 181 val against an official 407 / 91 -- and
    every per-object statement afterwards is over a dataset twice the intended
    size.
    """
    scenes = find_scenes(tree_with_variants)
    assert len(scenes) == 2 * sum(LAYOUT.values()), "fixture did not build both trees"

    catalog = build_catalog(scenes)
    assert len(catalog) == sum(LAYOUT.values())


def test_grouping_keeps_every_break_pattern(tree_with_variants):
    """Deduplicating must not mean discarding: the variant's modes are extra
    break patterns of the same shape, not copies of the same ones."""
    catalog = build_catalog(find_scenes(tree_with_variants))
    assert catalog.num_modes == 2 * 8 * sum(LAYOUT.values())
    for entry in catalog.objects:
        assert len(set(entry.modes)) == len(entry.modes), "duplicate (dir, mode)"


def test_an_object_split_cannot_leak_through_a_variant_directory(tree_with_variants):
    """
    The consequence that matters more than the count.

    Ungrouped, a shape is two objects, and nothing stops one of them being
    assigned to train and the other to val -- at which point the object split
    is not an object split and validation is partly memorisation, with no error
    anywhere.
    """
    catalog = build_catalog(find_scenes(tree_with_variants))
    official = _official(tree_with_variants)
    train = split_catalog(catalog, "train", official=official)
    val = split_catalog(catalog, "val", official=official)
    assert {e.key for e in train.objects} & {e.key for e in val.objects} == set()
    assert len(train) + len(val) == len(catalog)


def test_mode_filter_restricts_to_the_standard_patterns(tree_with_variants):
    catalog = build_catalog(find_scenes(tree_with_variants), mode_filter="fractured_")
    assert catalog.num_modes == 8 * sum(LAYOUT.values())


def test_objects_with_no_usable_mode_are_dropped(tmp_path):
    _make_tree(tmp_path, modes=8)
    catalog = build_catalog(find_scenes(tmp_path), mode_filter="nothing_matches_")
    assert len(catalog) == 0


# --------------------------------------------------------------------------
# 2. Two meanings of "held out"
# --------------------------------------------------------------------------

def test_object_split_holds_out_whole_shapes(tree):
    catalog = build_catalog(find_scenes(tree))
    official = _official(tree)
    train = split_catalog(catalog, "train", split_by="object", official=official)
    val = split_catalog(catalog, "val", split_by="object", official=official)

    assert {e.key for e in train.objects}.isdisjoint({e.key for e in val.objects})
    # Every mode of a val object is in val; none of them is in train.
    assert train.num_modes + val.num_modes == catalog.num_modes


def test_fracture_split_keeps_every_shape_and_holds_out_patterns(tree):
    """
    The split that answers "unseen breakage of a KNOWN shape".

    Train and val must share every object and no break pattern. Sharing a
    pattern would make it a memorisation test; not sharing the objects would
    make it the object split under a different name, and the comparison between
    the two -- which is the entire reason this mode exists -- would be
    measuring both effects at once.
    """
    catalog = build_catalog(find_scenes(tree))
    official = _official(tree)
    train = split_catalog(catalog, "train", split_by="fracture", official=official)
    val = split_catalog(catalog, "val", split_by="fracture", official=official)

    train_objects = {e.key for e in train.objects}
    val_objects = {e.key for e in val.objects}
    assert val_objects == train_objects, "the two splits must share every shape"
    assert val_objects, "the fracture split produced nothing"

    train_modes = {(e.key, m) for e in train.objects for _d, m in e.modes}
    val_modes = {(e.key, m) for e in val.objects for _d, m in e.modes}
    assert train_modes.isdisjoint(val_modes), "a break pattern is in both splits"


def test_fracture_split_draws_only_from_the_official_training_shapes(tree):
    """
    The official val and test shapes must stay untouched by this experiment,
    or the benchmark number is no longer available from the same dataset.
    """
    catalog = build_catalog(find_scenes(tree))
    official = _official(tree)
    held_out = {e.key for e in split_catalog(catalog, "val", split_by="object",
                                             official=official).objects}
    for split in ("train", "val", "test"):
        keys = {e.key for e in split_catalog(catalog, split, split_by="fracture",
                                             official=official).objects}
        assert keys.isdisjoint(held_out), f"{split} reached an official val shape"


def test_fracture_pool_all_uses_every_shape(tree):
    catalog = build_catalog(find_scenes(tree))
    official = _official(tree)
    keys = {e.key for e in split_catalog(catalog, "train", split_by="fracture",
                                         official=official,
                                         fracture_pool="all").objects}
    assert len(keys) == len(catalog)


def test_partition_modes_is_exact_disjoint_and_total(tree):
    """
    Positional assignment, not per-mode hashing.

    Hashing each mode name independently gives a *binomial* count per object:
    at 0.1 over eighty modes some objects would get no validation mode at all
    and others fourteen. Every object contributing in the intended proportion
    is the premise of this split mode, so the count has to be exact.
    """
    catalog = build_catalog(find_scenes(tree))
    for entry in catalog.objects:
        parts = partition_modes(entry, val_frac=0.25, test_frac=0.25, seed=0)
        sizes = {name: len(values) for name, values in parts.items()}
        assert sizes["val"] == 2 and sizes["test"] == 2 and sizes["train"] == 4
        union = [m for values in parts.values() for m in values]
        assert sorted(union) == sorted(entry.modes)
        assert len(set(union)) == len(union)


def test_partition_modes_is_deterministic(tree):
    catalog = build_catalog(find_scenes(tree))
    entry = catalog.objects[0]
    assert partition_modes(entry, seed=3) == partition_modes(entry, seed=3)
    assert partition_modes(entry, seed=3) != partition_modes(entry, seed=4)


def test_an_object_with_too_few_modes_goes_to_train_only(tmp_path):
    """
    Two modes cannot fill three disjoint splits. Emitting an empty val set
    would look like a result; sending the object to train and saying how many
    were affected does not.
    """
    _make_tree(tmp_path, modes=2)
    catalog = build_catalog(find_scenes(tmp_path))
    entry = catalog.objects[0]
    parts = partition_modes(entry)
    assert len(parts["train"]) == 2 and not parts["val"] and not parts["test"]

    report = fracture_split_report(catalog)
    assert report["objects_train_only"] == len(catalog)
    assert report["val"] == 0


def test_held_out_fractions_never_crowd_out_training(tree):
    catalog = build_catalog(find_scenes(tree))
    for entry in catalog.objects:
        parts = partition_modes(entry, val_frac=0.45, test_frac=0.45)
        assert parts["train"], "training was left with nothing"


# --------------------------------------------------------------------------
# 3. The category imbalance
# --------------------------------------------------------------------------

def _shares(catalog, scheme, temperature):
    weights = balance_weights(catalog, scheme, temperature)
    items = catalog.items()
    if weights is None:
        weights = [1.0] * len(items)
    total = sum(weights)
    share: Counter = Counter()
    for (index, _directory, _mode), weight in zip(items, weights):
        share[catalog.objects[index].category] += weight / total
    return share


def test_natural_sampling_is_not_neutral(tree):
    """The premise: without balancing, the model sees 3.4 bottles per cup."""
    catalog = build_catalog(find_scenes(tree))
    share = _shares(catalog, "none", 1.0)
    assert share["Bottle"] / share["Cup"] == pytest.approx(17 / 5, rel=1e-6)


def test_category_balancing_equalises_categories(tree):
    catalog = build_catalog(find_scenes(tree))
    share = _shares(catalog, "category", 1.0)
    for value in share.values():
        assert value == pytest.approx(1 / len(LAYOUT), rel=1e-9)


def test_temperature_interpolates(tree):
    """``0`` natural, ``1`` full balance, ``0.5`` between -- and the softened
    setting must actually reduce the spread rather than merely change it."""
    catalog = build_catalog(find_scenes(tree))
    natural = _shares(catalog, "category", 0.0)
    half = _shares(catalog, "category", 0.5)
    full = _shares(catalog, "category", 1.0)

    def spread(share):
        return max(share.values()) / min(share.values())

    assert spread(natural) == pytest.approx(17 / 3, rel=1e-6)
    assert spread(full) == pytest.approx(1.0, rel=1e-9)
    assert 1.0 < spread(half) < spread(natural)


def test_zero_temperature_disables_balancing(tree):
    catalog = build_catalog(find_scenes(tree))
    assert balance_weights(catalog, "category", 0.0) is None
    assert balance_weights(catalog, "none", 1.0) is None


def test_object_balancing_equalises_objects_not_categories(tmp_path):
    """
    ``object`` corrects for objects owning different numbers of break patterns
    and deliberately leaves the category proportions alone. A test where every
    object has the same mode count could not tell the two schemes apart, so the
    fixture gives one object twice as many.
    """
    _make_tree(tmp_path, modes=4)
    root = tmp_path / "everyday_compressed" / "everyday_compressed" / "Cup" / "cup000"
    for extra in range(4, 12):
        directory = root / f"fractured_{extra}"
        directory.mkdir()
        (directory / "compressed_fracture.npy").write_text("")

    catalog = build_catalog(find_scenes(tmp_path))
    weights = balance_weights(catalog, "object", 1.0)
    items = catalog.items()
    per_object: Counter = Counter()
    for (index, _d, _m), weight in zip(items, weights):
        per_object[index] += weight
    assert len(set(round(v, 12) for v in per_object.values())) == 1

    share = _shares(catalog, "object", 1.0)
    assert share["Bottle"] / share["Cup"] == pytest.approx(17 / 5, rel=1e-6)


def test_item_weights_counts_objects_not_items_per_category(tree):
    """
    A category's weight must divide by how many distinct OBJECTS it has, not
    how many items. Counting items lets an object with many break patterns
    stand in for several objects, which is the imbalance being removed.
    """
    categories = ["A"] * 10 + ["B"] * 2      # 1 object in A, 2 in B
    object_ids = [0] * 10 + [1, 2]
    weights = item_weights(categories, object_ids, "category", 1.0)
    total = sum(weights)
    share_a = sum(w for w, c in zip(weights, categories) if c == "A") / total
    assert share_a == pytest.approx(0.5, rel=1e-9)


def test_effective_sample_size_reports_the_cost_of_balancing(tree):
    """
    Balancing buys representativeness with variance. Kish's ESS says how much,
    so the trade is a number rather than a shrug.
    """
    catalog = build_catalog(find_scenes(tree))
    items = len(catalog.items())
    assert effective_sample_size([1.0] * items) == pytest.approx(items)

    weights = balance_weights(catalog, "category", 1.0)
    assert 0 < effective_sample_size(weights) < items


def test_unknown_scheme_is_rejected(tree):
    catalog = build_catalog(find_scenes(tree))
    with pytest.raises(ValueError, match="balance must be one of"):
        balance_weights(catalog, "sqrt", 1.0)


def test_unknown_split_mode_is_rejected(tree):
    catalog = build_catalog(find_scenes(tree))
    with pytest.raises(ValueError, match="split_by must be one of"):
        split_catalog(catalog, "train", split_by="object_or_something")
