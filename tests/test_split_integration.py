"""
The two new experiment controls, end to end through the real training loop.

The unit tests in ``test_catalog.py`` and ``test_sampling.py`` check the split
and the sampler in isolation. These check the parts nothing else can: that the
dataset, the loader and the epoch loop agree about what the split and the
weights mean, and that the per-category breakdown lines up with the fragments
it claims to describe.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

import reassembly.training as training
from reassembly.training import BreakingBadScenes, Config

CATEGORIES = {"Bottle": 6, "Cup": 2}


def _dataset(root: Path, modes: int = 6) -> Path:
    """A Breaking Bad tree with two categories of very different size."""
    from scipy.sparse import identity, save_npz

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])

    for category, count in CATEGORIES.items():
        for index in range(count):
            directory = (root / "everyday_compressed" / "everyday_compressed"
                         / category / f"{category.lower()}_{index}")
            directory.mkdir(parents=True)
            mesh.export(directory / "compressed_mesh.obj")
            save_npz(directory / "compressed_data.npz",
                     identity(len(vertices), format="csr"))
            for mode in range(modes):
                pieces = 2 + (mode % 3)
                mode_dir = directory / f"fractured_{mode}"
                mode_dir.mkdir()
                labels = np.floor((theta + np.pi) / (2 * np.pi) * pieces)
                np.save(mode_dir / "compressed_fracture.npy",
                        np.clip(labels, 0, pieces - 1).astype(np.int64))
    _write_official(root)
    return root


def _write_official(root: Path) -> None:
    """
    Official-style split lists, so the object split takes the same code path it
    takes on the real dataset rather than the hash fallback.

    One shape per category is held out. A proportional split over six and two
    objects would leave a category with none, and the per-category breakdown
    would then be untestable for the reason the test exists.
    """
    directory = root / "data_split" / "data_split"
    directory.mkdir(parents=True)
    listing = {"train": [], "val": []}
    for category, count in CATEGORIES.items():
        for index in range(count):
            name = f"everyday/{category}/{category.lower()}_{index}"
            listing["val" if index == count - 1 else "train"].append(name)
    for split, names in listing.items():
        (directory / f"everyday.{split}.txt").write_text("\n".join(names))


@pytest.fixture
def root(tmp_path):
    return _dataset(tmp_path / "data")


def _config(root, **kwargs):
    defaults = dict(
        root=str(root), out_dir=str(Path(root).parent / "out"),
        channels=16, heads=4, head_dim=4, embedding_dim=8, workers=0,
        tokens_per_scene=32, batch_size=2, epochs=1, modes_per_scene=None,
        schedule=("intra", "cross"), check_init=False, steps_per_epoch=0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


# --------------------------------------------------------------------------
# The fracture split, through the dataset
# --------------------------------------------------------------------------

def test_fracture_split_gives_the_dataset_shared_shapes_and_disjoint_patterns(root):
    """
    The property the experiment rests on, asserted where it can actually be
    violated: after ``modes_per_scene`` sampling and the item-list build, not
    only in the catalogue.
    """
    config = _config(root, split_by="fracture", val_frac=0.25, test_frac=0.0)
    train = BreakingBadScenes(config, "train")
    val = BreakingBadScenes(config, "val")

    train_objects = {entry.key for entry in train.scenes}
    val_objects = {entry.key for entry in val.scenes}
    assert val_objects and val_objects == train_objects

    def pairs(dataset):
        return {(dataset.scenes[index].key, mode) for index, mode in dataset.items}

    assert pairs(train).isdisjoint(pairs(val))


def test_object_split_gives_the_dataset_disjoint_shapes(root):
    config = _config(root, split_by="object", val_frac=0.25, test_frac=0.0)
    train = BreakingBadScenes(config, "train")
    val = BreakingBadScenes(config, "val")
    assert {e.key for e in train.scenes}.isdisjoint({e.key for e in val.scenes})


def test_modes_per_scene_does_not_leak_a_held_out_pattern(root):
    """
    ``modes_per_scene`` samples from the modes a split OWNS. Sampling from all
    of an object's modes and then filtering would be the obvious ordering and
    would quietly put validation patterns into training.
    """
    config = _config(root, split_by="fracture", val_frac=0.34, test_frac=0.0,
                     modes_per_scene=3)
    train = BreakingBadScenes(config, "train")
    val = BreakingBadScenes(config, "val")
    train_pairs = {(train.scenes[i].key, m) for i, m in train.items}
    val_pairs = {(val.scenes[i].key, m) for i, m in val.items}
    assert train_pairs and val_pairs
    assert train_pairs.isdisjoint(val_pairs)


def test_fracture_split_is_stable_across_epoch_seeds(root):
    """
    ``epoch_seed`` rotates which patterns an epoch SHOWS. It must not move the
    split itself -- a validation set that drifted every epoch would make the
    curve uninterpretable and nothing would say so.
    """
    config = _config(root, split_by="fracture", modes_per_scene=2, val_frac=0.34,
                     test_frac=0.0)
    first = BreakingBadScenes(config, "val", epoch_seed=0)
    second = BreakingBadScenes(config, "val", epoch_seed=7)
    train = BreakingBadScenes(config, "train", epoch_seed=7)
    train_pairs = {(train.scenes[i].key, m) for i, m in train.items}
    for dataset in (first, second):
        assert {(dataset.scenes[i].key, m) for i, m in dataset.items}.isdisjoint(
            train_pairs)


# --------------------------------------------------------------------------
# Balancing, through the loader
# --------------------------------------------------------------------------

def test_balanced_sampling_equalises_the_categories_actually_drawn(root):
    """
    Not "the weights are right" -- that is a unit test -- but that the weights
    reach the sampler aligned with the items, and the epoch the loader yields
    is the one that was asked for.
    """
    config = _config(root, balance="category", balance_temperature=1.0)
    dataset = BreakingBadScenes(config, "train")
    weights = dataset.sampling_weights()
    assert weights is not None and len(weights) == len(dataset.items)

    from reassembly.data.sampling import WeightedDistributedSampler

    sampler = WeightedDistributedSampler(weights, num_samples=40_000, seed=0)
    categories = dataset.categories()
    counts = Counter(categories[index] for index in sampler)
    total = sum(counts.values())
    for share in counts.values():
        assert share / total == pytest.approx(0.5, abs=0.02)


def test_unbalanced_sampling_is_skewed_by_category_size(root):
    """
    The premise, measured on the same path the model sees: without balancing,
    a category's share of an epoch is its share of the SHAPES, which for the
    training split here is five bottles against one cup.
    """
    config = _config(root, balance="none")
    dataset = BreakingBadScenes(config, "train")
    assert dataset.sampling_weights() is None

    shapes = Counter(entry.category for entry in dataset.scenes)
    counts = Counter(dataset.categories())
    assert shapes["Bottle"] > shapes["Cup"], "fixture is not imbalanced"
    assert counts["Bottle"] / counts["Cup"] == pytest.approx(
        shapes["Bottle"] / shapes["Cup"])


def test_validation_is_never_reweighted(root):
    """
    A val number that moved with the sampler could not be compared across
    settings -- the quantity would stop being "error on the val split" and
    become "error on the val split as this run happened to weight it".
    """
    config = _config(root, balance="category")
    val = BreakingBadScenes(config, "val")
    loader = training._loader(val, config, shuffle=False, rank=0, world=1, epoch=0)
    assert loader.sampler is None or not hasattr(loader.sampler, "weights")


def test_the_balanced_loader_yields_a_full_epoch(root):
    config = _config(root, balance="category")
    dataset = BreakingBadScenes(config, "train")
    loader = training._loader(dataset, config, shuffle=True, rank=0, world=1,
                              epoch=0)
    assert len(loader.sampler) == len(dataset)


@pytest.mark.parametrize("balance", ["none", "category"])
@pytest.mark.parametrize("world", [1, 2, 3])
def test_a_fixed_length_epoch_is_the_same_length_on_every_gpu(root, balance, world):
    """
    ``steps_per_epoch`` batches per GPU, whatever the split size, the balancing
    or the number of GPUs -- and the GPUs' shares never overlap. Equal counts are
    not a nicety: every GPU takes part in every optimizer step, so a GPU with
    one batch fewer would leave the others waiting at the last one.
    """
    config = _config(root, balance=balance, steps_per_epoch=7)
    dataset = BreakingBadScenes(config, "train")
    shares = [list(training._loader(dataset, config, shuffle=True, rank=r,
                                    world=world, epoch=3).sampler)
              for r in range(world)]
    assert all(len(share) == 7 * config.batch_size for share in shares)
    loaders = [training._loader(dataset, config, shuffle=True, rank=r, world=world,
                                epoch=3) for r in range(world)]
    assert all(len(loader) == 7 for loader in loaders)
    if balance == "none":
        # One seeded draw, sliced by rank: no index twice until the draw
        # wraps around the dataset.
        drawn = [i for share in shares for i in share]
        assert len(set(drawn)) == min(len(drawn), len(dataset))
    again = list(training._loader(dataset, config, shuffle=True, rank=0,
                                  world=world, epoch=3).sampler)
    assert again == shares[0], "an epoch's draw must be reproducible"


@pytest.mark.parametrize("world", [2, 3])
def test_validation_shards_cover_every_scene_exactly_once(root, world):
    """
    ``DistributedSampler`` pads the last round by repeating scenes, which is
    right for training and wrong for a measurement: the repeats are counted
    twice. Validation shards are unpadded and disjoint.
    """
    config = _config(root)
    val = BreakingBadScenes(config, "val")
    shards = [list(training._loader(val, config, shuffle=False, rank=r,
                                    world=world, epoch=0).sampler)
              for r in range(world)]
    drawn = sorted(i for shard in shards for i in shard)
    assert drawn == list(range(len(val)))


# --------------------------------------------------------------------------
# One real epoch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(split_by="object"),
    dict(split_by="fracture", val_frac=0.25, test_frac=0.0),
    dict(balance="category"),
    dict(balance="object", balance_temperature=0.5),
])
def test_a_real_epoch_runs_under_every_setting(root, overrides):
    """
    Every combination has to survive the actual loop, not just the dataset.
    Config knobs that only work in isolation are worse than none: they run, and
    the number they produce is wrong for a reason nobody looks for.
    """
    config = _config(root, **overrides)
    history = training.train(config)
    assert len(history) == 1
    assert np.isfinite(history[0]["val_geodesic_deg"])


def test_the_epoch_reports_tilt_twist_and_head_collinearity(root):
    history = training.train(_config(root))
    row = history[0]
    for key in ("val_tilt_deg", "val_twist_deg", "val_head_cos"):
        assert key in row and np.isfinite(row[key]), key
    assert 0.0 <= row["val_head_cos"] <= 1.0
    # An untrained frame's residual is close to uniform, so most of it is tilt.
    assert row["val_tilt_deg"] > 20.0


def test_the_category_breakdown_covers_every_fragment(root):
    """
    The breakdown's counts must add up to the fragments the mean was taken
    over. They would not if the labels were recovered from the loader's
    position, because a dropped sample shifts every label after it -- and the
    result would still look like a plausible table.
    """
    history = training.train(_config(root))
    breakdown = history[0]["val_by_category"]
    assert breakdown
    assert set(breakdown) <= set(CATEGORIES) | {"(uncategorised)"}
    counted = sum(entry["fragments"] for entry in breakdown.values())
    assert counted == pytest.approx(history[0]["val_fragments"])


def test_diagnostics_do_not_change_the_optimised_total(root):
    """
    tilt, twist and head|cos| are reported and must be inert. If any of them
    ever entered the total, the model would be optimising a diagnostic and the
    diagnostic would stop being one.
    """
    from reassembly.training import build_criterion

    config = _config(root)
    dataset = BreakingBadScenes(config, "train")
    loader = training._loader(dataset, config, shuffle=False, rank=0, world=1,
                              epoch=0)
    batch, _dropped = next(iter(loader))
    model = training.build_model(config)
    loss, report, _R = training._forward(model, batch, build_criterion(config),
                                         config)

    terms = sum(report[name] * getattr(config, f"w_{name}")
                for name in ("rotation", "position", "normal", "face", "embedding")
                if name in report)
    assert float(loss.detach()) == pytest.approx(terms, rel=1e-5)
    assert {"tilt_deg", "twist_deg", "head_cos"} <= set(report)
