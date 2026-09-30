"""
`max_fragments`: only the break patterns of `min_fragments` to `max_fragments`
pieces, for training, validation and evaluation alike -- the benchmark's 2-20
is `--max_fragments 20`.

Each property below fails without an error when broken: a limit applied after
the object subset trains on fewer objects than asked for, one applied before
the split moves patterns between train and validation, and one left out of
the resume logic compares best.pt across two different validation sets. All
of them still print plausible numbers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import write_scene  # noqa: E402

from vngat.config import Config, parse_config  # noqa: E402
from vngat.data.catalog import (  # noqa: E402
    bounded_subset, build_catalog, fixed_items, limit_fragments, piece_count, split_objects,
)
from vngat.data.dataset import BreakingBadDataset, dataset_kwargs  # noqa: E402
from vngat.data.io import load_scene  # noqa: E402

BASE = "everyday_compressed/everyday_compressed"
# Pieces in each break pattern, fractured_0, fractured_1, ...
LAYOUT = {
    "Bottle/b0": [2, 3, 6],        # train: keeps two
    "Bottle/b1": [5, 6],           # train: nothing in range, the object goes
    "Cup/c0": [1, 3],              # train: the single piece goes too
    "Bottle/b2": [6, 3],           # val: keeps one
    "Cup/c1": [6],                 # val: nothing in range
}
OFFICIAL = {"train": ["everyday/Bottle/b0", "everyday/Bottle/b1", "everyday/Cup/c0"],
            "val": ["everyday/Bottle/b2", "everyday/Cup/c1"]}
LOW, HIGH = 2, 4


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "data"
    for i, (rel, pieces) in enumerate(sorted(LAYOUT.items())):
        for k, count in enumerate(pieces):
            # Same seed per object: the same mesh, cut into `count` pieces.
            write_scene(root / BASE / rel, modes=(f"fractured_{k}",), pieces=count, seed=i)
    split_dir = root / "data_split" / "data_split"
    split_dir.mkdir(parents=True)
    for split, names in OFFICIAL.items():
        (split_dir / f"everyday.{split}.txt").write_text("\n".join(names) + "\n")
    return str(root)


def _objects(root, split="train", **kw):
    catalog = build_catalog(root, ["everyday_compressed"], "fractured_")
    return split_objects(catalog, split, root=root, split_source="official", **kw)


def _dataset(root, **kw):
    base = dict(subsets=["everyday_compressed"], fracture_pattern="fractured_",
                split_source="official", correspondence_tol=1e-5)
    base.update(kw)
    return BreakingBadDataset(root, **base)


def _pairs(objects):
    return {(scene, mode) for entry in objects for scene, mode in entry.modes}


def _names(objects):
    return sorted(entry.name for entry in objects)


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def test_the_label_file_count_is_what_the_loader_builds(tree):
    """The count reads one small file instead of decompressing; on patterns
    with no empty label it must equal the number of meshes the loader makes."""
    for entry in _objects(tree, "train") + _objects(tree, "val"):
        for scene, mode in entry.modes:
            assert piece_count(scene, mode) == len(load_scene(scene, mode)), (entry.name, mode)
    expected = {name.split("/")[1]: pieces for name, pieces in LAYOUT.items()}
    for entry in _objects(tree, "train"):
        counts = [piece_count(scene, mode) for scene, mode in entry.modes]
        assert counts == expected[entry.name]


# --------------------------------------------------------------------------
# What the limit keeps
# --------------------------------------------------------------------------

def test_no_limit_changes_nothing(tree):
    objects = _objects(tree)
    same, report = limit_fragments(objects, LOW, 0)
    assert same == objects and report is None
    # The dataset's own lists are exactly the unfiltered ones.
    val = _dataset(tree, split="val", fixed=True)
    assert val.objects == _objects(tree, "val")
    assert val.items == fixed_items(_objects(tree, "val"), 0, 0)


def test_the_limit_keeps_only_patterns_in_range_and_drops_emptied_objects(tree):
    kept, report = limit_fragments(_objects(tree), LOW, HIGH)
    assert {e.name: [mode for _, mode in e.modes] for e in kept} == {
        "b0": ["fractured_0", "fractured_1"],       # 2 and 3 pieces; 6 goes
        "c0": ["fractured_1"],                      # 3; the single piece goes
    }
    assert (report.patterns, report.patterns_kept) == (7, 3)
    assert (report.objects, report.objects_kept) == (3, 2)
    assert report.unreadable == 0
    assert report.describe() == "3 of 7 break patterns (2 of 3 objects)"


@pytest.mark.parametrize("split_by", ["object", "fracture"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_the_limit_never_moves_a_pattern_between_splits(tree, split_by, split):
    """Applied after the split, the limit can only remove: the patterns left
    in each split are that split's own patterns in range, nothing else."""
    kw = dict(split=split, split_by=split_by, fracture_pool="all")
    try:
        everything = _pairs(_dataset(tree, **kw).objects)
    except ValueError:                      # a split this small can be empty
        pytest.skip(f"no {split} patterns under split_by={split_by}")
    in_range = {p for p in everything if LOW <= piece_count(*p) <= HIGH}
    if not in_range:
        with pytest.raises(ValueError, match="no break pattern"):
            _dataset(tree, max_fragments=HIGH, **kw)
        return
    assert _pairs(_dataset(tree, max_fragments=HIGH, **kw).objects) == in_range


def test_max_objects_counts_objects_that_still_have_a_pattern(tree):
    """Filtering after the subset would pick b1, lose it to the limit, and
    train on one object where two were asked for."""
    unfiltered = _objects(tree)
    naive, _ = limit_fragments(bounded_subset(unfiltered, 2), LOW, HIGH)
    assert len(naive) == 1                  # the failure this guards against
    ds = _dataset(tree, split="train", max_objects=2, max_fragments=HIGH)
    assert _names(ds.objects) == ["b0", "c0"]


def test_an_unreadable_label_file_is_kept_for_the_loader_to_report(tree):
    objects = _objects(tree)
    scene, mode = objects[0].modes[0]
    piece_count.cache_clear()
    with open(f"{scene}/{mode}/compressed_fracture.npy", "wb") as fh:
        fh.write(b"not an array")
    kept, report = limit_fragments(objects, LOW, HIGH)
    assert (scene, mode) in _pairs(kept)
    assert report.unreadable == 1 and "unreadable" in report.describe()
    piece_count.cache_clear()


# --------------------------------------------------------------------------
# What the run sees
# --------------------------------------------------------------------------

def test_training_and_validation_only_see_patterns_in_range(tree):
    val = _dataset(tree, split="val", fixed=True, max_fragments=HIGH)
    assert _names(val.objects) == ["b2"] and len(val) == 1
    assert val[0]["target"].num_fragments == 3
    assert len(_dataset(tree, split="val", fixed=True)) == 2      # c1 without the limit

    train = _dataset(tree, split="train", nominal_length=4, max_fragments=HIGH)
    np.random.seed(0)
    drawn = {train[i]["target"].num_fragments for i in range(12)}
    assert drawn <= {2, 3} and drawn


def test_the_banner_says_what_the_limit_removed(tree):
    from vngat.training.trainer import _data_banner

    cfg = Config(root_dir=tree, data_subsets="everyday_compressed", fracture_pattern="fractured_",
                 split_source="official", max_fragments=HIGH)
    train = BreakingBadDataset(split="train", **dataset_kwargs(cfg))
    val = BreakingBadDataset(split="val", fixed=True, **dataset_kwargs(cfg))
    text = "\n".join(_data_banner(cfg, train, val, 1))
    assert "2-4 pieces per scene" in text
    assert "train keeps 3 of 7 break patterns (2 of 3 objects)" in text
    assert "val   keeps 1 of 3 break patterns (1 of 2 objects)" in text

    cfg.max_fragments = 0
    train = BreakingBadDataset(split="train", **dataset_kwargs(cfg))
    val = BreakingBadDataset(split="val", fixed=True, **dataset_kwargs(cfg))
    assert "no upper limit" in "\n".join(_data_banner(cfg, train, val, 1))


# --------------------------------------------------------------------------
# The setting
# --------------------------------------------------------------------------

def test_the_setting_is_validated_and_reaches_the_dataset():
    for bad in (dict(max_fragments=-1), dict(min_fragments=3, max_fragments=2)):
        with pytest.raises(ValueError, match="max_fragments"):
            Config(**bad).validate()
    Config(max_fragments=0).validate()
    cfg = parse_config(["--max_fragments", "20"])
    assert cfg.max_fragments == 20 and "max_fragments" in cfg._explicit
    assert dataset_kwargs(cfg)["max_fragments"] == 20


def test_a_resume_keeps_the_limit_and_a_changed_one_resets_the_best(tmp_path):
    from test_checkpoint import _write_stub_checkpoint

    from vngat.training.trainer import (
        _RESTORED_FIELDS, _fragment_limit_changed, adopt_checkpoint_config,
    )

    assert "max_fragments" in _RESTORED_FIELDS
    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(max_fragments=20))
    cfg = Config()
    object.__setattr__(cfg, "_explicit", frozenset())
    adopt_checkpoint_config(cfg, ckpt, is_main=False)
    assert cfg.max_fragments == 20                  # continued without the flag

    # A checkpoint from before the setting used every pattern.
    assert not _fragment_limit_changed({}, Config())
    assert _fragment_limit_changed({}, Config(max_fragments=20))
    assert not _fragment_limit_changed({"max_fragments": 20}, Config(max_fragments=20))
    assert _fragment_limit_changed({"max_fragments": 20}, Config())


def test_evaluate_scores_an_old_checkpoint_on_the_benchmark_range(tree, tmp_path):
    """A checkpoint from before the setting, scored on 2-4 pieces by flag:
    only the one validation pattern in range is scored."""
    from scripts import evaluate as E
    from vngat.training.trainer import build_model

    cfg = Config(root_dir=tree, hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                 head_dim=4, embed_dim=4, device="cpu", split_source="official",
                 data_subsets="everyday_compressed", fracture_pattern="fractured_")
    stored = cfg.to_dict()
    del stored["max_fragments"]                                     # an old checkpoint
    ckpt = tmp_path / "best.pt"
    torch.save({"model": build_model(cfg, torch.device("cpu")).state_dict(),
                "config": stored, "epoch": 1}, ckpt)

    out = tmp_path / "eval.json"
    common = ["--checkpoint", str(ckpt), "--device", "cpu", "--num_workers", "0",
              "--solve_translation", "false"]
    summary = E.evaluate(E.build_parser().parse_args(
        common + ["--max_fragments", str(HIGH), "--out", str(out)]))
    assert summary["num_fragments"] == 3                           # b2's 3-piece pattern only
    assert json.loads(out.read_text())["max_fragments"] == HIGH
    everything = E.evaluate(E.build_parser().parse_args(common))
    assert everything["num_fragments"] > 3 and math.isfinite(everything["geodesic_deg"])
    with pytest.raises(SystemExit):
        E.evaluate(E.build_parser().parse_args(common + ["--max_fragments", "1"]))
