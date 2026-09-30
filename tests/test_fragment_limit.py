"""
``--max_fragments``: only the break patterns of 2 to ``max_fragments`` pieces,
for training, validation and ``--evaluate`` alike -- the benchmark's 2-20 is
``--max_fragments 20``.

Each property below fails without an error when broken: a limit applied after
``max_objects`` trains on fewer objects than asked for, one applied before the
split moves patterns between train and val, and one left out of the resume
logic compares best.pt across two different validation sets. All of them
still print plausible numbers.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

import reassembly.data.scene as scene_module
import reassembly.training as training
from reassembly.data.catalog import limit_fragments
from reassembly.data.scene import SceneReader, piece_count
from reassembly.training import BreakingBadScenes, Config, Skipped

BASE = ("everyday_compressed", "everyday_compressed")
# Pieces in each break pattern, fractured_0, fractured_1, ...
LAYOUT = {
    ("Bottle", "b0"): [2, 3, 6],        # train: keeps two
    ("Cup", "c0"): [1, 3],              # train: the single piece goes
    ("Mug", "m0"): [5, 6],              # train: nothing in range, the object goes
    ("Bottle", "b2"): [6, 3],           # val: keeps one
    ("Cup", "c1"): [6],                 # val: nothing in range
}
VAL = {("Bottle", "b2"), ("Cup", "c1")}
HIGH = 4


def _tree(root: Path) -> Path:
    from scipy.sparse import identity, save_npz

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])
    for (category, name), pieces in LAYOUT.items():
        directory = root.joinpath(*BASE, category, name)
        directory.mkdir(parents=True)
        mesh.export(directory / "compressed_mesh.obj")
        save_npz(directory / "compressed_data.npz", identity(len(vertices), format="csr"))
        for mode, count in enumerate(pieces):
            (directory / f"fractured_{mode}").mkdir()
            # `count` sectors about the axis: each one a genuine piece.
            labels = np.floor((theta + np.pi) / (2 * np.pi) * count)
            np.save(directory / f"fractured_{mode}" / "compressed_fracture.npy",
                    np.clip(labels, 0, count - 1).astype(np.int64))
    split_dir = root / "data_split" / "data_split"
    split_dir.mkdir(parents=True)
    for split in ("train", "val"):
        names = [f"everyday/{c}/{n}" for c, n in LAYOUT if ((c, n) in VAL) == (split == "val")]
        (split_dir / f"everyday.{split}.txt").write_text("\n".join(names))
    return root


@pytest.fixture
def root(tmp_path):
    piece_count.cache_clear()
    yield _tree(tmp_path / "data")
    piece_count.cache_clear()


def _config(root, **kwargs):
    defaults = dict(
        root=str(root), out_dir=str(Path(root).parent / "out"),
        channels=16, heads=4, head_dim=4, embedding_dim=8, workers=0,
        tokens_per_scene=32, batch_size=1, accumulate=1, epochs=1, modes_per_scene=None,
        schedule=("intra",), check_init=False, steps_per_epoch=0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _keys(dataset):
    return {dataset.key(i) for i in range(len(dataset))}


def _count(key: str, root) -> int:
    *parts, mode = key.split("/")
    return piece_count(Path(root).joinpath(BASE[0], *parts), mode)


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def test_the_label_file_count_is_what_the_loader_builds(root):
    """The count reads one small file instead of decompressing; on patterns
    with no empty label it must equal the fragments the loader builds."""
    for (category, name), pieces in LAYOUT.items():
        directory = root.joinpath(*BASE, category, name)
        reader = SceneReader(directory)
        for mode, count in enumerate(pieces):
            assert piece_count(directory, f"fractured_{mode}") == count
            assert len(reader.load_mode(f"fractured_{mode}").fragments) == count


# --------------------------------------------------------------------------
# What the limit keeps
# --------------------------------------------------------------------------

def test_no_limit_changes_nothing(root):
    train = BreakingBadScenes(_config(root), "train")
    assert train.fragment_limit is None
    assert len(train) == 3 + 2 + 2                    # every pattern, as before


def test_the_limit_keeps_only_patterns_in_range_and_drops_emptied_objects(root):
    everything = BreakingBadScenes(_config(root), "train").catalog
    kept, report = limit_fragments(everything, 2, HIGH)
    assert {e.name: [mode for _, mode in e.modes] for e in kept.objects} == {
        "b0": ["fractured_0", "fractured_1"],        # 2 and 3 pieces; 6 goes
        "c0": ["fractured_1"],                       # 3; the single piece goes
    }
    assert (report.patterns, report.patterns_kept) == (7, 3)
    assert (report.objects, report.objects_kept) == (3, 2)
    assert report.describe() == "3 of 7 break patterns (2 of 3 objects)"


@pytest.mark.parametrize("split_by", ["object", "fracture"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_the_limit_never_moves_a_pattern_between_splits(root, split_by, split):
    """Applied after the split, the limit can only remove: the patterns left
    in each split are that split's own patterns in range, nothing else."""
    kw = dict(split_by=split_by, fracture_pool="all", val_frac=0.34, test_frac=0.0)
    try:
        everything = _keys(BreakingBadScenes(_config(root, **kw), split))
    except FileNotFoundError:               # a split this small can be empty
        pytest.skip(f"no {split} patterns under split_by={split_by}")
    in_range = {k for k in everything if 2 <= _count(k, root) <= HIGH}
    if not in_range:
        with pytest.raises(FileNotFoundError, match="no break pattern"):
            BreakingBadScenes(_config(root, max_fragments=HIGH, **kw), split)
        return
    assert _keys(BreakingBadScenes(_config(root, max_fragments=HIGH, **kw), split)) == in_range


def test_max_objects_counts_objects_that_still_have_a_pattern(root):
    """``max_objects`` strides over the sorted objects: first and last of
    b0, c0, m0. Limiting afterwards would take m0, lose it to the limit, and
    train on one object where two were asked for."""
    everything = BreakingBadScenes(_config(root, max_objects=2), "train")
    assert sorted(e.name for e in everything.catalog.objects) == ["b0", "m0"]
    limited = BreakingBadScenes(_config(root, max_objects=2, max_fragments=HIGH), "train")
    assert sorted(e.name for e in limited.catalog.objects) == ["b0", "c0"]


def test_an_unreadable_label_file_is_kept_for_the_loader_to_report(root):
    catalog = BreakingBadScenes(_config(root), "train").catalog
    directory, mode = catalog.objects[0].modes[0]
    (directory / mode / "compressed_fracture.npy").write_bytes(b"not an array")
    piece_count.cache_clear()
    kept, report = limit_fragments(catalog, 2, HIGH)
    assert (directory, mode) in {pair for e in kept.objects for pair in e.modes}
    assert report.unreadable == 1 and "unreadable" in report.describe()


# --------------------------------------------------------------------------
# What the run sees
# --------------------------------------------------------------------------

def test_training_and_validation_only_see_patterns_in_range(root):
    val = BreakingBadScenes(_config(root, max_fragments=HIGH), "val")
    assert len(val) == 1 and len(val.build(0).fragments) == 3
    assert len(BreakingBadScenes(_config(root), "val")) == 3

    train = BreakingBadScenes(_config(root, max_fragments=HIGH), "train")
    sizes = [len(train.build(i).fragments) for i in range(len(train))]
    assert sorted(sizes) == [2, 3, 3]


def test_a_scene_above_the_limit_is_skipped_when_built(root, monkeypatch):
    """The loaded count is checked too: were the label count ever to miss a
    pattern, the scene is skipped by name rather than trained on."""
    monkeypatch.setattr(scene_module, "piece_count", lambda *_: 2)
    train = BreakingBadScenes(_config(root, max_fragments=HIGH), "train")
    built = [train.build(i) for i in range(len(train))]
    skipped = [b for b in built if isinstance(b, Skipped)]
    assert {b.reason for b in skipped} >= {"6 fragments, above --max_fragments 4"}
    assert all(len(b.fragments) <= HIGH for b in built if not isinstance(b, Skipped))


def test_the_banner_says_what_the_limit_removed(root):
    config = _config(root, max_fragments=HIGH)
    lines = training._fragment_banner(config, BreakingBadScenes(config, "train"),
                                      BreakingBadScenes(config, "val"))
    text = "\n".join(lines)
    assert "2-4 per scene" in text
    assert "train keeps 3 of 7 break patterns (2 of 3 objects)" in text
    assert "val   keeps 1 of 3 break patterns (1 of 2 objects)" in text
    config = _config(root)
    lines = training._fragment_banner(config, BreakingBadScenes(config, "train"),
                                      BreakingBadScenes(config, "val"))
    assert "no upper limit" in "\n".join(lines)


# --------------------------------------------------------------------------
# The setting
# --------------------------------------------------------------------------

def test_the_flag_parses_validates_and_round_trips():
    from scripts.config_flags import config_flags
    from scripts.train import build_parser, config_from_args

    parse = lambda argv: config_from_args(build_parser().parse_args(argv))  # noqa: E731
    assert parse(["--max_fragments", "20"]).max_fragments == 20
    assert parse(["--max_fragments", "0"]).max_fragments is None     # 0 = no limit
    assert parse([]).max_fragments is None
    with pytest.raises(ValueError, match="max_fragments"):
        Config(max_fragments=1)
    flags = config_flags(Config(max_fragments=20))
    assert flags[flags.index("--max_fragments") + 1] == "20"
    assert parse(flags).max_fragments == 20


def test_a_changed_limit_is_announced_on_resume_and_resets_the_best(capsys):
    assert "max_fragments" in training._DATA
    # A checkpoint from before the setting used every pattern.
    assert not training._fragment_limit_changed({}, Config())
    assert training._fragment_limit_changed({}, Config(max_fragments=20))
    assert not training._fragment_limit_changed({"max_fragments": 20}, Config(max_fragments=20))
    assert training._fragment_limit_changed({"max_fragments": 20}, Config())

    old = {k: v for k, v in dataclasses.asdict(Config()).items() if k != "max_fragments"}
    training._check_resume_compatible(Config(max_fragments=20), old, announce=True)
    assert "max_fragments: None -> 20" in capsys.readouterr().out


def test_evaluate_scores_any_checkpoint_on_the_range_asked_for(root, capsys):
    """The checkpoint's data definition applies by default -- an explicit
    --max_fragments wins, so a model trained on every pattern can be scored
    on the benchmark's range without retraining."""
    from reassembly.training import build_model, save_checkpoint

    config = _config(root)
    path = Path(config.out_dir) / "last.pt"
    path.parent.mkdir(parents=True)
    save_checkpoint(path, build_model(config), None, config, epoch=0, step=1,
                    history=[], best=1.0)

    limited = dataclasses.replace(config, max_fragments=HIGH)
    capsys.readouterr()
    summary = training.evaluate(limited, checkpoint="last.pt", split="val", assemble=False)
    assert "using the checkpoint's --max_fragments None" in capsys.readouterr().out
    assert summary["fragments"] == 6 + 3 + 6 and summary["max_fragments"] is None

    summary = training.evaluate(limited, checkpoint="last.pt", split="val", assemble=False,
                                override=("max_fragments",))
    out = capsys.readouterr().out
    assert "fragments: 2-4 per scene, keeping 1 of 3 break patterns" in out
    assert summary["fragments"] == 3 and summary["max_fragments"] == HIGH
