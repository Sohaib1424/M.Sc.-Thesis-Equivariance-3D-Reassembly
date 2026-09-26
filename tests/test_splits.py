"""Deterministic hash-based splits."""
from __future__ import annotations

from collections import Counter

from vngat.data.splits import assign_split


def test_split_is_deterministic():
    names = [f"shape_{i}" for i in range(500)]
    first = [assign_split(n, 0.1, 0.1, 0) for n in names]
    second = [assign_split(n, 0.1, 0.1, 0) for n in names]
    assert first == second


def test_split_proportions_are_approximately_right():
    names = [f"shape_{i:06d}" for i in range(20000)]
    counts = Counter(assign_split(n, 0.1, 0.1, 0) for n in names)
    total = sum(counts.values())
    assert abs(counts["val"] / total - 0.1) < 0.01
    assert abs(counts["test"] / total - 0.1) < 0.01
    assert abs(counts["train"] / total - 0.8) < 0.02


def test_splits_are_disjoint_and_total():
    names = [f"shape_{i}" for i in range(3000)]
    assigned = {n: assign_split(n, 0.15, 0.15, 3) for n in names}
    buckets = {s: {n for n, v in assigned.items() if v == s} for s in ("train", "val", "test")}
    assert sum(len(b) for b in buckets.values()) == len(names)
    assert not (buckets["train"] & buckets["val"])
    assert not (buckets["train"] & buckets["test"])
    assert not (buckets["val"] & buckets["test"])


def test_seed_changes_the_partition():
    names = [f"shape_{i}" for i in range(1000)]
    a = [assign_split(n, 0.1, 0.1, 0) for n in names]
    b = [assign_split(n, 0.1, 0.1, 1) for n in names]
    assert a != b


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _build_tree(root, layout):
    from pathlib import Path

    for parent, count in layout.items():
        for i in range(count):
            d = Path(root) / parent / f"shape_{i:03d}"
            (d / "fractured_0").mkdir(parents=True)      # fracture sub-directory
            (d / "compressed_mesh.obj").write_text("")
            (d / "compressed_data.npz").write_text("")


def test_discovery_finds_every_subset_at_any_depth(tmp_path):
    """
    Regression guard. The previous scanner enumerated only
    everyday_compressed and artifact_compressed at hardcoded nesting depths,
    so the `volume_constrained-*` trees were invisible and a depth mismatch
    silently produced a too-small scene count with no error.
    """
    from vngat.data.splits import list_scene_directories

    layout = {
        "everyday_compressed/everyday_compressed/BeerBottle": 3,   # category/shape
        "everyday_compressed/everyday_compressed/Mug": 2,
        "artifact_compressed/artifact_compressed": 4,              # shape directly
        "volume_constrained-everyday_compressed/everyday/Bottle": 5,
        "volume_constrained-artifact_compressed": 6,
        "some/unexpectedly/deep/nesting": 2,
    }
    _build_tree(tmp_path, layout)
    found = list_scene_directories(str(tmp_path))
    assert len(found) == sum(layout.values())


def test_discovery_does_not_descend_into_fracture_directories(tmp_path):
    """Each scene holds ~100 fracture folders; treating them as scenes would
    inflate the count and walking them would be ~100x slower."""
    from vngat.data.splits import list_scene_directories

    _build_tree(tmp_path, {"everyday_compressed": 2})
    found = list_scene_directories(str(tmp_path))
    assert len(found) == 2
    assert all("fractured_0" not in str(p) for p in found)


def test_subset_filter_restricts_to_named_top_level_dirs(tmp_path):
    from vngat.data.splits import list_scene_directories

    _build_tree(tmp_path, {
        "everyday_compressed/cat": 3,
        "artifact_compressed": 4,
        "volume_constrained-everyday_compressed": 5,
    })
    standard = list_scene_directories(
        str(tmp_path), ["everyday_compressed", "artifact_compressed"])
    assert len(standard) == 7
    assert len(list_scene_directories(str(tmp_path))) == 12


def test_missing_root_returns_empty(tmp_path):
    from vngat.data.splits import list_scene_directories

    assert list_scene_directories(str(tmp_path / "nope")) == []


def _build_official(tmp_path, nested: bool):
    """`nested=True` reproduces the release's data_split/data_split/ layout."""
    objects = [("BeerBottle", "2927d6c8438f6e24fe6460d8d9bd16c6"),
               ("BeerBottle", "3f91158956ad7db0322747720d7d37e8"),
               ("Mug", "412289ded0359b326802d3678aa9d56b")]
    for category, obj in objects:
        d = tmp_path / "everyday_compressed" / "everyday_compressed" / category / obj
        (d / "fractured_0").mkdir(parents=True)
        (d / "compressed_mesh.obj").write_text("")
        (d / "compressed_data.npz").write_text("")
    split_dir = tmp_path / "data_split" / ("data_split" if nested else "")
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "everyday.train.txt").write_text(
        "\n".join(f"everyday/{c}/{o}" for c, o in objects[:2]))
    (split_dir / "everyday.val.txt").write_text(
        f"everyday/{objects[2][0]}/{objects[2][1]}")
    return objects


def test_official_split_found_when_nested_one_level_deeper(tmp_path):
    """
    Regression guard. The release ships the lists at
    data/data_split/data_split/everyday.train.txt -- nested exactly as
    everyday_compressed/everyday_compressed/ is. A non-recursive glob found
    nothing and raised "matched anything on disk" with zero entries parsed,
    which reads like a format mismatch rather than a lookup that never opened a
    file.
    """
    from vngat.data.splits import load_official_split

    _build_official(tmp_path, nested=True)
    assert len(load_official_split(str(tmp_path), "train", ["everyday_compressed"])) == 2
    assert len(load_official_split(str(tmp_path), "val", ["everyday_compressed"])) == 1


def test_official_split_also_found_when_flat(tmp_path):
    from vngat.data.splits import load_official_split

    _build_official(tmp_path, nested=False)
    assert len(load_official_split(str(tmp_path), "train", ["everyday_compressed"])) == 2


def test_missing_split_files_say_so_explicitly(tmp_path):
    """An empty data_split must not be reported as an id mismatch."""
    import pytest as _pytest

    from vngat.data.splits import load_official_split

    _build_official(tmp_path, nested=True)
    for f in (tmp_path / "data_split").rglob("*.txt"):
        f.unlink()
    with _pytest.raises(ValueError, match="No \\*.train.txt files found"):
        load_official_split(str(tmp_path), "train", ["everyday_compressed"])
