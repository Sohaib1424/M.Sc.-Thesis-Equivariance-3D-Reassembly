"""
Objects, split modes, balancing and the fixed validation set -- on a miniature
Breaking Bad tree written in the real compressed format.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from conftest import write_dataset
from vngat.data.catalog import (
    build_catalog, effective_sample_size, fixed_items, object_weights,
    partition_modes, split_objects,
)

EVERYDAY = "everyday_compressed/everyday_compressed"
VARIANT = "volume_constrained-everyday_compressed/everyday"

LAYOUT = {
    **{f"{EVERYDAY}/Bottle/b{i}": 6 for i in range(6)},
    **{f"{EVERYDAY}/Cup/c{i}": 6 for i in range(2)},
    **{f"{EVERYDAY}/Mug/m{i}": 6 for i in range(3)},
}
OFFICIAL = {
    "train": [f"everyday/Bottle/b{i}" for i in range(5)] + ["everyday/Cup/c0",
                                                           "everyday/Mug/m0", "everyday/Mug/m1"],
    "val": ["everyday/Bottle/b5", "everyday/Cup/c1", "everyday/Mug/m2"],
}


@pytest.fixture
def root(tmp_path):
    write_dataset(tmp_path, LAYOUT, OFFICIAL)
    return str(tmp_path)


def _catalog(root, subsets=None):
    return build_catalog(root, subsets, "fractured_")


def test_catalog_lists_objects_with_their_categories(root):
    objects = _catalog(root)
    assert len(objects) == 11
    assert Counter(e.category for e in objects) == {"Bottle": 6, "Mug": 3, "Cup": 2}
    assert all(len(e) == 6 for e in objects)


def test_variant_copies_are_one_object_with_the_union_of_patterns(tmp_path):
    """Counted as separate scenes, the vanilla and volume-constrained copies
    double the object count and can land one shape in two splits."""
    write_dataset(tmp_path, {f"{EVERYDAY}/Bottle/b0": 3, f"{VARIANT}/Bottle/b0": 2,
                             f"{EVERYDAY}/Cup/c0": 3})
    objects = build_catalog(str(tmp_path), None, "fractured_")
    assert len(objects) == 2
    bottle = next(e for e in objects if e.name == "b0")
    assert len(bottle) == 5 and bottle.category == "Bottle"
    assert len({scene for scene, _ in bottle.modes}) == 2


def test_official_object_split_matches_the_lists(root):
    objects = _catalog(root)
    train = split_objects(objects, "train", root=root, split_source="official")
    val = split_objects(objects, "val", root=root, split_source="official")
    assert {e.name for e in val} == {"b5", "c1", "m2"}
    assert len(train) == 8
    assert not ({e.key for e in train} & {e.key for e in val})
    with pytest.raises(ValueError, match="no official test split"):
        split_objects(objects, "test", root=root, split_source="official")


def test_unlisted_objects_belong_to_no_split(tmp_path):
    write_dataset(tmp_path, {**LAYOUT, f"{EVERYDAY}/Bottle/extra": 3}, OFFICIAL)
    objects = build_catalog(str(tmp_path), None, "fractured_")
    names = {e.name for s in ("train", "val")
             for e in split_objects(objects, s, root=str(tmp_path), split_source="official")}
    assert "extra" not in names


def test_hash_split_is_disjoint_and_keyed_on_the_name(root):
    objects = _catalog(root)
    parts = {s: {e.key for e in split_objects(objects, s, root=root, split_source="hash",
                                              val_frac=0.3, test_frac=0.3)}
             for s in ("train", "val", "test")
             if _nonempty(objects, s, root)}
    keys = list(parts.values())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (a & b)
    from vngat.data.splits import assign_split
    for split, members in parts.items():
        for key in members:
            assert assign_split(key.split("/")[-1], 0.3, 0.3, 0) == split


def _nonempty(objects, split, root):
    try:
        split_objects(objects, split, root=root, split_source="hash", val_frac=0.3, test_frac=0.3)
        return True
    except ValueError:
        return False


def test_fracture_split_holds_out_patterns_of_known_objects(root):
    objects = _catalog(root)
    kw = dict(root=root, split_by="fracture", split_source="official", val_frac=0.2,
              test_frac=0.2)
    train = split_objects(objects, "train", **kw)
    val = split_objects(objects, "val", **kw)
    test = split_objects(objects, "test", **kw)
    # the pool is the OFFICIAL TRAIN objects only: the held-out shapes stay clean
    assert {e.name for e in train} == {e.name for e in val} == {e.name for e in test}
    assert "b5" not in {e.name for e in train}
    for t in train:
        v = next(e for e in val if e.key == t.key)
        s = next(e for e in test if e.key == t.key)
        assert not (set(t.modes) & set(v.modes)) and not (set(t.modes) & set(s.modes))
        assert len(t) + len(v) + len(s) == 6
    everything = split_objects(objects, "val", **{**kw, "fracture_pool": "all"})
    assert "b5" in {e.name for e in everything}


def test_partition_is_exact_and_deterministic(root):
    entry = _catalog(root)[0]
    a = partition_modes(entry, 0.2, 0.2, seed=0)
    b = partition_modes(entry, 0.2, 0.2, seed=0)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (4, 1, 1)


def test_objects_with_too_few_patterns_are_train_only(tmp_path):
    write_dataset(tmp_path, {f"{EVERYDAY}/Bottle/b0": 2, f"{EVERYDAY}/Cup/c0": 5})
    objects = build_catalog(str(tmp_path), None, "fractured_")
    b0 = next(e for e in objects if e.name == "b0")
    parts = partition_modes(b0, 0.2, 0.2, 0)
    assert len(parts["train"]) == 2 and not parts["val"] and not parts["test"]


def test_max_objects_is_a_deterministic_subset(root):
    objects = _catalog(root)
    a = split_objects(objects, "train", root=root, split_source="official", max_objects=4)
    b = split_objects(objects, "train", root=root, split_source="official", max_objects=4)
    assert [e.key for e in a] == [e.key for e in b] and len(a) == 4


def test_category_balance_equalises_categories(root):
    objects = _catalog(root)
    assert object_weights(objects, "none") is None
    weights = object_weights(objects, "category", 1.0)
    mass = Counter()
    for e, w in zip(objects, weights):
        mass[e.category] += w
    values = list(mass.values())
    assert max(values) - min(values) < 1e-9
    # T = 0 is the natural, per-object draw
    assert object_weights(objects, "category", 0.0) is None
    # and the cost is visible: fewer effective objects per draw
    assert effective_sample_size(weights, len(objects)) < len(objects)
    assert effective_sample_size(None, len(objects)) == len(objects)


def test_fixed_validation_items_cover_every_object_before_repeating(root):
    objects = _catalog(root)
    one_each = fixed_items(objects, 0, seed=0)
    assert len(one_each) == len(objects)
    assert len({i for i, _, _ in one_each}) == len(objects)
    more = fixed_items(objects, len(objects) + 3, seed=0)
    counts = Counter(i for i, _, _ in more)
    assert max(counts.values()) == 2 and sum(counts.values()) == len(objects) + 3
    assert fixed_items(objects, 5, seed=0) == fixed_items(objects, 5, seed=0)


def test_a_small_fixed_set_still_spans_the_categories(root):
    """Round-robin in KEY order made a 4-scene set four bottles."""
    objects = _catalog(root)
    small = fixed_items(objects, 3, seed=0)
    assert {objects[i].category for i, _, _ in small} == {"Bottle", "Cup", "Mug"}


# --------------------------------------------------------------------------
# The dataset, end to end on the real decompression path
# --------------------------------------------------------------------------
def _dataset(root, **kw):
    from vngat.data.dataset import BreakingBadDataset
    base = dict(subsets=["everyday_compressed"], fracture_pattern="fractured_",
                split_source="official", correspondence_tol=1e-5)
    base.update(kw)
    return BreakingBadDataset(root, **base)


def test_fixed_validation_is_the_same_scene_and_rotation_every_time(root):
    ds = _dataset(root, split="val", fixed=True)
    assert len(ds) == 3 and ds.categories == ["Bottle", "Cup", "Mug"]
    a, b = ds[1], ds[1]
    assert (a["scene_dir"], a["mode"]) == (b["scene_dir"], b["mode"])
    assert torch.equal(a["rot"], b["rot"])
    assert torch.equal(a["target"].node_vec, b["target"].node_vec)


def test_training_draws_respect_the_split_and_carry_the_category(root):
    ds = _dataset(root, split="train", nominal_length=4)
    allowed = {e.key.split("/")[-1] for e in ds.objects}
    np.random.seed(0)
    for i in range(6):
        sample = ds[i]
        assert sample["scene_dir"].rstrip("/").split("/")[-1] in allowed
        assert sample["category"] in {"Bottle", "Cup", "Mug"}
        g = sample["target"]
        assert g.num_fragments >= 2
        radius = torch.stack([g.node_vec[g.node_frag == f, 0].norm(dim=-1).max()
                              for f in range(g.num_fragments)])
        assert abs(float(radius.max()) - 1.0) < 1e-4            # scene normalisation


def test_frac_input_shares_the_targets_units(root):
    """The network may see the fracture surface, but prediction and target must
    live in the same normalised units and carry the same size feature."""
    ds = _dataset(root, split="val", fixed=True, input_source="frac")
    sample = ds[0]
    target, model_input = sample["target"], sample["input"]
    assert model_input is not None
    assert model_input.num_fragments == target.num_fragments
    assert torch.equal(model_input.frag_unit, target.frag_unit)
    assert torch.equal(model_input.frag_log_scale, target.frag_log_scale)


def test_frac_input_is_smaller_but_in_the_targets_units(monkeypatch):
    """
    The plumbing of input_source='frac' with a fracture surface that is really
    smaller than the fragment (the synthetic pieces above have none, so the
    fallback hides the path). The network's input must share the TARGET's
    divisor and scale feature, or prediction and target live in different units.
    """
    import trimesh

    import vngat.data.dataset as dataset_module
    from vngat.data.dataset import build_graphs

    def half(mesh):
        out = trimesh.Trimesh(mesh.vertices, mesh.faces[: len(mesh.faces) // 2], process=False)
        out.remove_unreferenced_vertices()
        return out

    monkeypatch.setattr(dataset_module, "extract_fractures", half)
    rng = np.random.default_rng(0)
    meshes = []
    for scale in (1.0, 0.4, 0.7):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        meshes.append(trimesh.Trimesh(sphere.vertices * scale + rng.standard_normal(3),
                                      sphere.faces, process=False))
    target, model_input, info = build_graphs(meshes, input_source="frac")
    assert info["frac_fallbacks"] == 0
    assert model_input.num_fragments == target.num_fragments
    assert model_input.num_nodes < target.num_nodes
    assert torch.equal(model_input.frag_unit, target.frag_unit)
    assert torch.equal(model_input.frag_log_scale, target.frag_log_scale)
    # one divisor for the scene: the largest FULL fragment's radius
    assert torch.allclose(target.frag_unit, torch.full_like(target.frag_unit, 1.0), atol=1e-5)
