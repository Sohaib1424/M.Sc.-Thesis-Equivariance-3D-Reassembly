"""Preprocessing cache: fidelity, key invalidation, and the size budget."""
import numpy as np
import pytest

from reassembly.data.cache import ScenePreprocessCache


def sample(n_frag=3, n_vert=500, seed=0):
    rng = np.random.default_rng(seed)
    v = [rng.random((n_vert, 3)).astype(np.float64) for _ in range(n_frag)]
    f = [rng.integers(0, n_vert, (n_vert * 2, 3)).astype(np.int64) for _ in range(n_frag)]
    vc = [np.full(n_vert, -1, np.int64) for _ in range(n_frag)]
    ec = [np.full(n_vert * 3, -1, np.int64) for _ in range(n_frag)]
    return v, f, vc, ec


def test_round_trip_preserves_geometry(tmp_path):
    c = ScenePreprocessCache(str(tmp_path))
    v, f, vc, ec = sample()
    key = c.key("/scenes/abc", "fractured_0", decimate_to=6000)
    c.store(key, v, f, vc, ec, tol=1.5e-5)

    got = c.load(key)
    assert got is not None
    assert len(got["vertices"]) == len(v)
    assert got["tol"] == pytest.approx(1.5e-5)
    for a, b in zip(got["vertices"], v):
        # stored as float32 to halve the footprint
        assert np.allclose(a, b, atol=1e-6)
    for a, b in zip(got["faces"], f):
        assert np.array_equal(a, b)


def test_missing_entry_is_a_miss_not_an_error(tmp_path):
    c = ScenePreprocessCache(str(tmp_path))
    assert c.load(c.key("/nope", "fractured_0", decimate_to=6000)) is None
    assert c.misses == 1


def test_settings_are_part_of_the_key(tmp_path):
    """Changing decimate_to must MISS. Reusing geometry cached under different
    decimation settings would silently train on the wrong resolution."""
    c = ScenePreprocessCache(str(tmp_path))
    v, f, vc, ec = sample()
    k1 = c.key("/scenes/abc", "fractured_0", decimate_to=6000)
    k2 = c.key("/scenes/abc", "fractured_0", decimate_to=15000)
    assert k1 != k2
    c.store(k1, v, f, vc, ec, tol=1e-5)
    assert c.load(k1) is not None
    assert c.load(k2) is None


def test_fracture_is_part_of_the_key(tmp_path):
    c = ScenePreprocessCache(str(tmp_path))
    v, f, vc, ec = sample()
    k1 = c.key("/scenes/abc", "fractured_0", decimate_to=6000)
    k2 = c.key("/scenes/abc", "fractured_7", decimate_to=6000)
    assert k1 != k2
    c.store(k1, v, f, vc, ec, tol=1e-5)
    assert c.load(k2) is None


def test_budget_stops_writing_but_keeps_serving(tmp_path):
    """Kaggle's /kaggle/working quota also holds the checkpoints. An unbounded
    cache fills it and makes checkpoint saves fail, which loses the run."""
    c = ScenePreprocessCache(str(tmp_path), max_bytes=120_000, recheck_every=1)
    v, f, vc, ec = sample(n_frag=2, n_vert=800)

    keys = []
    for i in range(20):
        k = c.key("/scenes/abc", f"fractured_{i}", decimate_to=6000)
        keys.append(k)
        c.store(k, v, f, vc, ec, tol=1e-5)

    assert c.full
    assert c.skipped_full > 0, "budget never engaged"
    assert c.writes < 20, "wrote everything despite the budget"
    # whatever made it in is still readable
    assert c.load(keys[0]) is not None


def test_disabled_cache_is_a_no_op():
    c = ScenePreprocessCache(None)
    v, f, vc, ec = sample()
    key = c.key("/scenes/abc", "fractured_0", decimate_to=6000)
    c.store(key, v, f, vc, ec, tol=1e-5)
    assert c.load(key) is None
    assert c.writes == 0


def test_corrupt_entry_is_treated_as_a_miss(tmp_path):
    c = ScenePreprocessCache(str(tmp_path))
    v, f, vc, ec = sample()
    key = c.key("/scenes/abc", "fractured_0", decimate_to=6000)
    c.store(key, v, f, vc, ec, tol=1e-5)

    with open(c._path(key), "wb") as fh:
        fh.write(b"not an npz")

    assert c.load(key) is None
    assert c.errors == 1
    assert not c._path(key).exists(), "a corrupt entry should be removed"
