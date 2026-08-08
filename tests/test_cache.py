"""Preprocessing cache: fidelity, key invalidation, and the size budget."""
import numpy as np
import pytest

from reassembly.data.cache import (
    BaseMeshCache, DiskGuard, ScenePreprocessCache, free_gib,
)


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


# ---------------------------------------------------------------------------
# Disk protection. Filling /kaggle/working does not merely stop the cache --
# it makes checkpoint writes fail, which loses the run.
# ---------------------------------------------------------------------------
def test_guard_stops_on_the_free_space_floor(tmp_path):
    """The binding rule is FREE SPACE, not bytes written. A byte budget cannot
    see the dataset, the repo, or the checkpoints sharing the same quota."""
    guard = DiskGuard(str(tmp_path), min_free_gib=1e9, max_gib=None)
    assert not guard.allows()
    assert "free" in guard.stopped_reason


def test_guard_stops_on_the_byte_budget(tmp_path):
    guard = DiskGuard(str(tmp_path), min_free_gib=0.0, max_gib=1e-6,
                      recheck_every=1)
    assert guard.allows()
    guard.record(10 * 1024 ** 2)
    assert not guard.allows()
    assert "budget" in guard.stopped_reason


def test_both_caches_draw_from_ONE_budget(tmp_path):
    """Separate budgets silently sum: an 8 GiB scene cache plus a 4 GiB base
    cache is 12 GiB of a 20 GB quota, which is not what either number says."""
    guard = DiskGuard(str(tmp_path), min_free_gib=0.0, max_gib=1e-5,
                      recheck_every=1)
    scene = ScenePreprocessCache(str(tmp_path), guard=guard)
    base = BaseMeshCache(str(tmp_path), guard=guard)
    assert scene.guard is base.guard

    v, f, vc, ec = sample(n_frag=2, n_vert=900)
    for i in range(10):
        scene.store(scene.key("/s", f"frac_{i}", decimate_to=6000), v, f, vc, ec, 1e-5)

    # the base cache must now see the budget as spent, because it is shared
    assert not guard.allows()
    import numpy as np
    from scipy.sparse import csr_matrix
    before = base.writes
    base.store("/s", np.zeros((10, 3)), np.zeros((5, 3), dtype=np.int64),
               csr_matrix((10, 4)))
    assert base.writes == before, "base cache wrote past the shared budget"


def test_reads_keep_working_after_the_budget_is_spent(tmp_path):
    guard = DiskGuard(str(tmp_path), min_free_gib=0.0, max_gib=1e-5,
                      recheck_every=1)
    cache = ScenePreprocessCache(str(tmp_path), guard=guard)
    v, f, vc, ec = sample(n_frag=1, n_vert=500)

    key = cache.key("/s", "frac_0", decimate_to=6000)
    cache.store(key, v, f, vc, ec, 1e-5)
    assert cache.load(key) is not None

    guard.record(10 * 1024 ** 3)          # blow the budget
    assert not guard.allows()
    assert cache.load(key) is not None, "existing entries must stay readable"


def test_free_gib_reports_a_real_number(tmp_path):
    value = free_gib(str(tmp_path))
    assert value > 0 and value != float("inf")


def test_disabled_cache_never_touches_the_disk(tmp_path):
    cache = ScenePreprocessCache(None)
    v, f, vc, ec = sample()
    cache.store(cache.key("/s", "frac_0", decimate_to=6000), v, f, vc, ec, 1e-5)
    assert cache.writes == 0
    assert not any(tmp_path.iterdir())
