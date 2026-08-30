"""
Tests for fracture-surface patch grouping.

The property that matters most is rotation-invariance of the *partition*. If
patch membership changed under rotation, pooled patch features would stop being
equivariant -- and nothing would raise, because every tensor would keep its
shape and the loss would keep descending. That is the failure mode this file
exists to make impossible.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reassembly.mesh.patches import (
    allocate_budget,
    cross_fragment_pairs,
    fracture_patches,
    patch_centroids,
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


@pytest.fixture(scope="module")
def sphere():
    m = trimesh.creation.icosphere(subdivisions=3)
    return np.asarray(m.vertices), np.asarray(m.faces)


@pytest.fixture(scope="module")
def half_mask(sphere):
    """An upper-hemisphere 'fracture surface' -- one connected region."""
    V, _ = sphere
    return V[:, 2] > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_partition_is_rotation_invariant(sphere, half_mask, seed):
    """
    The partition must depend on the fragment's shape, not its pose. Same
    vertices, same patch ids, for a rotated copy.
    """
    V, F = sphere
    R = _rotation(seed)

    plain = fracture_patches(V, F, half_mask, max_patches=8)
    rotated = fracture_patches(V @ R.T, F, half_mask, max_patches=8)

    assert np.array_equal(plain.vertex_index, rotated.vertex_index)
    assert np.array_equal(plain.patch_of_vertex, rotated.patch_of_vertex)
    assert plain.num_patches == rotated.num_patches


def test_patch_centroids_are_equivariant(sphere, half_mask):
    """Pooling is a mean, so centroids must rotate with the fragment."""
    V, F = sphere
    R = _rotation(3)
    assignment = fracture_patches(V, F, half_mask, max_patches=8)

    plain = patch_centroids(V, assignment)
    rotated = patch_centroids(V @ R.T, fracture_patches(V @ R.T, F, half_mask,
                                                        max_patches=8))
    assert np.allclose(rotated, plain @ R.T, atol=1e-12)


def test_partition_is_invariant_to_translation_and_scale(sphere, half_mask):
    """
    The loader centres fragments and (optionally) rescales them into a unit
    sphere. Neither may change which vertex belongs to which patch, or the
    standardised and non-standardised copies of one scene would disagree.
    """
    V, F = sphere
    base = fracture_patches(V, F, half_mask, max_patches=8)
    moved = fracture_patches(V * 7.5 + np.array([3.0, -2.0, 11.0]), F,
                             half_mask, max_patches=8)
    assert np.array_equal(base.patch_of_vertex, moved.patch_of_vertex)


def test_budget_is_respected(sphere, half_mask):
    for budget in (1, 4, 16, 32):
        a = fracture_patches(*sphere[:2], half_mask, max_patches=budget) \
            if False else fracture_patches(sphere[0], sphere[1], half_mask,
                                           max_patches=budget)
        assert a.num_patches <= budget
        assert a.num_patches >= 1


def test_every_fracture_vertex_lands_in_exactly_one_patch(sphere, half_mask):
    V, F = sphere
    a = fracture_patches(V, F, half_mask, max_patches=8)
    assert a.vertex_index.shape == a.patch_of_vertex.shape
    assert np.array_equal(a.vertex_index, np.flatnonzero(half_mask))
    assert a.patch_of_vertex.min() >= 0
    assert a.patch_of_vertex.max() < a.num_patches
    # no empty patches
    assert len(np.unique(a.patch_of_vertex)) == a.num_patches


def test_separate_components_never_share_a_patch():
    """
    Two disjoint fracture regions must not be pooled together: they face
    different neighbouring fragments, and blending them would average away the
    signal that says which neighbour each belongs to.
    """
    a = trimesh.creation.icosphere(subdivisions=2)
    b = trimesh.creation.icosphere(subdivisions=2)
    b.apply_translation([10.0, 0, 0])
    V = np.vstack([a.vertices, b.vertices])
    F = np.vstack([a.faces, b.faces + len(a.vertices)])
    mask = np.ones(len(V), dtype=bool)

    assignment = fracture_patches(V, F, mask, max_patches=8)
    first = set(np.unique(assignment.patch_of_vertex[:len(a.vertices)]))
    second = set(np.unique(assignment.patch_of_vertex[len(a.vertices):]))
    assert first.isdisjoint(second)


def test_vertex_mode_reproduces_the_per_vertex_graph(sphere, half_mask):
    """The benchmark fallback: one patch per fracture vertex."""
    V, F = sphere
    a = fracture_patches(V, F, half_mask, mode="vertex")
    n = int(half_mask.sum())
    assert a.num_patches == n
    assert np.array_equal(a.patch_of_vertex, np.arange(n))


def test_empty_mask_is_handled(sphere):
    """
    Fragments with no fracture surface exist in the real data -- the full run
    reports frac_vertices min = 0. They must produce zero patches rather than
    raising, because the loader will meet them.
    """
    V, F = sphere
    a = fracture_patches(V, F, np.zeros(len(V), dtype=bool))
    assert a.num_patches == 0
    assert a.vertex_index.size == 0
    assert patch_centroids(V, a).shape == (0, 3)


def test_cross_fragment_pair_count():
    """Hand-checkable: 10*20 + 10*30 + 20*30 = 1100."""
    assert cross_fragment_pairs([10, 20, 30]) == 1100
    assert cross_fragment_pairs([5]) == 0
    assert cross_fragment_pairs([]) == 0


def test_patches_cut_the_graph_by_orders_of_magnitude():
    """
    The reason the mode exists, asserted rather than asserted-in-prose. Six
    fragments of 2,000 fracture vertices each, at a 32-patch budget.
    """
    per_vertex = cross_fragment_pairs([2000] * 6)
    per_patch = cross_fragment_pairs([32] * 6)
    assert per_vertex == 60_000_000
    assert per_patch == 15_360          # 1/2 (192^2 - 6*32^2)
    assert per_vertex / per_patch > 3_500


# --------------------------------------------------------------------------
# mode="sample" -- FPS selection of real vertices (the GARF-style option)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_mode_is_rotation_invariant(sphere, half_mask, seed):
    """
    Selection must pick the same vertices for a rotated fragment. Dart-thrown
    Poisson-disk sampling would not: its RNG stream has nothing to do with the
    geometry. FPS with an index tie-break does.
    """
    V, F = sphere
    R = _rotation(seed)
    plain = fracture_patches(V, F, half_mask, mode="sample", max_patches=32)
    rotated = fracture_patches(V @ R.T, F, half_mask, mode="sample", max_patches=32)
    assert np.array_equal(plain.vertex_index, rotated.vertex_index)


def test_sample_mode_returns_real_mesh_vertices(sphere, half_mask):
    """Unlike pooled patches, every token is an actual vertex on the surface."""
    V, F = sphere
    a = fracture_patches(V, F, half_mask, mode="sample", max_patches=16)
    fracture_ids = set(np.flatnonzero(half_mask).tolist())
    assert set(a.vertex_index.tolist()) <= fracture_ids
    assert len(set(a.vertex_index.tolist())) == len(a.vertex_index), "no duplicates"


def test_sample_mode_caps_but_does_not_pad(sphere, half_mask):
    """
    The count is a cap, not a guarantee. A fragment with fewer fracture vertices
    than the budget yields all of them -- which is why a per-scene budget, not a
    per-fragment cap, is what actually fixes the cross-attention cost.
    """
    V, F = sphere
    available = int(half_mask.sum())
    assert fracture_patches(V, F, half_mask, mode="sample",
                            max_patches=16).num_patches == 16
    big = fracture_patches(V, F, half_mask, mode="sample",
                           max_patches=available * 10)
    assert big.num_patches == available


def test_sample_mode_spreads_over_the_surface(sphere, half_mask):
    """
    FPS should cover the region, not cluster. Compare the mean nearest-neighbour
    distance among the selected points against a random subset of the same size:
    a spread-out set has the larger value.
    """
    from scipy.spatial import cKDTree

    V, F = sphere
    a = fracture_patches(V, F, half_mask, mode="sample", max_patches=24)
    chosen = V[a.vertex_index]

    def mean_nn(points):
        d, _ = cKDTree(points).query(points, k=2)
        return float(d[:, 1].mean())

    rng = np.random.default_rng(0)
    pool = np.flatnonzero(half_mask)
    random_gap = np.mean([
        mean_nn(V[rng.choice(pool, size=len(chosen), replace=False)])
        for _ in range(8)])
    assert mean_nn(chosen) > random_gap, "FPS should spread further than random"


def test_scene_budget_is_exactly_fixed():
    """
    The property the per-fragment cap does not have. Whatever the fragment
    sizes, the allocated tokens sum to the budget -- so the cross-attention
    pair count is a constant of the configuration, not of the mesh.
    """
    for weights in ([100, 200, 700], [5, 5, 5, 5], [1000, 1, 1],
                    [50] * 20, [1, 2, 3, 4, 5, 6, 7]):
        counts = allocate_budget(weights, total=256)
        assert counts.sum() == 256, weights
        assert (counts >= 1).all()


def test_scene_budget_favours_larger_fragments():
    counts = allocate_budget([10, 100, 1000], total=222)
    assert counts[2] > counts[1] > counts[0]


def test_the_total_is_a_hard_bound_even_when_it_cannot_cover_every_fragment():
    """
    More fragments than tokens. The total wins.

    This used to return one token per fragment regardless -- `total=4` over ten
    fragments gave ten tokens, and `total=5, minimum=10` over two gave ten,
    twice the budget. It never raised, and the numbers looked reasonable.

    It matters because the cost guarantee is `sum(t) <= total`: that inequality
    is the whole argument for why no per-fragment cap is needed. An allocator
    free to exceed it makes the bound a suggestion.

    The consequence is real and worth stating rather than hiding: the fragments
    that miss out get no cross-fragment edges at all. The remedy is a larger
    total, not a quietly overspent one.
    """
    counts = allocate_budget([1] * 10, total=4)
    assert counts.sum() == 4
    assert (counts >= 0).all()
    assert (counts > 0).sum() == 4, "four fragments served, six not"

    tight = allocate_budget([1.0, 1.0], total=5, minimum=10)
    assert tight.sum() <= 5


def test_a_starved_budget_serves_the_largest_fracture_surfaces_first():
    """Deterministic, and not arbitrary: whoever has the most to match wins."""
    counts = allocate_budget([1.0, 50.0, 2.0, 40.0], total=2)
    assert counts.sum() == 2
    assert counts[1] == 1 and counts[3] == 1, "the two largest are served"
    assert counts[0] == 0 and counts[2] == 0


@pytest.mark.parametrize("seed", range(6))
def test_the_allocation_never_exceeds_the_total(seed):
    """
    The invariant the cost bound rests on, fuzzed. `(T^2 - sum t_i^2)/2 < T^2/2`
    only bounds the graph if `sum t_i <= T` actually holds.
    """
    rng = np.random.default_rng(seed)
    for _ in range(400):
        n = int(rng.integers(1, 60))
        total = int(rng.integers(0, 2000))
        weights = rng.random(n) * rng.choice([1.0, 1e3, 1e-3])
        capacity = None if rng.random() < 0.5 else rng.integers(0, 40, n)
        counts = allocate_budget(weights, total,
                                 minimum=int(rng.integers(0, 3)), capacity=capacity)
        assert counts.sum() <= total, (n, total, counts.sum())
        assert (counts >= 0).all()


def test_a_degenerate_weight_does_not_poison_the_allocation():
    """
    `fracture_surface_area` returns NaN for a face with a NaN vertex, and
    Breaking Bad meshes do contain degenerate geometry. Left in, a NaN made
    every share NaN and `floor(NaN).astype(int64)` INT64_MIN -- a count of
    -9223372036854775808, which numpy reports only as a RuntimeWarning.
    """
    for bad in (np.nan, np.inf, -np.inf):
        counts = allocate_budget([bad, 1.0, 1.0], total=100)
        assert (counts >= 0).all(), f"{bad} produced {counts}"
        assert counts.sum() == 100


def test_fixed_scene_budget_makes_the_pair_count_constant():
    """
    The headline: with a per-scene budget the inter-fragment connection count
    depends only on the budget and the fragment count -- never on mesh
    resolution. Two scenes with wildly different meshes, same budget.
    """
    coarse = allocate_budget([120, 300, 80], total=256)
    fine = allocate_budget([12000, 30000, 8000], total=256)
    assert cross_fragment_pairs(coarse) == cross_fragment_pairs(fine)


# --------------------------------------------------------------------------
# eligible_mask -- fracture-ness as a FEATURE rather than a gate
# --------------------------------------------------------------------------

def test_eligible_mask_defaults_to_gating_on_fracture(sphere, half_mask):
    """Unchanged behaviour: tokens come from the fracture surface only."""
    V, F = sphere
    a = fracture_patches(V, F, half_mask, mode="sample", max_patches=16)
    assert set(a.vertex_index.tolist()) <= set(np.flatnonzero(half_mask).tolist())
    assert (a.fracture_fraction == 1.0).all()


def test_whole_surface_tokens_carry_fracture_ness_as_a_feature(sphere, half_mask):
    """
    The alternative GARF actually uses: attend over the whole surface, and let
    fracture-ness ride along as a per-token channel. Tokens now come from
    everywhere, and the flag separates them.
    """
    V, F = sphere
    everything = np.ones(len(V), dtype=bool)
    a = fracture_patches(V, F, half_mask, mode="sample", max_patches=64,
                         eligible_mask=everything)

    assert a.num_patches == 64
    on = a.fracture_fraction > 0.5
    assert on.any() and (~on).any(), "should sample from both regions"
    # the flag must agree with the mask it came from
    assert np.array_equal(on, half_mask[a.vertex_index])


def test_pooled_tokens_get_an_intermediate_fracture_fraction(sphere, half_mask):
    """
    A patch straddling the rim contains both kinds of vertex, and gets a
    fraction rather than being forced to one side -- which is the point of
    carrying it as a soft feature.
    """
    V, F = sphere
    everything = np.ones(len(V), dtype=bool)
    a = fracture_patches(V, F, half_mask, mode="patch", max_patches=12,
                         eligible_mask=everything)
    frac = a.fracture_fraction
    assert frac.shape == (a.num_patches,)
    assert ((frac >= 0.0) & (frac <= 1.0)).all()
    assert ((frac > 0.0) & (frac < 1.0)).any(), "expected a straddling patch"


def test_fracture_fraction_is_rotation_invariant(sphere, half_mask):
    """The feature channel must not depend on pose any more than the partition."""
    V, F = sphere
    R = _rotation(5)
    everything = np.ones(len(V), dtype=bool)
    kw = dict(mode="patch", max_patches=12, eligible_mask=everything)
    plain = fracture_patches(V, F, half_mask, **kw)
    rotated = fracture_patches(V @ R.T, F, half_mask, **kw)
    assert np.allclose(plain.fracture_fraction, rotated.fracture_fraction)


def test_gate_versus_feature_changes_the_graph_size(sphere, half_mask):
    """
    The cost of the feature route, stated rather than assumed: with the same
    per-fragment budget the token count is the same, so cross-attention costs
    the same -- what changes is which surface the tokens sample. A fixed budget
    is what makes the two comparable at all.
    """
    V, F = sphere
    everything = np.ones(len(V), dtype=bool)
    gated = fracture_patches(V, F, half_mask, mode="sample", max_patches=32)
    feature = fracture_patches(V, F, half_mask, mode="sample", max_patches=32,
                               eligible_mask=everything)
    assert gated.num_patches == feature.num_patches == 32
    assert cross_fragment_pairs([gated.num_patches] * 4) == \
           cross_fragment_pairs([feature.num_patches] * 4)


# --------------------------------------------------------------------------
# scene_tokens -- one budget, split across a scene, then sampled
# --------------------------------------------------------------------------

def _scene_of(n, seed=0):
    """n fragments of deliberately different sizes, each fully 'fracture'."""
    rng = np.random.default_rng(seed)
    frags, masks = [], []
    for i in range(n):
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0 + i)
        m.apply_translation(rng.normal(scale=6.0, size=3))
        frags.append((np.asarray(m.vertices), np.asarray(m.faces)))
        masks.append(np.ones(len(m.vertices), dtype=bool))
    return frags, masks


def test_scene_tokens_spends_the_whole_budget():
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(7)
    got = scene_tokens(frags, masks, total=256)
    assert sum(a.num_patches for a in got) == 256
    assert len(got) == 7


def test_scene_tokens_gives_larger_fragments_more():
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(5)
    got = scene_tokens(frags, masks, total=200)
    counts = [a.num_patches for a in got]
    assert counts == sorted(counts), "radius increases with index, so should the share"


def test_scene_tokens_never_starves_a_fragment_completely():
    """Every fragment must contribute at least one token or it is unreachable."""
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(12)
    got = scene_tokens(frags, masks, total=16)
    assert all(a.num_patches >= 1 for a in got)


def test_scene_tokens_is_pose_invariant():
    """
    The property that forces per-fragment sampling. Fragments arrive in random
    poses; if the token choice depended on where a fragment happened to be
    thrown, the same scene would produce a different graph every epoch.
    """
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(4)
    plain = scene_tokens(frags, masks, total=120)

    rng = np.random.default_rng(3)
    moved = []
    for V, F in frags:
        R = _rotation(int(rng.integers(0, 1000)))
        moved.append((V @ R.T + rng.normal(scale=20.0, size=3), F))
    shifted = scene_tokens(moved, masks, total=120)

    for a, b in zip(plain, shifted):
        assert np.array_equal(a.vertex_index, b.vertex_index)


def test_area_weighting_differs_from_vertex_count_weighting():
    """
    Why area is the default weight. A finely tessellated small fragment has many
    vertices but little surface; weighting by count would over-serve it.
    """
    from reassembly.mesh.patches import fracture_surface_area, scene_tokens

    # The coarse mesh needs enough vertices to actually receive its
    # area-proportional share, or the capacity cap -- not the weighting -- is
    # what decides the result.
    fine = trimesh.creation.icosphere(subdivisions=4, radius=0.5)   # many verts, small
    coarse = trimesh.creation.icosphere(subdivisions=3, radius=3.0)  # fewer verts, large
    frags = [(np.asarray(m.vertices), np.asarray(m.faces)) for m in (fine, coarse)]
    masks = [np.ones(len(m.vertices), bool) for m in (fine, coarse)]

    counts = [len(m.vertices) for m in (fine, coarse)]
    areas = [fracture_surface_area(v, f, m) for (v, f), m in zip(frags, masks)]
    assert counts[0] > counts[1], "fine mesh should have more vertices"
    assert areas[0] < areas[1], "but less surface area"

    by_area = [a.num_patches for a in scene_tokens(frags, masks, total=100)]
    by_count = [a.num_patches for a in scene_tokens(frags, masks, total=100,
                                                    weights=counts)]
    assert by_area[1] > by_area[0], "area weighting favours the larger surface"
    assert by_count[0] > by_count[1], "count weighting favours the denser mesh"


def test_fracture_surface_area_is_pose_invariant():
    from reassembly.mesh.patches import fracture_surface_area
    m = trimesh.creation.icosphere(subdivisions=2)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    mask = np.ones(len(V), bool)
    R = _rotation(1)
    a = fracture_surface_area(V, F, mask)
    b = fracture_surface_area(V @ R.T + 17.0, F, mask)
    assert a == pytest.approx(b, rel=1e-12)


def test_blocked_distances_match_the_direct_formulation():
    """
    The expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b is the same form that
    produced a real precision bug elsewhere in this project, so the equivalence
    is asserted rather than argued: same assignments, and residuals far below
    the tie tolerance the result feeds.
    """
    from reassembly.mesh.patches import _squared_distances, _tie_tolerance

    rng = np.random.default_rng(0)
    for n, k, scale in [(50, 7, 1.0), (2000, 64, 1.0), (500, 16, 1000.0),
                        (9000, 128, 0.001)]:
        pts = rng.normal(scale=scale, size=(n, 3))
        seeds = pts[rng.choice(n, size=k, replace=False)]

        blocked = _squared_distances(pts, seeds)
        direct = ((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)

        tol = _tie_tolerance(pts)
        assert np.abs(blocked - direct).max() < tol * 1e-3, (n, k, scale)
        assert np.array_equal(np.argmin(blocked, axis=1), np.argmin(direct, axis=1))
        assert (blocked >= 0).all(), "cancellation must not produce negatives"


def test_blocked_distances_handle_coincident_points():
    """Zero distance is where the expansion is weakest; it must still give 0."""
    from reassembly.mesh.patches import _squared_distances
    pts = np.array([[1e3, 1e3, 1e3], [1e3, 1e3, 1e3], [0.0, 0.0, 0.0]])
    d = _squared_distances(pts, pts[:2])
    assert d[0, 0] == 0.0 and d[1, 1] == 0.0
    assert (d >= 0).all()


def test_patch_mode_memory_stays_bounded():
    """
    The regression this guards: the direct formulation allocated n*k*3 floats,
    338 MB for a 10k-vertex fragment at k=1024, which a batched loader turns
    into an OOM.
    """
    import tracemalloc
    m = trimesh.creation.icosphere(subdivisions=5)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    mask = np.ones(len(V), bool)

    tracemalloc.start()
    fracture_patches(V, F, mask, mode="patch", max_patches=1024)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 120e6, f"peak {peak/1e6:.0f} MB -- the (n,k,3) temporary is back"


def test_scene_tokens_accepts_meshes_and_tuples_alike():
    from reassembly.mesh.patches import scene_tokens
    m = trimesh.creation.icosphere(subdivisions=2)
    mask = np.ones(len(m.vertices), bool)

    as_mesh = scene_tokens([m, m], [mask, mask], total=40)
    as_tuple = scene_tokens([(np.asarray(m.vertices), np.asarray(m.faces))] * 2,
                            [mask, mask], total=40)
    for a, b in zip(as_mesh, as_tuple):
        assert np.array_equal(a.vertex_index, b.vertex_index)


# --------------------------------------------------------------------------
# capacity-aware allocation
# --------------------------------------------------------------------------

def test_zero_capacity_fragments_get_no_budget():
    """
    The real dataset contains fragments with zero fracture surface. Giving one a
    token it cannot use under-spends the budget and quietly breaks the
    fixed-count promise.
    """
    alloc = allocate_budget([200.0, 150.0, 0.0, 100.0], total=64,
                            capacity=[210, 160, 0, 105])
    assert alloc[2] == 0
    assert alloc.sum() == 64, "the freed token must go to a fragment that can use it"


def test_allocation_never_exceeds_what_a_fragment_can_supply():
    alloc = allocate_budget([10.0, 10.0, 10.0], total=100, capacity=[5, 5, 200])
    assert alloc.tolist() == [5, 5, 90]
    assert alloc.sum() == 100


def test_budget_larger_than_total_capacity_is_capped_not_invented():
    """Asking for more tokens than exist yields everything, not padding."""
    alloc = allocate_budget([1.0, 1.0], total=1000, capacity=[7, 3])
    assert alloc.tolist() == [7, 3]


def test_scene_tokens_skips_a_fragment_with_no_fracture_surface():
    """End to end: an empty mask consumes no budget and yields no tokens."""
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(4)
    masks[2] = np.zeros_like(masks[2])            # this one has no fracture surface

    got = scene_tokens(frags, masks, total=100)
    assert got[2].num_patches == 0
    assert sum(a.num_patches for a in got) == 100, "budget must still be fully spent"


def test_all_zero_capacity_returns_nothing_rather_than_raising():
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(3)
    masks = [np.zeros_like(m) for m in masks]
    got = scene_tokens(frags, masks, total=64)
    assert all(a.num_patches == 0 for a in got)


def test_allocation_survives_float_noise_in_the_weights():
    """
    Regression for a silent pose-dependence. Areas computed from coordinates
    carry noise around 1e-16, and shares land on exact integers whenever
    fragment sizes are in a simple ratio -- radii 1:2:3:4 give areas 1:4:9:16,
    so a budget of 120 splits into exactly 4/16/36/64. Without quantisation
    `floor(3.9999999999)` and `floor(4.0000000001)` disagree, the allocation
    flips, and two copies of one scene get different graphs.
    """
    exact = np.array([1.0, 4.0, 9.0, 16.0]) * 12.3298486
    cap = [162] * 4
    base = allocate_budget(exact, 120, capacity=cap)

    rng = np.random.default_rng(0)
    for _ in range(40):
        jittered = exact * (1.0 + rng.uniform(-4e-16, 4e-16, size=4))
        assert np.array_equal(allocate_budget(jittered, 120, capacity=cap), base)


def test_scene_tokens_is_pose_invariant_with_far_away_fragments():
    """
    The end-to-end version: fragments rotated *and* translated a long way from
    the origin, where float precision in the area weights is at its worst.
    """
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(4)
    rng = np.random.default_rng(11)
    moved = [(V @ _rotation(int(rng.integers(0, 1000))).T
              + rng.normal(scale=500.0, size=3), F) for V, F in frags]

    plain = scene_tokens(frags, masks, total=120)
    shifted = scene_tokens(moved, masks, total=120)
    assert [a.num_patches for a in plain] == [a.num_patches for a in shifted]
    for a, b in zip(plain, shifted):
        assert np.array_equal(a.vertex_index, b.vertex_index)


# --------------------------------------------------------------------------
# the scene budget, and why there is no per-fragment cap
# --------------------------------------------------------------------------

def test_the_scene_total_alone_bounds_the_cost():
    """
    The arithmetic that makes a per-fragment cap unnecessary.

    For a budget `T` split into `t_i`, the undirected cross-fragment pair count
    is `(T^2 - sum t_i^2)/2 < T^2/2` for *any* split -- so the total alone caps
    the graph however lopsided the allocation, and a cap cannot tighten it.
    """
    garf_stack = 6 * (5000 * 4999 // 2)
    total = 2048
    for n in (2, 3, 8, 20, 99):
        even = [total // n] * n
        even[0] += total - sum(even)
        dominant = [total - (n - 1)] + [1] * (n - 1)
        for split in (even, dominant):
            assert sum(split) == total
            assert cross_fragment_pairs(split) <= total ** 2 / 2
        # The even split is the worst case, and it is still 0.03x GARF.
        assert cross_fragment_pairs(even) < garf_stack / 30


def test_a_cap_would_flatten_the_proportionality_it_is_meant_to_preserve():
    """
    Why the per-fragment cap was removed.

    A head statue with both ears, the nose and a piece of hair broken off: the
    head carries four mating surfaces and each small piece carries one. Weighted
    by fracture-surface area the head draws four times an ear's tokens, which is
    the point. A cap collapses them to equal -- the piece with the most
    interface to match gets the least resolution per unit of it.
    """
    area = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
    uncapped = allocate_budget(area, 2048)
    assert uncapped[0] == pytest.approx(4 * uncapped[1], rel=0.02)

    capped = allocate_budget(area, 2048, capacity=[128] * 5)
    assert capped[0] == capped[1], "a cap makes the head look like an ear"
    assert capped.sum() < uncapped.sum(), "and it throws the rest of the budget away"


def test_the_budget_goes_to_whichever_fragment_has_more_fracture_surface():
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(3)
    got = scene_tokens(frags, masks, total=120, weights=[9.0, 3.0, 1.0])
    counts = [a.num_patches for a in got]
    assert sum(counts) == 120
    assert counts[0] > counts[1] > counts[2]
    assert counts[0] == pytest.approx(3 * counts[1], rel=0.05)


def test_scene_total_binds_on_large_scenes():
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(20)
    got = scene_tokens(frags, masks, total=200)
    assert sum(a.num_patches for a in got) == 200


def test_a_cap_is_still_available_as_an_ablation():
    """Removed as a default, not as a capability."""
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(3)
    got = scene_tokens(frags, masks, total=2048, max_per_fragment=64)
    assert [a.num_patches for a in got] == [64, 64, 64]


def test_capacity_still_wins_over_any_limit():
    """A fragment cannot give more than it has, whatever the budget says."""
    from reassembly.mesh.patches import scene_tokens
    frags, masks = _scene_of(2)
    masks[0] = np.zeros_like(masks[0])
    masks[0][:5] = True                       # only 5 eligible vertices
    got = scene_tokens(frags, masks, total=500)
    assert got[0].num_patches == 5


# --------------------------------------------------------------------------
# Geodesic sampling -- distance along the break, not through the material
# --------------------------------------------------------------------------

def _hairpin(n=60, width=3, gap=0.10, arm=2.0):
    """
    A U-shaped strip whose two arms are close in space and far along the
    surface -- the shape a fracture surface actually takes on a mug shard,
    where the break runs down one side, round the base and back up the other.
    """
    t = np.linspace(0.0, arm, n)
    rows = []
    for j in range(width):
        rows.append(np.stack([t, np.zeros(n), np.full(n, j * 0.03)], -1))
    for j in range(width):
        rows.append(np.stack([t[::-1], np.full(n, gap), np.full(n, j * 0.03)], -1))
    vertices = np.concatenate(rows)

    total = 2 * width
    grid = np.arange(len(vertices)).reshape(total, n)
    faces = []
    for i in range(total - 1):
        for j in range(n - 1):
            a, b = grid[i, j], grid[i + 1, j]
            faces += [[a, b, grid[i, j + 1]], [b, grid[i + 1, j + 1], grid[i, j + 1]]]
    for j in range(width - 1):                       # bridge the fold
        faces += [[grid[j, -1], grid[width + j, 0], grid[j + 1, -1]],
                  [grid[width + j, 0], grid[width + j + 1, 0], grid[j + 1, -1]]]
    return vertices, np.array(faces), np.ones(len(vertices), bool)


def _surface_distances(vertices, faces, mask):
    """All-pairs shortest path along the fracture surface, for scoring."""
    from scipy.sparse.csgraph import dijkstra

    from reassembly.mesh.patches import _fracture_adjacency, _surface_graph
    from reassembly.mesh.topology import unique_edges

    index = np.flatnonzero(mask)
    remap = np.full(len(vertices), -1, np.int64)
    remap[index] = np.arange(index.size)
    local = remap[_fracture_adjacency(unique_edges(faces, len(vertices)), mask)]
    return dijkstra(_surface_graph(local, vertices[index]), directed=False), remap


def test_geodesic_spreads_tokens_further_apart_on_a_folded_surface():
    """
    The reason geodesic is the default.

    On a strip folded back on itself, straight-line distance measures *through
    the material*: two points on opposite arms read as neighbours, so the
    sampler declines to place tokens on both and the strip is covered unevenly.
    Minimum separation is the property "disk sampling" names -- it is the disk
    radius -- and geodesic roughly doubles it where the budget is tight.
    """
    vertices, faces, mask = _hairpin()
    distances, remap = _surface_distances(vertices, faces, mask)

    def min_separation(metric, k):
        assignment = fracture_patches(vertices, faces, mask, mode="sample",
                                      max_patches=k, metric=metric)
        selected = remap[assignment.vertex_index]
        block = distances[np.ix_(selected, selected)].copy()
        np.fill_diagonal(block, np.inf)
        return block.min()

    for k, factor in ((6, 1.8), (12, 1.5)):
        euclidean = min_separation("euclidean", k)
        geodesic = min_separation("geodesic", k)
        assert geodesic > factor * euclidean, (
            f"k={k}: geodesic separation {geodesic:.3f} vs euclidean "
            f"{euclidean:.3f} -- expected at least {factor}x"
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geodesic_sampling_is_rotation_invariant(seed):
    """
    Edge lengths do not change under a rigid motion, so surface distances do
    not either, and ties break by vertex index. The same vertices must be
    selected in every pose -- otherwise the two copies of a scene the loader
    builds would carry different tokens.
    """
    vertices, faces, mask = _hairpin(n=30)
    R = _rotation(seed)
    plain = fracture_patches(vertices, faces, mask, mode="sample",
                             max_patches=12, metric="geodesic")
    rotated = fracture_patches(vertices @ R.T, faces, mask, mode="sample",
                               max_patches=12, metric="geodesic")
    assert np.array_equal(plain.vertex_index, rotated.vertex_index)


def test_geodesic_sampling_covers_every_disconnected_patch_first():
    """
    A fragment that broke from three neighbours has three separate fracture
    patches, at infinite surface distance from each other. `argmax` therefore
    seeds each one before refining any -- the right priority, and it falls out
    of the metric with no special case for it.
    """
    meshes = [trimesh.creation.icosphere(subdivisions=1, radius=0.4) for _ in range(3)]
    for i, mesh in enumerate(meshes):
        mesh.apply_translation([3.0 * i, 0.0, 0.0])
    combined = trimesh.util.concatenate(meshes)
    vertices, faces = np.asarray(combined.vertices), np.asarray(combined.faces)
    mask = np.ones(len(vertices), bool)

    per_patch = len(meshes[0].vertices)
    for k in (3, 6, 9):
        assignment = fracture_patches(vertices, faces, mask, mode="sample",
                                      max_patches=k, metric="geodesic")
        patch_of_token = assignment.vertex_index // per_patch
        assert len(np.unique(patch_of_token)) == 3, (
            f"k={k} put tokens on only {len(np.unique(patch_of_token))} of 3 patches"
        )


def test_euclidean_metric_is_still_available():
    """Kept as the ablation: the claim above is comparative, so both must run."""
    vertices, faces, mask = _hairpin(n=30)
    for metric in ("euclidean", "geodesic"):
        assignment = fracture_patches(vertices, faces, mask, mode="sample",
                                      max_patches=8, metric=metric)
        assert assignment.vertex_index.size == 8
        assert mask[assignment.vertex_index].all()
