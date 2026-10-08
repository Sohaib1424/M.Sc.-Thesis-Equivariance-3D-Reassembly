"""
Stage two, placement from the verified pair fits
(:mod:`reassembly.assembly.placement`).

The failure it fixes, reproduced first: a stack of slabs whose embedding
matches are partly wrong -- pairs of break points on slabs that do not touch,
as a trained network produces in many-piece scenes. With the rotations exact,
the global solve (v6's only placement) still pulls the stack together; the
checked placement puts every slab back.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from reassembly.assembly import (  # noqa: E402
    PairRotation,
    agreement_deg,
    check_placement,
    default_placement,
    match_rotations,
    pairwise_rotations,
    ransac_motion,
    ransac_rotation,
    score_batch,
    solve_with_held,
    verified_pairs,
)
from reassembly.nn.losses import geodesic_angle  # noqa: E402

from test_assembly import (  # noqa: E402
    _broken_scene,
    _exact_matches,
    _fragments_with_known_translations,
    _oracle,
)

DTYPE = torch.float64


def _rotations(n: int, seed: int) -> torch.Tensor:
    from reassembly.data.transforms import random_rotations

    return torch.as_tensor(random_rotations(n, np.random.default_rng(seed)), dtype=DTYPE)


def _degrees(a, b) -> torch.Tensor:
    return torch.rad2deg(geodesic_angle(a, b))


# ------------------------------------------------------- a stack of slabs --

def _grid_faces(n: int, flip: bool = False) -> np.ndarray:
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    b, c, d = a + 1, a + n, a + n + 1
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return faces[:, ::-1] if flip else faces


def _rim(n: int) -> np.ndarray:
    top = np.arange(n)
    right = np.arange(1, n) * n + (n - 1)
    bottom = (n - 1) * n + np.arange(n - 2, -1, -1)
    left = np.arange(n - 2, 0, -1) * n
    return np.concatenate([top, right, bottom, left])


def _slab(top: np.ndarray, bottom: np.ndarray, n: int):
    """A closed solid between two ``n x n`` height fields, ``top`` above ``bottom``."""
    count = len(top)
    rim = _rim(n)
    after = np.roll(rim, -1)
    walls = np.concatenate([np.stack([rim, after, rim + count], 1),
                            np.stack([after, after + count, rim + count], 1)])
    faces = np.vstack([_grid_faces(n), _grid_faces(n, flip=True) + count, walls])
    return trimesh.Trimesh(np.vstack([top, bottom]), faces, process=False)


THICKNESS = (0.6, 0.7, 0.8, 1.2, 0.75, 0.65, 0.7, 0.6)   # the thickest, slab 3, anchors


def _stack(n: int = 10, seed: int = 0, thickness=THICKNESS):
    """
    Slabs stacked along z, each touching only the ones above and below it
    through a rough interface whose vertices both sides share -- a break, as
    Breaking Bad stores one. Like ``conftest.fractured_solid``, with more pieces.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    levels = np.concatenate([[0.0], np.cumsum(thickness)])
    surfaces = []
    for k, level in enumerate(levels):
        gz = np.full((n, n), level)
        if 0 < k < len(levels) - 1:
            noise = rng.normal(scale=0.08, size=(n, n))
            noise[0, :] = noise[-1, :] = noise[:, 0] = noise[:, -1] = 0.0
            gz = gz + noise
        surfaces.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))
    return [_slab(surfaces[k + 1], surfaces[k], n) for k in range(len(thickness))]


def _stacked_scene():
    from reassembly.data.features import build_scene, collate
    from reassembly.data.transforms import random_rotations
    from reassembly.mesh.fracture import fracture_vertex_masks

    meshes = _stack()
    sample = build_scene([np.asarray(m.vertices) for m in meshes],
                         [np.asarray(m.faces) for m in meshes], fracture_vertex_masks(meshes),
                         rotations=random_rotations(len(meshes), np.random.default_rng(3)),
                         tokens_per_scene=64)
    return collate([sample], dtype=DTYPE)


def _with_wrong_matches(batch, rotation, decoys: int, seed: int = 5, isolate=()):
    """
    The oracle's prediction (exact matches across every break), with
    ``decoys`` wrong matches added: two break vertices on slabs that do not
    touch get one shared code in four extra embedding dimensions, so each is
    the other's nearest neighbour. Every break vertex of a slab in ``isolate``
    gets a code of its own first, so none of its true matches survives; the
    wrong ones may still use it.
    """
    prediction = _oracle(batch, rotation)
    fragment = batch.vertex_fragment.long()
    fracture = batch.fracture.bool()
    count = int(batch.fragment_ptr[-1])
    rng = np.random.default_rng(seed)
    extra = torch.zeros(fragment.numel(), 4, dtype=DTYPE)
    for f in isolate:
        own = (fragment == f) & fracture
        extra[own] = torch.as_tensor(rng.standard_normal((int(own.sum()), 4)) * 1e3)
    pools = [np.flatnonzero(((fragment == f) & fracture).numpy()) for f in range(count)]
    used, made = set(), 0
    while made < decoys:
        i, j = (int(v) for v in rng.choice(count, 2, replace=False))
        if abs(i - j) < 2:
            continue
        a, b = int(rng.choice(pools[i])), int(rng.choice(pools[j]))
        if a in used or b in used:
            continue
        used |= {a, b}
        extra[a] = extra[b] = torch.as_tensor(rng.standard_normal(4) * 1e3)
        made += 1
    embedding = torch.cat([prediction.vertex_embedding, extra], dim=1)
    return prediction._replace(vertex_embedding=embedding)


def _spread(translations) -> float:
    t = np.asarray(translations)
    return float(np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()))


def test_wrong_matches_pull_the_global_solve_together_but_not_the_checked_one():
    """
    The bug, in miniature: about a quarter of the matches join slabs that do
    not touch. The rotations come out exact either way -- the chain only
    follows the well-supported fits -- but the global solve pulls the stack to
    60% of its height and places no slab, while the checked placement, with
    the same rotations, places every slab exactly.
    """
    batch = _stacked_scene()
    prediction = _with_wrong_matches(batch, _rotations(len(THICKNESS), 9), decoys=150)
    global_ = score_batch(batch, prediction, rotations="matched", placement="global")[0]
    checked = score_batch(batch, prediction, rotations="matched", placement="checked")[0]

    assert global_["_rotation"] == checked["_rotation"]
    assert checked["geodesic_deg"] < 1e-6 and checked["matched_share"] == 1.0
    true = (batch.centroid - batch.centroid[checked["_anchor"]]).numpy()
    assert _spread(global_["_translation"]) < 0.8 * _spread(true)
    assert global_["part_accuracy"] < 0.5 and global_["rmse_t"] > 0.1
    assert checked["part_accuracy"] == 1.0 and checked["rmse_t"] < 1e-6
    assert np.allclose(checked["_translation"], true, atol=1e-6)
    # Every match is still counted; the verified ones are the true matches the
    # decoys left.
    assert checked["matches"] == global_["matches"]
    assert 0 < checked["verified_matches"] < checked["matches"]


def test_without_wrong_matches_both_placements_are_exact():
    batch = _stacked_scene()
    prediction = _oracle(batch, _rotations(len(THICKNESS), 9))
    for placement in ("global", "checked"):
        scene = score_batch(batch, prediction, rotations="matched", placement=placement)[0]
        assert scene["part_accuracy"] == 1.0 and scene["rmse_t"] < 1e-6, placement
    assert scene["verified_matches"] == scene["matches"]


def test_a_fragment_the_chain_cannot_reach_misplaces_only_itself():
    """
    Slab 0 keeps no true match, only wrong ones, so the chain cannot reach it
    and it keeps the head's wrong rotation. It is placed from those matches
    with every other slab held: wherever it lands, the others stay exact.
    """
    batch = _stacked_scene()
    head = _rotations(len(THICKNESS), 9)
    prediction = _with_wrong_matches(batch, head, decoys=40, isolate=(0,))
    scene = score_batch(batch, prediction, rotations="matched", placement="checked")[0]
    assert scene["_reached"] == [False] + [True] * (len(THICKNESS) - 1)
    assert scene["_scored_geodesic_deg"][0] > 10.0
    true = (batch.centroid - batch.centroid[scene["_anchor"]]).numpy()
    others = np.asarray(scene["_translation"])[1:]
    assert np.allclose(others, true[1:], atol=1e-6)
    assert max(scene["_part_chamfer"][1:]) < 1e-12


def test_the_mean_gauge_centres_the_checked_placement(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    scene = score_batch(batch, _oracle(batch, batch.target_rotation), rotations="matched",
                        anchor=False)[0]
    assert scene["rmse_t"] < 1e-9 and scene["part_accuracy"] == 1.0
    assert np.allclose(np.mean(scene["_translation"], axis=0), 0.0, atol=1e-12)


def test_collisions_still_measure_from_the_anchor(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    scene = score_batch(batch, _oracle(batch, batch.target_rotation), rotations="matched",
                        collision=True)[0]
    assert scene["_translation"][scene["_anchor"]] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------ the pieces --

def test_the_fit_keeps_its_offset_and_its_inliers():
    R = _rotations(1, 20)[0]
    generator = torch.Generator().manual_seed(21)
    source = torch.randn(100, 3, generator=generator, dtype=DTYPE)
    shift = torch.tensor([0.2, -0.4, 1.0], dtype=DTYPE)
    target = source @ R.T + shift
    target[:30] = torch.randn(30, 3, generator=generator, dtype=DTYPE)       # 30% wrong
    rotation, fitted, inliers = ransac_motion(source, target,
                                              generator=torch.Generator().manual_seed(22))
    assert float(_degrees(rotation[None], R[None])) < 1e-8
    assert torch.allclose(fitted, shift, atol=1e-10)
    assert not bool(inliers[:30].any()) and bool(inliers[30:].all())
    # The rotation-only form makes the same draws, so it finds the same fit.
    same, count = ransac_rotation(source, target, generator=torch.Generator().manual_seed(22))
    assert torch.equal(same, rotation) and count == 70


def test_each_pair_carries_the_matches_its_motion_moves_into_place(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    points = batch.node_features[:, 0, :] * batch.unit[batch.vertex_fragment, None]
    points = points / batch.unit.max()
    oracle = _oracle(batch, batch.target_rotation)
    pairs = pairwise_rotations(points, batch.vertex_fragment, oracle.vertex_embedding, 2,
                               candidates=batch.fracture,
                               generator=torch.Generator().manual_seed(0))
    (pair,) = pairs
    assert pair.source.numel() == pair.target.numel() == pair.inliers >= 6
    assert bool((batch.vertex_fragment[pair.source] == pair.i).all())
    assert bool((batch.vertex_fragment[pair.target] == pair.j).all())
    moved = points[pair.source] @ pair.rotation.T + pair.shift
    assert float((moved - points[pair.target]).norm(dim=-1).max()) < 1e-9


def test_held_fragments_stay_and_the_free_ones_are_solved():
    local, owner, truth = _fragments_with_known_translations(seed=5)
    local, matches = _exact_matches(local, owner, truth, seed=6)
    held = torch.zeros(4, 3, dtype=DTYPE)
    held[0] = truth[0]
    free = torch.tensor([False, True, True, True])
    solved = solve_with_held(local, owner, matches, free, held)
    assert torch.equal(solved[0], truth[0])
    assert torch.allclose(solved, truth, atol=1e-6)


def test_a_fragment_no_match_reaches_goes_to_the_held_mean():
    local, owner, truth = _fragments_with_known_translations(seed=7, fragments=3)
    local, matches = _exact_matches(local, owner, truth, seed=8)
    local = torch.cat([local, torch.randn(10, 3, dtype=DTYPE)])
    owner = torch.cat([owner, torch.full((10,), 3)])
    held = torch.zeros(4, 3, dtype=DTYPE)
    held[0] = truth[0]
    free = torch.tensor([False, True, True, True])
    solved = solve_with_held(local, owner, matches, free, held)
    assert torch.allclose(solved[:3], truth, atol=1e-6)
    assert torch.allclose(solved[3], truth[0], atol=1e-9)


def _fit(truth, i, j, rotation=None, inliers=10):
    index = torch.arange(inliers)
    rotation = truth[j].T @ truth[i] if rotation is None else rotation
    return PairRotation(i, j, rotation, inliers, inliers, torch.zeros(3, dtype=DTYPE),
                        index, index)


def _turn(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    return torch.tensor([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]], dtype=DTYPE)


def test_only_the_pairs_the_chain_agrees_with_are_kept():
    truth = _rotations(5, 23)
    reached = torch.tensor([True, True, True, True, False])
    pairs = [
        _fit(truth, 0, 1), _fit(truth, 1, 2), _fit(truth, 1, 3),   # the chain's edges
        _fit(truth, 0, 2),                                          # a loop that closes
        _fit(truth, 2, 3, rotation=_rotations(1, 24)[0]),           # a wrong fit
        _fit(truth, 0, 3, inliers=4),                               # too few inliers
        _fit(truth, 3, 4),                                          # 4 is not reached
        PairRotation(1, 4, truth[4].T @ truth[1], 50, 50),          # no inliers kept
    ]
    kept = verified_pairs(pairs, truth, reached)
    assert [(p.i, p.j) for p in kept] == [(0, 1), (1, 2), (1, 3), (0, 2)]


def test_agreement_is_the_angle_between_the_fit_and_the_chain():
    truth = _rotations(2, 25)
    exact = truth[1].T @ truth[0]
    assert agreement_deg(_fit(truth, 0, 1), truth) < 1e-6
    off = _fit(truth, 0, 1, rotation=exact @ _turn(3.0))
    assert agreement_deg(off, truth) == pytest.approx(3.0, abs=1e-6)
    reached = torch.ones(2, dtype=torch.bool)
    assert len(verified_pairs([off], truth, reached)) == 1
    assert not verified_pairs([_fit(truth, 0, 1, rotation=exact @ _turn(8.0))], truth, reached)


# ------------------------------------------------------------ the switch --

def test_the_placement_defaults_follow_the_rotations(fractured_solid):
    assert default_placement("matched") == "checked"
    assert default_placement("network") == "global"
    assert check_placement("matched", None) == "checked"
    assert check_placement("matched", "global") == "global"
    batch, _ = _broken_scene(fractured_solid)
    oracle = _oracle(batch, batch.target_rotation)
    assert "verified_matches" in score_batch(batch, oracle, rotations="matched")[0]
    network = score_batch(batch, oracle)[0]
    assert "verified_matches" not in network
    assert network == score_batch(batch, oracle, placement="global")[0]


def test_a_placement_that_cannot_be_used_is_rejected(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    oracle = _oracle(batch, batch.target_rotation)
    with pytest.raises(ValueError, match="placement 'checked' places"):
        score_batch(batch, oracle, placement="checked")
    with pytest.raises(ValueError, match="placement must be one of"):
        score_batch(batch, oracle, rotations="matched", placement="sideways")


def test_the_root_is_held_where_it_is(fractured_solid):
    """``place`` answers relative to the root, which no solve moves -- the
    scorer's anchor gauge is then exact, not a correction."""
    from reassembly.assembly import place
    from reassembly.nn.model import apply_rotation

    batch, _ = _broken_scene(fractured_solid)
    fragment = batch.vertex_fragment
    unit = batch.unit[fragment, None]
    rotation = batch.target_rotation
    points = apply_rotation(batch.node_features[:, 0, :], rotation, fragment) * unit
    normals = apply_rotation(batch.node_features[:, 1, :], rotation, fragment)
    raw = batch.node_features[:, 0, :] * unit / batch.unit.max()
    oracle = _oracle(batch, rotation)
    root = int(torch.argmax(batch.log_scale.flatten()))
    chained, reached, pairs = match_rotations(
        raw, fragment, oracle.vertex_embedding, rotation.clone(), root,
        candidates=batch.fracture, generator=torch.Generator().manual_seed(0))
    placed = place(points, normals, fragment, oracle.vertex_embedding, 2, pairs, chained,
                   reached, root, candidates=batch.fracture)
    assert torch.equal(placed.translations[root], torch.zeros(3, dtype=DTYPE))
    true = batch.centroid - batch.centroid[root]
    assert torch.allclose(placed.translations, true, atol=1e-9)
    assert placed.links.source.numel() == pairs[0].inliers


def test_the_chain_and_the_pairs_agree_on_a_real_scene(fractured_solid):
    """The pairs ``match_rotations`` returns are the ones the placement reads."""
    batch, _ = _broken_scene(fractured_solid)
    points = batch.node_features[:, 0, :] * batch.unit[batch.vertex_fragment, None]
    points = points / batch.unit.max()
    oracle = _oracle(batch, batch.target_rotation)
    root = int(torch.argmax(batch.log_scale.flatten()))
    start = _rotations(2, 26)
    start[root] = batch.target_rotation[root]
    chained, reached, pairs = match_rotations(
        points, batch.vertex_fragment, oracle.vertex_embedding, start, root,
        candidates=batch.fracture, generator=torch.Generator().manual_seed(0))
    kept = verified_pairs(pairs, chained, reached)
    assert len(kept) == 1 and kept[0].inliers == pairs[0].inliers
