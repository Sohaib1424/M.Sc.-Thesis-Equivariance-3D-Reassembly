"""
Stage two, rotations: fitted from the embedding matches and chained from the
anchor (:mod:`reassembly.assembly.rotation`).

Reference values first, as for the translation solver: with exact
correspondences the fits must be exact, the chain must compose them the right
way round, and a rotation head that is wrong must not matter to any fragment
the chain reaches.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.assembly import (  # noqa: E402
    PairRotation,
    chain_rotations,
    kabsch,
    match_rotations,
    ransac_rotation,
    score_batch,
)
from reassembly.nn.losses import geodesic_angle  # noqa: E402

from test_assembly import _broken_scene, _oracle  # noqa: E402

DTYPE = torch.float64


def _rotations(n: int, seed: int) -> torch.Tensor:
    from reassembly.data.transforms import random_rotations

    return torch.as_tensor(random_rotations(n, np.random.default_rng(seed)), dtype=DTYPE)


def _degrees(a, b) -> torch.Tensor:
    return torch.rad2deg(geodesic_angle(a, b))


# ------------------------------------------------------------------ fitting --

def test_kabsch_recovers_a_rigid_motion_exactly():
    R = _rotations(1, 0)[0]
    source = torch.randn(50, 3, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    target = source @ R.T + torch.tensor([0.3, -1.0, 2.0], dtype=DTYPE)
    fitted, shift = kabsch(source, target)
    assert float(_degrees(fitted[None], R[None])) < 1e-8
    assert torch.allclose(shift, torch.tensor([0.3, -1.0, 2.0], dtype=DTYPE), atol=1e-10)


def test_kabsch_returns_a_rotation_not_a_reflection_for_flat_points():
    """A break surface can be nearly flat. The best orthogonal fit to coplanar
    points is ambiguous up to a reflection, which must never be returned."""
    R = _rotations(1, 2)[0]
    source = torch.randn(40, 3, generator=torch.Generator().manual_seed(3), dtype=DTYPE)
    source[:, 2] = 0.0
    fitted, _ = kabsch(source, source @ R.T)
    assert float(torch.linalg.det(fitted)) == pytest.approx(1.0, abs=1e-10)
    assert float(_degrees(fitted[None], R[None])) < 1e-6


def test_ransac_ignores_a_majority_of_wrong_matches():
    R = _rotations(1, 4)[0]
    generator = torch.Generator().manual_seed(5)
    source = torch.randn(200, 3, generator=generator, dtype=DTYPE)
    target = source @ R.T + 0.5
    target[:120] = torch.randn(120, 3, generator=generator, dtype=DTYPE)   # 60% wrong
    fitted, inliers = ransac_rotation(source, target, generator=torch.Generator().manual_seed(6))
    assert float(_degrees(fitted[None], R[None])) < 1e-8
    assert inliers == 80


def test_ransac_refuses_fewer_than_three_matches():
    with pytest.raises(ValueError, match="at least three"):
        ransac_rotation(torch.zeros(2, 3, dtype=DTYPE), torch.zeros(2, 3, dtype=DTYPE))


# ----------------------------------------------------------------- chaining --

def _true_pairs(truth, edges, inliers=50):
    """Exact fits: R_ij = T_j^T T_i carries i's input frame onto j's."""
    return [PairRotation(i, j, truth[j].T @ truth[i], inliers, inliers) for i, j in edges]


def test_the_chain_composes_the_fits_the_right_way_round():
    truth = _rotations(4, 7)
    pairs = _true_pairs(truth, [(0, 1), (1, 2), (2, 3)])
    start = _rotations(4, 8)                     # the head's: wrong everywhere
    start[2] = truth[2]                          # the root's is trusted
    chained, reached = chain_rotations(pairs, start, root=2)
    assert bool(reached.all())
    assert float(_degrees(chained, truth).max()) < 1e-8


def test_the_chain_takes_the_best_supported_path():
    """A fragment two ways from the root is placed through the edge with more
    agreeing matches, not the first one found."""
    truth = _rotations(3, 9)
    wrong = _rotations(1, 10)[0]
    pairs = _true_pairs(truth, [(0, 1)], inliers=60) + _true_pairs(truth, [(1, 2)], inliers=40)
    pairs.append(PairRotation(0, 2, wrong, 7, 7))           # weak and wrong
    chained, _ = chain_rotations(pairs, truth.clone(), root=0)
    assert float(_degrees(chained, truth).max()) < 1e-8


def test_unreached_fragments_keep_their_rotation():
    truth = _rotations(3, 11)
    start = _rotations(3, 12)
    start[0] = truth[0]
    pairs = _true_pairs(truth, [(0, 1)]) + [PairRotation(1, 2, truth[2].T @ truth[1], 4, 4)]
    chained, reached = chain_rotations(pairs, start, root=0, min_inliers=6)
    assert reached.tolist() == [True, True, False]
    assert torch.equal(chained[2], start[2])
    assert float(_degrees(chained[1:2], truth[1:2])) < 1e-8


def test_an_invalid_root_changes_nothing():
    start = _rotations(2, 13)
    chained, reached = chain_rotations([], start, root=5)
    assert torch.equal(chained, start) and not bool(reached.any())


# ----------------------------------------------------------- whole scenes --

def test_matched_rotations_are_exact_from_exact_matches(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    points = batch.node_features[:, 0, :] * batch.unit[batch.vertex_fragment, None]
    points = points / batch.unit.max()
    oracle = _oracle(batch, batch.target_rotation)
    root = int(torch.argmax(batch.log_scale.flatten()))
    start = _rotations(2, 14)
    start[root] = batch.target_rotation[root]
    chained, reached, pairs = match_rotations(
        points, batch.vertex_fragment, oracle.vertex_embedding, start, root,
        candidates=batch.fracture, generator=torch.Generator().manual_seed(0))
    assert len(pairs) == 1 and pairs[0].inliers >= 6
    assert bool(reached.all())
    assert float(_degrees(chained, batch.target_rotation).max()) < 1e-6


def test_the_scorer_recovers_a_rotation_the_head_got_wrong(fractured_solid):
    """The same wrong head that fails the network route scores perfectly once
    the rotations come from the matches -- and the head's rotation for a
    reached fragment does not enter the result at all."""
    batch, _ = _broken_scene(fractured_solid)
    turn = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    wrong = batch.target_rotation.clone()
    scored = 1 - int(torch.argmax(batch.log_scale.flatten()))
    wrong[scored] = turn @ wrong[scored]

    network = score_batch(batch, _oracle(batch, wrong))[0]
    matched = score_batch(batch, _oracle(batch, wrong), rotations="matched")[0]
    assert network["part_accuracy"] < 1.0 and network["geodesic_deg"] > 80.0
    assert matched["geodesic_deg"] < 1e-5
    assert matched["part_accuracy"] == 1.0 and matched["rmse_t"] < 1e-6
    assert matched["matched_share"] == 1.0 and matched["_reached"] == [True, True]
    assert len(matched["_scored_geodesic_deg"]) == 1

    other = batch.target_rotation.clone()
    other[scored] = _rotations(1, 15)[0]
    again = score_batch(batch, _oracle(batch, other), rotations="matched")[0]
    assert np.allclose(again["_rotation"], matched["_rotation"], atol=1e-9)


def test_the_matched_route_leaves_the_prediction_untouched(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    wrong = _rotations(2, 16)
    prediction = _oracle(batch, wrong)
    before = prediction.rotation.clone()
    score_batch(batch, prediction, rotations="matched", anchor=False)
    assert torch.equal(prediction.rotation, before)


def test_an_unknown_rotation_source_is_rejected(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    with pytest.raises(ValueError, match="rotations must be one of"):
        score_batch(batch, _oracle(batch, batch.target_rotation), rotations="head")


def test_the_draws_are_tied_to_the_scene_not_the_run(fractured_solid):
    """Two evaluations of one checkpoint must agree: RANSAC is seeded per scene
    by name, not by a stream that depends on what was scored before."""
    batch, _ = _broken_scene(fractured_solid)
    noisy = _oracle(batch, batch.target_rotation)
    generator = torch.Generator().manual_seed(17)
    embedding = noisy.vertex_embedding + 0.05 * torch.randn(
        noisy.vertex_embedding.shape, generator=generator, dtype=DTYPE)
    noisy = noisy._replace(vertex_embedding=embedding)
    first = score_batch(batch, noisy, rotations="matched")[0]
    second = score_batch(batch, noisy, rotations="matched")[0]
    assert first["_rotation"] == second["_rotation"]
    assert math.isfinite(first["geodesic_deg"])
