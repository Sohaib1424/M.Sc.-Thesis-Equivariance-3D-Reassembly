"""
Stage two, rotations: fitted from the embedding matches and chained from the
anchor (:mod:`reassembly.assembly.rotation`).

Reference values first, as for the translation solver: with exact
correspondences the fits must be exact, the chain must compose them the right
way round, a wrong starting rotation must not matter to any fragment the chain
reaches, and the batch-level matching must read nothing but what the method is
given at inference.
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
    match_batch,
    match_rotations,
    ransac_rotation,
    score_batch,
)
from reassembly.nn.losses import geodesic_angle  # noqa: E402

from test_assembly import _broken_scene, _given, _oracle  # noqa: E402

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
    start = _rotations(4, 8)                     # wrong everywhere
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
    oracle = _oracle(batch)
    root = int(torch.argmax(batch.log_scale.flatten()))
    start = _rotations(2, 14)
    start[root] = batch.target_rotation[root]
    chained, reached, pairs = match_rotations(
        points, batch.vertex_fragment, oracle.vertex_embedding, start, root,
        candidates=batch.fracture, generator=torch.Generator().manual_seed(0))
    assert len(pairs) == 1 and pairs[0].inliers >= 6
    assert bool(reached.all())
    assert float(_degrees(chained, batch.target_rotation).max()) < 1e-6


def test_the_scorer_fits_what_a_given_wrong_rotation_gets_wrong(fractured_solid):
    """A fragment tipped onto its side fails when its rotation is given; fitted
    from the matches, the same scene scores perfectly."""
    batch, _ = _broken_scene(fractured_solid)
    turn = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    wrong = batch.target_rotation.clone()
    scored = 1 - int(torch.argmax(batch.log_scale.flatten()))
    wrong[scored] = turn @ wrong[scored]

    given = score_batch(batch, _oracle(batch), matched=_given(batch, wrong),
                        placement="global")[0]
    matched = score_batch(batch, _oracle(batch))[0]
    assert given["part_accuracy"] < 1.0 and given["geodesic_deg"] > 80.0
    assert matched["geodesic_deg"] < 1e-5
    assert matched["part_accuracy"] == 1.0 and matched["rmse_t"] < 1e-6
    assert matched["matched_share"] == 1.0 and matched["_reached"] == [True, True]
    assert len(matched["_scored_geodesic_deg"]) == 1


def test_the_batch_matching_reads_no_truth(fractured_solid):
    """Every target field gone -- rotations, positions, normals, centroids,
    clusters -- and the matched rotations are the same to the bit."""
    batch, _ = _broken_scene(fractured_solid)
    embedding = _oracle(batch).vertex_embedding
    plain = match_batch(batch, embedding)
    hidden = batch._replace(
        **{name: torch.full_like(getattr(batch, name), float("nan"))
           for name in ("target_rotation", "target_vertices", "target_normals",
                        "target_edge_normals", "centroid")},
        cluster=torch.full_like(batch.cluster, -1), num_clusters=0)
    again = match_batch(hidden, embedding)
    assert torch.equal(plain.rotation, again.rotation)
    assert torch.equal(plain.reached, again.reached) and bool(plain.reached.all())


def test_the_root_is_the_identity_and_an_unreached_fragment_keeps_it(fractured_solid):
    """Truth-free: the largest fragment is its own frame's identity, and a
    fragment no fit reaches keeps the identity too -- for the anchor protocol
    to align, not a guess."""
    batch, _ = _broken_scene(fractured_solid)
    root = int(torch.argmax(batch.log_scale.flatten()))
    lonely = match_batch(batch, _oracle(batch).vertex_embedding, min_matches=10 ** 6)
    eye = torch.eye(3, dtype=DTYPE).expand(2, 3, 3)
    assert torch.equal(lonely.rotation, eye)
    assert lonely.reached.tolist() == [i == root for i in range(2)]
    assert lonely.roots == [root] and lonely.pairs == [[]]
    fitted = match_batch(batch, _oracle(batch).vertex_embedding)
    assert torch.equal(fitted.rotation[root], eye[0])


def test_the_draws_are_tied_to_the_scene_not_the_run(fractured_solid):
    """Two evaluations of one checkpoint must agree: RANSAC is seeded per scene
    by name, not by a stream that depends on what was scored before."""
    batch, _ = _broken_scene(fractured_solid)
    noisy = _oracle(batch)
    generator = torch.Generator().manual_seed(17)
    embedding = noisy.vertex_embedding + 0.05 * torch.randn(
        noisy.vertex_embedding.shape, generator=generator, dtype=DTYPE)
    noisy = noisy._replace(vertex_embedding=embedding)
    first = score_batch(batch, noisy)[0]
    second = score_batch(batch, noisy)[0]
    assert first["_rotation"] == second["_rotation"]
    assert math.isfinite(first["geodesic_deg"])
