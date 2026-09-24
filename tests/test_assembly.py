"""
Stage two: the translation solver and the benchmark scores built on it.

Reference values first, as everywhere else: with the true rotations and exact
correspondences the solver must recover the true translations to round-off,
and a perfect assembly must score RMSE(T) 0, Chamfer 0, part accuracy 1.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.assembly import (  # noqa: E402
    Matches,
    mean_over_scenes,
    mutual_nearest_neighbours,
    score_batch,
    solve_translations,
    subsample_per_fragment,
)

DTYPE = torch.float64


def _fragments_with_known_translations(seed=0, fragments=4, points=40):
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(fragments * points, 3, generator=generator, dtype=DTYPE)
    owner = torch.arange(fragments).repeat_interleave(points)
    truth = torch.randn(fragments, 3, generator=generator, dtype=DTYPE)
    truth = truth - truth.mean(0, keepdim=True)
    return local, owner, truth


def _exact_matches(local, owner, truth, count=60, seed=1):
    """
    Pairs (i on fragment a, j on fragment b) that coincide once placed: move j
    so that local_j + t_b == local_i + t_a.
    """
    generator = torch.Generator().manual_seed(seed)
    local = local.clone()
    order = torch.randperm(local.shape[0], generator=generator).tolist()
    sources, targets = [], []
    # Disjoint pairs: moving j for one pair must not break an earlier one.
    for i, j in zip(order[0::2], order[1::2]):
        if len(sources) == count or owner[i] == owner[j]:
            continue
        local[j] = local[i] + truth[owner[i]] - truth[owner[j]]
        sources.append(i)
        targets.append(j)
    source, target = torch.tensor(sources), torch.tensor(targets)
    return local, Matches(source, target, torch.ones(len(sources), dtype=DTYPE))


def test_exact_correspondences_recover_the_translations():
    local, owner, truth = _fragments_with_known_translations()
    local, matches = _exact_matches(local, owner, truth)
    solved = solve_translations(local, owner, matches, 4)
    assert torch.allclose(solved, truth, atol=1e-8)
    assert torch.allclose(solved.mean(0), torch.zeros(3, dtype=DTYPE), atol=1e-12)


def test_a_few_wrong_matches_do_not_drag_the_assembly():
    """The Huber reweighting is what makes the solver usable on real matches."""
    local, owner, truth = _fragments_with_known_translations(seed=2)
    local, matches = _exact_matches(local, owner, truth, count=120, seed=3)
    generator = torch.Generator().manual_seed(4)
    wrong = matches.source.numel() // 10
    shuffled = matches.target.clone()
    shuffled[:wrong] = torch.randint(0, local.shape[0], (wrong,), generator=generator)
    noisy = Matches(matches.source, shuffled, matches.weight)
    once = solve_translations(local, owner, noisy, 4, iterations=1)
    robust = solve_translations(local, owner, noisy, 4, iterations=10, huber=0.05)
    assert (robust - truth).norm() < 0.25 * (once - truth).norm()


def test_matching_pairs_different_fragments_only_and_each_pair_once():
    embeddings = torch.tensor([[0.0], [0.01], [5.0], [5.02], [9.0]])
    owner = torch.tensor([0, 0, 1, 1, 2])
    source, target = mutual_nearest_neighbours(embeddings, owner)
    pairs = set(zip(source.tolist(), target.tolist()))
    assert all(owner[a] != owner[b] for a, b in pairs)
    assert all((b, a) not in pairs for a, b in pairs)


def test_subsampling_is_deterministic_and_prefers_the_candidates():
    owner = torch.tensor([0] * 10 + [1] * 6)
    fracture = torch.zeros(16, dtype=torch.bool)
    fracture[2:8] = True                      # fragment 1 has none: falls back
    keep = subsample_per_fragment(owner, 3, fracture)
    assert torch.equal(keep, subsample_per_fragment(owner, 3, fracture))
    assert set(keep[owner[keep] == 0].tolist()) <= set(range(2, 8))
    assert (owner[keep] == 1).sum() == 3


def _broken_scene(fractured_solid):
    from reassembly.data.features import build_scene, collate
    from reassembly.data.transforms import random_rotations
    from reassembly.mesh.fracture import fracture_vertex_masks

    meshes = list(fractured_solid)
    vertices = [np.asarray(m.vertices, dtype=np.float64) + [0.3, -0.2, 0.1] for m in meshes]
    faces = [np.asarray(m.faces) for m in meshes]
    masks = fracture_vertex_masks(meshes)
    sample = build_scene(vertices, faces, masks,
                         rotations=random_rotations(2, np.random.default_rng(3)),
                         tokens_per_scene=64)
    return collate([sample], dtype=DTYPE), vertices


def _oracle(batch, rotation):
    """A 'prediction' that knows the answer: exact rotations, and the assembled
    world position as the (invariant) embedding, so matches are the truly
    coincident interface points."""
    from reassembly.nn.model import Prediction

    world = (batch.target_vertices * batch.unit[batch.vertex_fragment, None]
             + batch.centroid[batch.vertex_fragment])
    return Prediction(rotation=rotation, frame=rotation.transpose(-1, -2),
                      vertex_embedding=world, vertex_features=None)


def test_a_perfect_prediction_scores_perfectly(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    scenes = score_batch(batch, _oracle(batch, batch.target_rotation))
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene["matches"] > 0
    assert scene["rmse_t"] < 1e-9
    assert scene["chamfer"] < 1e-12 and scene["part_chamfer"] < 1e-12
    assert scene["part_accuracy"] == 1.0


def test_a_wrong_rotation_is_not_rescued_by_the_solver(fractured_solid):
    """
    Translations cannot fix orientation: a fragment tipped 90 degrees onto its
    side fails. (About the vertical axis the square test slab would map onto
    itself up to the interface noise -- a symmetry, not a pass.)
    """
    batch, _ = _broken_scene(fractured_solid)
    turn = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    rotation = batch.target_rotation.clone()
    rotation[1] = turn @ rotation[1]
    scene = score_batch(batch, _oracle(batch, rotation))[0]
    assert scene["part_accuracy"] < 1.0
    assert scene["chamfer"] > 1e-3


def test_scores_average_per_scene_then_over_scenes():
    scenes = [{"rmse_t": 1.0, "part_accuracy": 1.0},
              {"rmse_t": 3.0, "part_accuracy": float("nan")}]
    mean = mean_over_scenes(scenes)
    assert mean["rmse_t"] == 2.0 and mean["part_accuracy"] == 1.0
