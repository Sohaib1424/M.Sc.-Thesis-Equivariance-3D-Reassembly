"""
The anchor protocol (``reassembly.nn.anchor``): each scene's largest fragment
is set to its true pose, and every other fragment is scored relative to it.

Each property below fails silently when broken -- a wrong side of
multiplication, an anchor chosen per batch instead of per scene, an anchor left
in the average -- and every one of those still produces a plausible number.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from reassembly.data.features import build_scene, collate
from reassembly.data.transforms import random_rotations
from reassembly.nn.anchor import (
    align_to_anchor,
    anchor_alignment,
    anchor_fragments,
    scored_count,
)
from reassembly.nn.losses import geodesic_angle
from reassembly.nn.model import Prediction
from reassembly.training import CHANCE, Config, _forward, _loss_fragments, build_criterion

DTYPE = torch.float64


def _rotations(n: int, seed: int) -> torch.Tensor:
    return torch.as_tensor(random_rotations(n, np.random.default_rng(seed)), dtype=DTYPE)


def _scene_of(sizes) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.as_tensor(sizes))


def _degrees(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(geodesic_angle(a, b))


def _sample(seed: int, fragments: int):
    """Dented spheres of decreasing size: fragment 0 is the largest."""
    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(fragments):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0 - 0.15 * i)
        v = np.asarray(mesh.vertices).copy()
        # Dented about the sphere's own centre, THEN moved into place: denting
        # after the move would scale the offset too, and the far fragments
        # would come out the "largest".
        v[rng.choice(len(v), 8, replace=False)] *= 1.1
        v += [2.5 * i, 0.0, 0.0]
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), 12, replace=False)] = True
        vertices.append(v)
        faces.append(np.asarray(mesh.faces))
        masks.append(mask)
    total = sum(len(v) for v in vertices)
    cluster = np.full(total, -1, np.int64)
    cluster[:12] = np.arange(12) // 3
    return build_scene(vertices, faces, masks, tokens_per_scene=24, cluster=cluster,
                       rotations=random_rotations(fragments, np.random.default_rng(50 + seed)))


class _Oracle(torch.nn.Module):
    """Answers with a fixed per-vertex embedding."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = torch.nn.Parameter(embedding)

    def forward(self, *args, **kwargs):
        return Prediction(vertex_embedding=self.embedding, vertex_features=None)


def _world(batch):
    """Each vertex where it sits in the assembled object: as an embedding, the
    truly coincident break vertices are each other's nearest neighbours, so
    stage two's fits are exact."""
    return (batch.target_vertices * batch.unit[batch.vertex_fragment, None]
            + batch.centroid[batch.vertex_fragment])


def _solid_batch(fractured_solid, seeds=(3, 4)):
    """The fractured solid twice, each scene scattered by its own rotations."""
    from reassembly.mesh.fracture import fracture_vertex_masks

    meshes = list(fractured_solid)
    vertices = [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
    faces = [np.asarray(m.faces) for m in meshes]
    masks = fracture_vertex_masks(meshes)
    return collate([build_scene(vertices, faces, masks, tokens_per_scene=64,
                                rotations=random_rotations(2, np.random.default_rng(seed)))
                    for seed in seeds], dtype=DTYPE)


# --------------------------------------------------------------------------
# Which fragment
# --------------------------------------------------------------------------

def test_the_anchor_is_each_scenes_largest_fragment():
    log_scale = torch.tensor([0.1, 0.5, 0.2,          # scene 0: fragment 1
                              0.3,                    # scene 1: its only fragment
                              0.9, 0.9, 0.2, 0.1])    # scene 2: a tie, the first
    anchor = anchor_fragments(log_scale[:, None], _scene_of([3, 1, 4]), 4)
    # Scene 3 has no fragments: one past the end, which means "none".
    assert anchor.tolist() == [1, 3, 4, 8]

    rotation = _rotations(8, 0)
    _, scored = align_to_anchor(rotation, rotation, _scene_of([3, 1, 4]), anchor)
    assert (~scored).nonzero().flatten().tolist() == [1, 3, 4]


def test_the_anchor_is_chosen_per_scene_not_per_batch():
    """Fragment ids are global across a batch; the largest overall must not
    become every scene's anchor."""
    first, second = _sample(0, 3), _sample(1, 4)
    batch = collate([first, second], dtype=DTYPE)
    anchor = anchor_fragments(batch.log_scale, batch.fragment_scene, batch.num_scenes)
    assert anchor.tolist() == [0, 3]      # each scene's own fragment 0
    assert scored_count(batch) == 3 + 4 - 2


# --------------------------------------------------------------------------
# What the alignment removes, and what it keeps
# --------------------------------------------------------------------------

def test_one_rotation_per_scene_is_removed_exactly():
    sizes = [3, 5, 2]
    scene = _scene_of(sizes)
    target = _rotations(10, 1)
    shared = _rotations(3, 2)                       # a different one per scene
    predicted = shared[scene] @ target
    anchor = anchor_fragments(torch.randn(10, dtype=DTYPE), scene, 3)
    aligned, scored = align_to_anchor(predicted, target, scene, anchor)

    assert float(_degrees(predicted, target).mean()) > 60.0
    assert float(_degrees(aligned, target).max()) < 1e-9
    assert int(scored.sum()) == 10 - 3


def test_the_aligned_error_is_the_error_relative_to_the_anchor():
    """The geodesic is bi-invariant, so the aligned error IS the error of the
    predicted rotation relative to the anchor's -- the benchmark's question."""
    scene = _scene_of([4, 3])
    predicted, target = _rotations(7, 3), _rotations(7, 4)
    anchor = anchor_fragments(torch.randn(7, dtype=DTYPE), scene, 2)
    aligned, _ = align_to_anchor(predicted, target, scene, anchor)
    a = anchor[scene]
    relative = _degrees(predicted[a].transpose(-1, -2) @ predicted,
                        target[a].transpose(-1, -2) @ target)
    assert torch.allclose(_degrees(aligned, target), relative, atol=1e-9)


def test_chance_is_unchanged_on_the_fragments_that_are_scored():
    """
    A non-anchor fragment's aligned error is the angle between two independent
    uniform rotations, as the absolute error is -- so 126.5 deg is still what a
    model that has learned nothing reads. The anchor itself reads 0, which is
    why it is left out of every average.
    """
    scenes = 3000
    scene = _scene_of([3] * scenes)
    predicted, target = _rotations(3 * scenes, 5), _rotations(3 * scenes, 6)
    anchor = anchor_fragments(torch.randn(3 * scenes, dtype=DTYPE), scene, scenes)
    aligned, scored = align_to_anchor(predicted, target, scene, anchor)
    angle = _degrees(aligned, target)
    # 6000 samples of a quantity with a 37 deg spread: 3 sigma is ~1.5 deg.
    assert abs(float(angle[scored].mean()) - CHANCE["geodesic_deg"]) < 1.5
    assert float(angle[~scored].max()) < 1e-9


# --------------------------------------------------------------------------
# The scores of the matched rotations
# --------------------------------------------------------------------------

def test_the_scores_vanish_for_exact_matches_in_any_frame(fractured_solid):
    """
    Right answer, wrong global frame: the matching holds each scene's root at
    the identity, not at its true pose, and every score of the rotations it
    fits from exact correspondences is zero once the anchor protocol has
    aligned them -- and large before. An alignment applied on the wrong side
    of the rotations fails it.
    """
    batch = _solid_batch(fractured_solid)
    config = Config()
    criterion = build_criterion(config)
    _, report, R = _forward(_Oracle(_world(batch)), batch, criterion, config, score=True)
    for term in ("rotation", "position", "normal", "face"):
        assert report[term] < 1e-6, (term, report[term])
    _, unaligned = criterion(R, batch.target_rotation)
    assert unaligned["rotation"] > 0.1


def test_the_step_is_weighted_by_the_fragments_the_scores_count():
    batch = collate([_sample(0, 3), _sample(1, 4)], dtype=DTYPE)
    assert _loss_fragments(batch, Config()) == 5


def test_validation_metrics_and_the_rotation_score_are_one_number():
    """The metrics are the anchor protocol's, and so is the rotation score:
    same matched rotations, same fragments, same number. Random embeddings,
    so that number is not zero."""
    from reassembly.training import run_epoch

    batch = collate([_sample(3, 3), _sample(4, 4)], dtype=torch.float32)
    embedding = torch.randn(batch.vertex_fragment.numel(), 4,
                            generator=torch.Generator().manual_seed(9))
    config = Config(workers=0)
    summary, _, _ = run_epoch(_Oracle(embedding), [(batch, [])], build_criterion(config),
                              config, label="val", show_progress=False)
    assert summary["geodesic_deg"] > 10.0
    assert summary["geodesic_fragments"] == 7 - 2
    assert summary["fragments"] == 7
    assert summary["rotation_degrees"] == pytest.approx(summary["geodesic_deg"], abs=1e-3)
