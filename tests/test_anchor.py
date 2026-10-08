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
    """Answers every scene with one rotation `G[scene]` applied to the truth."""

    def __init__(self, batch, global_rotation):
        super().__init__()
        self.rotation = global_rotation[batch.fragment_scene] @ batch.target_rotation
        self.embedding = torch.nn.Parameter(
            torch.randn(batch.vertex_fragment.numel(), 4,
                        dtype=batch.target_rotation.dtype))

    def forward(self, *args, **kwargs):
        return Prediction(rotation=self.rotation, frame=self.rotation.transpose(-1, -2),
                          vertex_embedding=self.embedding, vertex_features=None)


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
# The loss under the anchor target
# --------------------------------------------------------------------------

def _loss_terms(target: str, global_rotation):
    batch = collate([_sample(0, 3), _sample(1, 4)], dtype=DTYPE)
    config = Config(rotation_target=target)
    model = _Oracle(batch, global_rotation)
    _, report, _ = _forward(model, batch, build_criterion(config), config)
    return report


def test_the_anchor_loss_vanishes_under_any_rotation_of_each_scene():
    """
    Right answer, wrong global frame: every rotation-dependent term is zero
    under the anchor target and large under the absolute one. This is the
    whole point of the target, and a correction applied on the wrong side of
    the prediction fails it.
    """
    shared = _rotations(2, 7)
    anchored = _loss_terms("anchor", shared)
    absolute = _loss_terms("absolute", shared)
    for term in ("rotation", "position", "normal", "face"):
        assert anchored[term] < 1e-6, (term, anchored[term])
        assert absolute[term] > 0.1, (term, absolute[term])
    # And the truth itself scores zero under both.
    identity = torch.eye(3, dtype=DTYPE).expand(2, 3, 3)
    for target in ("anchor", "absolute"):
        report = _loss_terms(target, identity)
        assert report["rotation"] < 1e-6 and report["normal"] < 1e-6


def test_the_anchor_is_trained_through_the_fragments_scored_against_it():
    """
    The anchor is left out of the average, but it is the reference its scene is
    scored against, so its prediction must still receive gradient -- from every
    other fragment. Without it the anchor would never be trained at all.
    """
    batch = collate([_sample(2, 3)], dtype=DTYPE)
    rotation = _rotations(3, 8).requires_grad_(True)
    anchor = anchor_fragments(batch.log_scale, batch.fragment_scene, 1)
    aligned, scored = align_to_anchor(rotation, batch.target_rotation,
                                      batch.fragment_scene, anchor)
    loss = geodesic_angle(aligned, batch.target_rotation)[scored].mean()
    loss.backward()
    assert float(rotation.grad[int(anchor)].norm()) > 1e-3


def test_the_step_is_weighted_by_the_fragments_the_loss_scores():
    batch = collate([_sample(0, 3), _sample(1, 4)], dtype=DTYPE)
    assert _loss_fragments(batch, Config(rotation_target="anchor")) == 5
    assert _loss_fragments(batch, Config(rotation_target="absolute")) == 7
    with pytest.raises(ValueError, match="rotation_target"):
        Config(rotation_target="relative")


def test_validation_scores_both_protocols_from_the_same_predictions():
    """The metrics are the anchor protocol's whatever the target, with the
    absolute error beside them; and the loss agrees with the metric it is
    meant to be -- same predictions, same fragments, same number."""
    from reassembly.training import run_epoch

    samples = [_sample(3, 3), _sample(4, 4)]
    batch = collate(samples, dtype=torch.float32)
    shared = _rotations(2, 9).float()
    for target in ("anchor", "absolute"):
        config = Config(rotation_target=target, workers=0)
        model = _Oracle(batch, shared)
        loader = [(batch, [])]
        summary, _, _ = run_epoch(model, loader, build_criterion(config), config,
                                  label="val", show_progress=False)
        assert summary["geodesic_deg"] < 1e-3              # anchor protocol
        assert summary["absolute_geodesic_deg"] > 10.0
        assert summary["geodesic_fragments"] == 7 - 2
        assert summary["fragments"] == 7
        loss_frame = "geodesic_deg" if target == "anchor" else "absolute_geodesic_deg"
        assert summary["rotation_degrees"] == pytest.approx(summary[loss_frame], abs=1e-3)
