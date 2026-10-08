"""
Evaluation without Breaking Bad's shared break vertices
(:mod:`reassembly.evaluation.noise`; ``--evaluate --jitter/--drop``).

Off by default, and then nothing changes. On, the method sees the noise and
the score does not, and every draw belongs to its scene.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.assembly import score_batch  # noqa: E402
from reassembly.data.features import complete_batch  # noqa: E402
from reassembly.evaluation.noise import (  # noqa: E402
    check_noise,
    drop_candidates,
    jitter_inputs,
    noise_suffix,
)

from test_assembly import _broken_scene, _oracle  # noqa: E402
from test_assembly_placement import _stack  # noqa: E402

DTYPE = torch.float64


def _sample(meshes, key: str, normalize_mode: str = "scene"):
    from reassembly.data.features import build_scene
    from reassembly.data.transforms import random_rotations
    from reassembly.mesh.fracture import fracture_vertex_masks

    return build_scene([np.asarray(m.vertices) for m in meshes],
                       [np.asarray(m.faces) for m in meshes], fracture_vertex_masks(meshes),
                       rotations=random_rotations(len(meshes), np.random.default_rng(3)),
                       tokens_per_scene=64, key=key, normalize_mode=normalize_mode)


def _batch(*samples):
    from reassembly.data.features import collate

    return complete_batch(collate(list(samples), dtype=DTYPE))


def _noise(batch, sigma: float, seed: int = 0) -> torch.Tensor:
    """What jitter_inputs added, in world units."""
    moved = jitter_inputs(batch, sigma, seed).node_features[:, 0, :] - batch.node_features[:, 0, :]
    return moved * batch.unit[batch.vertex_fragment, None]


# ------------------------------------------------------------ off is off --

def test_no_noise_is_no_change(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    batch = complete_batch(batch)
    assert jitter_inputs(batch, 0.0) is batch
    mask = batch.fracture.clone()
    assert drop_candidates(mask, 0.0, torch.Generator()) is mask
    assert noise_suffix(0.0, 0.0) == ""
    oracle = _oracle(batch, batch.target_rotation)
    for rotations in ("network", "matched"):
        plain = score_batch(batch, oracle, rotations=rotations)[0]
        assert score_batch(batch, oracle, rotations=rotations, drop=0.0)[0] == plain
        seen = score_batch(batch, oracle, rotations=rotations, observed=batch)[0]
        for key in ("rmse_t", "chamfer", "part_chamfer", "geodesic_deg", "matches"):
            assert seen[key] == pytest.approx(plain[key], abs=1e-12), key


# ---------------------------------------------------------------- jitter --

def test_jitter_moves_the_input_vertices_and_nothing_else():
    batch = _batch(_sample(_stack(), "stack"))
    noisy = jitter_inputs(batch, 0.01, seed=0)
    world = _noise(batch, 0.01)
    assert float(world.std()) == pytest.approx(0.01 * float(batch.unit.max()), rel=0.1)
    assert torch.equal(noisy.node_features[:, 1, :], batch.node_features[:, 1, :])
    assert torch.equal(noisy.edge_attr[:, :2], batch.edge_attr[:, :2])
    source, target = batch.edge_index
    assert torch.allclose(noisy.edge_attr[:, 2, :],
                          noisy.node_features[source, 0, :] - noisy.node_features[target, 0, :])
    for name in ("target_vertices", "target_rotation", "target_normals", "centroid", "unit",
                 "fracture", "token_index"):
        assert torch.equal(getattr(noisy, name), getattr(batch, name)), name


def test_jitter_is_in_largest_fragment_radii_under_either_normalisation():
    """Per-fragment normalisation divides each piece by its own radius; the
    noise must still be one size in the world, the largest piece's."""
    batch = _batch(_sample(_stack(thickness=(0.3, 2.5)), "two", normalize_mode="fragment"))
    assert float(batch.unit.min()) < 0.8 * float(batch.unit.max())
    world = _noise(batch, 0.01)
    for f in range(2):
        own = batch.vertex_fragment == f
        assert float(world[own].std()) == pytest.approx(0.01 * float(batch.unit.max()),
                                                        rel=0.1), f


def test_the_noise_belongs_to_the_scene_not_the_batch(fractured_solid):
    alone = _batch(_sample(list(fractured_solid), "a"))
    together = _batch(_sample(_stack(), "b"), _sample(list(fractured_solid), "a"))
    start = int(together.vertex_ptr[together.fragment_ptr[1]])
    assert torch.equal(_noise(together, 0.01, seed=4)[start:], _noise(alone, 0.01, seed=4))
    assert not torch.allclose(_noise(alone, 0.01, seed=5), _noise(alone, 0.01, seed=4))


def test_the_method_sees_the_noise_and_the_score_does_not(fractured_solid):
    """The solve reads the noisy vertices; the Chamfer distance reads the
    predicted pose on the clean fragments, so it carries no noise floor."""
    batch, _ = _broken_scene(fractured_solid)
    batch = complete_batch(batch)
    noisy = jitter_inputs(batch, 0.02, seed=1)
    oracle = _oracle(batch, batch.target_rotation)
    scored = score_batch(batch, oracle, observed=noisy)[0]
    on_noise = score_batch(noisy, oracle)[0]
    clean = score_batch(batch, oracle)[0]
    assert scored["_translation"] == on_noise["_translation"]
    assert scored["_translation"] != clean["_translation"]
    assert scored["rmse_t"] == on_noise["rmse_t"]
    assert scored["chamfer"] < 0.5 * on_noise["chamfer"]
    with pytest.raises(ValueError, match="same scenes"):
        score_batch(batch, oracle, observed=_batch(_sample(_stack(), "other")))


# ------------------------------------------------------------------ drop --

def test_drop_leaves_break_vertices_out_of_the_matching(fractured_solid):
    batch, _ = _broken_scene(fractured_solid)
    oracle = _oracle(batch, batch.target_rotation)
    full = score_batch(batch, oracle, rotations="matched")[0]
    half = score_batch(batch, oracle, rotations="matched", drop=0.5)[0]
    assert half["matches"] < 0.5 * full["matches"]
    assert half == score_batch(batch, oracle, rotations="matched", drop=0.5)[0]
    assert half["part_accuracy"] == 1.0


def test_settings_that_cannot_be_used_are_refused(fractured_solid):
    check_noise(0.01, 0.5)
    for jitter, drop in ((-0.01, 0.0), (float("nan"), 0.0), (0.0, 1.0), (0.0, -0.1)):
        with pytest.raises(ValueError):
            check_noise(jitter, drop)
    assert noise_suffix(0.01, 0.5) == "_jitter0.01_drop0.5"
    assert noise_suffix(0.0, 0.25) == "_drop0.25"
    batch, _ = _broken_scene(fractured_solid)
    with pytest.raises(ValueError, match="--drop"):
        score_batch(batch, _oracle(batch, batch.target_rotation), drop=1.0)


def test_the_flags_belong_to_evaluate(monkeypatch, capsys):
    from scripts.train import main

    for argv in (["--jitter", "0.01"], ["--evaluate", "--drop", "1"],
                 ["--evaluate", "--jitter", "-1"]):
        monkeypatch.setattr(sys, "argv", ["train.py", "--root_dir", "nowhere", *argv])
        with pytest.raises(SystemExit) as stop:
            main()
        assert stop.value.code == 2, argv
    error = capsys.readouterr().err
    assert "--jitter and --drop apply to --evaluate only" in error
    assert "--drop must be at least 0 and below 1" in error
