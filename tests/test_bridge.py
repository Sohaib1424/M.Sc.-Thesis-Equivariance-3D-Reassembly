"""Input/target/prediction bridge and the training-loop conventions."""
from __future__ import annotations

import torch

from conftest import make_scene, random_rotation
from vngat.training.bridge import (
    apply_rotation_per_entity, build_predictions, build_targets,
    ground_truth_rotation, prepare_scene,
)


def test_ground_truth_rotation_is_the_transpose():
    A = random_rotation(5)
    R_gt = ground_truth_rotation(A)
    assert torch.allclose(R_gt, A.transpose(-1, -2))
    eye = torch.eye(3).expand_as(A)
    assert torch.allclose(torch.bmm(R_gt, A), eye, atol=1e-5)


def test_perfect_prediction_reconstructs_the_clean_geometry():
    """
    The end-to-end consistency check: diffuse by A, predict A^T, and every
    geometric loss term must be zero. If any convention in the chain is
    transposed, this fails.
    """
    scene = make_scene(seed=5)
    A = random_rotation(scene.num_fragments)
    diffused = scene.rotate_per_fragment(A)

    R_pred = ground_truth_rotation(A)
    preds = build_predictions(diffused, R_pred)
    targets = build_targets(scene, A, scene)

    assert torch.allclose(preds["x_pred"], targets["x_gt"], atol=1e-4)
    assert torch.allclose(preds["n_pred"], targets["n_gt"], atol=1e-4)
    assert torch.allclose(preds["mid_pred"], targets["mid_gt"], atol=1e-4)
    assert torch.allclose(preds["n1_pred"], targets["n1_gt"], atol=1e-4)
    assert torch.allclose(preds["n2_pred"], targets["n2_gt"], atol=1e-4)


def test_apply_rotation_per_entity_matches_manual_loop():
    v = torch.randn(20, 3)
    frag = torch.randint(0, 4, (20,))
    R = random_rotation(4)
    out = apply_rotation_per_entity(v, R, frag)
    for i in range(20):
        assert torch.allclose(out[i], R[frag[i]] @ v[i], atol=1e-5)


def test_prepare_scene_derives_the_diffused_view_on_device():
    scene = make_scene(seed=6)
    A = random_rotation(scene.num_fragments)
    batch = {"target": scene, "input": None, "rot": A,
             "trans": torch.zeros(scene.num_fragments, 3)}
    out = prepare_scene(batch, torch.device("cpu"), non_blocking=False)

    assert out["diffused_input"] is out["diffused_target"]      # no duplicate work
    manual = scene.rotate_per_fragment(A)
    assert torch.allclose(out["diffused_target"].node_vec, manual.node_vec, atol=1e-5)
    # edge lengths are invariant, so the diffused view must reuse them verbatim
    assert torch.equal(out["diffused_target"].edge_len, scene.edge_len)


def test_targets_take_geometry_from_full_and_clusters_from_input():
    full = make_scene(((10, 20), (8, 16)), seed=7)
    partial = make_scene(((6, 11), (5, 9)), seed=8)
    A = random_rotation(full.num_fragments)
    targets = build_targets(full, A, partial)
    assert torch.equal(targets["x_gt"], full.node_vec[:, 0])
    assert torch.equal(targets["vertex_cluster_id"], partial.vertex_cluster_id)
    assert torch.equal(targets["edge_cluster_id"], partial.edge_cluster_id)
