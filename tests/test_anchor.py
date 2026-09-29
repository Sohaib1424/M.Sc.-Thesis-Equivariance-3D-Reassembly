"""
The anchor protocol (`vngat.evaluation.anchor`): each scene's largest fragment
is set to its true pose, and every other fragment is scored relative to it.

Every property here fails silently when broken -- a correction on the wrong
side, an anchor chosen per batch instead of per scene, an anchor left in the
average, a weighting that still counts it -- and each of those still produces
a plausible number.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_scene, random_rotation  # noqa: E402

from vngat.data.graph import collate_scenes  # noqa: E402
from vngat.evaluation.anchor import (  # noqa: E402
    align_to_anchor, anchor_alignment, anchor_fragments, scored_fragments,
)
from vngat.evaluation.metrics import CHANCE_GEODESIC_DEG, evaluate_scene, geodesic_angle  # noqa: E402



def _scene_of(sizes):
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.as_tensor(sizes))


def _degrees(a, b):
    """This package's `geodesic_angle` is already in degrees. Its atan2 carries
    a sqrt(1e-12) floor (`geodesic_rotation_loss`), so an exact prediction reads
    ~3e-5 deg, not 0 -- hence the 1e-4 deg tolerances below."""
    return geodesic_angle(a, b)


def _micro(scenes, seed):
    """One batch dict as the loader yields it, with a known scatter rotation."""
    target = collate_scenes(scenes) if len(scenes) > 1 else scenes[0]
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(target.num_fragments, 3, 3, generator=g, dtype=torch.float64))
    q[:, :, 0] *= torch.linalg.det(q).sign().unsqueeze(-1)
    return {"target": target, "input": None, "rot": q.float(),
            "scene_dirs": [f"s{seed}_{i}" for i in range(len(scenes))],
            "categories": ["Bottle", "Cup", "Bottle", "Cup"][:len(scenes)],
            "num_scenes": len(scenes)}


class _Oracle(torch.nn.Module):
    """Answers with the truth turned by one rotation per scene."""

    def __init__(self, R):
        super().__init__()
        self.R = R
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.grad_checkpointing = False

    def forward(self, node_vec, edge_index, **_):
        return {"R_pred": self.R + 0.0 * self.p,
                "vertex_embedding": torch.randn(node_vec.shape[0], 4) + 0.0 * self.p,
                "edge_embedding": torch.randn(edge_index.shape[1], 4) + 0.0 * self.p,
                "head_cos": torch.tensor(0.5)}


# --------------------------------------------------------------------------
# Which fragment
# --------------------------------------------------------------------------
def test_the_anchor_is_each_scenes_largest_fragment():
    log_scale = torch.tensor([0.1, 0.5, 0.2,          # scene 0: fragment 1
                              0.3,                    # scene 1: its only fragment
                              0.9, 0.9, 0.2, 0.1])    # scene 2: a tie, the first
    anchor = anchor_fragments(log_scale[:, None], _scene_of([3, 1, 4]), 4)
    assert anchor.tolist() == [1, 3, 4, 8]            # scene 3 empty: one past the end
    R = random_rotation(8)
    _, scored = align_to_anchor(R, R, _scene_of([3, 1, 4]), anchor)
    assert (~scored).nonzero().flatten().tolist() == [1, 3, 4]


def test_the_anchor_is_chosen_per_scene_not_per_batch(batch):
    """Fragment ids are global across a batch; the largest overall must not
    become every scene's anchor."""
    anchor = anchor_fragments(batch.frag_log_scale, batch.frag_scene, batch.num_scenes)
    scale = batch.frag_log_scale.flatten()
    for s in range(batch.num_scenes):
        members = (batch.frag_scene == s).nonzero().flatten()
        assert int(anchor[s]) == int(members[scale[members].argmax()])
    assert scored_fragments(batch) == batch.num_fragments - batch.num_scenes


# --------------------------------------------------------------------------
# What the alignment removes, and what it keeps
# --------------------------------------------------------------------------
def test_one_rotation_per_scene_is_removed_exactly():
    scene = _scene_of([3, 5, 2])
    R_gt = random_rotation(10, torch.float64)
    shared = random_rotation(3, torch.float64)        # a different one per scene
    R_pred = shared[scene] @ R_gt
    anchor = anchor_fragments(torch.randn(10, dtype=torch.float64), scene, 3)
    aligned, scored = align_to_anchor(R_pred, R_gt, scene, anchor)
    assert float(_degrees(R_pred, R_gt).mean()) > 60.0
    assert float(_degrees(aligned, R_gt).max()) < 1e-4
    assert int(scored.sum()) == 10 - 3


def test_the_aligned_error_is_the_error_relative_to_the_anchor():
    scene = _scene_of([4, 3])
    R_pred, R_gt = random_rotation(7, torch.float64), random_rotation(7, torch.float64)
    anchor = anchor_fragments(torch.randn(7, dtype=torch.float64), scene, 2)
    aligned, _ = align_to_anchor(R_pred, R_gt, scene, anchor)
    a = anchor[scene]
    relative = _degrees(R_pred[a].transpose(-1, -2) @ R_pred, R_gt[a].transpose(-1, -2) @ R_gt)
    assert torch.allclose(_degrees(aligned, R_gt), relative, atol=1e-6)


def test_chance_is_unchanged_on_the_fragments_that_are_scored():
    """A non-anchor fragment's aligned error is the angle between two
    independent uniform rotations, so 126.5 deg is still chance; the anchor
    reads 0, which is why it is left out of every average."""
    import numpy as np

    from vngat.data.io import random_rotation_matrices

    # The data pipeline's sampler, which is uniform. `conftest.random_rotation`
    # is not -- QR without the sign fix on R's diagonal; two of its draws are
    # 100.8 deg apart on average, not 126.5 -- which is harmless where a test
    # needs *a* rotation and wrong here, where it needs chance.
    scenes = 3000
    scene = _scene_of([3] * scenes)
    R_pred = torch.as_tensor(random_rotation_matrices(3 * scenes, np.random.default_rng(0)))
    R_gt = torch.as_tensor(random_rotation_matrices(3 * scenes, np.random.default_rng(1)))
    anchor = anchor_fragments(torch.randn(3 * scenes, dtype=torch.float64), scene, scenes)
    aligned, scored = align_to_anchor(R_pred, R_gt, scene, anchor)
    angle = _degrees(aligned, R_gt)
    assert abs(float(angle[scored].mean()) - CHANCE_GEODESIC_DEG) < 1.5   # 3 sigma
    assert float(angle[~scored].max()) < 1e-4


def test_the_anchor_is_trained_through_the_fragments_scored_against_it(scene):
    """Left out of the average, the anchor is still the reference its scene
    is scored against -- so it must receive gradient from the others."""
    R_gt = random_rotation(scene.num_fragments)
    R_pred = random_rotation(scene.num_fragments).requires_grad_(True)
    aligned, scored = anchor_alignment(R_pred, R_gt, scene)
    geodesic_angle(aligned, R_gt)[scored].mean().backward()
    anchor = int((~scored).nonzero()[0])
    assert float(R_pred.grad[anchor].norm()) > 1e-3


# --------------------------------------------------------------------------
# The loss and the step under the anchor target
# --------------------------------------------------------------------------
def _terms(target, shared, micro):
    from vngat.losses.composite import CompositeLoss
    from vngat.training import trainer as T

    scene = T.prepare_scene(micro, torch.device("cpu"))
    R_gt = scene["rot"].transpose(-1, -2)
    model = _Oracle(shared[scene["clean_target"].frag_scene] @ R_gt)
    return T._forward_loss(model, CompositeLoss(rotation_target=target), scene,
                           torch.device("cpu"), False)


def test_the_anchor_loss_vanishes_under_any_rotation_of_each_scene():
    """Right answer, wrong global frame: every rotation-dependent term is zero
    under the anchor target and large under the absolute one."""
    micro = _micro([make_scene(seed=4), make_scene(((9, 17), (6, 11)), seed=5)], seed=6)
    shared = random_rotation(2)
    anchored, absolute = _terms("anchor", shared, micro), _terms("absolute", shared, micro)
    for key in ("rot", "pos", "node", "face"):
        assert float(anchored[key]) < 1e-4, (key, float(anchored[key]))
        assert float(absolute[key]) > 0.1, (key, float(absolute[key]))
    # The protocol metrics do not depend on the target.
    for out in (anchored, absolute):
        assert float(out["_anchor_deg"].max()) < 0.05
        assert float(out["_absolute_deg"].mean()) > 10.0
        assert out["_anchor_deg"].numel() == 5 - 2


def test_the_anchor_is_left_out_of_the_geometric_terms():
    """
    Its aligned prediction is exact, so counting the anchor would add a free
    zero to every per-fragment mean -- a third of the median scene's score.
    """
    from vngat.losses.composite import CompositeLoss
    from vngat.training import trainer as T
    from vngat.training.bridge import build_predictions, build_targets

    cpu = torch.device("cpu")
    scene = T.prepare_scene(_micro([make_scene(seed=12)], seed=13), cpu)
    R = random_rotation(3)
    out = T._forward_loss(_Oracle(R), CompositeLoss(rotation_target="anchor"), scene, cpu, False)

    targets = build_targets(scene["clean_target"], scene["rot"], scene["diffused_input"])
    aligned, keep = anchor_alignment(R, targets["R_gt"], scene["clean_target"])
    x_pred = build_predictions(scene["diffused_target"], aligned)["x_pred"]
    frag = targets["node_frag"]
    squared = ((x_pred - targets["x_gt"]) ** 2).sum(-1)
    per_fragment = torch.zeros(3).index_add_(0, frag, squared) / torch.bincount(frag, minlength=3)
    assert float(per_fragment[~keep].max()) < 1e-8                 # the anchor: exact
    assert float(out["pos"]) == pytest.approx(float(per_fragment[keep].mean()), rel=1e-4)
    assert float(out["pos"]) > 1.2 * float(per_fragment.mean())


def test_the_step_is_weighted_by_the_fragments_the_loss_scores(batch):
    from vngat.config import Config
    from vngat.training import trainer as T

    assert T.loss_fragments(batch, "anchor") == batch.num_fragments - batch.num_scenes
    assert T.loss_fragments(batch, "absolute") == batch.num_fragments
    with pytest.raises(ValueError, match="rotation_target"):
        Config(rotation_target="relative").validate()


class _Bar:
    def update(self, *_):
        pass

    def set_postfix_str(self, *_):
        pass


@pytest.mark.parametrize("target", ["anchor", "absolute"])
def test_validation_reports_both_protocols_exactly(target):
    """
    The epoch's anchor_deg and absolute_deg are exact means over their
    fragments, the category breakdown covers exactly the scored ones, and the
    loss-side rotation agrees with the protocol figure it is meant to be --
    same predictions, same fragments, same number.
    """
    from vngat.config import Config
    from vngat.losses.composite import CompositeLoss
    from vngat.models.vn_gat import VNGATModel
    from vngat.training import trainer as T

    torch.manual_seed(0)
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4)
    cfg = Config(device="cpu", amp=False, micro_batch_scenes=1, rotation_target=target)
    loss_fn = CompositeLoss(rotation_target=target)
    loader = [_micro([make_scene(seed=7), make_scene(((5, 9), (13, 27)), seed=8)], seed=9),
              _micro([make_scene(((9, 17), (6, 11), (8, 14), (12, 22)), seed=10)], seed=11)]
    metrics, diag = T.run_phase(model, loader, loss_fn, None, T.make_scaler(torch.device("cpu"), False),
                                torch.device("cpu"), cfg, train=False, epoch=0, bar=_Bar(),
                                categories=("Bottle", "Cup"))

    anchor_all, absolute_all = [], []
    with torch.no_grad():
        for micro in loader:
            for piece in T._make_micro_batches(micro, 1):
                scene = T.prepare_scene(piece, torch.device("cpu"))
                out = T._forward_loss(model, loss_fn, scene, torch.device("cpu"), False)
                anchor_all.append(out["_anchor_deg"])
                absolute_all.append(out["_absolute_deg"])
    anchor_all, absolute_all = torch.cat(anchor_all), torch.cat(absolute_all)
    assert anchor_all.numel() == (3 + 2 + 4) - 3 and absolute_all.numel() == 9
    assert metrics["anchor_deg"] == pytest.approx(float(anchor_all.mean()), abs=1e-4)
    assert metrics["absolute_deg"] == pytest.approx(float(absolute_all.mean()), abs=1e-4)
    assert diag["category_counts"]["Bottle"] + diag["category_counts"]["Cup"] == 6
    loss_frame = "anchor_deg" if target == "anchor" else "absolute_deg"
    assert metrics["rot_deg"] == pytest.approx(metrics[loss_frame], abs=1e-3)


# --------------------------------------------------------------------------
# Old checkpoints, resume and best.pt
# --------------------------------------------------------------------------
def test_a_checkpoint_from_before_the_setting_resumes_on_the_absolute_target(tmp_path, capsys):
    from test_checkpoint import _write_stub_checkpoint

    from vngat.config import Config
    from vngat.training.trainer import adopt_checkpoint_config

    ckpt = tmp_path / "last.pt"
    _write_stub_checkpoint(ckpt, dict(checkpoint_monitor="rot"))   # as v5 wrote it
    cfg = Config()
    object.__setattr__(cfg, "_explicit", frozenset())
    adopt_checkpoint_config(cfg, ckpt, is_main=True)
    assert cfg.rotation_target == "absolute" and cfg.checkpoint_monitor == "rot"
    printed = capsys.readouterr()
    assert "predates --rotation_target" in printed.out + printed.err

    switched = Config(rotation_target="anchor")
    object.__setattr__(switched, "_explicit", frozenset({"rotation_target"}))
    adopt_checkpoint_config(switched, ckpt, is_main=True)
    assert switched.rotation_target == "anchor"
    printed = capsys.readouterr()
    assert "switches it to 'anchor'" in printed.out + printed.err


def test_best_is_reset_when_the_monitored_value_changes_meaning():
    from vngat.training.trainer import _monitor_changed_meaning

    # Old run, same target, loss-side monitor: continues exactly.
    assert not _monitor_changed_meaning("rot", "absolute", "absolute", False)
    # The target changed: the loss terms are a different quantity now.
    assert _monitor_changed_meaning("rot", "absolute", "anchor", False)
    # Protocol figures mean one thing whatever the target, but did not exist
    # (or were measured without the anchor) before the protocol.
    assert not _monitor_changed_meaning("anchor_deg", "anchor", "absolute", True)
    assert _monitor_changed_meaning("tilt", "absolute", "absolute", False)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def test_evaluation_leaves_the_anchor_out_and_reports_the_absolute_error():
    R_gt = random_rotation(4, torch.float64)
    R_pred = R_gt.clone()
    R_pred[2] = random_rotation(1, torch.float64)[0]
    keep = torch.tensor([True, False, True, True])
    turned = random_rotation(1, torch.float64)[0] @ R_gt           # a global turn
    metrics = evaluate_scene(R_pred, R_gt, keep=keep, R_absolute=turned)
    expected = float(_degrees(R_pred[keep], R_gt[keep]).mean())
    assert metrics["geodesic_deg"] == pytest.approx(expected)
    assert metrics["scored_fragments"] == 3.0 and metrics["num_fragments"] == 4.0
    assert metrics["absolute_geodesic_deg"] > 1.0
    alone = evaluate_scene(R_gt[:1], R_gt[:1], keep=torch.tensor([False]))
    assert math.isnan(alone["geodesic_deg"])                       # nothing to score


def test_the_evaluate_script_scores_an_old_checkpoint_with_the_anchor(tmp_path):
    """
    End to end on a miniature dataset, with a checkpoint written the way v5
    wrote them before the anchor existed (no `rotation_target`): the script
    loads it, aligns every scene to its largest fragment, measures translations
    from the anchor's, and reports both errors.
    """
    from conftest import write_dataset

    from scripts import evaluate as E
    from vngat.config import Config
    from vngat.training.trainer import build_model

    root = tmp_path / "data"
    base = "everyday_compressed/everyday_compressed"
    write_dataset(root, {f"{base}/Bottle/b0": 2, f"{base}/Bottle/b1": 2, f"{base}/Cup/c0": 2},
                  official={"train": ["everyday/Bottle/b0", "everyday/Cup/c0"],
                            "val": ["everyday/Bottle/b1"]})
    cfg = Config(root_dir=str(root), hidden_channels=8, num_layers=2, num_vn_slots=3,
                 heads=2, head_dim=4, embed_dim=4, device="cpu", split_source="official",
                 data_subsets="everyday_compressed", min_fragments=2)
    model = build_model(cfg, torch.device("cpu"))
    stored = cfg.to_dict()
    del stored["rotation_target"]                                  # an old checkpoint
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": stored, "epoch": 3}, ckpt)

    args = E.build_parser().parse_args(["--checkpoint", str(ckpt), "--device", "cpu",
                                        "--num_workers", "0", "--split", "val"])
    summary = E.evaluate(args)
    for key in ("geodesic_deg", "absolute_geodesic_deg", "rmse_T", "part_accuracy"):
        assert key in summary and math.isfinite(summary[key]), key
    assert summary["scored_fragments"] < summary["num_fragments"]
