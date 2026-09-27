"""
The command-line tools, end to end on a tiny dataset: each one runs, and the
parts with arithmetic in them -- the reassembly poses above all -- are right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_training_recovery import _config, _dataset  # noqa: E402

import reassembly.training as training  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A dataset on disk and a checkpoint trained on it for two steps."""
    base = tmp_path_factory.mktemp("scripts")
    root = _dataset(base / "data")
    config = _config(root, epochs=1, steps_per_epoch=2, schedule=("intra", "cross"))
    training.train(config)
    return root, config, Path(config.out_dir) / "last.pt"


def _flags(root, config):
    return ["--root_dir", str(root), "--hidden_channels", str(config.channels), "--heads",
            str(config.heads), "--head_dim", str(config.head_dim), "--embed_dim",
            str(config.embedding_dim), "--tokens_per_scene", str(config.tokens_per_scene),
            "--num_workers", "0", "--val_frac", str(config.val_frac), "--test_frac", "0.0"]


def _val_key(root, config):
    val = training.BreakingBadScenes(config, "val")
    return val.key(0)


def test_check_version_passes_on_this_copy():
    from scripts.check_version import main

    assert main() == 0


def test_the_dump_round_trips_and_its_poses_are_exact(trained, tmp_path, capsys):
    """
    The last frame of ``truth`` is the object; the first frame of anything is
    the scatter; and a perfect prediction's last frame is the object too.
    """
    from reassembly.viz.reassembly import load_dump, pose_at, posed_vertices
    from scripts.dump_prediction import main

    root, config, checkpoint = trained
    out = tmp_path / "pred.npz"
    assert main(_flags(root, config) + ["--checkpoint", str(checkpoint), "--scene",
                                        _val_key(root, config), "--out", str(out)]) == 0
    dump = load_dump(out)
    assert np.isfinite(dump["rmse_t"]) and 0.0 <= dump["part_accuracy"] <= 1.0

    truth = posed_vertices(dump, 1.0, "truth")
    for fragment, placed in zip(dump["fragments"], truth):
        assert np.allclose(placed, fragment["vertices"], atol=1e-5)
    for i, fragment in enumerate(dump["fragments"]):
        M = pose_at(dump, i, 0.0, "reassemble")
        c = fragment["centroid"]
        expected = (fragment["vertices"] - c) @ dump["A"][i].T + c + dump["shift"][i]
        assert np.allclose(fragment["vertices"] @ M[:3, :3].T + M[:3, 3], expected, atol=1e-5)

    perfect = dict(dump, R_pred=dump["R_gt"],
                   placement=np.stack([f["centroid"] for f in dump["fragments"]]))
    for fragment, placed in zip(dump["fragments"], posed_vertices(perfect, 1.0, "reassemble")):
        assert np.allclose(placed, fragment["vertices"], atol=1e-5)


def test_render_gif_writes_a_still_and_an_animation(trained, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from scripts.dump_prediction import main as dump_main
    from scripts.render_gif import main

    root, config, checkpoint = trained
    dump = tmp_path / "pred.npz"
    dump_main(_flags(root, config) + ["--checkpoint", str(checkpoint), "--scene",
                                      _val_key(root, config), "--out", str(dump)])
    assert main(["--dump", str(dump), "--still", "1.0", "--out", str(tmp_path / "f.png"),
                 "--width", "120", "--height", "120"]) == 0
    assert (tmp_path / "f.png").stat().st_size > 0
    assert main(["--dump", str(dump), "--dump", str(dump), "--label", "a", "--label", "b",
                 "--mode", "compare", "--steps", "3", "--hold", "1", "--width", "100",
                 "--height", "100", "--out", str(tmp_path / "a.gif")]) == 0
    from PIL import Image

    with Image.open(tmp_path / "a.gif") as gif:
        # Pillow merges identical consecutive frames (the held ends) into one
        # longer frame, so count time rather than frames: 8 frames at 20 fps.
        total = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            total += gif.info["duration"]
        assert total == 8 * 50


def test_the_ping_pong_holds_both_ends():
    from reassembly.viz.reassembly import frame_parameter

    s = [frame_parameter(i, steps=5, hold=2) for i in range(14)]
    assert s[:2] == [0.0, 0.0] and s[6:9] == [1.0, 1.0, 1.0]
    assert s[2:7] == sorted(s[2:7]) and s[9:] == sorted(s[9:], reverse=True)


def test_check_scene_is_clean_on_a_healthy_scene(trained, capsys):
    from scripts.check_scene import main

    root, config, checkpoint = trained
    assert main(_flags(root, config) + ["--scene", _val_key(root, config),
                                        "--checkpoint", str(checkpoint),
                                        "--trials", "2"]) == 0
    assert "0 of 2 trials failed" in capsys.readouterr().out


def test_check_scene_finds_the_module_that_goes_non_finite(trained, tmp_path, capsys):
    """Poison one layer's weights: the report must name that stage and module."""
    from scripts.check_scene import main

    root, config, checkpoint = trained
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["model"]["pool_proj.weight"][0, 0] = float("inf")
    poisoned = tmp_path / "poisoned.pt"
    torch.save(state, poisoned)
    flags = _flags(root, config) + ["--scene", _val_key(root, config),
                                    "--checkpoint", str(poisoned), "--trials", "1"]
    assert main(flags) == 1
    assert "MODEL in float32" in capsys.readouterr().out
    assert main(flags + ["--locate"]) == 1
    assert "first module: pool_proj" in capsys.readouterr().out


def test_benchmark_data_runs_every_measurement(trained, capsys):
    from scripts.benchmark_data import main

    root, config, _ = trained
    assert main(_flags(root, config) + ["--scenes", "4", "--throughput", "2",
                                        "--batch_size", "2", "--micro_batch_scenes", "2",
                                        "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    for heading in ("per stage", "where the batch is finished", "loader throughput"):
        assert heading in out
    assert "copy + pairs on the device" in out


def test_the_sweep_leaves_truncated_runs_out_of_the_verdict(capsys):
    from scripts.scaling_sweep import _slope, summarise

    assert _slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    results = [
        {"objects": 2, "seed": 0, "steps_per_object": 400, "requested_steps_per_object": 400,
         "truncated": False, "first_train_deg": 120, "best_train_deg": 40,
         "final_train_deg": 45, "tail_slope_deg_per_epoch": 0.0, "epochs": 10},
        {"objects": 8, "seed": 0, "steps_per_object": 100, "requested_steps_per_object": 400,
         "truncated": True, "first_train_deg": 120, "best_train_deg": 90,
         "final_train_deg": 95, "tail_slope_deg_per_epoch": -2.0, "epochs": 3},
    ]
    summarise(results, seeds=[0])
    out = capsys.readouterr().out
    assert "TRUNCATED" in out and "INVALID: 1 run(s)" in out
    table = out.split("best of seeds")[1]
    assert "       8" not in table, "a truncated count must not reach the verdict"


def test_max_objects_trains_on_that_many_shapes(trained):
    root, config, _ = trained
    import dataclasses

    few = training.BreakingBadScenes(dataclasses.replace(config, max_objects=2), "train")
    every = training.BreakingBadScenes(config, "train")
    assert len(few.scenes) == 2 < len(every.scenes)
    assert {e.key for e in few.scenes} <= {e.key for e in every.scenes}
    assert len(training.BreakingBadScenes(dataclasses.replace(config, max_objects=2),
                                          "val").scenes) == len(
        training.BreakingBadScenes(config, "val").scenes), "validation is untouched"


def test_offenders_and_history_are_written(trained):
    root, config, _ = trained
    history = json.loads((Path(config.out_dir) / "history.json").read_text())
    row = history[-1]
    for key in ("train_steps", "train_grad_norm", "gpus", "val_attempted"):
        assert key in row, key
