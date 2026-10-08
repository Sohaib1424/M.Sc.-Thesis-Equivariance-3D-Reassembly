"""
Results read back for figures: object types line up across subsets, every
per-piece number lands on its own piece, and the figures draw from them.

No torch: the results layer is numpy only, so it runs wherever the files are.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from reassembly.viz.results import (by_piece_count, distribution, errors_by_type,
                                    load_evaluation, load_history, markdown_table,
                                    object_type, summary_row, type_order, write_csv)


def test_one_name_per_kind_of_object_whatever_the_subset():
    assert object_type("Bottle", "everyday_compressed/Bottle/x/fractured_0") == "Bottle"
    # A volume-constrained copy nests the base subset: its categories carry it.
    assert object_type("everyday_compressed/Bottle") == "Bottle"
    assert object_type("everyday_compressed\\Mug") == "Mug"
    # No categories: the subset's family, for the base subset and for its copy.
    assert object_type("(uncategorised)", "artifact_compressed//obj/fractured_1") == "Artifact"
    assert object_type("artifact_compressed",
                       "artifact_compressed/artifact_compressed/obj/fractured_1") == "Artifact"
    assert object_type("", "volume_constrained-artifact_compressed/obj/m") == "Artifact"
    assert object_type("", "") == "Uncategorised"


def _scene(category, key, angles, anchor, reached=None, network=None, part=None,
           part_accuracy=1.0):
    count = len(angles) + (1 if anchor >= 0 else 0)
    record = {"geodesic_deg": float(np.mean(angles)), "euler_rmse_deg": 1.0, "rmse_t": 0.01,
              "chamfer": 0.001, "part_chamfer": 0.002, "part_accuracy": part_accuracy,
              "matches": 10.0, "fragments": float(count),
              "_part_chamfer": part if part is not None else [0.001] * count,
              "_scored_geodesic_deg": angles, "_anchor": anchor,
              "_scene": key, "_category": category}
    if reached is not None:
        record["_reached"] = reached
        record["matched_share"] = 0.5
    if network is not None:
        record["_network_geodesic_deg"] = network
    return record


def _evaluation_file(tmp_path, name="val_metrics_matched.json", network=True):
    scenes = [
        # anchor in the middle: scored fragments are 0 and 2
        _scene("Mug", "everyday_compressed/Mug/a/fractured_0", [0.5, 30.0], anchor=1,
               reached=[True, True, False], network=[100.0, 120.0] if network else None,
               part=[0.001, 0.0, 0.5], part_accuracy=0.5),
        _scene("everyday_compressed/Bottle", "everyday_compressed/Bottle/b/fractured_1",
               [0.1, 0.2, 0.3], anchor=0, reached=[True, True, True, True],
               network=[90.0, 95.0, 99.0] if network else None),
        _scene("(uncategorised)", "artifact_compressed//c/fractured_0", [2.0], anchor=0,
               reached=[True, True], network=[80.0] if network else None),
    ]
    summary = {"geodesic_deg": 100.0, "geodesic_median_deg": 99.0, "geodesic_fragments": 6,
               "match@1": 0.6, "rotations": "matched",
               "matched": {"geodesic_deg": 5.5, "geodesic_median_deg": 0.4, "reached": 0.83,
                           "fragments": 6, "acc@5deg": 0.83, "acc@10deg": 0.83,
                           "acc@30deg": 0.83},
               "assembly": {"part_accuracy": 0.8, "rmse_t": 0.01, "chamfer": 0.001},
               "evaluation": {"checkpoint": "best.pt", "epoch": 7},
               "assembly_scenes": scenes}
    folder = tmp_path / "W10-everyday"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(summary))
    return path


def test_every_number_lands_on_its_own_piece(tmp_path):
    evaluation = load_evaluation(_evaluation_file(tmp_path))
    assert evaluation.label == "W10-everyday"                  # the folder, by default
    assert evaluation.sources() == ["network", "matched"]
    pieces = evaluation.pieces
    assert pieces["type"].tolist() == ["Mug", "Mug", "Bottle", "Bottle", "Bottle", "Artifact"]
    assert pieces["matched"].tolist() == [0.5, 30.0, 0.1, 0.2, 0.3, 2.0]
    assert pieces["network"].tolist() == [100.0, 120.0, 90.0, 95.0, 99.0, 80.0]
    # The Mug scene's anchor is fragment 1: the scored pieces are 0 and 2, so
    # their reach and Chamfer are fragments 0 and 2's, not 0 and 1's.
    assert pieces["reached"][:2].tolist() == [1.0, 0.0]
    assert pieces["part_chamfer"][:2].tolist() == [0.001, 0.5]
    assert pieces["pieces"].tolist() == [3, 3, 4, 4, 4, 2]
    assert evaluation.scenes["type"].tolist() == ["Mug", "Bottle", "Artifact"]
    assert evaluation.families == {"Artifact"}


def test_a_matched_file_from_before_the_head_errors_were_kept(tmp_path):
    evaluation = load_evaluation(_evaluation_file(tmp_path, network=False), label="old")
    assert evaluation.sources() == ["matched"]
    assert np.isnan(evaluation.errors("network")).all()


def test_tables_order_types_by_error_with_the_families_last(tmp_path):
    evaluation = load_evaluation(_evaluation_file(tmp_path), label="E")
    assert type_order([evaluation], "matched") == ["Bottle", "Mug", "Artifact"]
    rows = errors_by_type([evaluation], "matched")
    assert [r["type"] for r in rows] == ["Bottle", "Mug", "Artifact"]
    assert rows[0]["n"] == 3 and rows[0]["median"] == pytest.approx(0.2)
    stats = distribution(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
    assert stats["n"] == 4 and stats["mean"] == 2.5 and stats["acc@5"] == 1.0
    assert distribution(np.array([np.nan])) == {"n": 0}

    counts = {row["pieces"]: row for row in by_piece_count(evaluation)}
    assert counts["3"]["scenes"] == 1 and counts["3"]["part_accuracy"] == 0.5
    assert counts["4"]["matched_median_deg"] == pytest.approx(0.2)
    assert counts["2"]["reached"] == 1.0

    row = summary_row(evaluation)
    assert row["checkpoint"] == "best.pt (epoch 7)" and row["matched_median_deg"] == 0.4
    table = markdown_table([row], (("dataset", "dataset", ""), ("reached", "placed", ".0%"),
                                   ("rmse_t", "RMSE", ".3f"), ("chamfer", "CD", "")))
    assert "| E | 83% | 0.010 | 0.001 |" in table
    path = write_csv(tmp_path / "t" / "summary.csv", [row, {"dataset": "x", "reached": math.nan}])
    lines = path.read_text().splitlines()
    assert lines[0].startswith("dataset,file,checkpoint") and lines[2].startswith("x,")


def test_a_rerun_epoch_keeps_its_last_row(tmp_path):
    rows = [{"epoch": 0, "val_geodesic_deg": 120.0, "val_acc@5deg": 0.2, "train_total": 9.0},
            {"epoch": 1, "val_geodesic_deg": 115.0, "val_acc@5deg": 0.9, "partial": 1},
            {"epoch": 1, "val_geodesic_deg": 110.0, "val_acc@5deg": 0.5, "train_total": 7.0},
            {"epoch": 2, "val_geodesic_deg": 112.0, "val_acc@5deg": 0.7,
             "val_by_category": {"everyday_compressed/Mug": {"geodesic_deg": 100.0}}}]
    path = tmp_path / "run" / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rows))
    history = load_history(path)
    assert history.label == "run"
    assert history.epochs().tolist() == [1, 2, 3]
    assert history.series("val_geodesic_deg").tolist() == [120.0, 110.0, 112.0]
    assert history.best_epoch() == 3            # best.pt: the highest acc@5
    assert np.isnan(history.series("train_total")[2])
    assert history.categories() == ["everyday_compressed/Mug"]

    # A run with the rotation head kept best.pt by the lowest geodesic.
    rows[0]["val_head_cos"] = 0.5
    path.write_text(json.dumps(rows))
    assert load_history(path).has_head and load_history(path).best_epoch() == 2


def test_every_figure_draws_from_these_results(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from reassembly.viz import figures as fg

    rows = [{"epoch": e, "train_total": 10.0 - e, "val_total": 10.5 - e,
             **{f"{s}_{t}": 1.0 / (e + 1) for s in ("train", "val")
                for t in ("rotation", "position", "normal", "face", "embedding", "match@1")},
             "train_rotation_degrees": 120.0 - e, "val_geodesic_deg": 121.0 - e,
             "val_geodesic_median_deg": 122.0 - e,
             "val_by_category": {"Mug": {"geodesic_deg": 100.0 + e}}} for e in range(6)]
    path = tmp_path / "h" / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rows))
    history = load_history(path, "run")
    other = load_history(path, "other")
    evaluation = load_evaluation(_evaluation_file(tmp_path), "Everyday")
    writer = fg.FigureWriter(tmp_path / "figs", formats=("png",), dpi=60)
    assert fg.plot_total_loss(history, writer)
    assert fg.plot_loss_terms(history, writer)
    assert fg.plot_embedding(history, writer)
    assert fg.plot_rotation_error(history, writer)
    assert fg.plot_category_heatmap(history, writer)
    assert fg.plot_comparison([history, other], "val", writer)
    for source in ("matched", "network"):
        for kind in ("box", "violin"):
            assert fg.plot_errors_by_type([evaluation], source, writer, kind=kind)
    assert fg.plot_accuracy_curves([evaluation], writer)
    assert fg.plot_part_accuracy([evaluation], writer)
    assert fg.plot_by_piece_count([evaluation], writer, minimum_scenes=1)
    written = sorted(p.relative_to(tmp_path / "figs").as_posix()
                     for p in (tmp_path / "figs").rglob("*.png"))
    assert "training/run/loss_terms.png" in written
    assert "evaluation/rotation_error_by_type_matched_violin.png" in written
    assert len(written) == len(writer.written) == 13
    with pytest.raises(ValueError, match="at most 8"):
        fg.plot_comparison([history] * 9, "val", writer)
