#!/usr/bin/env python3
"""
Is the copy you are about to run the copy that was edited?

    python -m scripts.check_version

This project is edited in one place and run in another (a Kaggle container, a
second machine), and a partial copy produces symptoms that look like logic
bugs: a traceback whose line numbers do not match the source, a test failing
against code that is already correct, a fix that "did not work". This answers
in a second, with no GPU, no dataset and no torch:

* every source, script and test file **compiles** -- a truncated copy does not;
* every significant fix is **present**, by a line only the fixed file contains;
* no test file defines the same test twice (Python keeps the last one
  silently, so the first never runs).
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (what, file, a line only the fixed version contains)
MARKERS = [
    # -- several GPUs --------------------------------------------------------
    ("one gradient all-reduce per step, not DDP", "src/reassembly/distributed.py",
     "def sum_gradients"),
    ("a gradient nobody produced stays None", "src/reassembly/distributed.py",
     "flags[index] = 1.0"),
    ("replicas start identical", "src/reassembly/distributed.py",
     "def broadcast_parameters"),
    ("rescale after the all-reduce, by the global count", "src/reassembly/training.py",
     "inverse = 1.0 / total"),
    ("GPUs meet only at step boundaries", "src/reassembly/training.py",
     "if in_group == config.accumulate:"),
    ("trailing partial group is stepped", "src/reassembly/training.py",
     "if training and in_group:"),
    ("stop is voted, on every GPU", "src/reassembly/training.py",
     "signal_watch = StopSignal().install()"),
    ("SIGTERM forwarded to the GPU processes", "src/reassembly/training.py",
     "class _ForwardTerminate"),
    ("validation sharded without padding", "src/reassembly/data/sampling.py",
     "class ShardSampler"),
    ("epoch summaries gathered from every GPU", "src/reassembly/training.py",
     "tally = _merge_tallies(dist.gather(tally))"),
    ("per-step counts not multiplied by the GPU count", "src/reassembly/training.py",
     "_PER_STEP = ("),
    ("only rank 0 carries the resume point", "src/reassembly/training.py",
     "if main and load_path.resolve() != last_path.resolve():"),
    ("stable scene seed (no salted hash)", "src/reassembly/training.py",
     "def stable_seed"),
    # -- a batch that fails ----------------------------------------------------
    ("gradient set aside per micro-batch", "src/reassembly/training.py",
     "def _take_gradients"),
    ("OOM retried with checkpointing", "src/reassembly/training.py",
     "class _checkpointing"),
    ("non-finite gradient dropped", "src/reassembly/training.py",
     'outcome = "nonfinite-grad"'),
    ("empty step does not move the weights", "src/reassembly/training.py",
     'tally["empty_steps"] += 1'),
    ("failures named by scene and tallied across epochs", "src/reassembly/training.py",
     "def _record_offenders"),
    ("repairs counted", "src/reassembly/data/features.py", "class Repairs"),
    ("non-finite coordinates skipped by name", "src/reassembly/training.py",
     '"non-finite vertex coordinates"'),
    # -- epochs and the data path -------------------------------------------
    ("fixed-length epochs", "src/reassembly/training.py", "steps_per_epoch: int = 1600"),
    ("fixed-length sampler", "src/reassembly/data/sampling.py", "class EpochSampler"),
    ("restarted epoch rewinds its step", "src/reassembly/training.py",
     'state.get("epoch_step", state["step"])'),
    ("perturbed copy made on the device", "src/reassembly/data/features.py",
     "def perturb_on_device"),
    ("pair lists built on the device", "src/reassembly/data/features.py",
     "def pair_on_device"),
    # -- assembly and evaluation --------------------------------------------
    ("translation solver", "src/reassembly/assembly/translation.py",
     "def solve_translations"),
    ("benchmark scores in world units", "src/reassembly/assembly/scoring.py",
     "def score_batch"),
    ("evaluation assembles", "src/reassembly/training.py", "def _assembly_lines"),
    ("evaluation uses the checkpoint's data", "src/reassembly/training.py",
     "def _adopt_checkpoint_settings"),
    ("Chamfer memory bounded", "src/reassembly/evaluation/metrics.py",
     "(1 << 24) // (3 * max(other, 1))"),
    # -- scripts -------------------------------------------------------------
    ("shared Config flags", "scripts/config_flags.py", "def add_config_arguments"),
    ("data pipeline benchmark", "scripts/benchmark_data.py", "def device_side"),
    ("NaN locator", "scripts/check_scene.py", "def locate"),
    ("prediction dump with placement", "scripts/dump_prediction.py", "placement="),
    ("reassembly viewer", "scripts/visualize_reassembly.py", "build_trimesh_scene"),
    ("GIF renderer", "scripts/render_gif.py", "def render_matplotlib"),
    ("object-count sweep", "scripts/scaling_sweep.py", "--max_objects"),
    # -- earlier fixes that must not regress --------------------------------
    ("non-finite loss caught before backward", "src/reassembly/training.py",
     'return "nonfinite-loss", report, value'),
    ("fragment-weighted micro-batches", "src/reassembly/training.py",
     "scaled = loss * weight"),
    ("objects grouped across variant directories", "src/reassembly/data/catalog.py",
     "def build_catalog"),
    ("multi-process tests", "tests/test_distributed.py",
     "def test_the_step_is_the_global_mean_and_the_replicas_stay_identical"),
    ("tests independent of the machine's GPU count", "tests/conftest.py",
     "def no_gpu(monkeypatch)"),
    ("OOM-group test exact, on one thread", "tests/test_training_recovery.py",
     "def _identical(a, b)"),
    # -- Thesis 1's flags, one checkpointing switch ----------------------------
    ("whole-layer gradient checkpointing", "src/reassembly/nn/model.py",
     "torch.utils.checkpoint.checkpoint(layer, *inputs,"),
    ("Thesis 1's flag names and meanings", "scripts/config_flags.py",
     "THESIS1_ONLY = {"),
    ("--lr_schedule constant", "src/reassembly/training.py",
     'if config.lr_schedule == "constant":'),
    ("--split_source official is enforced", "src/reassembly/training.py",
     'if config.split_source == "official" and not official:'),
    ("fewer GPUs than --num_gpus is announced", "src/reassembly/training.py",
     "GPU(s) visible -- using {requested}."),
    ("time budget read between epochs", "src/reassembly/training.py",
     "votes = dist.sum_scalars([float(val_stopped), float(time.time() >= deadline)],"),
    # -- Thesis v6: the largest fragment as the anchor --------------------------
    ("the anchor: one rotation per scene, on the left", "src/reassembly/nn/anchor.py",
     "aligned = torch.matmul(correction[fragment_scene.long()], rotation)"),
    ("--rotation_target anchor|absolute", "src/reassembly/training.py",
     'ROTATION_TARGETS = ("anchor", "absolute")'),
    ("the loss scores every fragment but the anchors", "src/reassembly/training.py",
     "compared, scored = anchor_alignment(batch, R)"),
    ("the step weighted by the fragments scored", "src/reassembly/training.py",
     "weight = float(_loss_fragments(batch, config))"),
    ("metrics use the anchor, absolute error beside them", "src/reassembly/training.py",
     'tally["absolute_predictions"].append(R.detach().float().cpu())'),
    ("old checkpoints read as the absolute target", "src/reassembly/training.py",
     'LEGACY_ROTATION_TARGET = "absolute"'),
    ("stage two measured from the anchor", "src/reassembly/assembly/scoring.py",
     "t_true = centroid - centroid[reference]"),
    # -- Thesis v6: the benchmark's 2-20 pieces ---------------------------------
    ("--max_fragments (0 = no limit)", "src/reassembly/training.py",
     "max_fragments: Optional[int] = None"),
    ("pieces counted from the label file, not by loading", "src/reassembly/data/scene.py",
     "def piece_count(scene_dir: str | Path, mode: str) -> int:"),
    ("the limit applied after the split, before max_objects", "src/reassembly/training.py",
     "self.catalog, self.fragment_limit = limit_fragments("),
    ("a resume onto another limit resets the best-so-far", "src/reassembly/training.py",
     'if _fragment_limit_changed(state.get("config") or {}, config):'),
    ("--evaluate --max_fragments overrides the checkpoint", "scripts/train.py",
     'override = ("max_fragments",) if args.max_fragments is not None else ()'),
    # -- Thesis v6: the optional Hugging Face Hub mirror -------------------------
    ("hub mirror off unless all three --hf_* are given", "src/reassembly/hub.py",
     "return bool(self.repo_id and self.local_dir and self._token)"),
    ("hub pushed after every epoch, before the stop", "src/reassembly/training.py",
     'hub.push(f"epoch {epoch + 1}: val geodesic {score:.3f}")'),
    ("hub token kept out of Config and checkpoints", "scripts/train.py",
     "train(config, hub=HubSync(args.hf_repo_id, args.hf_local_dir, args.hf_token))"),
    # -- Thesis v6: rotations from the matches, and two diagnostics fixed --------
    ("rotations fitted from the embedding matches", "src/reassembly/assembly/rotation.py",
     "def chain_rotations"),
    ("--evaluate --rotations matched", "src/reassembly/assembly/scoring.py",
     'ROTATION_SOURCES = ("network", "matched")'),
    ("tilt/twist read in the object's frame", "src/reassembly/evaluation/metrics.py",
     "residual = torch.matmul(predicted, target.transpose(-1, -2))"),
    ("'no limit' survives config_flags", "scripts/config_flags.py",
     'out += [flag(name), "0"]'),
    # -- Thesis v6: reports and figures from saved results -----------------------
    ("the report built from the summary alone", "src/reassembly/training.py",
     "def format_evaluation"),
    ("the report saved beside the metrics", "src/reassembly/training.py",
     '(out_dir / f"{split}_report{suffix}.txt").write_text('),
    ("metrics written one line per scene", "src/reassembly/training.py", "def dump_metrics"),
    ("the head's own angles kept in a matched evaluation", "src/reassembly/assembly/scoring.py",
     'entry["_network_geodesic_deg"] = network_angle[own].tolist()'),
    ("a report rebuilt from a metrics file", "scripts/report_metrics.py",
     "from reassembly.training import format_evaluation"),
    ("results read without torch", "src/reassembly/viz/results.py", "def load_evaluation"),
    ("figures and tables from results", "scripts/make_figures.py",
     "from reassembly.viz import figures as fg"),
    # -- Thesis v7: placement from the verified pair fits ------------------------
    ("each pair fit keeps its offset and its inliers", "src/reassembly/assembly/rotation.py",
     "def ransac_motion"),
    ("only the pair fits the chain agrees with place", "src/reassembly/assembly/placement.py",
     "def verified_pairs"),
    ("the anchor held, not the zero mean", "src/reassembly/assembly/placement.py",
     "def solve_with_held"),
    ("unreached fragments placed with the reached held", "src/reassembly/assembly/placement.py",
     "free = ~reached"),
    ("--placement checked|global", "src/reassembly/assembly/scoring.py",
     'PLACEMENTS = ("checked", "global")'),
    ("--evaluate --placement", "scripts/train.py", "placement=args.placement"),
    ("the evaluation records its placement", "src/reassembly/training.py",
     'summary["placement"] = placement'),
    ("the dump records its placement", "scripts/dump_prediction.py",
     "placement_method=np.array(method)"),
    ("placement tests", "tests/test_assembly_placement.py",
     "def test_wrong_matches_pull_the_global_solve_together_but_not_the_checked_one"),
    # -- Thesis v7: evaluation without shared break vertices -----------------------
    ("--jitter: noise on the inputs, per scene", "src/reassembly/evaluation/noise.py",
     "def jitter_inputs"),
    ("--drop: break vertices left out of the matching", "src/reassembly/evaluation/noise.py",
     "def drop_candidates"),
    ("the model is shown the noisy copy", "src/reassembly/training.py",
     'on_prediction(batch, holder["prediction"], holder["batch"])'),
    ("the score reads the clean fragments", "src/reassembly/assembly/scoring.py",
     "shape = apply_rotation(batch.node_features[v0:v1, 0, :], rotation[f0:f1],"),
    ("--evaluate --jitter/--drop", "scripts/train.py", "jitter=args.jitter, drop=args.drop"),
    ("noise tests", "tests/test_evaluation_noise.py",
     "def test_the_method_sees_the_noise_and_the_score_does_not"),
]


def main() -> int:
    problems = []
    print(f"checking {ROOT}\n")

    files = sorted(p for folder in ("src", "scripts", "tests")
                   for p in (ROOT / folder).rglob("*.py"))
    broken = []
    for path in files:
        try:
            # In memory: no .pyc written into the checked copy.
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, ValueError, UnicodeDecodeError) as error:
            broken.append((path, error))
    for path, error in broken:
        print(f"  DOES NOT COMPILE  {path.relative_to(ROOT)}: {error}")
        problems.append(str(path))
    print(f"  {len(files) - len(broken)}/{len(files)} files compile")

    for label, relative, marker in MARKERS:
        path = ROOT / relative
        if not path.is_file():
            print(f"  ABSENT   {label:<50} {relative}")
            problems.append(label)
            continue
        present = marker in path.read_text(encoding="utf-8")
        print(f"  {'ok     ' if present else 'STALE  '}  {label:<50} {relative}")
        if not present:
            problems.append(label)

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue                         # already reported as not compiling
        names = [n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        repeated = [name for name, count in Counter(names).items() if count > 1]
        if repeated:
            print(f"  DUPLICATE tests in {path.name}: {repeated} -- only the last runs")
            problems.append(f"duplicate tests in {path.name}")

    print()
    if problems:
        print(f"{len(problems)} problem(s): re-copy the project before running it.")
        return 1
    print(f"all {len(MARKERS)} markers present, every file compiles, no shadowed tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
