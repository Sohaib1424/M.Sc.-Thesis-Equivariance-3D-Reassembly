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
    ("evaluation assembles", "src/reassembly/training.py", "def _print_assembly"),
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
