#!/usr/bin/env python
"""
Verify that a deployed copy of this project actually contains every fix,
without needing a GPU, a dataset, or a training run.

    python -m scripts.check_version

Exists because this project is edited in one place and executed in another
(a Kaggle container), and a partial file copy produces confusing symptoms:
a traceback whose line numbers do not match the source, or a stale test
failing against correct code. One command that says which files are behind is
faster than reading tracebacks.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, file, marker that only the fixed version contains)
MARKERS = [
    ("rotation convention (F^T, not F)",
     "vngat/models/vn_layers.py", "return gram_schmidt_frame(a1, a2, eps=eps).transpose(-1, -2)"),
    ("Gram-Schmidt normalises safely",
     "vngat/models/vn_layers.py", "a1.pow(2).sum(-1, keepdim=True) + eps * eps"),
    ("geodesic angle via atan2",
     "vngat/models/vn_layers.py", "torch.atan2(sin_theta, cos_theta)"),
    ("segment reductions accumulate in fp32",
     "vngat/models/segment_ops.py", "def _accum_dtype"),
    ("per-fragment equivariance (invariant exchange)",
     "vngat/models/virtual_nodes.py", "self.slot_invariant"),
    ("attention weights cast to value dtype",
     "vngat/models/gat_layer.py", "alpha.to(v.dtype)"),
    ("AMP dtype safety in the embedding loss",
     "vngat/losses/composite.py", "acc = torch.float32"),
    ("embedding repulsion (anti-collapse)",
     "vngat/losses/composite.py", "push_weight * push"),
    ("embeddings normalised (bounded loss)",
     "vngat/losses/composite.py", "emb = emb / torch.sqrt(emb.pow(2)"),
    ("scheduler watches rot, not total",
     "vngat/config.py", "lr_monitor"),
    ("lr visible in the epoch table",
     "vngat/utils/progress.py", '("lr", 10)'),
    ("last.pt for lossless resume",
     "vngat/training/checkpoint.py", 'LAST_NAME = "last.pt"'),
    ("swing-twist diagnostic",
     "vngat/evaluation/metrics.py", "def swing_twist_error"),
    ("scaling sweep script",
     "scripts/scaling_sweep.py", "SCALING SUMMARY"),
    ("VNLeakyReLU inner products in fp32",
     "vngat/models/vn_layers.py", "xf = _at_least_float32(x)"),
    ("chamfer without the cdist sqrt round trip",
     "vngat/evaluation/metrics.py", "clamp_min(0).amin(dim=1)"),
    ("checkpoint best_val ordering",
     "vngat/training/checkpoint.py", "improved_best = val_loss < self.best_val"),
    ("gradient sync on a fixed placeholder step",
     "vngat/training/trainer.py", "def _sync_gradients"),
    ("fresh sync context per OOM retry",
     "vngat/training/trainer.py", "with make_sync():"),
    ("single-GPU device resolution",
     "vngat/training/distributed.py", "def resolve_device"),
    ("embedding margins in Config",
     "vngat/config.py", "emb_push_margin"),
    ("collapse regression test",
     "tests/test_losses.py", "test_collapsed_embeddings_are_penalised"),
    ("OOM recovery tests",
     "tests/test_trainer_recovery.py", "test_oom_retry_rebuilds_the_sync_context"),
    ("Drive round-trip verification",
     "vngat/training/drive.py", "def verify"),
    ("Drive pre-flight script",
     "scripts/check_drive.py", "storagequota"),
    ("assumption-free dataset discovery",
     "vngat/data/splits.py", "for dirpath, dirnames, filenames in os.walk"),
    ("data_subsets in Config",
     "vngat/config.py", "data_subsets"),
    ("NaN-safe norms (zero-vector gradient)",
     "vngat/models/vn_layers.py", "axis.pow(2).sum(-1) + 1e-12"),
    ("non-finite loss guard",
     "vngat/training/trainer.py", "def _first_non_finite_parameter"),
    ("official Breaking Bad splits",
     "vngat/data/splits.py", "def load_official_split"),
]


def main() -> int:
    missing = []
    print(f"checking {ROOT}\n")
    for label, relative, marker in MARKERS:
        path = ROOT / relative
        if not path.is_file():
            print(f"  ABSENT   {label:<46} {relative}")
            missing.append(label)
            continue
        ok = marker in path.read_text()
        print(f"  {'ok     ' if ok else 'STALE  '}  {label:<46} {relative}")
        if not ok:
            missing.append(label)

    # Duplicate test names silently shadow each other -- Python keeps the last.
    print()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        names = [n.name for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        dupes = [n for n, c in Counter(names).items() if c > 1]
        if dupes:
            print(f"  DUPLICATE test names in {path.name}: {dupes}")
            missing.append(f"duplicate tests in {path.name}")

    # Configs must expose exactly the fields Config declares.
    import yaml

    cls = next(n for n in ast.parse((ROOT / "vngat/config.py").read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Config")
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    for cfg in sorted((ROOT / "configs").glob("*.yaml")):
        keys = set(yaml.safe_load(cfg.read_text()) or {})
        if keys != fields:
            print(f"  CONFIG MISMATCH {cfg.name}: {sorted(keys ^ fields)}")
            missing.append(f"config {cfg.name}")

    print()
    if missing:
        print(f"{len(missing)} item(s) out of date -- re-copy the project before running.")
        return 1
    print(f"all {len(MARKERS)} checks passed; {len(fields)} config fields consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
