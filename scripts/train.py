#!/usr/bin/env python
"""
Training entry point.

    python -m scripts.train --config configs/kaggle_2xt4_full.yaml
    python -m scripts.train --root_dir data --num_gpus 2 --batch_size 2

From a notebook, call `main([...])` with an explicit argument list -- reading
`sys.argv` there would pick up the Jupyter kernel's own launch flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

# Both must run before torch/numpy do their thread-pool setup.
limit_blas_threads(1)
configure_warnings()

from vngat.config import parse_config  # noqa: E402
from vngat.training.trainer import launch  # noqa: E402


def main(argv=None) -> None:
    cfg = parse_config(argv)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(str(Path(cfg.checkpoint_dir) / "config.yaml"))
    launch(cfg)


if __name__ == "__main__":
    main()
