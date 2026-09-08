"""
Process-level environment hygiene: warning filters, thread limits, seeding.

All three have to be applied BEFORE the heavy imports and worker spawns to have
any effect, so they live together and every entry point calls them first.
"""
from __future__ import annotations

import os
import random
import warnings


def limit_blas_threads(num_threads: int = 1) -> None:
    """
    Pin BLAS/OpenMP to `num_threads`.

    Colab and Kaggle expose few vCPUs. Without this, every data-loading worker
    spins up its own OpenMP pool sized to the full core count and they fight
    each other -- the numpy/scipy geometry work ends up SLOWER than
    single-threaded. Must be set before numpy/TF are imported to take effect.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(num_threads))


# NOTE: every entry uses a SINGLE category class. `filterwarnings(category=(A, B))`
# silently does nothing useful -- `category` must be a class, not a tuple -- and
# is an easy way to "fix" warnings while fixing nothing.
_MESSAGE_FILTERS = [
    # trimesh emits these on meshes with degenerate/duplicate faces, which the
    # Breaking Bad fracture surfaces legitimately contain.
    #
    # "invalid value encountered in divide" is deliberately NOT filtered: it is
    # trimesh's signal that a zero-area triangle produced a NaN normal, and
    # silencing it hid a real data defect for an entire project.
    # `vngat.data.features._sanitise` repairs and COUNTS those values instead.
    (r".*divide by zero encountered.*", RuntimeWarning),
    (r".*Mean of empty slice.*", RuntimeWarning),
    # scipy 1.15 renamed Rotation.random(random_state=) -> rng=; a compat shim
    # handles it, but third-party code may still hit the old spelling.
    (r".*`random_state` is deprecated.*", DeprecationWarning),
]


def configure_warnings(strict: bool = False) -> None:
    """
    Silence the specific, known-benign warnings this pipeline provokes.

    Message-scoped rather than a blanket ignore: a blanket filter would also
    hide the warnings that indicate a real bug (shape mismatches, deprecated
    behaviour, non-finite values). `strict=True` turns anything not on the
    known-benign list into an error, which is what the test suite runs with.
    """
    for message, category in _MESSAGE_FILTERS:
        warnings.filterwarnings("ignore", message=message, category=category)
    if strict:
        warnings.filterwarnings("error", category=UserWarning)
        warnings.filterwarnings("error", category=RuntimeWarning)


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed python/numpy/TF for one process. `rank` offsets so replicas draw
    different scenes instead of duplicating each other's work."""
    import numpy as np

    effective = seed + 1000 * rank
    random.seed(effective)
    np.random.seed(effective % (2**32))
    try:
        import tensorflow as tf

        tf.random.set_seed(effective)
    except ImportError:      # data-only tools do not need TF
        pass


def dataloader_worker_init(worker_id: int, base_seed: int = 0) -> None:
    """
    Re-seed python's `random` and numpy's GLOBAL RNG inside each worker.

    This pipeline draws scenes with `random.choice` and rotations with
    `scipy...Rotation.random` (numpy's global RNG). Without a per-worker reseed
    every worker produces the IDENTICAL stream of scenes and rotations --
    silently cutting effective data diversity by a factor of `num_workers`
    while looking like it is working fine.
    """
    import numpy as np

    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
