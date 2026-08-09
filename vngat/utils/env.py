"""
Process-level environment hygiene: warning filters, thread limits, seeding.

Three separate concerns live here because all three have to be applied *before*
the heavy imports / worker spawns to have any effect, so it is easier to keep
them in one place that every entry point calls first.
"""
from __future__ import annotations

import os
import random
import warnings


# ---------------------------------------------------------------------------
# Thread limits
# ---------------------------------------------------------------------------
def limit_blas_threads(num_threads: int = 1) -> None:
    """
    Pin BLAS/OpenMP to `num_threads`.

    Kaggle's 2xT4 notebooks expose ~4 vCPUs. Without this, every one of
    (2 ranks x N dataloader workers x main processes) spins up its own
    OpenMP pool sized to the full core count, and they fight each other --
    numpy/scipy geometry work in the loader workers ends up *slower* than
    single-threaded. Must be set before numpy/torch are imported to take
    effect, which is why entry points call this at the very top.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(num_threads))


# ---------------------------------------------------------------------------
# Warning filters
# ---------------------------------------------------------------------------
#
# NOTE: every entry below is applied with a SINGLE category class.
# `warnings.filterwarnings(..., category=(A, B))` silently does nothing useful
# -- `category` must be a class, not a tuple -- and a tuple there is an easy
# way to "fix" warnings while actually fixing nothing. One call per category.
_MESSAGE_FILTERS = [
    # trimesh emits these on meshes with degenerate/duplicate faces, which the
    # Breaking Bad fracture surfaces legitimately contain.
    (r".*invalid value encountered in (divide|true_divide).*", RuntimeWarning),
    (r".*divide by zero encountered.*", RuntimeWarning),
    (r".*Mean of empty slice.*", RuntimeWarning),
    # torch.load without weights_only (we always pass it explicitly, but some
    # third-party code inside torch does not).
    (r".*You are using `torch\.load` with `weights_only=False`.*", FutureWarning),
    # torch.cuda.amp.* -> torch.amp.* migration noise on torch>=2.4.
    (r".*`torch\.cuda\.amp\.(autocast|GradScaler)\(.*\)` is deprecated.*", FutureWarning),
    # index_reduce_ is flagged beta by torch; it is used deliberately in
    # vngat/models/segment_ops.py and its semantics are pinned by tests.
    (r".*index_reduce\(\) is in beta.*", UserWarning),
    # scipy 1.15 renamed Rotation.random(random_state=) -> rng=; we use a
    # compat shim, but trimesh/other libs may still hit it.
    (r".*`random_state` is deprecated.*", DeprecationWarning),
]


def configure_warnings(strict: bool = False) -> None:
    """
    Silence the specific, known-benign warnings this pipeline provokes.

    Deliberately message-scoped rather than blanket `ignore`: a blanket filter
    would also hide the warnings that actually indicate a bug (shape
    mismatches, non-writable tensor aliasing, deprecated autograd behaviour).
    `strict=True` turns everything that is *not* on the known-benign list into
    an error, which is what the test-suite runs with.
    """
    for message, category in _MESSAGE_FILTERS:
        warnings.filterwarnings("ignore", message=message, category=category)
    if strict:
        warnings.filterwarnings("error", category=UserWarning)
        warnings.filterwarnings("error", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed python/numpy/torch for one process. `rank` offsets so DDP ranks
    draw different scenes instead of duplicating each other's work."""
    import numpy as np
    import torch

    effective = seed + 1000 * rank
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def dataloader_worker_init(worker_id: int) -> None:
    """
    Re-seed python's `random` and numpy's *global* RNG inside each DataLoader
    worker.

    PyTorch seeds `torch`'s RNG per worker automatically but does NOT touch
    numpy or python `random`. This pipeline draws scenes with
    `random.choice` and rotations with `scipy...Rotation.random` (numpy global
    RNG), so without this every worker would produce the *identical* stream of
    scenes and rotations -- silently cutting effective data diversity by a
    factor of `num_workers` while looking like it was working fine.
    """
    import numpy as np
    import torch

    base = torch.initial_seed() % (2**31)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
