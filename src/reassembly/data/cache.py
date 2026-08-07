"""
A disk cache for the deterministic half of scene preparation.

WHY THIS EXISTS
---------------
Measured on a real single-GPU run (1x T4, batch 4, ~22.5k vertices per scene):

    train   data (blocking wait) 82.3s   compute 125.0s
    val     data (blocking wait) 42.5s   compute   6.3s
    -----------------------------------------------------
    data    124.8s = 49% of the 256.5s epoch

Half the wall clock went to the data loader with four workers already running,
which means preprocessing was a genuine bottleneck rather than a rounding
error. On two GPUs it gets worse, not better: the same CPU cores are split
across twice as many ranks, so the second GPU buys much less than 2x.

None of that work needs redoing. Mesh parsing, piece splitting, duplicate-face
resolution, decimation and correspondence detection are all a pure function of
``(scene, fracture, decimation settings)``. Only the scattering transform is
random per sample, and that is cheap.

WHAT IS AND IS NOT CACHED
-------------------------
Cached:      fragment vertices/faces after decimation, the correspondence
             cluster ids, and the tolerance actually used.
Not cached:  the diffusion transform, feature construction, and every tensor
             derived from them -- those must stay random per epoch or the
             model sees the same scattering every time.

The key includes every setting that changes the result, so changing
``decimate_to`` produces different entries rather than silently reusing stale
geometry. That is the failure mode a cache has to avoid, and it is why the key
is derived from values rather than from the path alone.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ScenePreprocessCache:
    """Content-addressed cache of preprocessed scenes, as ``.npz`` files.

    Safe across DataLoader workers and DDP ranks: writes go to a
    process-unique temporary file and are then atomically renamed, so a reader
    either sees a complete entry or no entry at all. Two workers racing on the
    same scene both write, and the rename makes the last one win -- wasteful
    once, never corrupt.
    """

    VERSION = 1

    def __init__(self, root: Optional[str], enabled: bool = True,
                 max_bytes: Optional[int] = 8 * 1024 ** 3,
                 recheck_every: int = 256):
        """``max_bytes`` bounds the cache. Once exceeded, entries are still
        READ but no new ones are written.

        This bound is not optional in practice. Measured on a real scene at
        ``decimate_to: 6000``, one entry is ~437 KB, so Kaggle's 20 GB
        ``/kaggle/working`` quota holds about 45,000 of them. With ~100
        fractures per scene the number of distinct (scene, fracture) pairs runs
        into the hundreds of thousands, and random draws would reach the quota
        within the first hour of training.

        Filling that quota does not just stop the cache -- it makes CHECKPOINT
        WRITES FAIL, which loses the run. Stopping at a budget and continuing
        to serve whatever is already cached degrades instead.

        The total is re-measured every ``recheck_every`` writes rather than
        tracked purely in-process, because several DataLoader workers and DDP
        ranks write to the same directory and each would otherwise only see
        its own share.
        """
        self.root = Path(root) if root else None
        self.enabled = bool(enabled and root)
        self.max_bytes = max_bytes
        self.recheck_every = max(1, recheck_every)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self.skipped_full = 0
        self._bytes = 0
        self._since_check = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._bytes = self._measure()

    def _measure(self) -> int:
        total = 0
        try:
            for sub in os.scandir(self.root):
                if not sub.is_dir():
                    continue
                for entry in os.scandir(sub.path):
                    if entry.is_file():
                        total += entry.stat().st_size
        except OSError:
            pass
        return total

    @property
    def full(self) -> bool:
        return self.max_bytes is not None and self._bytes >= self.max_bytes

    # -- keying --------------------------------------------------------------
    def key(self, scene_dir: str, fracture_id: str, **settings) -> str:
        parts = [f"v{self.VERSION}", str(scene_dir), str(fracture_id)]
        parts += [f"{k}={settings[k]!r}" for k in sorted(settings)]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        # Two-level fan-out: a single directory with hundreds of thousands of
        # entries is slow to stat on most filesystems, and unusable on a
        # network mount.
        return self.root / key[:2] / f"{key}.npz"

    # -- read / write --------------------------------------------------------
    def load(self, key: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                n = int(z["num_fragments"])
                out = {
                    "vertices": [z[f"v{i}"] for i in range(n)],
                    "faces": [z[f"f{i}"] for i in range(n)],
                    "vertex_cluster_ids": [z[f"vc{i}"] for i in range(n)],
                    "edge_cluster_ids": [z[f"ec{i}"] for i in range(n)],
                    "tol": float(z["tol"]),
                }
            self.hits += 1
            return out
        except Exception:                                    # noqa: BLE001
            # A truncated or unreadable entry is a cache miss, never a crash.
            self.errors += 1
            self.misses += 1
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def store(self, key: str, vertices: Sequence[np.ndarray],
              faces: Sequence[np.ndarray],
              vertex_cluster_ids: Sequence[np.ndarray],
              edge_cluster_ids: Sequence[np.ndarray], tol: float) -> None:
        if not self.enabled:
            return
        if self._since_check >= self.recheck_every:
            self._bytes = self._measure()
            self._since_check = 0
        if self.full:
            self.skipped_full += 1
            return

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"num_fragments": np.int64(len(vertices)), "tol": np.float64(tol)}
        for i, (v, f, vc, ec) in enumerate(
            zip(vertices, faces, vertex_cluster_ids, edge_cluster_ids)
        ):
            payload[f"v{i}"] = np.asarray(v, dtype=np.float32)
            payload[f"f{i}"] = np.asarray(f, dtype=np.int32)
            payload[f"vc{i}"] = np.asarray(vc, dtype=np.int64)
            payload[f"ec{i}"] = np.asarray(ec, dtype=np.int64)

        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, **payload)
            size = tmp.stat().st_size
            os.replace(tmp, path)        # atomic within a filesystem
            self.writes += 1
            self._bytes += size
            self._since_check += 1
        except Exception:                                    # noqa: BLE001
            self.errors += 1
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        gb = self._bytes / 1024 ** 3
        cap = f"/{self.max_bytes / 1024 ** 3:.1f}" if self.max_bytes else ""
        extra = f", {self.skipped_full} skipped (budget reached)" if self.skipped_full else ""
        return (f"cache: {self.hits} hits / {total} lookups ({rate:.1f}%), "
                f"{self.writes} written, {gb:.2f}{cap} GiB, "
                f"{self.errors} errors{extra}")
