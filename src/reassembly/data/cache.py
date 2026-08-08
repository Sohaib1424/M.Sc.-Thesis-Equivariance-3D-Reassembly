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
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def free_gib(path) -> float:
    """Free space on the filesystem holding ``path``, in GiB."""
    try:
        return shutil.disk_usage(str(path)).free / 1024 ** 3
    except OSError:
        return float("inf")


class DiskGuard:
    """Shared stop-writing rule for every cache on one filesystem.

    A per-cache byte budget is not enough on its own. It only knows what THAT
    cache wrote, so two caches with separate budgets can sum to more than
    intended -- and neither can see the dataset, the repository, or the
    checkpoints sharing the same 20 GB quota.

    Filling that quota does not merely stop the cache. It makes CHECKPOINT
    WRITES FAIL, which is the one failure that costs a whole run. So the
    binding rule here is FREE SPACE, not bytes written: caching stops while
    the filesystem still has ``min_free_gib`` in hand, whatever consumed it.

    Checked periodically rather than on every write, because ``disk_usage`` is
    a syscall and this sits in the data path.
    """

    def __init__(self, root, min_free_gib: float = 4.0, max_gib: Optional[float] = 10.0,
                 recheck_every: int = 64):
        self.root = Path(root) if root else None
        self.min_free_gib = min_free_gib
        self.max_bytes = int(max_gib * 1024 ** 3) if max_gib else None
        self.recheck_every = max(1, recheck_every)
        self._written = 0
        self._since_check = 0
        self._free = free_gib(root) if root else float("inf")
        self.stopped_reason = ""

    def allows(self, extra_bytes: int = 0) -> bool:
        if self.root is None:
            return False
        self._since_check += 1
        if self._since_check >= self.recheck_every:
            self._free = free_gib(self.root)
            self._since_check = 0

        if self._free <= self.min_free_gib:
            self.stopped_reason = (
                f"only {self._free:.1f} GiB free, floor is {self.min_free_gib:.1f}")
            return False
        if self.max_bytes is not None and self._written >= self.max_bytes:
            self.stopped_reason = (
                f"cache budget of {self.max_bytes / 1024 ** 3:.1f} GiB reached")
            return False
        return True

    def record(self, size: int) -> None:
        self._written += size
        self._free -= size / 1024 ** 3

    @property
    def written_gib(self) -> float:
        return self._written / 1024 ** 3


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
                 guard: Optional["DiskGuard"] = None,
                 max_bytes: Optional[int] = None,
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
        self.enabled = bool(enabled and root)   # '' or None disables
        # A shared guard so every cache on this filesystem draws from ONE
        # budget and one free-space floor, instead of each believing it has the
        # whole quota to itself.
        self.guard = guard if guard is not None else DiskGuard(
            root, max_gib=(max_bytes / 1024 ** 3) if max_bytes else 10.0)
        self.recheck_every = max(1, recheck_every)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self.skipped_full = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def full(self) -> bool:
        return not self.guard.allows()

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
        if not self.guard.allows():
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
            np.savez_compressed(tmp, **payload)
            size = tmp.stat().st_size
            os.replace(tmp, path)        # atomic within a filesystem
            self.writes += 1
            self.guard.record(size)
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
        extra = (f", {self.skipped_full} skipped ({self.guard.stopped_reason})"
                 if self.skipped_full else "")
        return (f"cache: {self.hits} hits / {total} lookups ({rate:.1f}%), "
                f"{self.writes} written, {self.errors} errors{extra}")


class BaseMeshCache:
    """Cache of the parsed base mesh, keyed by SCENE only.

    Separate from ``ScenePreprocessCache`` because of a hit-rate asymmetry that
    is easy to miss.

    ``compressed_mesh.obj`` and ``compressed_data.npz`` are identical across
    every fracture of a scene, and parsing them costs ~105 ms per sample
    (measured: 18,949 vertices, 35,274 faces). ``load_scene`` re-parses both on
    every single draw.

    The (scene, fracture) cache does not help here. With ~100 fractures per
    scene and random draws, the same PAIR rarely recurs until a large fraction
    of all pairs has been seen -- so early training misses almost every time.
    The same SCENE recurs constantly. Keying on scene alone turns a 105 ms
    parse into a 2.3 ms array load: **45x**, hitting from the first repeat.

    At batch 16 that is 1.67 s -> 0.04 s per batch, before any decimation or
    correspondence work.
    """

    VERSION = 1

    def __init__(self, root: Optional[str], enabled: bool = True,
                 guard: Optional["DiskGuard"] = None, compress: bool = True):
        self.compress = compress
        self.root = Path(root) / "base_meshes" if root else None
        self.enabled = bool(enabled and root)
        self.guard = guard if guard is not None else DiskGuard(root)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, scene_dir: str) -> Path:
        key = hashlib.sha1(f"v{self.VERSION}|{scene_dir}".encode()).hexdigest()
        return self.root / key[:2] / f"{key}.npz"

    def load(self, scene_dir: str):
        """Returns ``(vertices, faces, piece_to_fine_vertices)`` or None."""
        if not self.enabled:
            return None
        path = self._path(scene_dir)
        if not path.exists():
            self.misses += 1
            return None
        try:
            from scipy.sparse import csr_matrix
            with np.load(path, allow_pickle=False) as z:
                vertices = z["v"].astype(np.float64)
                faces = z["f"].astype(np.int64)
                matrix = csr_matrix((z["md"], z["mi"], z["mp"]),
                                    shape=(int(z["ms"][0]), int(z["ms"][1])))
            self.hits += 1
            return vertices, faces, matrix
        except Exception:                                    # noqa: BLE001
            self.misses += 1
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def store(self, scene_dir: str, vertices, faces, matrix) -> None:
        if not self.enabled:
            return
        if not self.guard.allows():
            return
        path = self._path(scene_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        csr = matrix.tocsr()
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            # Compressed by default. Measured on a real scene: 933 KB -> 308 KB
            # (3.0x), write 6.6 ms -> 41 ms, read 2.0 ms -> 5.0 ms. The write
            # cost is paid ONCE per scene; the read is still 21x faster than
            # the 105 ms parse it replaces. Tripling coverage matters more than
            # 3 ms on a hit, because for uniform random draws the hit rate is
            # roughly (scenes cached) / (scenes in the split) -- so size IS the
            # hit rate.
            writer = np.savez_compressed if self.compress else np.savez
            writer(
                tmp,
                v=np.asarray(vertices, dtype=np.float32),
                f=np.asarray(faces, dtype=np.int32),
                md=csr.data, mi=csr.indices, mp=csr.indptr,
                ms=np.asarray(csr.shape, dtype=np.int64),
            )
            size = tmp.stat().st_size
            os.replace(tmp, path)
            self.writes += 1
            self.guard.record(size)
        except Exception:                                    # noqa: BLE001
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        return (f"base-mesh cache: {self.hits}/{total} ({rate:.1f}%)")


def _dir_size(root: Path) -> int:
    total = 0
    try:
        for sub in os.scandir(root):
            if sub.is_dir():
                for entry in os.scandir(sub.path):
                    if entry.is_file():
                        total += entry.stat().st_size
    except OSError:
        pass
    return total
