"""Shared fixtures. Tests run in float64 so a correct implementation gives
residuals ~1e-15; in float32 the same checks give ~1e-7, which is
indistinguishable from a real bug."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.data.graph import FragmentGraph, collate_scenes, merge_fragments  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    tf.keras.backend.set_floatx("float64")
    yield
    tf.keras.backend.set_floatx("float32")


def haar_rotations(num: int, seed: int = 0) -> np.ndarray:
    """Uniform on SO(3) via unit quaternions -- exactly Haar, unlike raw QR."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(num, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def make_fragment(num_v: int = 12, num_e: int = 25, seed: int = 0) -> FragmentGraph:
    r = np.random.default_rng(seed)
    pos = r.normal(size=(num_v, 3))
    pos -= pos.mean(0)                       # centralised, as the real pipeline does
    nrm = r.normal(size=(num_v, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    a, b = r.integers(0, num_v, num_e), r.integers(0, num_v, num_e)
    return FragmentGraph(
        node_vec=np.stack([pos, nrm], 1),
        edge_index=np.stack([a, b]).astype(np.int64),
        edge_len=np.linalg.norm(pos[a] - pos[b], axis=1)[:, None],
        edge_vec=np.stack([(pos[a] + pos[b]) / 2, r.normal(size=(num_e, 3)),
                           r.normal(size=(num_e, 3))], 1),
        centroid=r.normal(size=3),
        vertex_cluster_id=r.integers(-1, 4, num_v),
        edge_cluster_id=r.integers(-1, 4, num_e))


def make_scene(specs=((10, 21), (7, 15), (13, 29)), seed: int = 0):
    return merge_fragments([make_fragment(nv, ne, seed * 100 + i)
                            for i, (nv, ne) in enumerate(specs)])


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def batch():
    """Three DIFFERENT objects: different fragment counts and different
    per-fragment vertex counts, which is the real-world case."""
    return collate_scenes([
        make_scene(((11, 23), (7, 15), (19, 31)), seed=1),
        make_scene(((5, 9), (13, 27)), seed=2),
        make_scene(((9, 17), (6, 11), (8, 14), (12, 22)), seed=3)])
