"""
Shared fixtures.

Tests are organized in three tiers:

  pure       numpy/scipy only -- runs anywhere, no torch, no dataset
  torch      needs torch + torch_geometric (skipped if absent)
  needs_data needs the Breaking Bad dataset on disk (opt in with --root-dir)

Run everything available:      pytest
Only what runs without torch:  pytest -m "not torch and not needs_data"
Include the dataset tests:     pytest --root-dir data
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def pytest_addoption(parser):
    parser.addoption("--root-dir", action="store", default=None,
                     help="Breaking Bad dataset root; enables the needs_data tests.")


def pytest_configure(config):
    for marker in ("slow", "needs_data", "cuda", "torch"):
        config.addinivalue_line("markers", f"{marker}: see tests/conftest.py")


@pytest.fixture(scope="session")
def root_dir(request):
    value = request.config.getoption("--root-dir")
    if value is None or not Path(value).exists():
        pytest.skip("needs --root-dir pointing at the Breaking Bad dataset")
    return value


@pytest.fixture(scope="session")
def torch_mod():
    return pytest.importorskip("torch", reason="torch not installed")


@pytest.fixture(scope="session")
def pyg():
    pytest.importorskip("torch", reason="torch not installed")
    return pytest.importorskip("torch_geometric", reason="torch_geometric not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# --------------------------------------------------------------------------
# Geometry helpers usable without trimesh
# --------------------------------------------------------------------------
def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation matrix."""
    u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def grid_mesh(n: int = 6, z: float = 0.0, offset=(0.0, 0.0, 0.0)):
    """A flat triangulated grid as (vertices, faces) arrays."""
    g = np.linspace(0, 1, n)
    Y, X = np.meshgrid(g, g, indexing="ij")
    verts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=-1) + np.asarray(offset)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, c], [b, d, c]]
    return verts, np.asarray(faces)


@pytest.fixture
def trimesh_mod():
    return pytest.importorskip("trimesh", reason="trimesh not installed")
