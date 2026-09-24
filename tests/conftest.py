"""
Shared fixtures.

The important one is :func:`fractured_solid`. Earlier ad-hoc checks in this
project split a sphere by triangle centroid, which produces two *open surface
patches* sharing only a one-dimensional rim. Real Breaking Bad fragments are
closed solids, and the break creates a genuine two-dimensional surface that
exists twice, once on each side, with coincident vertices. A fixture that does
not reproduce that cannot test fracture extraction at all -- it will report
zero fracture faces and look like a pass.

:func:`no_gpu` is for the few tests whose meaning would otherwise depend on
how many GPUs the machine running them has.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh


def _grid_faces(n: int, flip: bool = False) -> np.ndarray:
    """Triangulation of an ``n x n`` vertex grid, row-major indexing."""
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    b, c, d = a + 1, a + n, a + n + 1
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return faces[:, ::-1] if flip else faces


def _rim_indices(n: int) -> np.ndarray:
    """Border vertices of an ``n x n`` grid, in a single closed loop."""
    top = np.arange(n)
    right = np.arange(1, n) * n + (n - 1)
    bottom = (n - 1) * n + np.arange(n - 2, -1, -1)
    left = np.arange(n - 2, 0, -1) * n
    return np.concatenate([top, right, bottom, left])


def _build_half(interface: np.ndarray, n: int, z_far: float, upward: bool):
    """A closed solid bounded by ``interface`` on one side and a flat plane on the other."""
    far = interface.copy()
    far[:, 2] = z_far
    vertices = np.vstack([interface, far])

    interface_faces = _grid_faces(n, flip=not upward)
    far_faces = _grid_faces(n, flip=upward) + len(interface)

    rim = _rim_indices(n)
    nxt = np.roll(rim, -1)
    walls = np.concatenate([
        np.stack([rim, nxt, rim + len(interface)], 1),
        np.stack([nxt, nxt + len(interface), rim + len(interface)], 1),
    ])
    if not upward:
        walls = walls[:, ::-1]

    faces = np.vstack([interface_faces, far_faces, walls])
    return trimesh.Trimesh(vertices, faces, process=False)


@pytest.fixture(scope="session")
def solid():
    return trimesh.creation.icosphere(subdivisions=4)


@pytest.fixture(scope="session")
def fractured_solid():
    """
    Two closed solids sharing a rough interface -- a synthetic fracture.

    Built explicitly rather than by slicing, because ``trimesh.slice_plane``
    needs ``shapely`` to cap, and an environment without it would skip exactly
    the tests that matter most. Both halves are watertight and both contain
    the *same* interface vertices, which is the property the coincidence
    labelling relies on. The interface is noisy while the far faces are flat,
    so the dihedral heuristic has something real to find.
    """
    n = 24
    rng = np.random.default_rng(7)
    x = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    gz = rng.normal(scale=0.08, size=(n, n))
    gz[0, :] = gz[-1, :] = gz[:, 0] = gz[:, -1] = 0.0    # rim must be planar
    interface = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    lower = _build_half(interface, n, z_far=-1.5, upward=True)
    upper = _build_half(interface, n, z_far=+1.5, upward=False)
    return lower, upper


@pytest.fixture(scope="session")
def simple_meshes():
    return {
        "box": trimesh.creation.box(),
        "cylinder": trimesh.creation.cylinder(radius=1.0, height=2.0, sections=24),
        "annulus": trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=1.0),
        "torus": trimesh.creation.torus(1.0, 0.3),
        "icosphere": trimesh.creation.icosphere(subdivisions=3),
    }


@pytest.fixture(scope="session")
def open_patch(solid):
    """An open surface patch -- the degenerate case that must not crash."""
    from reassembly.arrays import compact_indices

    faces = np.asarray(solid.faces)
    vertices = np.asarray(solid.vertices)
    keep = np.asarray(solid.triangles_center)[:, 2] > 0
    kept, new_faces = compact_indices(faces[keep], vertices.shape[0])
    return trimesh.Trimesh(vertices[kept], new_faces, process=False)


# --------------------------------------------------------------------------
# The machine
# --------------------------------------------------------------------------

@pytest.fixture
def no_gpu(monkeypatch):
    """
    Hide every GPU -- from this process and from any process it spawns.

    For the tests that need ``train(devices=N)`` to mean N processes. It means
    that only on a machine without a GPU: with GPUs, ``N`` is capped at how
    many there are, so on a one-GPU Colab machine ``devices=2`` is one process
    on ``cuda:0``, and a test asserting two processes fails there while passing
    on a laptop. Hidden, it is N processes over gloo on every machine.

    Both layers are needed. The spawned processes are fresh interpreters and
    see only the environment variable. This process may already have started
    CUDA in an earlier test, after which torch no longer reads the variable,
    so here the two calls the launcher makes are patched instead.
    """
    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
