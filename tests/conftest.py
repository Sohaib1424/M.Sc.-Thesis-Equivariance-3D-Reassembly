"""Shared fixtures and synthetic-graph builders."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.data.graph import FragmentGraph, SceneBatch, collate_scenes, merge_fragments  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(1234)


def random_rotation(num: int = 1, dtype=torch.float32) -> torch.Tensor:
    """(num, 3, 3) proper rotations, via QR with a determinant fix."""
    q, _ = torch.linalg.qr(torch.randn(num, 3, 3, dtype=torch.float64))
    det = torch.linalg.det(q)
    q[:, :, 0] *= det.sign().unsqueeze(-1)
    return q.to(dtype)


def make_fragment(num_v: int = 12, num_e: int = 25, seed: int = 0) -> FragmentGraph:
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(num_v, 3, generator=g)
    pos = pos - pos.mean(0, keepdim=True)              # centralised, as the real pipeline does
    normal = torch.nn.functional.normalize(torch.randn(num_v, 3, generator=g), dim=-1)
    src = torch.randint(0, num_v, (num_e,), generator=g)
    dst = torch.randint(0, num_v, (num_e,), generator=g)
    edge_index = torch.stack([src, dst])
    n1 = torch.nn.functional.normalize(torch.randn(num_e, 3, generator=g), dim=-1)
    n2 = torch.nn.functional.normalize(torch.randn(num_e, 3, generator=g), dim=-1)
    delta = pos[src] - pos[dst]                        # source minus destination
    return FragmentGraph(
        node_vec=torch.stack([pos, normal], dim=1),
        edge_index=edge_index,
        edge_attr=torch.stack([n1, n2, delta], dim=1),
        centroid=torch.randn(3, generator=g),
        radius=float(pos.norm(dim=-1).max()),
        vertex_cluster_id=torch.randint(-1, 3, (num_v,), generator=g),
        edge_cluster_id=torch.randint(-1, 3, (num_e,), generator=g),
    )


def make_scene(frag_specs=((10, 21), (7, 15), (13, 29)), seed: int = 0,
               normalize_mode: str = "scene") -> SceneBatch:
    return merge_fragments([
        make_fragment(nv, ne, seed=seed * 100 + i) for i, (nv, ne) in enumerate(frag_specs)
    ], normalize_mode=normalize_mode)


@pytest.fixture
def scene() -> SceneBatch:
    return make_scene()


@pytest.fixture
def batch() -> SceneBatch:
    """A batch of three DIFFERENT objects: different fragment counts and
    different per-fragment vertex counts, which is the real-world case."""
    return collate_scenes([
        make_scene(((11, 23), (7, 15), (19, 31)), seed=1),
        make_scene(((5, 9), (13, 27)), seed=2),
        make_scene(((9, 17), (6, 11), (8, 14), (12, 22)), seed=3),
    ])


def test_no_duplicate_test_names():
    """
    Two functions with the same name in one module: Python keeps the LAST, so
    the earlier one never runs. That silently reverted an updated assertion
    back to a stale one, which then failed against correct code. Cheap to
    check, easy to miss by eye.
    """
    import ast
    from collections import Counter

    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        names = [n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        dupes = [n for n, c in Counter(names).items() if c > 1]
        assert not dupes, f"{path.name} defines these twice: {dupes}"


# --------------------------------------------------------------------------
# A miniature Breaking Bad tree on disk, in the real compressed format
# --------------------------------------------------------------------------
def write_scene(scene_dir, modes=("fractured_0", "fractured_1", "fractured_2"),
                pieces: int = 3, seed: int = 0) -> None:
    """
    One object directory the real loader can read: `compressed_mesh.obj` (an
    irregular icosphere), `compressed_data.npz` (identity cell -> vertex map)
    and, per mode, `compressed_fracture.npy` (a piece label per cell, cut by
    random planes so the pieces are genuine fragments of one surface).
    """
    import numpy as np
    import scipy.sparse
    import trimesh

    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sphere = trimesh.creation.icosphere(subdivisions=2)
    vertices = sphere.vertices * (1.0 + 0.1 * rng.standard_normal((len(sphere.vertices), 1)))
    trimesh.Trimesh(vertices, sphere.faces, process=False).export(scene_dir / "compressed_mesh.obj")
    scipy.sparse.save_npz(scene_dir / "compressed_data.npz",
                          scipy.sparse.identity(len(vertices), format="csr"))
    for k, mode in enumerate(modes):
        (scene_dir / mode).mkdir(exist_ok=True)
        axis = rng.standard_normal(3)
        proj = vertices @ (axis / np.linalg.norm(axis))
        edges = np.quantile(proj, np.linspace(0, 1, pieces + 1)[1:-1])
        labels = np.searchsorted(edges, proj).astype(np.int64)
        np.save(scene_dir / mode / "compressed_fracture.npy", labels)


def write_dataset(root, layout, official=None) -> None:
    """
    `layout`: {"everyday_compressed/everyday_compressed/Bottle/b0": n_modes, ...}.
    `official`: {"train": ["everyday/Bottle/b0", ...], "val": [...]} written to
    data_split/data_split/everyday.<split>.txt, nested the way the release is.
    """
    root = Path(root)
    for i, (rel, n_modes) in enumerate(sorted(layout.items())):
        write_scene(root / rel, modes=tuple(f"fractured_{k}" for k in range(n_modes)), seed=i)
    if official:
        split_dir = root / "data_split" / "data_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split, entries in official.items():
            (split_dir / f"everyday.{split}.txt").write_text("\n".join(entries) + "\n")
