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
    edge_len = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
    edge_mid = (pos[src] + pos[dst]) / 2
    n1 = torch.nn.functional.normalize(torch.randn(num_e, 3, generator=g), dim=-1)
    n2 = torch.nn.functional.normalize(torch.randn(num_e, 3, generator=g), dim=-1)
    return FragmentGraph(
        node_vec=torch.stack([pos, normal], dim=1),
        edge_index=edge_index,
        edge_len=edge_len,
        edge_vec=torch.stack([edge_mid, n1, n2], dim=1),
        centroid=torch.randn(3, generator=g),
        vertex_cluster_id=torch.randint(-1, 3, (num_v,), generator=g),
        edge_cluster_id=torch.randint(-1, 3, (num_e,), generator=g),
    )


def make_scene(frag_specs=((10, 21), (7, 15), (13, 29)), seed: int = 0) -> SceneBatch:
    return merge_fragments([
        make_fragment(nv, ne, seed=seed * 100 + i) for i, (nv, ne) in enumerate(frag_specs)
    ])


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
