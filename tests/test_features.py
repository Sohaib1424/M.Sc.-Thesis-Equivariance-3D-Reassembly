"""
Batch construction: the perturbed copy made on the device, the world-unit
fields the assembly needs, the repair counts, and the scene seed.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from reassembly.data.features import build_scene, collate, perturb_on_device  # noqa: E402
from reassembly.data.transforms import random_rotations  # noqa: E402

DTYPE = torch.float64


def _meshes(seed: int = 0, count: int = 3):
    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(count):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0 - 0.2 * i)
        v = np.asarray(mesh.vertices) * (1 + 0.1 * rng.standard_normal((len(mesh.vertices), 1)))
        vertices.append(v + [2.0 * i, 0.3 * i, -0.5 * i])
        faces.append(np.asarray(mesh.faces))
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), 24, replace=False)] = True
        masks.append(mask)
    return vertices, faces, masks


def _pair(seed: int = 0, **kwargs):
    """The same scene built both ways, from the same rotations and seeds."""
    v, f, m = _meshes(seed)
    rotations = random_rotations(len(v), np.random.default_rng(seed + 7))
    cluster = np.full(sum(len(x) for x in v), -1, np.int64)
    cluster[:30] = np.arange(30) // 3
    common = dict(rotations=rotations, tokens_per_scene=32, cluster=cluster, **kwargs)
    shipped = build_scene(v, f, m, rng=np.random.default_rng(seed), perturb=True, **common)
    derived = build_scene(v, f, m, rng=np.random.default_rng(seed), perturb=False, **common)
    return shipped, derived


def test_the_device_path_builds_the_same_batch():
    """
    ``perturb=False`` ships only the assembled copy; ``perturb_on_device``
    rotates it into the perturbed one. Exact up to round-off: the worker made
    the perturbed copy by rotating too, just in numpy and float64.
    """
    shipped, derived = _pair()
    a = collate([shipped], dtype=DTYPE)
    b = collate([derived], dtype=DTYPE)
    assert b.node_features is None and b.edge_attr is None
    b = perturb_on_device(b)
    assert torch.allclose(a.node_features, b.node_features, atol=1e-12)
    assert torch.allclose(a.edge_attr, b.edge_attr, atol=1e-12)
    for name in ("edge_index", "token_index", "token_query", "token_key",
                 "vertex_fragment", "cluster"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for name in ("target_rotation", "target_vertices", "target_normals",
                 "target_edge_normals", "log_scale", "unit", "centroid"):
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=1e-12), name
    assert perturb_on_device(a) is a, "a batch that has its copy is left alone"


def test_the_device_path_gives_the_same_loss():
    from reassembly.training import Config, _forward, build_criterion, build_model

    config = Config(channels=16, heads=4, head_dim=4, embedding_dim=8,
                    schedule=("intra", "cross", "intra"))
    torch.manual_seed(0)
    model = build_model(config)
    shipped, derived = _pair(3)
    loss_a, report_a, _ = _forward(model, collate([shipped]), build_criterion(config), config)
    loss_b, report_b, _ = _forward(model, collate([derived]), build_criterion(config), config)
    assert float(loss_a.detach()) == pytest.approx(float(loss_b.detach()), rel=1e-5)
    for name in ("rotation", "position", "normal", "face"):
        assert report_a[name] == pytest.approx(report_b[name], rel=1e-5, abs=1e-6), name


def test_world_units_can_be_recovered_from_the_batch():
    """
    The assembly is scored in world units -- the part-accuracy threshold is
    0.01 of them -- so the batch must carry what undoes the normalisation.
    """
    v, f, m = _meshes(1)
    for mode in ("scene", "fragment"):
        sample = build_scene(v, f, m, rotations=random_rotations(3, np.random.default_rng(1)),
                             normalize_mode=mode, tokens_per_scene=16)
        batch = collate([sample], dtype=DTYPE)
        world = (batch.target_vertices * batch.unit[batch.vertex_fragment, None]
                 + batch.centroid[batch.vertex_fragment])
        assert torch.allclose(world, torch.as_tensor(np.concatenate(v)), atol=1e-12), mode
        if mode == "scene":
            assert torch.allclose(batch.unit, batch.unit[:1].expand_as(batch.unit))


def test_the_fracture_mask_rides_along_aligned_with_the_vertices():
    v, f, m = _meshes(2)
    batch = collate([build_scene(v, f, m, rotations=random_rotations(3, np.random.default_rng(2)),
                                 tokens_per_scene=16)])
    assert batch.fracture.dtype == torch.bool
    assert torch.equal(batch.fracture, torch.as_tensor(np.concatenate(m)))


def test_repairs_are_counted_not_hidden():
    """
    Zero-area faces get a zero normal and so do vertices no face reaches.
    Nothing is NaN -- which is exactly why the count has to be reported: a
    repaired zero trains silently.
    """
    v, f, m = _meshes(4)
    clean = build_scene(v, f, m, rotations=random_rotations(3, np.random.default_rng(4)),
                        tokens_per_scene=16, key="obj/clean")
    assert clean.repairs.total == 0 and clean.key == "obj/clean"

    v0 = np.vstack([v[0], v[0][:1] + 5.0])               # an unreferenced vertex
    f0 = np.vstack([f[0], [[0, 0, 1]]])                  # a zero-area face
    m0 = np.concatenate([m[0], [False]])
    broken = build_scene([v0] + v[1:], [f0] + f[1:], [m0] + m[1:],
                         rotations=random_rotations(3, np.random.default_rng(4)),
                         tokens_per_scene=16)
    assert broken.repairs.degenerate_faces == 1
    assert broken.repairs.zero_normals == 1
    batch = collate([broken])
    assert batch.repairs[0].degenerate_faces == 1
    assert torch.isfinite(batch.target_normals).all()
    assert torch.isfinite(perturb_on_device(batch).edge_attr).all()


def test_the_scene_seed_is_the_same_in_every_interpreter():
    """
    The seed used to be ``hash((key, epoch))``, and Python salts string
    hashes per interpreter -- so every session and every GPU process drew a
    different perturbation for the same scene, validation included.
    """
    from reassembly.training import stable_seed

    here = stable_seed("Bottle/bottle_3/fractured_12", 0, 0)
    code = ("import sys; sys.path.insert(0, 'src'); "
            "from reassembly.training import stable_seed; "
            "print(stable_seed('Bottle/bottle_3/fractured_12', 0, 0))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for salt in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code], cwd=root, check=True,
                             capture_output=True, text=True,
                             env={**os.environ, "PYTHONHASHSEED": salt})
        assert int(out.stdout.strip()) == here
    assert stable_seed("a", 1, 0) != stable_seed("a", 2, 0)
    assert stable_seed("a", 1, 0) != stable_seed("a", 1, 1)
