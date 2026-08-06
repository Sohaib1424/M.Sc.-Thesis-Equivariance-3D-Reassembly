"""Voxel-cluster decimation: validity, monotonicity, and interface survival."""
import numpy as np
import pytest

from reassembly.data.decimate import (
    decimate_arrays, suggested_correspondence_tol, voxel_cluster_ids,
)
from conftest import grid_mesh


def sphere(n_theta=24, n_phi=48, radius=1.0, centre=(0, 0, 0)):
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    V = V.reshape(-1, 3) * radius + np.asarray(centre)
    F = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = i * n_phi + j; b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j; d = (i + 1) * n_phi + (j + 1) % n_phi
            F += [[a, b, c], [b, d, c]]
    return V, np.asarray(F)


def assert_valid_mesh(V, F):
    assert np.isfinite(V).all()
    if len(F) == 0:
        return
    assert F.min() >= 0 and F.max() < len(V), "face index out of range"
    degenerate = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    assert not degenerate.any(), "degenerate face survived the collapse"
    keyed = np.sort(F, axis=1)
    assert len(np.unique(keyed, axis=0)) == len(F), "duplicate faces"
    assert len(np.unique(F)) == len(V), "unreferenced vertices left behind"


@pytest.mark.parametrize("voxel", [0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
def test_output_is_a_valid_mesh(voxel):
    V, F = sphere()
    nv, nf, vmap = decimate_arrays(V, F, V.min(0), voxel)
    assert_valid_mesh(nv, nf)
    assert vmap.shape == (len(V),)
    live = vmap >= 0
    if live.any():
        assert vmap[live].max() < len(nv)


def test_vertex_count_is_monotone_in_voxel_size():
    V, F = sphere()
    origin = V.min(0)
    counts = [len(decimate_arrays(V, F, origin, h)[0]) for h in (0.02, 0.05, 0.1, 0.2, 0.4)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_geometric_error_bounded_by_voxel_diagonal():
    from scipy.spatial import cKDTree
    V, F = sphere()
    for h in (0.05, 0.1, 0.2):
        nv, _, _ = decimate_arrays(V, F, V.min(0), h)
        d, _ = cKDTree(nv).query(V)
        assert d.max() <= h * np.sqrt(3), f"error exceeds one voxel diagonal at h={h}"


def test_shared_grid_preserves_the_interface_better_than_per_fragment_grids():
    """The reason decimate_scene anchors every fragment on ONE grid.

    Two fragments meeting on an oblique plane, with different bounding boxes so
    their own local grids are genuinely offset. On the shared grid the two
    copies of an interface vertex land on the same cell; on per-fragment grids
    they do not, and the correspondence supervision degrades.
    """
    rng = np.random.default_rng(11)
    n = 22
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    e1 = np.array([0.9, 0.3, -0.2]); e2 = np.array([-0.15, 0.8, 0.55])
    interface = np.array([0.137, -0.291, 0.443]) + u.ravel()[:, None] * e1 + v.ravel()[:, None] * e2
    nrm = np.cross(e1, e2); nrm /= np.linalg.norm(nrm)

    def fragment(side, density):
        bulk = interface + side * nrm * rng.uniform(0.05, 1.3, size=(len(interface), 1))
        bulk = bulk[rng.random(len(bulk)) < density]
        V = np.concatenate([interface, bulk], axis=0)
        idx = np.arange(len(V)); rng.shuffle(idx)
        F = np.stack([idx[:-2], idx[1:-1], idx[2:]], axis=1)
        return V, F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]

    VA, FA = fragment(+1, 0.9)
    VB, FB = fragment(-1, 0.35)
    n_iface = len(interface)
    shared_origin = np.minimum(VA.min(0), VB.min(0))

    for h in (0.05, 0.1, 0.2):
        tol = suggested_correspondence_tol(h)

        aV, _, amap = decimate_arrays(VA, FA, shared_origin, h)
        bV, _, bmap = decimate_arrays(VB, FB, shared_origin, h)
        live = (amap[:n_iface] >= 0) & (bmap[:n_iface] >= 0)
        gap_shared = np.linalg.norm(
            aV[amap[:n_iface][live]] - bV[bmap[:n_iface][live]], axis=1
        ).max()

        aV2, _, amap2 = decimate_arrays(VA, FA, VA.min(0), h)
        bV2, _, bmap2 = decimate_arrays(VB, FB, VB.min(0), h)
        live2 = (amap2[:n_iface] >= 0) & (bmap2[:n_iface] >= 0)
        gap_per = np.linalg.norm(
            aV2[amap2[:n_iface][live2]] - bV2[bmap2[:n_iface][live2]], axis=1
        ).max()

        assert gap_shared <= tol, f"shared grid lost the interface at h={h}"
        assert gap_shared <= gap_per, f"shared grid was not better at h={h}"


def test_empty_input_is_handled():
    nv, nf, vmap = decimate_arrays(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                                   np.zeros(3), 0.1)
    assert len(nv) == 0 and len(nf) == 0 and len(vmap) == 0


def test_no_faces_input_is_handled():
    V = np.random.default_rng(0).normal(size=(50, 3))
    nv, nf, vmap = decimate_arrays(V, np.zeros((0, 3), dtype=np.int64), V.min(0), 0.5)
    assert len(nf) == 0 and len(vmap) == len(V) and len(nv) > 0


def test_rejects_non_positive_voxel_size():
    with pytest.raises(ValueError):
        voxel_cluster_ids(np.zeros((3, 3)), np.zeros(3), 0.0)


def test_tolerance_scales_with_voxel_size():
    assert suggested_correspondence_tol(0.0) == pytest.approx(1e-5)
    assert suggested_correspondence_tol(0.2) > suggested_correspondence_tol(0.05)


# ---------------------------------------------------------------------------
# Regression tests for a failure found on a real 53-fragment scene.
# ---------------------------------------------------------------------------
class _Mesh:
    """Minimal mesh-like object; decimate_scene only reads .vertices/.faces."""

    def __init__(self, v, f):
        self.vertices, self.faces = np.asarray(v), np.asarray(f)


def _many_small_fragments(rng, n_frag=40, per_frag=25, spread=1.0):
    """A scene of fragments already at or below the per-fragment floor.

    This is the shape that broke the original search: every fragment is pinned
    by min_vertices_per_fragment, so no voxel size can reduce the total.
    """
    meshes = []
    for i in range(n_frag):
        V = rng.normal(size=(per_frag, 3)) * 0.05 + rng.normal(size=3) * spread
        idx = np.arange(per_frag)
        F = np.stack([idx[:-2], idx[1:-1], idx[2:]], axis=1)
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        meshes.append(_Mesh(V, F))
    return meshes


def test_unreachable_target_is_reported_not_claimed(rng):
    """The original bisection ran to its upper bound, decimated nothing, and
    still reported applied=True with a voxel size larger than the object.
    Observed on real data: 11,303 -> 11,303 vertices at voxel 0.527 on an
    object 1.0 across."""
    from reassembly.data.decimate import decimate_scene

    meshes = _many_small_fragments(rng)
    before = sum(len(m.vertices) for m in meshes)
    _out, info = decimate_scene(meshes, target_vertices=before // 10,
                                min_vertices_per_fragment=32, mesh_factory=_Mesh)

    assert not info.target_met, "unreachable target must be reported as unmet"
    if not info.reduced:
        assert info.voxel_size == 0.0, \
            "a search that decimated nothing must not report a usable voxel size"


def test_voxel_size_never_exceeds_the_object_when_reduced(rng):
    """The damaging consequence: suggested_correspondence_tol scales by
    voxel_size, so a runaway voxel makes every vertex match every other."""
    from reassembly.data.decimate import decimate_scene, scene_diagonal

    meshes = _many_small_fragments(rng)
    diag = scene_diagonal(meshes)
    for target in (500, 200, 50):
        _out, info = decimate_scene(meshes, target_vertices=target,
                                    min_vertices_per_fragment=32, mesh_factory=_Mesh)
        if info.reduced:
            assert info.voxel_size < diag * 0.5, \
                f"voxel {info.voxel_size} is a large fraction of the object ({diag})"
        else:
            assert info.voxel_size == 0.0


def test_reduced_implies_the_count_actually_moved(rng):
    from reassembly.data.decimate import decimate_scene

    meshes = _many_small_fragments(rng, n_frag=8, per_frag=400)
    before = sum(len(m.vertices) for m in meshes)
    out, info = decimate_scene(meshes, target_vertices=before // 4, mesh_factory=_Mesh)
    assert len(out) == len(meshes), "fragment count must never change"
    if info.reduced:
        assert info.vertices_after < info.vertices_before
    else:
        assert info.vertices_after == info.vertices_before
