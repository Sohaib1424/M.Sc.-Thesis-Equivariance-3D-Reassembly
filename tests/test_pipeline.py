"""Feature construction, collation, and the dataset/model bridge."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")
trimesh = pytest.importorskip("trimesh", reason="trimesh not installed")

from torch_geometric.data import Data  # noqa: E402

from reassembly.data.collate import collate_scenes, merge_fragments  # noqa: E402
from reassembly.data.features import get_features  # noqa: E402
from reassembly.training.bridge import (  # noqa: E402
    apply_rotation_per_fragment, build_model_inputs, build_predictions, build_targets,
)

pytestmark = pytest.mark.torch


def box(offset=(0, 0, 0), extent=1.0):
    m = trimesh.creation.box(extents=(extent,) * 3)
    m.apply_translation(np.asarray(offset, dtype=float))
    return m


def scene(n=3):
    return [box(offset=(i * 1.0, 0, 0)) for i in range(n)]


# ------------------------------------------------------------------ features
def test_feature_shapes_and_bidirectional_edges():
    m = box()
    g = get_features(m)
    E = len(m.edges_unique)
    assert g.x.shape == (len(m.vertices), 6)
    assert g.edge_index.shape == (2, 2 * E)
    assert g.edge_attr.shape == (2 * E, 10)
    assert g.is_forward_edge.shape == (2 * E,)
    assert int(g.is_forward_edge.sum()) == E
    assert g.inc_index.shape == (2, 4 * E)


def test_positions_are_centralized():
    g = get_features(box(offset=(5.0, -3.0, 2.0)))
    assert torch.allclose(g.x[:, 0:3].mean(0), torch.zeros(3), atol=1e-5)


def test_centroid_is_carried_through():
    offset = (5.0, -3.0, 2.0)
    g = get_features(box(offset=offset))
    assert torch.allclose(g.centroid.reshape(3), torch.tensor(offset), atol=1e-5)


def test_backward_edges_mirror_the_face_normal_slots():
    m = box()
    g = get_features(m)
    E = len(m.edges_unique)
    fwd, bwd = g.edge_attr[:E], g.edge_attr[E:]
    assert torch.allclose(fwd[:, 0:4], bwd[:, 0:4])       # length + midpoint unchanged
    assert torch.allclose(fwd[:, 4:7], bwd[:, 7:10])      # n1 <-> n2 swapped
    assert torch.allclose(fwd[:, 7:10], bwd[:, 4:7])


def test_edge_length_matches_the_geometry():
    m = box()
    g = get_features(m)
    E = len(m.edges_unique)
    u, v = m.edges_unique[:, 0], m.edges_unique[:, 1]
    expected = np.linalg.norm(m.vertices[u] - m.vertices[v], axis=1)
    assert np.allclose(g.edge_attr[:E, 0].numpy(), expected, atol=1e-5)


def test_edge_cluster_ids_are_doubled():
    m = box()
    E = len(m.edges_unique)
    ecid = np.arange(E)
    g = get_features(m, edge_cluster_ids=ecid)
    assert g.edge_cluster_id.shape == (2 * E,)
    assert torch.equal(g.edge_cluster_id[:E], g.edge_cluster_id[E:])


def test_mismatched_edge_cluster_length_is_rejected():
    with pytest.raises(ValueError):
        get_features(box(), edge_cluster_ids=np.arange(3))


# ------------------------------------------------------------------- merging
def test_merge_offsets_indices_correctly():
    meshes = scene(3)
    graphs = [get_features(m) for m in meshes]
    merged = merge_fragments(graphs)

    assert merged.x.shape[0] == sum(g.x.shape[0] for g in graphs)
    assert merged.edge_attr.shape[0] == sum(g.edge_attr.shape[0] for g in graphs)
    assert merged.num_fragments == 3
    assert int(merged.edge_index.max()) < merged.x.shape[0]
    assert int(merged.inc_index[0].max()) < merged.x.shape[0]
    assert int(merged.inc_index[1].max()) < merged.edge_attr.shape[0]


def test_merge_carries_an_explicit_fragment_count():
    """Not fragment_id.max()+1: an empty fragment never appears in
    fragment_id, and everything per-fragment downstream would misalign."""
    graphs = [get_features(m) for m in scene(3)]
    empty = Data(x=torch.zeros((0, 6)), edge_index=torch.empty((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, 10)), inc_index=torch.empty((2, 0), dtype=torch.long),
                 num_nodes=0)
    empty.is_forward_edge = torch.zeros(0, dtype=torch.bool)
    empty.centroid = torch.zeros((1, 3))
    merged = merge_fragments(graphs + [empty])
    assert merged.num_fragments == 4
    assert int(merged.fragment_id.max()) == 2


def test_edges_never_cross_a_fragment_boundary():
    """build_predictions depends on this instead of carrying a per-edge
    fragment id."""
    merged = merge_fragments([get_features(m) for m in scene(4)])
    src = merged.fragment_id[merged.edge_index[0]]
    dst = merged.fragment_id[merged.edge_index[1]]
    assert torch.equal(src, dst)


def test_is_forward_edge_is_not_a_positional_half_slice():
    """After merging, the layout is per-fragment [fwd; bwd] blocks -- code that
    slices the first half of the merged array is silently wrong for 2+
    fragments. This asserts the two differ."""
    graphs = [get_features(m) for m in scene(3)]
    merged = merge_fragments(graphs)
    total = merged.is_forward_edge.numel()
    positional = torch.zeros(total, dtype=torch.bool)
    positional[: total // 2] = True
    assert not torch.equal(merged.is_forward_edge, positional)
    assert int(merged.is_forward_edge.sum()) == total // 2


# ----------------------------------------------------------------- collation
def build_scene_graph(n_frag=3, offset=0.0):
    meshes = [box(offset=(offset + i, 0, 0)) for i in range(n_frag)]
    return merge_fragments([get_features(m) for m in meshes])


def test_collate_makes_fragment_ids_globally_unique():
    a, b = build_scene_graph(3), build_scene_graph(2, offset=50.0)
    batched = collate_scenes([a, b])
    assert batched.num_fragments == 5
    assert int(batched.fragment_id.max()) == 4
    assert batched.fragment_scene_id.tolist() == [0, 0, 0, 1, 1]


def test_collate_offsets_edges_and_incidence():
    a, b = build_scene_graph(2), build_scene_graph(2, offset=50.0)
    n_a = a.x.shape[0]
    batched = collate_scenes([a, b])
    assert int(batched.edge_index.max()) < batched.x.shape[0]
    assert int(batched.inc_index[0].max()) < batched.x.shape[0]
    assert int(batched.inc_index[1].max()) < batched.edge_attr.shape[0]
    assert int(batched.inc_index[0].max()) >= n_a          # second scene really shifted


def test_collate_offsets_cluster_ids_independently():
    def with_clusters(offset):
        meshes = [box(offset=(offset + i, 0, 0)) for i in range(2)]
        graphs = []
        for i, m in enumerate(meshes):
            vcid = np.full(len(m.vertices), -1, dtype=np.int64)
            vcid[:4] = i                                   # 2 clusters per scene
            ecid = np.full(len(m.edges_unique), -1, dtype=np.int64)
            ecid[:3] = i
            graphs.append(get_features(m, vertex_cluster_ids=vcid, edge_cluster_ids=ecid))
        return merge_fragments(graphs)

    batched = collate_scenes([with_clusters(0.0), with_clusters(50.0)])
    vcid = batched.vertex_cluster_id
    assert int(vcid.max()) == 3                            # 2 + 2, offset applied
    assert int((vcid == -1).sum()) > 0                     # -1 left untouched


def test_collate_concatenates_fragment_centroids():
    a, b = build_scene_graph(3), build_scene_graph(2, offset=50.0)
    batched = collate_scenes([a, b])
    assert batched.fragment_centroid.shape == (5, 3)
    assert batched.fragment_centroid[3, 0] > 40            # the second scene


def test_collate_of_nothing_returns_none():
    assert collate_scenes([]) is None
    assert collate_scenes([None, None]) is None


# -------------------------------------------------------------------- bridge
def rand_rot_np(seed=0):
    rng = np.random.default_rng(seed)
    u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def diffused_pair(n_frag=3, seed=0):
    meshes = scene(n_frag)
    mats, diffused = [], []
    rng = np.random.default_rng(seed)
    for i, m in enumerate(meshes):
        M = np.eye(4)
        M[:3, :3] = rand_rot_np(seed + i)
        M[:3, 3] = rng.normal(size=3) * 2
        diffused.append(m.copy().apply_transform(M))
        mats.append(M)
    clean = merge_fragments([get_features(m) for m in meshes])
    diff = merge_fragments([get_features(m) for m in diffused])
    clean.fragment_scene_id = torch.zeros(n_frag, dtype=torch.long)
    diff.fragment_scene_id = torch.zeros(n_frag, dtype=torch.long)
    t = torch.stack([torch.from_numpy(M).float() for M in mats])
    return clean, diff, t


def test_build_targets_inverts_the_diffusion_rotation():
    clean, diff, t = diffused_pair()
    targets = build_targets(clean, t, input_graph=diff)
    R_gt = targets["R_gt"]
    R_diffuse = t[:, :3, :3]
    eye = torch.eye(3).expand_as(R_gt)
    assert torch.allclose(R_gt @ R_diffuse, eye, atol=1e-5)


def test_true_rotation_reconstructs_the_clean_geometry():
    clean, diff, t = diffused_pair()
    targets = build_targets(clean, t, input_graph=diff)
    predicted = build_predictions(diff, targets["R_gt"])
    assert torch.allclose(predicted["x_pred"], targets["x_gt"], atol=1e-4)
    assert torch.allclose(predicted["n_pred"], targets["n_gt"], atol=1e-4)
    assert torch.allclose(predicted["mid_pred"], targets["mid_gt"], atol=1e-4)


def test_a_wrong_rotation_does_not_reconstruct():
    clean, diff, t = diffused_pair()
    targets = build_targets(clean, t, input_graph=diff)
    wrong = targets["R_gt"].flip(0)
    predicted = build_predictions(diff, wrong)
    assert not torch.allclose(predicted["x_pred"], targets["x_gt"], atol=1e-2)


def test_centralization_isolates_rotation_from_translation():
    """The algebraic fact R_gt depends on: centralization cancels the
    translation exactly, leaving a pure rotation."""
    clean, diff, t = diffused_pair()
    R_d = t[:, :3, :3]
    frag = clean.fragment_id
    lhs = diff.x[:, 0:3]
    rhs = torch.einsum("nij,nj->ni", R_d[frag], clean.x[:, 0:3])
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_build_model_inputs_shapes():
    _clean, diff, _t = diffused_pair()
    inputs = build_model_inputs(diff)
    assert inputs["x"].shape == (diff.x.shape[0], 2, 3)
    assert inputs["edge_scalar"].shape == (diff.edge_attr.shape[0], 1)
    assert inputs["edge_vec"].shape == (diff.edge_attr.shape[0], 3, 3)
    assert inputs["num_fragments"] == diff.num_fragments


def test_build_model_inputs_tolerates_a_missing_scene_id():
    _clean, diff, _t = diffused_pair()
    del diff.fragment_scene_id
    assert build_model_inputs(diff)["fragment_scene_id"] is None


def test_build_targets_defaults_cluster_ids_when_absent():
    clean, diff, t = diffused_pair()
    targets = build_targets(clean, t, input_graph=diff)
    assert (targets["vertex_cluster_id"] == -1).all()
    assert targets["edge_cluster_id"].numel() == int(diff.is_forward_edge.sum())


def test_apply_rotation_per_fragment():
    R = torch.stack([torch.eye(3), torch.diag(torch.tensor([-1.0, -1.0, 1.0]))])
    v = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    out = apply_rotation_per_fragment(v, R, torch.tensor([0, 1]))
    assert torch.allclose(out[0], v[0])
    assert torch.allclose(out[1], torch.tensor([-1.0, -2.0, 3.0]))
