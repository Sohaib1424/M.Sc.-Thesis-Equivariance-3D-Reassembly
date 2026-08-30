"""
The whole pipeline, from meshes to a loss with gradients.

The unit tests prove each layer is equivariant. These prove the *assembly* is:
that the rotation label matches the perturbation, that the head's transpose
points the right way, that every parameter receives gradient, and that a batch
of two scenes gives each scene exactly what it would have got alone.

The transpose test is the one worth reading. `R_pred = frame.T` is the single
easiest thing in this codebase to get backwards, and getting it backwards is
invisible at initialisation -- chance is chance in either direction -- so it
would surface only as a model that never converges, weeks later.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from reassembly.data.features import build_scene, clustered_vertices, collate
from reassembly.data.transforms import random_rotations
from reassembly.nn.losses import ReassemblyLoss, geodesic_angle
from reassembly.nn.model import ReassemblyNet, apply_rotation

DTYPE = torch.float64


def _fragments(seed: int, count: int = 3):
    """A scene of spheres at different scales, with a random fracture mask."""
    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(count):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0 - 0.2 * i)
        mesh.apply_translation([2.0 * i, 0.0, 0.0])
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), 24, replace=False)] = True
        vertices.append(v)
        faces.append(f)
        masks.append(mask)
    return vertices, faces, masks


def _sample(seed: int, count: int = 3, rotations=None, **kwargs):
    v, f, m = _fragments(seed, count)
    rng = np.random.default_rng(seed + 100)
    return build_scene(v, f, m, rotations=rotations, rng=rng,
                       max_tokens_per_fragment=8, **kwargs)


def _net(**kwargs):
    torch.manual_seed(0)
    return ReassemblyNet(channels=16, heads=4, **kwargs).to(DTYPE)


def _forward(net, batch):
    return net(batch.node_features, batch.edge_index, batch.edge_attr,
               batch.vertex_fragment, batch.num_fragments,
               log_scale=batch.log_scale, token_index=batch.token_index,
               token_query=batch.token_query, token_key=batch.token_key)


# --------------------------------------------------------------------------
# The rotation convention
# --------------------------------------------------------------------------

def test_the_label_undoes_the_perturbation():
    """
    `v_perturbed @ R_label.T == v_assembled`, exactly.

    This is the definition of the target, and it has to survive normalisation:
    centring removes the perturbation's translation, and the per-scene divisor
    is a radius, which a rotation does not change. So both copies share one
    divisor and the relation holds in normalised coordinates too.
    """
    batch = collate([_sample(1)], dtype=DTYPE)
    perturbed = batch.node_features[:, 0, :]
    recovered = apply_rotation(perturbed, batch.target_rotation, batch.vertex_fragment)
    assert torch.allclose(recovered, batch.target_vertices, atol=1e-12)


def test_the_label_is_the_transpose_of_the_applied_rotation():
    rotations = random_rotations(3, np.random.default_rng(5))
    batch = collate([_sample(2, rotations=rotations)], dtype=DTYPE)
    expected = torch.as_tensor(rotations.transpose(0, 2, 1), dtype=DTYPE)
    assert torch.allclose(batch.target_rotation, expected, atol=1e-12)


def test_the_head_returns_a_rotation_that_points_the_right_way():
    """
    The frame is equivariant, `M(x Q^T) = Q M(x)`, so the rotation that undoes
    the perturbation is `M^T` -- and the network's job reduces to mapping an
    already-assembled fragment to the identity frame. Applying the head's
    output to a perturbed fragment must therefore land it at `v M_assembled`,
    which is the assembled pose exactly when the network has learned that.
    """
    net = _net()
    rotations = random_rotations(3, np.random.default_rng(9))
    identity = np.repeat(np.eye(3)[None], 3, axis=0)

    assembled = collate([_sample(3, rotations=identity)], dtype=DTYPE)
    perturbed = collate([_sample(3, rotations=rotations)], dtype=DTYPE)

    Q = torch.as_tensor(rotations, dtype=DTYPE)
    frame_assembled = _forward(net, assembled).frame
    prediction = _forward(net, perturbed)

    # The frame co-rotates with the input.
    assert torch.allclose(prediction.frame, Q @ frame_assembled, atol=1e-9)

    # And the predicted rotation carries the perturbed coordinates onto the
    # assembled ones, up to the learned frame.
    landed = apply_rotation(perturbed.node_features[:, 0, :],
                            prediction.rotation, perturbed.vertex_fragment)
    target = apply_rotation(assembled.node_features[:, 0, :],
                            frame_assembled.transpose(-1, -2), assembled.vertex_fragment)
    assert torch.allclose(landed, target, atol=1e-9)


def test_a_perfect_frame_gives_zero_rotation_loss():
    """Closing the loop: if the network output the identity frame on assembled
    fragments, the geodesic loss on perturbed ones would be exactly zero."""
    rotations = random_rotations(4, np.random.default_rng(11))
    batch = collate([_sample(4, count=4, rotations=rotations)], dtype=DTYPE)
    # A hypothetical perfect model: frame = Q, so rotation = Q^T.
    perfect = torch.as_tensor(rotations.transpose(0, 2, 1), dtype=DTYPE)
    assert geodesic_angle(perfect, batch.target_rotation).max().item() < 1e-12


# --------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------

def test_forward_produces_proper_rotations():
    batch = collate([_sample(1), _sample(2)], dtype=DTYPE)
    prediction = _forward(_net(), batch)

    assert prediction.rotation.shape == (batch.num_fragments, 3, 3)
    identity = torch.eye(3, dtype=DTYPE).expand_as(prediction.rotation)
    R = prediction.rotation
    assert torch.allclose(R.transpose(-1, -2) @ R, identity, atol=1e-10)
    assert torch.allclose(torch.det(R), torch.ones(batch.num_fragments, dtype=DTYPE),
                          atol=1e-10)
    assert torch.isfinite(prediction.vertex_embedding).all()


def test_the_whole_model_is_equivariant_per_fragment():
    """
    The end-to-end claim. Re-posing one fragment rotates its own prediction and
    leaves the others alone -- which is what makes the per-fragment label
    learnable at all. Anything that sends a *vector* between fragments breaks
    the second half while still training.
    """
    net = _net()
    base_rotations = random_rotations(3, np.random.default_rng(21))
    extra = random_rotations(1, np.random.default_rng(22))[0]

    changed = base_rotations.copy()
    changed[0] = extra @ base_rotations[0]

    a = _forward(net, collate([_sample(6, rotations=base_rotations)], dtype=DTYPE))
    b = _forward(net, collate([_sample(6, rotations=changed)], dtype=DTYPE))

    E = torch.as_tensor(extra, dtype=DTYPE)
    assert torch.allclose(b.frame[0], E @ a.frame[0], atol=1e-9)
    assert torch.allclose(b.frame[1:], a.frame[1:], atol=1e-9)


def test_batching_does_not_change_a_scene():
    """
    A scene must predict the same thing alone and in company. The failure mode
    is cross-scene attention leakage, which makes the prediction depend on what
    else happened to land in the batch.
    """
    net = _net()
    alone = _forward(net, collate([_sample(7)], dtype=DTYPE))
    together = _forward(net, collate([_sample(7), _sample(8)], dtype=DTYPE))
    assert torch.allclose(together.frame[:3], alone.frame, atol=1e-9)


def test_a_scene_with_no_tokens_still_predicts():
    """
    Coincidence labelling reports fragments with zero fracture vertices, and a
    single-fragment mode has no cross-fragment pairs at all. Neither may crash
    or produce NaN -- the intra-fragment stream alone still has to give an
    answer.
    """
    vertices, faces, _ = _fragments(9, count=2)
    empty = [np.zeros(len(v), bool) for v in vertices]
    batch = collate([build_scene(vertices, faces, empty,
                                 rng=np.random.default_rng(0))], dtype=DTYPE)
    assert batch.token_index.numel() == 0
    prediction = _forward(_net(), batch)
    assert torch.isfinite(prediction.rotation).all()


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------

def test_every_parameter_receives_gradient():
    """
    A layer initialised to exactly the identity by zeroing its final weight has
    no gradient into anything upstream of it. That silently disabled the entire
    cross-fragment pathway -- the architecture's whole contribution -- for the
    first optimiser steps, and nothing about the loss curve would have shown it.
    """
    net = _net()
    batch = collate([_sample(1), _sample(2)], dtype=DTYPE)
    prediction = _forward(net, batch)

    keep, cluster, count = clustered_vertices(batch.vertex_fragment)  # stand-in clusters
    total, _ = ReassemblyLoss()(
        prediction.rotation, batch.target_rotation,
        vertices=apply_rotation(batch.node_features[:, 0, :], prediction.rotation,
                                batch.vertex_fragment),
        target_vertices=batch.target_vertices,
        normals=apply_rotation(batch.node_features[:, 1, :], prediction.rotation,
                               batch.vertex_fragment),
        target_normals=batch.target_normals,
        vertex_batch=batch.vertex_fragment,
        embeddings=prediction.vertex_embedding[keep], cluster=cluster,
        num_clusters=count,
    )
    total.backward()

    dead = [name for name, p in net.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reaches: {dead}"
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_loss_at_initialisation_is_near_chance():
    """
    An untrained model should read chance, and a term that does not is
    measuring something other than what its name says. The tolerance is wide
    because a handful of fragments is a small sample -- this catches a term
    that is wrong by a factor, not one that is off by noise.
    """
    net = _net()
    batch = collate([_sample(s, count=4) for s in range(12)], dtype=DTYPE)
    prediction = _forward(net, batch)
    _, report = ReassemblyLoss()(
        prediction.rotation, batch.target_rotation,
        normals=apply_rotation(batch.node_features[:, 1, :], prediction.rotation,
                               batch.vertex_fragment),
        target_normals=batch.target_normals,
        vertex_batch=batch.vertex_fragment,
    )
    assert 90.0 < report["rotation_degrees"] < 160.0     # chance 126.5
    assert 0.6 < report["normal"] < 1.4                  # chance 1.0


# --------------------------------------------------------------------------
# The batch itself
# --------------------------------------------------------------------------

def test_collate_shifts_every_index():
    batch = collate([_sample(1), _sample(2, count=4)], dtype=DTYPE)
    n = batch.node_features.shape[0]
    assert int(batch.edge_index.max()) < n
    assert int(batch.token_index.max()) < n
    assert batch.num_fragments == 3 + 4
    assert batch.fragment_scene.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert batch.vertex_ptr[-1].item() == n
    assert batch.fragment_ptr.tolist() == [0, 3, 7]


def test_collate_keeps_fragments_contiguous():
    """
    Sorted by (scene, fragment), so each fragment's vertices form one block.
    Variable-length attention kernels need that, and so does `ptr`.
    """
    batch = collate([_sample(1), _sample(2)], dtype=DTYPE)
    fragment = batch.vertex_fragment
    assert (fragment[1:] >= fragment[:-1]).all(), "vertices must be grouped by fragment"
    assert batch.vertex_ptr.tolist() == (
        [0] + torch.cumsum(torch.bincount(fragment), 0).tolist()
    )


def test_cross_fragment_pairs_never_cross_a_scene():
    batch = collate([_sample(1), _sample(2, count=4)], dtype=DTYPE)
    scene_of_token = batch.fragment_scene[batch.vertex_fragment[batch.token_index]]
    fragment_of_token = batch.vertex_fragment[batch.token_index]
    assert (scene_of_token[batch.token_query] == scene_of_token[batch.token_key]).all()
    assert (fragment_of_token[batch.token_query]
            != fragment_of_token[batch.token_key]).all()


def test_both_copies_share_one_topology():
    """
    The losses match vertices, edges and normals *by index*. Two independently
    built copies would compare unrelated rows without raising anything --
    `resolve_duplicated_faces` reorders faces lexicographically, so it really
    can happen.
    """
    sample = _sample(1)
    for fragment in sample.fragments:
        assert fragment.vertices.shape == fragment.target_vertices.shape
        assert fragment.normals.shape == fragment.target_normals.shape
        assert fragment.edge_normals.shape == fragment.target_edge_normals.shape


def test_log_scale_is_the_log_of_the_world_radius():
    """
    Log, not raw. Fragments span 4 to 83,039 vertices, so raw scale is
    heavy-tailed enough to dominate whatever it is concatenated with; GARF
    applies a positional encoding for the same reason.
    """
    sample = _sample(1)
    batch = collate([sample], dtype=DTYPE)
    expected = torch.tensor([[np.log(f.radius)] for f in sample.fragments], dtype=DTYPE)
    assert torch.allclose(batch.log_scale, expected, atol=1e-12)


def test_vertex_mode_keeps_every_masked_vertex_as_a_token():
    """`vertex` is the unreduced baseline: every fracture vertex is a token."""
    vertices, faces, masks = _fragments(11, count=2)
    sample = build_scene(vertices, faces, masks, rng=np.random.default_rng(0),
                         token_mode="vertex", tokens_per_scene=None,
                         max_tokens_per_fragment=None)
    for fragment, mask in zip(sample.fragments, masks):
        assert fragment.token_vertices.size == int(mask.sum())
        assert mask[fragment.token_vertices].all()


def test_sample_mode_selects_real_vertices_within_budget():
    vertices, faces, masks = _fragments(12, count=2)
    sample = build_scene(vertices, faces, masks, rng=np.random.default_rng(0),
                         token_mode="sample", tokens_per_scene=None,
                         max_tokens_per_fragment=6)
    for fragment, mask in zip(sample.fragments, masks):
        assert fragment.token_vertices.size <= 6
        assert mask[fragment.token_vertices].all(), "a token must be a fracture vertex"


def test_patch_mode_is_rejected_rather_than_silently_degraded():
    """
    `patch` tokens are centroids, which are not mesh vertices and so cannot be
    a vertex index. Falling back would ignore the budget entirely and emit one
    token per fracture vertex while looking like it had worked.
    """
    vertices, faces, masks = _fragments(13, count=2)
    with pytest.raises(ValueError, match="patch"):
        build_scene(vertices, faces, masks, rng=np.random.default_rng(0),
                    token_mode="patch")


# --------------------------------------------------------------------------
# The token budget -- a fixed number per scene
# --------------------------------------------------------------------------

def _budget_scene(count: int, seed: int = 0, mask_size: int = 200):
    """Fragments large enough that the budget, not the mesh, is what binds."""
    rng = np.random.default_rng(seed)
    vertices, faces, masks = [], [], []
    for i in range(count):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0 + 0.1 * i)
        mesh.apply_translation([3.0 * i, 0.0, 0.0])
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        mask = np.zeros(len(v), bool)
        mask[rng.choice(len(v), mask_size, replace=False)] = True
        vertices.append(v)
        faces.append(f)
        masks.append(mask)
    return vertices, faces, masks


def _token_counts(count, total, cap, seed=0):
    v, f, m = _budget_scene(count, seed)
    sample = build_scene(v, f, m, rng=np.random.default_rng(seed),
                         tokens_per_scene=total, max_tokens_per_fragment=cap)
    return [fragment.token_vertices.size for fragment in sample.fragments]


def test_the_scene_budget_is_a_ceiling_on_the_whole_scene():
    """
    The number that makes the inter-fragment graph a property of the
    configuration rather than of whatever mesh arrived. Add fragments and the
    scene's token count stops growing.
    """
    for count in (8, 20, 40):
        assert sum(_token_counts(count, 512, 128)) == 512


def test_a_big_fragment_draws_more_tokens_than_a_small_one():
    """
    The head-statue case, and the reason there is no per-fragment cap by
    default: a fragment carrying several mating surfaces needs more tokens to
    describe them, and the area weighting gives it exactly that. A cap would
    hand it the same count as the smallest chip in the scene.
    """
    counts = _token_counts(20, 512, None)
    assert counts[-1] > counts[0], "larger fracture surfaces should draw more"
    assert max(counts) > 2 * min(counts), "the split should be genuinely uneven"
    assert min(counts) >= 1, "and nothing gets zero"


def test_the_scene_total_alone_bounds_the_graph():
    """
    No cap needed for cost. For a budget T the undirected pair count is
    `(T^2 - sum t_i^2)/2 < T^2/2` for any split at all, so the total is a
    complete bound on its own -- while a cap alone bounds nothing.
    """
    for count in (8, 20, 40):
        tokens = _token_counts(count, 512, None)
        assert sum(tokens) == 512
        pairs = (sum(tokens) ** 2 - sum(t * t for t in tokens)) / 2
        assert pairs < 512 ** 2 / 2

    uncapped_no_total = sum(_token_counts(40, None, 128))
    assert uncapped_no_total > 4 * 512, "a cap alone does not bound the scene"


def test_sampling_is_per_fragment_and_survives_perturbation():
    """
    Tokens are chosen inside each fragment, from its own shape, so the same
    scene under two different perturbations selects the same vertices. One
    farthest-point pass over the pooled scene would not: at input time the
    pieces are wherever they were thrown, so the sample would change with the
    perturbation and the equivariance would be gone.
    """
    vertices, faces, masks = _budget_scene(4, seed=3)
    a = build_scene(vertices, faces, masks,
                    rotations=np.repeat(np.eye(3)[None], 4, axis=0))
    b = build_scene(vertices, faces, masks,
                    rotations=random_rotations(4, np.random.default_rng(17)))
    for x, y in zip(a.fragments, b.fragments):
        assert np.array_equal(x.token_vertices, y.token_vertices)


def test_tokens_come_only_from_the_fracture_mask():
    """
    The mask gates the *token pool*, not the graph: all vertices stay nodes with
    full features and all mesh edges stay edges. It decides only which vertices
    may be sampled as cross-fragment tokens.
    """
    vertices, faces, masks = _budget_scene(3, seed=5)
    sample = build_scene(vertices, faces, masks, rng=np.random.default_rng(0))
    for fragment, mask, v in zip(sample.fragments, masks, vertices):
        assert mask[fragment.token_vertices].all()
        assert fragment.vertices.shape[0] == len(v), "every vertex stays a node"


def test_mesh_edges_never_cross_a_fragment_boundary():
    """
    A guard on an assumption the equivariance argument silently depends on.

    Intra-fragment attention aggregates along `edge_index`. If one edge joined
    two fragments, that layer would mix them, and re-posing one fragment would
    move another fragment's prediction -- per-fragment equivariance gone, with
    nothing raising. The collate builds `edge_index` per fragment and offsets
    it, so it cannot happen; this asserts the property rather than the code
    path, so a future refactor cannot quietly lose it.
    """
    batch = collate([_sample(1), _sample(2, count=4)], dtype=DTYPE)
    source = batch.vertex_fragment[batch.edge_index[0]]
    destination = batch.vertex_fragment[batch.edge_index[1]]
    assert torch.equal(source, destination), "a mesh edge joined two fragments"


def test_a_cross_fragment_mesh_edge_would_break_equivariance():
    """
    Why the guard above is not paranoia -- the failure demonstrated rather than
    asserted. One edge joining two fragments is enough to make fragment 1's
    prediction depend on fragment 0's pose.
    """
    net = _net(schedule=("intra",))
    n, fragment = 24, torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    within = [[f * 8 + i, f * 8 + (i + 1) % 8] for f in range(3) for i in range(8)]

    def frames(edges):
        torch.manual_seed(0)
        edge_index = torch.tensor(edges).T
        attr = torch.zeros(edge_index.shape[1], 3, 3, dtype=DTYPE)
        x = torch.randn(n, 2, 3, generator=torch.Generator().manual_seed(4), dtype=DTYPE)
        spin = torch.eye(3, dtype=DTYPE).expand(3, 3, 3).clone()
        spin[0] = torch.as_tensor(random_rotations(1, np.random.default_rng(3))[0])
        empty = torch.empty(0, dtype=torch.long)
        plain = net(x, edge_index, attr, fragment, 3, token_index=empty,
                    token_query=empty, token_key=empty)
        turned = torch.einsum("nij,nkj->nki", spin[fragment], x)
        rotated = net(turned, edge_index, attr, fragment, 3, token_index=empty,
                      token_query=empty, token_key=empty)
        return (rotated.frame[1:] - plain.frame[1:]).abs().max().item()

    assert frames(within) < 1e-9, "fragment-local edges keep the others fixed"
    assert frames(within + [[0, 8]]) > 1e-3, (
        "a single cross-fragment edge should visibly move another fragment's "
        "prediction -- if this no longer holds, the guard above is testing nothing"
    )


def test_edges_carry_the_relative_position_of_the_neighbour():
    """
    The third edge channel: `p_source - p_destination`.

    Without it a message -- a linear map of the *source* node's features and the
    edge's -- can say what the neighbour looks like but not which way it lies.
    Every standard equivariant message-passing construction supplies it, and it
    is what the reference V-GAT/VN diagrams call `r_ij`.
    """
    batch = collate([_sample(1)], dtype=DTYPE)
    assert batch.edge_attr.shape[1] == 3, "n1, n2, delta"

    source = batch.node_features[batch.edge_index[0], 0, :]
    destination = batch.node_features[batch.edge_index[1], 0, :]
    assert torch.allclose(batch.edge_attr[:, 2, :], source - destination, atol=1e-12)


def test_the_relative_position_is_negated_on_the_reverse_edge():
    """
    Both directed copies of an undirected mesh edge exist, and the vector must
    flip between them -- otherwise half the messages would point the wrong way.
    Taking it from `edge_index` rather than from a second convention is what
    makes that automatic.
    """
    sample = _sample(1)
    for fragment in sample.fragments:
        half = fragment.edge_index.shape[1] // 2
        forward = fragment.edge_normals[:half, 2, :]
        reverse = fragment.edge_normals[half:, 2, :]
        assert np.allclose(forward, -reverse, atol=1e-12)


def test_the_relative_position_rotates_with_the_fragment():
    """It is a genuine vector feature, not a fixed anchor: it must co-rotate."""
    rotations = random_rotations(3, np.random.default_rng(31))
    identity = np.repeat(np.eye(3)[None], 3, axis=0)
    assembled = _sample(5, rotations=identity)
    perturbed = _sample(5, rotations=rotations)
    for fragment, spun, Q in zip(assembled.fragments, perturbed.fragments, rotations):
        assert np.allclose(spun.edge_normals[:, 2, :],
                           fragment.edge_normals[:, 2, :] @ Q.T, atol=1e-12)
