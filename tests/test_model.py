"""
End-to-end equivariance of every layer and of the full model.

These are the tests that make the thesis's central claim checkable rather than
asserted: rotate the input, and the output must rotate the same way (or, for
the rotation head, transform the way `predict_rotation` documents).
"""
from __future__ import annotations

import pytest
import torch

from conftest import random_rotation
from vngat.models.gat_layer import VNGraphAttention, VNGraphAttentionBlock
from vngat.models.heads import InvariantEmbeddingHead, symmetric_edge_features
from vngat.models.virtual_nodes import VirtualNodeBlock
from vngat.models.vn_gat import VNGATModel
from vngat.models.vn_layers import (
    VNInvariant, VNLayerNorm, VNLeakyReLU, VNLinear, VNScaleGate,
)

TOL = 2e-4


def rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Row-vector convention: (..., C, 3) @ R^T."""
    return x @ R.T


@pytest.mark.parametrize("layer_factory", [
    lambda: VNLinear(6, 10),
    lambda: VNLeakyReLU(6),
    lambda: VNLayerNorm(6),
])
def test_primitive_layers_are_equivariant(layer_factory):
    layer = layer_factory().double()
    x = torch.randn(40, 6, 3, dtype=torch.float64)
    R = random_rotation(1, dtype=torch.float64)[0]
    lhs = rotate(layer(x), R)
    rhs = layer(rotate(x, R))
    assert torch.allclose(lhs, rhs, atol=1e-9)


def test_vn_invariant_is_invariant_and_bottlenecked():
    layer = VNInvariant(32, bottleneck=8).double()
    x = torch.randn(20, 32, 3, dtype=torch.float64)
    R = random_rotation(1, dtype=torch.float64)[0]
    assert torch.allclose(layer(x), layer(rotate(x, R)), atol=1e-9)
    assert layer.out_features == 64          # 8*8, not 32*32


def test_gat_layer_is_equivariant(scene):
    layer = VNGraphAttention(2, 3, heads=2, head_dim=4).double()
    R = random_rotation(1, dtype=torch.float64)[0]
    ei, ea = VNGATModel.symmetrise_edges(scene.edge_index, scene.edge_attr.double())
    x = scene.node_vec.double()

    lhs = rotate(layer(x, ei, ea), R)
    rhs = layer(rotate(x, R), ei, rotate(ea, R))
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_gat_layer_is_equivariant_per_fragment(scene):
    """Mesh edges never cross fragments, so independent per-fragment rotations
    must commute with the layer exactly -- the property training relies on."""
    block = VNGraphAttentionBlock(2, 3, heads=2, head_dim=4).double()
    R = random_rotation(scene.num_fragments, dtype=torch.float64)
    ei, ea = VNGATModel.symmetrise_edges(scene.edge_index, scene.edge_attr.double())
    edge_frag = scene.node_frag[ei[0]]
    x = scene.node_vec.double()
    lhs = _rotate_per_fragment(block(x, ei, ea), R, scene.node_frag)
    rhs = block(_rotate_per_fragment(x, R, scene.node_frag), ei,
                _rotate_per_fragment(ea, R, edge_frag))
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_attention_score_reads_the_edge(scene):
    """
    The previous layer's key depended on the source vertex only, so replacing
    every face normal left the attention weights EXACTLY unchanged. Here the key
    includes the edge, so the weights must move.
    """
    layer = VNGraphAttention(2, 3, heads=2, head_dim=4).double()
    ei, ea = VNGATModel.symmetrise_edges(scene.edge_index, scene.edge_attr.double())
    x = scene.node_vec.double()
    other = ea.clone()
    other[:, :2] = torch.nn.functional.normalize(torch.randn_like(other[:, :2]), dim=-1)
    with torch.no_grad():
        a = layer._scores(x, ea, ei[0], ei[1])
        b = layer._scores(x, other, ei[0], ei[1])
    assert not torch.allclose(a, b, atol=1e-6)


def test_isolated_vertex_keeps_its_own_state():
    """No self term and no residual made an isolated vertex exactly zero before."""
    layer = VNGraphAttention(4, 3, heads=2, head_dim=4).double()
    x = torch.randn(3, 4, 3, dtype=torch.float64)
    ei = torch.tensor([[0], [1]])
    ea = torch.randn(1, 3, 3, dtype=torch.float64)
    out = layer(x, ei, ea)
    assert float(out[2].detach().norm()) > 0
    block = VNGraphAttentionBlock(4, 3, heads=2, head_dim=4).double()
    assert torch.allclose(block(x, ei, ea)[2] - x[2], block.attention(x, ei, ea)[2])


def test_split_projections_keep_the_wide_layer_initialisation():
    """`W [a; b; c] = W_a a + W_b b + W_c c` is exact without a bias, and
    `fan_in` keeps each piece at the scale the single wide layer would have."""
    layer = VNGraphAttention(64, 3, heads=4, head_dim=8)
    key_bound = 1.0 / (64 + 64 + 3) ** 0.5
    value_bound = 1.0 / (64 + 3) ** 0.5
    for name in ("key_src", "key_dst", "key_edge"):
        assert float(getattr(layer, name).map.weight.detach().abs().max()) <= key_bound + 1e-7
    for name in ("value_src", "value_edge"):
        assert float(getattr(layer, name).map.weight.detach().abs().max()) <= value_bound + 1e-7


def test_score_width_is_independent_of_hidden_width():
    layer = VNGraphAttention(16, 3, heads=4, head_dim=6)
    assert layer.query.map.weight.shape == (24, 16)
    assert layer.value_src.map.weight.shape == (16, 16)


def test_scale_gate_is_equivariant_and_starts_near_identity():
    gate = VNScaleGate(8).double()
    x = torch.randn(30, 8, 3, dtype=torch.float64)
    s = torch.randn(30, 1, dtype=torch.float64)
    R = random_rotation(1, dtype=torch.float64)[0]
    assert torch.allclose(rotate(gate(x, s), R), gate(rotate(x, R), s), atol=1e-10)
    # softplus(0.5413) = 1: an untrained gate barely changes the signal ...
    assert torch.allclose(gate(x, s), x, atol=0.05 * float(x.abs().max()))
    # ... but it is not dead: the scale input reaches the output.
    assert not torch.allclose(gate(x, s), gate(x, s + 3.0), atol=1e-9)


def test_virtual_node_block_is_equivariant(scene):
    block = VirtualNodeBlock(8, num_slots=4, heads=2).double()
    x = torch.randn(scene.num_nodes, 8, 3, dtype=torch.float64)
    R = random_rotation(1, dtype=torch.float64)[0]
    args = (scene.node_frag, scene.num_fragments, scene.frag_scene)
    assert torch.allclose(rotate(block(x, *args), R), block(rotate(x, R), *args), atol=1e-8)


def _rotate_per_fragment(x, R, node_frag):
    return torch.einsum('nij,ncj->nci', R[node_frag], x)


def test_virtual_node_block_is_equivariant_per_fragment(scene):
    """
    THE property the training target depends on, and the one a global-rotation
    check cannot see.

    Diffusion rotates every fragment INDEPENDENTLY, so fragment f's output must
    be equivariant to its own rotation and invariant to the others'. A block
    that lets slots of different fragments attend to each other as VECTORS
    passes the global check above and fails this one -- and with it goes the
    whole "learn G(clean) = I once, get A^T for free" argument.
    """
    block = VirtualNodeBlock(8, num_slots=4, heads=2).double()
    x = torch.randn(scene.num_nodes, 8, 3, dtype=torch.float64)
    R = random_rotation(scene.num_fragments, dtype=torch.float64)
    args = (scene.node_frag, scene.num_fragments, scene.frag_scene)

    lhs = _rotate_per_fragment(block(x, *args), R, scene.node_frag)
    rhs = block(_rotate_per_fragment(x, R, scene.node_frag), *args)
    assert torch.allclose(lhs, rhs, atol=1e-8)


def test_rotating_one_fragment_leaves_the_others_alone(scene):
    """The invariance half of the same property, stated directly."""
    block = VirtualNodeBlock(8, num_slots=4, heads=2).double().eval()
    x = torch.randn(scene.num_nodes, 8, 3, dtype=torch.float64)
    args = (scene.node_frag, scene.num_fragments, scene.frag_scene)

    R = torch.eye(3, dtype=torch.float64).expand(scene.num_fragments, 3, 3).clone()
    R[-1] = random_rotation(1, dtype=torch.float64)[0]
    others = scene.node_frag < scene.num_fragments - 1

    with torch.no_grad():
        base = block(x, *args)
        after = block(_rotate_per_fragment(x, R, scene.node_frag), *args)
    assert torch.allclose(base[others], after[others], atol=1e-9)


def test_cross_fragment_information_still_flows(scene):
    """
    Guards the fix from being 'achieved' by cutting the connection entirely:
    perturbing one fragment's SHAPE must still change another's output, even
    though rotating it must not.
    """
    block = VirtualNodeBlock(8, num_slots=4, heads=2).double().eval()
    x = torch.randn(scene.num_nodes, 8, 3, dtype=torch.float64)
    args = (scene.node_frag, scene.num_fragments, scene.frag_scene)
    perturbed = x.clone()
    perturbed[scene.node_frag == scene.num_fragments - 1] += 3.0

    with torch.no_grad():
        base = block(x, *args)
        after = block(perturbed, *args)
    first = scene.node_frag == 0
    assert not torch.allclose(base[first], after[first], atol=1e-6)


def _model_kwargs(g, node_vec=None, edge_attr=None, dtype=torch.float64):
    return dict(
        node_vec=(g.node_vec if node_vec is None else node_vec).to(dtype),
        edge_index=g.edge_index,
        edge_attr=(g.edge_attr if edge_attr is None else edge_attr).to(dtype),
        node_frag=g.node_frag, num_fragments=g.num_fragments,
        frag_scene=g.frag_scene, frag_log_scale=g.frag_log_scale.to(dtype),
    )


def test_full_model_is_equivariant_per_fragment(scene):
    """End-to-end: R_pred_f(A_f x_f) == R_pred_f(x) A_f^T for independent A_f."""
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4, gram_bottleneck=4).double().eval()
    R = random_rotation(scene.num_fragments, dtype=torch.float64)
    with torch.no_grad():
        base = model(**_model_kwargs(scene))
        rotated = model(**_model_kwargs(
            scene,
            _rotate_per_fragment(scene.node_vec.double(), R, scene.node_frag),
            _rotate_per_fragment(scene.edge_attr.double(), R, scene.edge_frag),
        ))
    assert torch.allclose(rotated["R_pred"], base["R_pred"] @ R.transpose(-1, -2), atol=TOL)
    assert torch.allclose(rotated["vertex_embedding"], base["vertex_embedding"], atol=TOL)
    assert torch.allclose(rotated["edge_embedding"], base["edge_embedding"], atol=TOL)


def test_fragment_size_reaches_the_prediction(scene):
    """The per-scene normalisation removes absolute size from the coordinates;
    the scale gate is how it gets back in. It must actually get in."""
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4, gram_bottleneck=4).double().eval()
    with torch.no_grad():
        # make the gate non-trivial, as training would
        torch.nn.init.normal_(model.scale_gate.mlp[-1].weight, std=0.5)
        a = model(**_model_kwargs(scene))["node_features"]
        kwargs = _model_kwargs(scene)
        kwargs["frag_log_scale"] = kwargs["frag_log_scale"] + 2.0
        b = model(**kwargs)["node_features"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_virtual_node_block_does_not_leak_across_scenes(batch):
    """
    Perturbing scene 1 must leave scene 0's output untouched. Without
    scene-scoped masking in stage 2 it does not, and the model's prediction for
    one object then depends on what else happened to be in the batch.
    """
    block = VirtualNodeBlock(8, num_slots=4, heads=2).double().eval()
    x = torch.randn(batch.num_nodes, 8, 3, dtype=torch.float64)
    args = (batch.node_frag, batch.num_fragments, batch.frag_scene)

    scene_of_node = batch.frag_scene[batch.node_frag]
    mask = scene_of_node == 1
    perturbed = x.clone()
    perturbed[mask] += 5.0

    with torch.no_grad():
        base = block(x, *args)
        after = block(perturbed, *args)
    unaffected = scene_of_node == 0
    assert torch.allclose(base[unaffected], after[unaffected], atol=1e-9)
    assert not torch.allclose(base[mask], after[mask], atol=1e-6)


def test_edge_features_are_symmetric_in_endpoints(scene):
    """The edge embedding must not depend on which endpoint is stored first:
    reversing an edge swaps n1/n2 and negates the relative position."""
    h = torch.randn(scene.num_nodes, 5, 3)
    ei, ea = VNGATModel.symmetrise_edges(scene.edge_index, scene.edge_attr)
    E = scene.num_edges
    forward = symmetric_edge_features(h, ei[:, :E], ea[:E])
    reverse = symmetric_edge_features(h, ei[:, E:], ea[E:])
    assert torch.allclose(forward, reverse, atol=1e-6)
    assert forward.shape[1] == 5 + 1


def test_embedding_head_is_invariant():
    head = InvariantEmbeddingHead(12, 6, bottleneck=6).double()
    x = torch.randn(30, 12, 3, dtype=torch.float64)
    R = random_rotation(1, dtype=torch.float64)[0]
    assert torch.allclose(head(x), head(rotate(x, R)), atol=1e-8)


def test_full_model_rotation_output_is_right_equivariant(scene):
    """
    R_pred(A x) == R_pred(x) A^T. Combined with the training target R_gt = A^T,
    this is what makes a single learned canonicalisation generalise to every
    orientation.
    """
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4, gram_bottleneck=4).double().eval()
    R = random_rotation(1, dtype=torch.float64)[0]
    with torch.no_grad():
        base = model(**_model_kwargs(scene))
        rotated = model(**_model_kwargs(scene, rotate(scene.node_vec.double(), R),
                                        rotate(scene.edge_attr.double(), R)))
    assert torch.allclose(rotated["R_pred"], base["R_pred"] @ R.T, atol=TOL)
    assert torch.allclose(rotated["vertex_embedding"], base["vertex_embedding"], atol=TOL)
    assert torch.allclose(rotated["edge_embedding"], base["edge_embedding"], atol=TOL)
    assert torch.allclose(rotate(base["node_features"], R), rotated["node_features"], atol=TOL)


def test_model_defaults_to_four_layers():
    model = VNGATModel()
    assert model.num_layers == 4
    assert len(model.mesh_layers) == 4 and len(model.vn_blocks) == 4


def test_model_rejects_indivisible_head_split():
    with pytest.raises(ValueError, match="divisible"):
        VNGATModel(hidden_channels=10, heads=4)


def test_symmetrise_edges_layout(scene):
    ei, ea = VNGATModel.symmetrise_edges(scene.edge_index, scene.edge_attr)
    E = scene.num_edges
    assert ei.shape[1] == 2 * E
    assert torch.equal(ei[:, :E], scene.edge_index)              # forward block first
    assert torch.equal(ei[:, E:], scene.edge_index.flip(0))      # then the reverse block
    assert torch.equal(ea[:E], scene.edge_attr)
    assert torch.equal(ea[E:, 0], scene.edge_attr[:, 1])         # face normals swapped
    assert torch.equal(ea[E:, 1], scene.edge_attr[:, 0])
    assert torch.equal(ea[E:, 2], -scene.edge_attr[:, 2])        # relative position negated
    # and it is still "source minus destination" on the reverse copy
    p = scene.node_vec[:, 0]
    assert torch.allclose(ea[:, 2], p[ei[0]] - p[ei[1]], atol=1e-6)


def test_variable_fragment_and_vertex_counts_run(batch):
    """The user-facing requirement: arbitrary batches -- different objects,
    different fragment counts, different vertex counts per fragment."""
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4)
    out = model(**_model_kwargs(batch, dtype=torch.float32))
    assert out["R_pred"].shape == (batch.num_fragments, 3, 3)
    assert out["vertex_embedding"].shape[0] == batch.num_nodes
    assert out["edge_embedding"].shape[0] == batch.num_edges
    eye = torch.eye(3).expand_as(out["R_pred"])
    assert torch.allclose(out["R_pred"] @ out["R_pred"].transpose(-1, -2), eye, atol=1e-4)


def test_gradient_checkpointing_matches_plain_forward(scene):
    model = VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                       head_dim=4, embed_dim=4)
    kwargs = _model_kwargs(scene, dtype=torch.float32)
    model.train()
    model.grad_checkpointing = False
    a = model(**kwargs)["R_pred"]
    model.grad_checkpointing = True
    b = model(**kwargs)["R_pred"]
    assert torch.allclose(a, b, atol=1e-5)


# --------------------------------------------------------------------------
# Half-precision overflow
# --------------------------------------------------------------------------
def test_attention_scores_survive_large_activations():
    """
    THE regression guard for the NaN that stopped real training runs.

    Autocast puts `.sum` on its float32 list but NOT the elementwise product
    feeding it, so `q * k` ran in float16 and overflowed to inf once both
    operands passed ~256 (256^2 > 65504). `segment_softmax` then subtracts the
    per-segment max for stability, and inf - inf = NaN -- after which R_pred and
    both embedding heads are NaN together, which is exactly the signature seen
    in training (all nine loss terms failing at once).
    """
    from vngat.models.segment_ops import at_least_float32

    big = torch.full((64, 48), 400.0, dtype=torch.float16)
    naive = (big * big).sum(-1)
    assert torch.isinf(naive).any(), "expected fp16 overflow in the naive product"

    safe = (at_least_float32(big) * at_least_float32(big)).sum(-1)
    assert torch.isfinite(safe).all()


def test_at_least_float32_preserves_double():
    """An unconditional `.float()` would downcast float64 and silently drop the
    equivariance tests' residual from ~1e-15 to ~1e-8."""
    from vngat.models.segment_ops import at_least_float32

    assert at_least_float32(torch.zeros(2, dtype=torch.float16)).dtype == torch.float32
    assert at_least_float32(torch.zeros(2, dtype=torch.float32)).dtype == torch.float32
    assert at_least_float32(torch.zeros(2, dtype=torch.float64)).dtype == torch.float64


def test_gat_layer_finite_with_large_inputs():
    """End-to-end: the layer must not emit NaN on large inputs."""
    torch.manual_seed(0)
    layer = VNGraphAttention(2, 3, heads=2, head_dim=4)
    n, e = 40, 90
    x = torch.randn(n, 2, 3) * 500.0
    edge_index = torch.randint(0, n, (2, e))
    edge_attr = torch.randn(e, 3, 3) * 500.0
    out = layer(x, edge_index, edge_attr)
    assert torch.isfinite(out).all()


def test_layernorm_does_not_collapse_on_large_inputs():
    """
    In half precision a channel norm above ~256 squares to inf, the RMS becomes
    inf and the scale collapses to zero -- silently ZEROING the features instead
    of normalising them. That is not NaN, so nothing downstream flags it.
    """
    from vngat.models.vn_layers import VNLayerNorm

    layer = VNLayerNorm(6)
    out = layer(torch.randn(20, 6, 3, dtype=torch.float16) * 400.0)
    assert torch.isfinite(out).all()
    assert float(out.detach().abs().max()) > 0, "features were zeroed, not normalised"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
def test_full_model_finite_under_autocast_with_large_activations(scene):
    """The whole forward pass under real AMP, with inputs scaled to the regime
    that used to overflow."""
    device = torch.device("cuda")
    graph = scene.to(device)
    model = VNGATModel(hidden_channels=16, num_layers=4, num_vn_slots=4,
                       heads=2, head_dim=4, embed_dim=8).to(device).eval()
    graph.node_vec.mul_(300.0)
    graph.edge_attr.mul_(300.0)
    with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
        out = model(**_model_kwargs(graph, dtype=torch.float32))
    for key, value in out.items():
        assert torch.isfinite(value).all(), f"{key} went non-finite under AMP"
