"""
Virtual-node cross-fragment communication.

The two things that must not regress:
  * segment attention is NUMERICALLY IDENTICAL to the dense masked version it
    replaced (the memory optimization must not change the model);
  * stage 2 does not leak information between scenes in the same batch.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from reassembly.models.virtual_nodes import (  # noqa: E402
    VirtualNodeCommunicationBlock, VNDenseCrossAttention, VNSlotAttention,
    VNVirtualNodeInit, scatter_mean_vectors,
)

pytestmark = pytest.mark.torch

C, K, H = 8, 4, 2
FRAG_SIZES = [7, 11, 5, 9]


def random_rotation_t(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def rot(t, Q):
    return torch.einsum("ij,...j->...i", Q, t)


def make_scene():
    fragment_id = torch.cat([torch.full((n,), i) for i, n in enumerate(FRAG_SIZES)]).long()
    return fragment_id, sum(FRAG_SIZES), len(FRAG_SIZES)


def copy_weights(dst, src):
    for name in ("lin_q", "lin_k", "lin_v", "lin_out"):
        getattr(dst, name).map.weight.data.copy_(getattr(src, name).map.weight.data)


def test_segment_upward_matches_dense_masked():
    torch.manual_seed(0)
    fragment_id, N, F = make_scene()

    seg = VNSlotAttention(C, C, C, num_slots=K, heads=H, direction="up")
    dense = VNDenseCrossAttention(C, C, C, heads=H)
    copy_weights(dense, seg)

    slots = torch.randn(F, K, C, 3)
    verts = torch.randn(N, C, 3)

    out_seg = seg(slots, verts, fragment_id, F).reshape(F * K, C, 3)

    slot_frag = torch.arange(F).repeat_interleave(K)
    mask = slot_frag.unsqueeze(1) == fragment_id.unsqueeze(0)          # (F*K, N)
    out_dense = dense(slots.reshape(F * K, C, 3), verts, mask=mask)

    assert torch.allclose(out_seg, out_dense, atol=1e-5), \
        (out_seg - out_dense).abs().max()


def test_segment_downward_matches_dense_masked():
    torch.manual_seed(1)
    fragment_id, N, F = make_scene()

    seg = VNSlotAttention(C, C, C, num_slots=K, heads=H, direction="down")
    dense = VNDenseCrossAttention(C, C, C, heads=H)
    copy_weights(dense, seg)

    slots = torch.randn(F, K, C, 3)
    verts = torch.randn(N, C, 3)

    out_seg = seg(slots, verts, fragment_id, F)

    slot_frag = torch.arange(F).repeat_interleave(K)
    mask = fragment_id.unsqueeze(1) == slot_frag.unsqueeze(0)          # (N, F*K)
    out_dense = dense(verts, slots.reshape(F * K, C, 3), mask=mask)

    assert torch.allclose(out_seg, out_dense, atol=1e-5), \
        (out_seg - out_dense).abs().max()


@pytest.mark.parametrize("direction", ["up", "down"])
def test_slot_attention_is_equivariant(direction):
    torch.manual_seed(2)
    fragment_id, N, F = make_scene()
    layer = VNSlotAttention(C, C, C, num_slots=K, heads=H, direction=direction)
    slots, verts = torch.randn(F, K, C, 3), torch.randn(N, C, 3)
    Q = random_rotation_t(9)

    base = layer(slots, verts, fragment_id, F)
    rotated = layer(rot(slots, Q), rot(verts, Q), fragment_id, F)
    assert torch.allclose(rotated, rot(base, Q), atol=1e-4)


def test_virtual_node_init_is_equivariant():
    torch.manual_seed(3)
    fragment_id, N, F = make_scene()
    init = VNVirtualNodeInit(C, C, num_slots=K)
    x = torch.randn(N, C, 3)
    Q = random_rotation_t(10)
    assert torch.allclose(init(rot(x, Q), fragment_id, F),
                          rot(init(x, fragment_id, F), Q), atol=1e-5)


def test_scatter_mean_vectors_is_equivariant_and_correct():
    fragment_id, N, F = make_scene()
    x = torch.randn(N, C, 3)
    pooled = scatter_mean_vectors(x, fragment_id, F)
    assert pooled.shape == (F, C, 3)
    assert torch.allclose(pooled[0], x[fragment_id == 0].mean(0), atol=1e-5)
    Q = random_rotation_t(11)
    assert torch.allclose(scatter_mean_vectors(rot(x, Q), fragment_id, F),
                          rot(pooled, Q), atol=1e-5)


def test_block_is_equivariant():
    torch.manual_seed(4)
    fragment_id, N, F = make_scene()
    block = VirtualNodeCommunicationBlock(C, num_slots=K, heads=H).eval()
    x = torch.randn(N, C, 3)
    scene_id = torch.zeros(F, dtype=torch.long)
    Q = random_rotation_t(12)
    base = block(x, fragment_id, F, fragment_scene_id=scene_id)
    rotated = block(rot(x, Q), fragment_id, F, fragment_scene_id=scene_id)
    assert torch.allclose(rotated, rot(base, Q), atol=1e-4)


def test_no_cross_scene_leakage_when_scene_ids_are_supplied():
    """Regression for the cross-scene leakage bug.

    Two independent scenes concatenated into one batch. Perturbing scene B's
    input must leave scene A's output bit-identical. Without
    fragment_scene_id, stage 2 attends across the whole batch and it does not.
    """
    torch.manual_seed(5)
    sizes = [6, 8, 7, 9]
    fragment_id = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)]).long()
    F, N = len(sizes), sum(sizes)
    scene_id = torch.tensor([0, 0, 1, 1])           # frags 0,1 -> scene A; 2,3 -> scene B
    scene_a_nodes = fragment_id < 2

    block = VirtualNodeCommunicationBlock(C, num_slots=K, heads=H).eval()
    x = torch.randn(N, C, 3)

    out_a = block(x, fragment_id, F, fragment_scene_id=scene_id)[scene_a_nodes]

    x2 = x.clone()
    x2[~scene_a_nodes] = torch.randn_like(x2[~scene_a_nodes]) * 50.0
    out_a2 = block(x2, fragment_id, F, fragment_scene_id=scene_id)[scene_a_nodes]

    assert torch.allclose(out_a, out_a2, atol=1e-6), \
        "scene B's input changed scene A's output -- stage 2 mask is not scoped"


def test_leak_is_reproduced_without_scene_ids():
    """The companion of the test above: with fragment_scene_id=None the block
    treats the whole batch as one scene, and the leak is real. This documents
    that passing None is only safe for genuinely single-scene input."""
    torch.manual_seed(5)
    sizes = [6, 8, 7, 9]
    fragment_id = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)]).long()
    F, N = len(sizes), sum(sizes)
    scene_a_nodes = fragment_id < 2

    block = VirtualNodeCommunicationBlock(C, num_slots=K, heads=H).eval()
    x = torch.randn(N, C, 3)
    out_a = block(x, fragment_id, F, fragment_scene_id=None)[scene_a_nodes]

    x2 = x.clone()
    x2[~scene_a_nodes] = torch.randn_like(x2[~scene_a_nodes]) * 50.0
    out_a2 = block(x2, fragment_id, F, fragment_scene_id=None)[scene_a_nodes]

    assert not torch.allclose(out_a, out_a2, atol=1e-4)


def test_upward_attention_respects_fragment_boundaries():
    """A fragment's slots must be a function of that fragment's vertices only."""
    torch.manual_seed(6)
    fragment_id, N, F = make_scene()
    layer = VNSlotAttention(C, C, C, num_slots=K, heads=H, direction="up")
    slots, verts = torch.randn(F, K, C, 3), torch.randn(N, C, 3)

    base = layer(slots, verts, fragment_id, F)
    verts2 = verts.clone()
    verts2[fragment_id == 1] = torch.randn_like(verts2[fragment_id == 1]) * 20
    changed = layer(slots, verts2, fragment_id, F)

    assert torch.allclose(base[0], changed[0], atol=1e-6)
    assert not torch.allclose(base[1], changed[1], atol=1e-4)


def test_head_dim_must_divide_out_channels():
    with pytest.raises(ValueError):
        VNSlotAttention(8, 8, 9, num_slots=4, heads=4)


# ---------------------------------------------------------------------------
# Stage 2 must not be quadratic in BATCH size.
# ---------------------------------------------------------------------------
def test_stage2_per_scene_matches_dense_masked():
    """The optimization must change cost, not output.

    Stage 2 used to build one (F*K, F*K) score matrix across the whole batch
    and mask it to block-diagonal -- quadratic in batch size, with every
    cross-scene entry computed and then discarded. Running each scene as its
    own block gives the same numbers: masked-away entries contribute nothing.
    """
    torch.manual_seed(11)
    sizes = [5, 7, 6, 9, 4, 8]
    fragment_id = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)]).long()
    F, N = len(sizes), sum(sizes)
    scene_id = torch.tensor([0, 0, 1, 1, 2, 2])

    block = VirtualNodeCommunicationBlock(C, num_slots=K, heads=H).eval()
    x = torch.randn(N, C, 3)
    with torch.no_grad():
        out = block(x, fragment_id, F, fragment_scene_id=scene_id)

    # Reference: run each scene entirely on its own, which is what the
    # block-diagonal mask was meant to express.
    parts = torch.zeros_like(out)
    for scene in (0, 1, 2):
        frags = torch.nonzero(scene_id == scene, as_tuple=True)[0]
        node_mask = torch.isin(fragment_id, frags)
        local_fragment_id = torch.searchsorted(frags, fragment_id[node_mask])
        with torch.no_grad():
            parts[node_mask] = block(
                x[node_mask], local_fragment_id, len(frags),
                fragment_scene_id=torch.zeros(len(frags), dtype=torch.long),
            )

    assert torch.allclose(out, parts, atol=1e-5), (out - parts).abs().max()


def test_stage2_output_is_independent_of_batch_composition():
    """A scene's result must not depend on which other scenes share its batch.
    If it does, training and single-scene inference disagree."""
    torch.manual_seed(12)
    block = VirtualNodeCommunicationBlock(C, num_slots=K, heads=H).eval()

    sizes_a = [5, 7]
    x_a = torch.randn(sum(sizes_a), C, 3)
    frag_a = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes_a)]).long()

    with torch.no_grad():
        alone = block(x_a, frag_a, 2, fragment_scene_id=torch.zeros(2, dtype=torch.long))

    sizes_b = [6, 9, 4]
    x_b = torch.randn(sum(sizes_b), C, 3) * 30.0
    frag_all = torch.cat([frag_a, torch.cat(
        [torch.full((n,), i + 2) for i, n in enumerate(sizes_b)]).long()])
    with torch.no_grad():
        together = block(torch.cat([x_a, x_b]), frag_all, 5,
                         fragment_scene_id=torch.tensor([0, 0, 1, 1, 1]))

    assert torch.allclose(alone, together[: x_a.shape[0]], atol=1e-5)
