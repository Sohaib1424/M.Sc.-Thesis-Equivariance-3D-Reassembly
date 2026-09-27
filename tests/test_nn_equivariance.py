"""
Equivariance is the thesis. These tests are where it is actually checked.

The claim `L(x R^T) = L(x) R^T` is an algebraic identity, so it should hold to
machine precision, not approximately. Everything here runs in float64 with a
1e-10 tolerance; a failure at 1e-6 would mean a real break, not accumulated
round-off.

The test that matters most is the last group. A layer can be perfectly
equivariant and still make the task unlearnable, if it is equivariant to the
*wrong thing* -- see `test_cross_fragment_is_invariant_to_other_fragments`.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.nn.cross import VNCrossFragmentAttention, cross_fragment_index
from reassembly.nn.gat import VNGraphAttention, VNGraphAttentionBlock
from reassembly.nn.segment import (
    batch_to_counts,
    counts_to_ptr,
    ptr_to_batch,
    segment_max,
    segment_mean,
    segment_softmax,
    segment_sum,
)
from reassembly.nn.vn import (
    VNInvariant,
    VNLayerNorm,
    VNLeakyReLU,
    VNLinear,
    VNMLP,
    VNScaleGate,
    gram_schmidt,
    normalize,
    safe_norm,
)

TOL = 1e-10
DTYPE = torch.float64


def random_rotation(seed: int) -> "torch.Tensor":
    """A proper rotation, via QR with the sign fix that forces det = +1."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=generator, dtype=DTYPE)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------
# Vector Neuron primitives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["linear", "leaky_relu", "layer_norm", "mlp"])
def test_vn_modules_are_equivariant(name, seed):
    module = {
        "linear": lambda: VNLinear(8, 5),
        "leaky_relu": lambda: VNLeakyReLU(8),
        "layer_norm": lambda: VNLayerNorm(8),
        "mlp": lambda: VNMLP(8, 16, 5),
    }[name]().to(DTYPE)

    x = torch.randn(17, 8, 3, dtype=DTYPE)
    R = random_rotation(seed)
    assert torch.allclose(module(x) @ R.T, module(x @ R.T), atol=TOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vn_invariant_is_invariant(seed):
    module = VNInvariant(8).to(DTYPE)
    x = torch.randn(17, 8, 3, dtype=DTYPE)
    assert torch.allclose(module(x), module(x @ random_rotation(seed).T), atol=TOL)


def test_vn_linear_has_no_bias():
    """
    A bias adds a fixed vector, and a fixed vector does not rotate. This is the
    single easiest way to break the whole model, so it is asserted rather than
    left to code review.
    """
    layer = VNLinear(4, 4)
    assert not any(name.endswith("bias") for name, _ in layer.named_parameters())
    zero = torch.zeros(3, 4, 3)
    assert torch.equal(layer(zero), zero), "f(0) must be 0 for a linear equivariant map"


def test_scale_gate_is_equivariant_and_starts_at_identity():
    gate = VNScaleGate(8).to(DTYPE)
    x = torch.randn(11, 8, 3, dtype=DTYPE)
    scale = torch.randn(11, 1, dtype=DTYPE)
    R = random_rotation(0)
    assert torch.allclose(gate(x, scale) @ R.T, gate(x @ R.T, scale), atol=TOL)
    # softplus(0.5413248546) = 1, so an untrained gate is within ~1% of a
    # no-op. Not exactly one: the final weight is small rather than zero, so
    # that the layers feeding it are not born with zero gradient.
    assert torch.allclose(gate(x, scale), x, rtol=0.05, atol=0.05 * x.abs().max())


def test_scale_gate_actually_uses_the_scalar():
    """A gate that ignores its conditioning is a gate in name only."""
    gate = VNScaleGate(8).to(DTYPE)
    torch.nn.init.normal_(gate.mlp[-1].weight, std=0.5)
    x = torch.randn(11, 8, 3, dtype=DTYPE)
    a = gate(x, torch.zeros(11, 1, dtype=DTYPE))
    b = gate(x, torch.ones(11, 1, dtype=DTYPE))
    assert not torch.allclose(a, b)


def test_safe_norm_is_exact_and_differentiable_at_zero():
    """
    Both halves matter. `sqrt(s + eps)` would be differentiable but wrong by
    ~5e-9 everywhere, which is what left `gram_schmidt` non-orthonormal.
    """
    x = torch.randn(5, 3, dtype=DTYPE)
    assert torch.allclose(safe_norm(x), torch.linalg.norm(x, dim=-1), atol=1e-15)

    zero = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    safe_norm(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.isfinite(normalize(torch.zeros(2, 3, dtype=DTYPE))).all()


# --------------------------------------------------------------------------
# The rotation head
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gram_schmidt_gives_a_proper_rotation(seed):
    v = torch.randn(6, 2, 3, dtype=DTYPE)
    M = gram_schmidt(v)
    identity = torch.eye(3, dtype=DTYPE).expand_as(M)
    assert torch.allclose(M.transpose(-1, -2) @ M, identity, atol=1e-12)
    assert torch.allclose(torch.det(M), torch.ones(6, dtype=DTYPE), atol=1e-12)


def test_gram_schmidt_is_equivariant():
    """Rotating the two input vectors rotates every column of the result."""
    v = torch.randn(6, 2, 3, dtype=DTYPE)
    R = random_rotation(0)
    assert torch.allclose(R @ gram_schmidt(v), gram_schmidt(v @ R.T), atol=TOL)


def test_gram_schmidt_survives_degenerate_input():
    """Parallel or zero inputs must give a finite matrix, not NaN."""
    parallel = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=DTYPE)
    zeros = torch.zeros(1, 2, 3, dtype=DTYPE)
    for v in (parallel, zeros):
        assert torch.isfinite(gram_schmidt(v)).all()


# --------------------------------------------------------------------------
# Segment reductions
# --------------------------------------------------------------------------

def test_segment_reductions_against_a_loop():
    src = torch.randn(12, 3, dtype=DTYPE)
    index = torch.tensor([0, 0, 2, 2, 2, 4, 4, 0, 1, 1, 4, 2])
    n = 6                                        # segments 3 and 5 are empty
    total, mean, peak = (segment_sum(src, index, n), segment_mean(src, index, n),
                         segment_max(src, index, n))
    for s in range(n):
        rows = src[index == s]
        if rows.numel() == 0:
            # Empty segments are zero everywhere -- never NaN, never -inf.
            assert torch.equal(total[s], torch.zeros(3, dtype=DTYPE))
            assert torch.equal(mean[s], torch.zeros(3, dtype=DTYPE))
            assert torch.equal(peak[s], torch.zeros(3, dtype=DTYPE))
        else:
            assert torch.allclose(total[s], rows.sum(0), atol=1e-12)
            assert torch.allclose(mean[s], rows.mean(0), atol=1e-12)
            assert torch.allclose(peak[s], rows.max(0).values, atol=1e-12)


def test_segment_softmax_sums_to_one_per_segment():
    logits = torch.randn(20, 4, dtype=DTYPE) * 50.0     # large: overflow bait
    index = torch.randint(0, 5, (20,))
    alpha = segment_softmax(logits, index, 7)
    assert torch.isfinite(alpha).all()
    for s in index.unique().tolist():
        assert torch.allclose(alpha[index == s].sum(0),
                              torch.ones(4, dtype=DTYPE), atol=1e-12)


def test_ptr_round_trip_keeps_empty_segments():
    """
    A zero-length segment is a real case -- a fragment with no fracture surface
    contributes no tokens -- and it survives in `ptr` where `batch` cannot
    represent it at all.
    """
    counts = torch.tensor([3, 0, 2, 0])
    ptr = counts_to_ptr(counts)
    assert ptr.tolist() == [0, 3, 3, 5, 5]
    batch = ptr_to_batch(ptr)
    assert batch.tolist() == [0, 0, 0, 2, 2]
    assert batch_to_counts(batch, num_segments=4).tolist() == counts.tolist()


# --------------------------------------------------------------------------
# Intra-fragment attention
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_graph_attention_is_equivariant(seed):
    layer = VNGraphAttention(16, edge_channels=3, heads=4).to(DTYPE)
    x = torch.randn(20, 16, 3, dtype=DTYPE)
    edge_index = torch.randint(0, 20, (2, 60))
    edge_attr = torch.randn(60, 3, 3, dtype=DTYPE)
    R = random_rotation(seed)
    a = layer(x, edge_index, edge_attr) @ R.T
    b = layer(x @ R.T, edge_index, edge_attr @ R.T)
    assert torch.allclose(a, b, atol=TOL)


def test_graph_attention_block_is_equivariant_and_residual():
    block = VNGraphAttentionBlock(16, heads=4).to(DTYPE)
    x = torch.randn(20, 16, 3, dtype=DTYPE)
    edge_index = torch.randint(0, 20, (2, 60))
    edge_attr = torch.randn(60, 3, 3, dtype=DTYPE)
    R = random_rotation(0)
    a = block(x, edge_index, edge_attr) @ R.T
    b = block(x @ R.T, edge_index, edge_attr @ R.T)
    assert torch.allclose(a, b, atol=TOL)


def test_graph_attention_handles_isolated_nodes():
    """
    Vertex 0 is never a destination. It must come out of the softmax as its own
    self-transform, not as 0/0. Breaking Bad meshes really do contain vertices
    with no incident edges after cell extraction.
    """
    layer = VNGraphAttention(16, heads=4).to(DTYPE)
    x = torch.randn(20, 16, 3, dtype=DTYPE)
    edge_index = torch.stack([torch.randint(1, 20, (50,)), torch.randint(1, 20, (50,))])
    edge_attr = torch.randn(50, 3, 3, dtype=DTYPE)
    out = layer(x, edge_index, edge_attr)
    assert torch.isfinite(out).all()


def test_graph_attention_is_permutation_equivariant():
    """Relabelling vertices must permute the output, not change it."""
    layer = VNGraphAttention(16, heads=4).to(DTYPE)
    x = torch.randn(20, 16, 3, dtype=DTYPE)
    edge_index = torch.randint(0, 20, (2, 60))
    edge_attr = torch.randn(60, 3, 3, dtype=DTYPE)

    perm = torch.randperm(20)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(20)
    permuted = layer(x[perm], inverse[edge_index], edge_attr)
    assert torch.allclose(permuted, layer(x, edge_index, edge_attr)[perm], atol=TOL)


# --------------------------------------------------------------------------
# Cross-fragment attention -- the constraint that decides the design
# --------------------------------------------------------------------------

def _scene():
    """Two scenes: one with three fragments, one with two."""
    fragment = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4])
    scene = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return fragment, scene


def test_cross_fragment_index_pairs_within_scene_across_fragments():
    fragment, scene = _scene()
    query, key = cross_fragment_index(fragment, scene, 2)
    assert (scene[query] == scene[key]).all(), "no pair may cross scenes"
    assert (fragment[query] != fragment[key]).all(), "no pair may stay in a fragment"

    # Scene 0 has fragment sizes 3,2,4 -> 9^2 - (9+4+16) = 52 ordered pairs.
    # Scene 1 has 2,3 -> 5^2 - (4+9) = 12.
    assert query.numel() == 52 + 12
    assert len({(a, b) for a, b in zip(query.tolist(), key.tolist())}) == 64


def test_cross_fragment_is_equivariant_to_its_own_fragment():
    fragment, scene = _scene()
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    torch.nn.init.normal_(layer.mix[-1].weight, std=0.3)       # leave the identity
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)

    R = random_rotation(0)
    rotated = x.clone()
    rotated[fragment == 0] = x[fragment == 0] @ R.T
    out = layer(rotated, query, key)
    base = layer(x, query, key)
    assert torch.allclose(out[fragment == 0], base[fragment == 0] @ R.T, atol=TOL)


def test_cross_fragment_is_invariant_to_other_fragments():
    """
    The property the whole layer is built around, and the one that is easy to
    lose: fragment i's output must not move when fragment j is re-posed.

    Each fragment is perturbed independently, so the label for fragment i does
    not depend on how fragment j landed. If the prediction did, the network
    would be asked to predict a quantity its input does not determine, and it
    would converge to the conditional mean instead -- the exact symptom the
    earlier model showed, where validation error sat at the "axis right,
    azimuth random" floor.

    The residual is round-off, not leakage. Nothing directional crosses the
    gap: what fragment j sends is a set of inner products, and an inner product
    of rotated vectors differs from the unrotated one only in the last bits.
    That shows up here as ~3e-16 -- thirteen orders of magnitude below the
    output's own scale, and it does not grow with the rotation angle. A real
    leak would be the same order as the output itself.
    """
    fragment, scene = _scene()
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    torch.nn.init.normal_(layer.mix[-1].weight, std=0.3)
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)
    base = layer(x, query, key)
    scale = base.abs().max().item()

    for target in (0, 1, 2):
        rotated = x.clone()
        rotated[fragment == target] = x[fragment == target] @ random_rotation(target).T
        out = layer(rotated, query, key)
        others = fragment != target
        drift = (out[others] - base[others]).abs().max().item()
        assert drift < 1e-12 * scale, (
            f"re-posing fragment {target} moved another fragment's output by "
            f"{drift:.2e}, which is too large to be round-off"
        )


def test_cross_fragment_does_not_leak_between_scenes():
    """
    A batch concatenates unrelated objects. This bug is in the project's
    history: fragment ids are globally unique, so an unmasked layer let a
    fragment of one object inform a fragment of another and made the prediction
    depend on batch composition.
    """
    fragment, scene = _scene()
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    torch.nn.init.normal_(layer.mix[-1].weight, std=0.3)
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)
    base = layer(x, query, key)

    perturbed = x.clone()
    perturbed[scene == 0] = torch.randn_like(x[scene == 0])
    out = layer(perturbed, query, key)
    assert torch.equal(out[scene == 1], base[scene == 1])


def test_cross_fragment_survives_a_scene_with_one_fragment():
    """
    Coincidence labelling reports fragments with zero fracture vertices, and a
    single-fragment mode has no cross-fragment pairs at all. The layer must
    return the input, not NaN from an empty softmax.
    """
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    fragment = torch.tensor([0, 0, 0])
    scene = torch.tensor([0, 0, 0])
    query, key = cross_fragment_index(fragment, scene, 1)
    assert query.numel() == 0
    out = layer(torch.randn(3, 16, 3, dtype=DTYPE), query, key)
    assert torch.isfinite(out).all()


def test_cross_fragment_handles_an_empty_token_set():
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    empty = torch.empty(0, dtype=torch.long)
    query, key = cross_fragment_index(empty, empty, 0)
    out = layer(torch.empty(0, 16, 3, dtype=DTYPE), query, key)
    assert out.shape == (0, 16, 3)


def test_cross_fragment_starts_near_the_identity_without_going_dead():
    """
    Two requirements that pull against each other.

    An untrained cross layer should pass the intra-fragment representation
    through, so the first steps train the backbone rather than fight noise
    injected on top of it -- but a layer initialised to *exactly* the identity
    by zeroing its final gate weight has no gradient into anything upstream of
    that weight, which is the entire cross-fragment pathway. It would sit dead
    for the first steps and wake only as the bias drifted.
    """
    fragment, scene = _scene()
    layer = VNCrossFragmentAttention(16, heads=4).to(DTYPE)
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)

    out = layer(x, query, key)
    deviation = (out - layer.norm(x)).abs().max() / x.abs().max()
    assert deviation < 0.02, "an untrained cross layer should be near the identity"

    out.sum().backward()
    dead = [name for name, p in layer.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reaches {dead} at initialisation"


# --------------------------------------------------------------------------
# Gradient checkpointing
#
# Recomputing the gathered q/k/v in the backward pass instead of storing them
# is what makes a 2048-token scene fit on a 16 GB T4: measured 4.00 GB down to
# 0.42 GB per cross layer. That is only a legitimate trade if it changes
# nothing else, so these tests pin all three properties -- identical values,
# identical gradients, and the memory actually saved.
# --------------------------------------------------------------------------

def _retained_bytes(function):
    """
    Bytes of tensor actually kept alive for the backward pass.

    ``torch.cuda.max_memory_allocated`` cannot answer this: it is a peak, so it
    also counts temporaries that are freed again, and the first call warms the
    allocator. Counting distinct tensors as autograd packs them is the direct
    measurement, and it works identically on CPU.
    """
    seen, total = set(), 0

    def pack(tensor):
        nonlocal total
        if tensor.data_ptr() not in seen:
            seen.add(tensor.data_ptr())
            total += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        function()
    return total


def _cross_pair(**kwargs):
    """The same layer twice, weights tied, checkpointing on in one of them."""
    torch.manual_seed(7)
    plain = VNCrossFragmentAttention(16, heads=4, checkpoint=False, **kwargs).to(DTYPE)
    torch.nn.init.normal_(plain.mix[-1].weight, std=0.3)      # leave the identity
    torch.manual_seed(7)
    checkpointed = VNCrossFragmentAttention(16, heads=4, checkpoint=True, **kwargs).to(DTYPE)
    checkpointed.load_state_dict(plain.state_dict())
    return plain, checkpointed


def test_cross_fragment_checkpointing_changes_nothing():
    """
    Bitwise, not approximately. Recomputation re-runs the *same* ops on the
    *same* inputs, so any difference at all would mean the recomputed forward
    is not the one that ran -- a stale buffer, a dropout mask, a different
    branch -- and would show up as a silent training bug, not a crash.
    """
    fragment, scene = _scene()
    plain, checkpointed = _cross_pair()
    query, key = cross_fragment_index(fragment, scene, 2)

    outputs, grads = [], []
    for layer in (plain, checkpointed):
        x = torch.randn(14, 16, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(3))
        x.requires_grad_(True)
        out = layer(x, query, key)
        (out * out).sum().backward()
        outputs.append(out.detach())
        grads.append({name: p.grad.clone() for name, p in layer.named_parameters()})
        grads[-1]["__input__"] = x.grad.clone()

    assert torch.equal(outputs[0], outputs[1]), "checkpointing moved the output"
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), f"gradient differs: {name}"


def test_cross_fragment_checkpointing_keeps_the_symmetries():
    """
    The properties the layer exists for must survive the memory optimisation:
    equivariant to its own fragment's pose, invariant to every other's.
    """
    fragment, scene = _scene()
    _, layer = _cross_pair()
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)
    base = layer(x, query, key)

    R = random_rotation(0)
    rotated = x.clone()
    rotated[fragment == 0] = x[fragment == 0] @ R.T
    out = layer(rotated, query, key)
    assert torch.allclose(out[fragment == 0], base[fragment == 0] @ R.T, atol=TOL)
    others = fragment != 0
    assert (out[others] - base[others]).abs().max() < 1e-12 * base.abs().max()


def test_cross_fragment_checkpointing_actually_saves_memory():
    """
    The whole point, measured as the quantity that actually decides whether a
    scene fits: **bytes per pair**, not bytes.

    A total would be the wrong test. The layer's non-pair tensors are ``(T, C)``
    and dominate at toy sizes, so a total could look fine on 64 pairs while the
    real 3.5-million-pair scene still OOMs. The slope between two scene sizes
    isolates the pair dimension exactly, and it is the number the Kaggle
    arithmetic uses: measured on the real model, 1144 B/pair storing the
    gathers against 120 B/pair recomputing them, which is 4.00 GB against
    0.42 GB per cross layer.
    """
    plain, checkpointed = _cross_pair()

    def bytes_per_pair(layer):
        measurements = []
        for per_fragment in (8, 40):
            fragment = torch.arange(3).repeat_interleave(per_fragment)
            scene = torch.zeros_like(fragment)
            query, key = cross_fragment_index(fragment, scene, 1)
            x = torch.randn(fragment.numel(), 16, 3, dtype=DTYPE)
            measurements.append(
                (query.numel(), _retained_bytes(lambda: layer(x, query, key)))
            )
        (p0, b0), (p1, b1) = measurements
        return (b1 - b0) / (p1 - p0)

    stored = bytes_per_pair(plain)
    recomputed = bytes_per_pair(checkpointed)
    assert recomputed < stored / 4, (
        f"recomputing retained {recomputed:.0f} B/pair against {stored:.0f} "
        f"B/pair stored -- the gathers are being kept after all"
    )


def test_intra_fragment_checkpointing_changes_nothing():
    """Same contract for the intra layer, including the no-edge-features path."""
    for edge_attr in (torch.randn(60, 3, 3, dtype=DTYPE), None):
        torch.manual_seed(11)
        plain = VNGraphAttention(16, heads=4, checkpoint=False).to(DTYPE)
        torch.manual_seed(11)
        checkpointed = VNGraphAttention(16, heads=4, checkpoint=True).to(DTYPE)
        checkpointed.load_state_dict(plain.state_dict())
        edge_index = torch.randint(0, 20, (2, 60), generator=torch.Generator().manual_seed(5))

        outputs, grads = [], []
        for layer in (plain, checkpointed):
            x = torch.randn(20, 16, 3, dtype=DTYPE,
                            generator=torch.Generator().manual_seed(4))
            x.requires_grad_(True)
            out = layer(x, edge_index, edge_attr)
            (out * out).sum().backward()
            outputs.append(out.detach())
            # value_edge sees no gradient when there are no edge features --
            # that is the point of the second pass, so record None as None
            # rather than crashing on it.
            grads.append({n: None if p.grad is None else p.grad.clone()
                          for n, p in layer.named_parameters()})
            grads[-1]["__input__"] = x.grad.clone()

        assert torch.equal(outputs[0], outputs[1])
        for name, expected in grads[0].items():
            actual = grads[1][name]
            assert (expected is None) == (actual is None), f"differs: {name}"
            assert expected is None or torch.equal(expected, actual), f"differs: {name}"


def test_checkpointing_is_inert_under_no_grad():
    """
    Validation runs under ``no_grad``, where there is nothing to recompute for.
    ``torch.utils.checkpoint`` warns and returns a detached result in that
    context, so the layer must take the plain path instead -- otherwise every
    validation batch pays for a spurious second forward.
    """
    fragment, scene = _scene()
    plain, checkpointed = _cross_pair()
    x = torch.randn(14, 16, 3, dtype=DTYPE)
    query, key = cross_fragment_index(fragment, scene, 2)
    with torch.no_grad():
        assert torch.equal(plain(x, query, key), checkpointed(x, query, key))
