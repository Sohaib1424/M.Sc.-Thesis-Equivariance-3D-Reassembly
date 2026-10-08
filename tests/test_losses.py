"""
Every loss term, checked against the value it must read at chance.

A geometric loss that is silently wrong still descends, so "the loss went down"
proves nothing. What does prove something is knowing what each term reads when
the prediction is a random rotation, and comparing. These are Monte Carlo
checks with a fixed seed and a tolerance sized to the sample count, not
assertions about exact numbers.

Two chance baselines, not one
-----------------------------
"Chance" is ambiguous for rotations, so both are measured:

* **random prediction** -- an independent Haar rotation. Geodesic 126.48 deg,
  Euler RMSE 83.25 deg.
* **identity prediction** -- always output no rotation at all. Geodesic is the
  *same* 126.48 deg, because a Haar rotation's angle is distributed the same
  way whether it is measured against the identity or against another Haar
  rotation. Euler RMSE is 83.18 deg -- also the same, within Monte-Carlo noise.

That second row used to read 83.14 against the random prediction's 86.29, and
the 3 deg gap was documented here and in the design document as a trap in
GARF's headline metric: a model collapsing towards the identity could appear to
beat chance having learned nothing. The gap was ours, not GARF's. ``euler_rmse``
computed ``euler(predicted) - euler(target)`` componentwise, which is not a
metric on SO(3) -- Euler angles are chart coordinates and subtracting two charts
weights the same physical error differently depending on where in the chart the
pair sits. Measuring the angles of the residual ``predicted^T @ target`` closes
the gap to 0.07 deg.

The geodesic angle is still the primary number, because it is the quantity the
loss optimises and it needs no chart at all. Euler RMSE is reported for
comparability with published tables, and the comparable row for this project is
GARF's **vanilla Everyday supplementary** table -- SE(3)-Equiv 79.30 deg,
GARF-mini 10.41 deg -- not the headline one.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.nn.losses import (
    ReassemblyLoss,
    correspondence_loss,
    _euler_xyz,
    cosine_loss,
    embedding_consistency_loss,
    euler_rmse,
    geodesic_angle,
    position_loss,
    rotation_loss,
)

DTYPE = torch.float64
SAMPLES = 60_000
CHANCE_GEODESIC = math.pi / 2 + 2 / math.pi          # 2.2074 rad = 126.48 deg


def haar(n: int, seed: int) -> "torch.Tensor":
    """``(n, 3, 3)`` Haar-uniform rotations."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=generator, dtype=DTYPE)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    flipped = torch.det(q) < 0
    q[flipped, :, 0] *= -1
    return q


# --------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------

def test_geodesic_angle_matches_the_analytic_chance_value():
    """
    For two independent Haar rotations the relative angle has density
    ``(1 - cos t)/pi`` on ``[0, pi]``, whose mean is ``pi/2 + 2/pi``. Any
    disagreement here means the loss is not measuring the angle it claims to.
    """
    measured = geodesic_angle(haar(SAMPLES, 1), haar(SAMPLES, 2)).mean().item()
    assert measured == pytest.approx(CHANCE_GEODESIC, abs=0.01)
    assert rotation_loss(haar(SAMPLES, 1), haar(SAMPLES, 2)).item() == pytest.approx(
        CHANCE_GEODESIC, abs=0.01
    )


def test_geodesic_angle_is_zero_for_a_perfect_prediction():
    """
    Exactly zero, not merely small.

    It was 5e-9 rad until the axis norm's floor was dropped from 1e-8 to 1e-20
    in this one path -- see the comment in `geodesic_angle`. The larger floor
    bought no stability, because `atan2`'s derivative cancels the norm's, and
    it put a 5e-9 rad error at both ends of the range.
    """
    R = haar(1000, 3)
    assert geodesic_angle(R, R).abs().max().item() < 1e-14


def test_geodesic_angle_recovers_a_known_rotation():
    for degrees in (1.0, 30.0, 90.0, 179.0, 180.0):
        t = math.radians(degrees)
        about_z = torch.tensor(
            [[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0],
             [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        identity = torch.eye(3, dtype=DTYPE)
        assert geodesic_angle(identity, about_z).item() == pytest.approx(t, abs=1e-9)


def test_geodesic_gradient_stays_finite_where_arccos_would_not():
    """
    The reason for ``atan2``. ``arccos((tr - 1)/2)`` has an infinite derivative
    at a perfect fit, so its gradient grows without bound exactly as the model
    converges, swamping every other term.
    """
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    for t in (1e-7, 1e-4, 1e-2):
        angle = torch.tensor(t, dtype=DTYPE, requires_grad=True)
        K = torch.zeros(3, 3, dtype=DTYPE)
        K[0, 1], K[1, 0] = -axis[2], axis[2]
        R = torch.eye(3, dtype=DTYPE) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        geodesic_angle(torch.eye(3, dtype=DTYPE), R).backward()
        assert torch.isfinite(angle.grad) and angle.grad.abs() < 10.0


def test_axis_correct_azimuth_random_reference():
    """
    The ~90 degree level the previous model's validation error sat at: the
    axis recovered, the rotation about it uniform.

    A LANDMARK, NOT A FLOOR. It was described in this project as a structural
    limit of a per-fragment canonicaliser on surfaces of revolution -- something
    no amount of compute would move. That is false, and the earlier design's own
    later measurements refute it: a fragment of a symmetric object is not itself
    symmetric, because its fracture boundary is jagged and unique, and a scaling
    run on eight Everyday objects (bottles, bowls, mugs) reached 30.9 degrees
    training error.

    What the number is still good for is reading a stalled run, and
    ``reassembly.evaluation.metrics.swing_twist_error`` is what makes it
    readable: tilt ~ 0 with twist ~ 90 is this situation, tilt ~ 90 is "nothing
    learned", and the mean geodesic angle is the same for both.
    """
    generator = torch.Generator().manual_seed(7)
    axis = torch.nn.functional.normalize(
        torch.randn(SAMPLES, 3, generator=generator, dtype=DTYPE), dim=-1
    )
    angle = torch.rand(SAMPLES, generator=generator, dtype=DTYPE) * 2 * math.pi
    K = torch.zeros(SAMPLES, 3, 3, dtype=DTYPE)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    spin = (torch.eye(3, dtype=DTYPE) + torch.sin(angle)[:, None, None] * K
            + (1 - torch.cos(angle))[:, None, None] * (K @ K))
    base = haar(SAMPLES, 8)
    measured = math.degrees(geodesic_angle(base, base @ spin).mean().item())
    assert measured == pytest.approx(89.9, abs=0.6)


# --------------------------------------------------------------------------
# Euler RMSE -- the comparability metric, and its trap
# --------------------------------------------------------------------------

def test_euler_extraction_matches_scipy():
    """`_euler_xyz` is extrinsic xyz. Pinned against an independent library."""
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    R = haar(2000, 11)
    reference = scipy_rotation.from_matrix(R.numpy()).as_euler("xyz")
    difference = np.abs(((_euler_xyz(R).numpy() - reference + np.pi) % (2 * np.pi)) - np.pi)
    assert difference.max() < 1e-10


def test_euler_rmse_chance_for_a_random_prediction():
    measured = euler_rmse(haar(SAMPLES, 1), haar(SAMPLES, 2)).item()
    assert measured == pytest.approx(83.25, abs=0.5)


def test_collapsing_to_identity_does_not_beat_guessing_on_euler_rmse():
    """
    The regression guard for the convention fix.

    Under the old componentwise convention a model that collapsed to the
    identity scored 83.14 against a random prediction's 86.29 -- three degrees
    *better* while having learned nothing -- and that artefact was reported as
    a property of the metric. Measuring the residual rotation closes the gap to
    noise, so there is nothing to be gained by collapsing. Both still read
    126.48 deg geodesic, which never paid for the collapse in the first place.
    """
    target = haar(SAMPLES, 2)
    identity = torch.eye(3, dtype=DTYPE).expand_as(target)
    random_guess = haar(SAMPLES, 1)

    collapsed = euler_rmse(identity, target).item()
    guessed = euler_rmse(random_guess, target).item()
    assert collapsed == pytest.approx(83.18, abs=0.5)
    assert abs(collapsed - guessed) < 0.5, (
        f"collapsing to the identity scored {collapsed:.2f} against a random "
        f"prediction's {guessed:.2f}. A gap here means the metric rewards "
        f"collapse again -- check that euler_rmse measures the RESIDUAL."
    )

    assert math.degrees(geodesic_angle(identity, target).mean().item()) == pytest.approx(
        126.48, abs=0.6
    )
    assert math.degrees(geodesic_angle(random_guess, target).mean().item()) == pytest.approx(
        126.48, abs=0.6
    )


def test_euler_rmse_is_invariant_to_a_global_frame_change():
    """
    The property the componentwise convention lacked, stated directly.

    Rotating BOTH the prediction and the target by the same fixed rotation
    leaves the error between them unchanged, so any honest error metric must
    return the same number. Euler charts are not frame-invariant, so
    subtracting two of them is not either: the old convention moves by tens of
    degrees under this, which is why its "chance" value depended on where in
    the chart the pairs happened to sit.
    """
    predicted, target = haar(4000, 3), haar(4000, 4)
    frame = haar(1, 5)[0]
    base = euler_rmse(predicted, target).item()
    moved = euler_rmse(frame @ predicted, frame @ target).item()
    assert moved == pytest.approx(base, abs=1e-9)


def test_euler_rmse_wraps_the_angle_difference():
    """359 degrees and 1 degree differ by 2, not by 358."""
    def about_z(degrees):
        t = math.radians(degrees)
        return torch.tensor([[[math.cos(t), -math.sin(t), 0.0],
                              [math.sin(t), math.cos(t), 0.0],
                              [0.0, 0.0, 1.0]]], dtype=DTYPE)
    measured = euler_rmse(about_z(179.0), about_z(-179.0)).item()
    assert measured == pytest.approx(math.sqrt(2.0 ** 2 / 3), abs=1e-6)


# --------------------------------------------------------------------------
# Geometric terms
# --------------------------------------------------------------------------

def test_normal_cosine_chance_is_one():
    generator = torch.Generator().manual_seed(4)
    v = torch.nn.functional.normalize(
        torch.randn(SAMPLES, 3, generator=generator, dtype=DTYPE), dim=-1)
    rotated = torch.einsum("nij,nj->ni", haar(SAMPLES, 5), v)
    assert cosine_loss(rotated, v).item() == pytest.approx(1.0, abs=0.01)
    assert cosine_loss(v, v).item() == pytest.approx(0.0, abs=1e-12)


def test_face_normal_chance_is_two():
    """Two adjacent normals per edge, so the term is twice the single-normal one."""
    generator = torch.Generator().manual_seed(6)
    f = torch.nn.functional.normalize(
        torch.randn(SAMPLES, 2, 3, generator=generator, dtype=DTYPE), dim=-1)
    rotated = torch.einsum("nij,nkj->nki", haar(SAMPLES, 7), f)
    assert 2.0 * cosine_loss(rotated, f).item() == pytest.approx(2.0, abs=0.02)


def test_cosine_loss_normalises_its_inputs():
    """A non-unit input must not inflate the term -- vertex normals are not
    guaranteed unit length after averaging over degenerate faces."""
    v = torch.randn(100, 3, dtype=DTYPE)
    assert cosine_loss(v * 17.0, v).item() == pytest.approx(0.0, abs=1e-12)


def test_position_chance_on_the_unit_sphere():
    """``E||Rx - x||`` for unit ``x`` and Haar ``R`` is 4/3."""
    generator = torch.Generator().manual_seed(9)
    p = torch.nn.functional.normalize(
        torch.randn(SAMPLES, 3, generator=generator, dtype=DTYPE), dim=-1)
    rotated = torch.einsum("nij,nj->ni", haar(SAMPLES, 10), p)
    assert position_loss(rotated, p).item() == pytest.approx(4 / 3, abs=0.01)
    assert position_loss(p, p).item() < 1e-7


def test_per_fragment_means_do_not_let_a_big_fragment_dominate():
    """
    Fragments span 4 to 83,039 vertices. Without segmenting, the loss is a
    per-*vertex* mean and one large fragment sets the gradient for the scene.
    """
    batch = torch.tensor([0] * 1000 + [1] * 2)
    predicted = torch.zeros(1002, 3, dtype=DTYPE)
    target = torch.zeros(1002, 3, dtype=DTYPE)
    target[1000:] = 1.0                                   # only the small fragment is wrong

    flat = position_loss(predicted, target)
    segmented = position_loss(predicted, target, batch, 2)
    # The 1e-8 tolerance is the `safe_norm` floor: the 1000 correct vertices
    # each contribute a distance of 1e-8 rather than exactly 0.
    assert flat.item() == pytest.approx(2 * math.sqrt(3) / 1002, abs=1e-7)
    assert segmented.item() == pytest.approx(math.sqrt(3) / 2, abs=1e-7)
    assert segmented > 100 * flat


# --------------------------------------------------------------------------
# Embedding consistency
# --------------------------------------------------------------------------

def test_embedding_loss_is_zero_when_a_cluster_agrees():
    z = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0], [5.0, 5.0]], dtype=DTYPE)
    cluster = torch.tensor([0, 0, 1, 1])
    assert embedding_consistency_loss(z, cluster, 2).item() == pytest.approx(0.0, abs=1e-14)


def test_embedding_loss_rejects_the_design_documents_formula():
    """
    The design document's ``||sum_i z_i||^2`` is minimised by embeddings that
    *cancel*: two opposite vectors score 0 and two identical ones score
    ``4||z||^2``, so it rewards exactly the configuration it is meant to
    penalise. The centroid-variance form gets both cases the right way round.
    """
    cluster = torch.tensor([0, 0])
    agreeing = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    cancelling = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=DTYPE)

    def design_document_form(z):
        return (z.sum(0) ** 2).sum()

    assert design_document_form(cancelling) < design_document_form(agreeing)
    assert (embedding_consistency_loss(agreeing, cluster, 1)
            < embedding_consistency_loss(cancelling, cluster, 1))
    assert embedding_consistency_loss(agreeing, cluster, 1).item() == pytest.approx(0.0, abs=1e-14)


def test_embedding_loss_handles_empty_input():
    empty = torch.empty(0, 4, dtype=DTYPE)
    index = torch.empty(0, dtype=torch.long)
    assert embedding_consistency_loss(empty, index, 0).item() == 0.0


# --------------------------------------------------------------------------
# The composite
# --------------------------------------------------------------------------

def test_composite_loss_reports_every_term_at_its_chance_value():
    criterion = ReassemblyLoss()
    n_fragments, n_vertices = 200, 4000
    fragment = torch.arange(n_fragments).repeat_interleave(n_vertices // n_fragments)

    generator = torch.Generator().manual_seed(12)
    normals = torch.nn.functional.normalize(
        torch.randn(n_vertices, 3, generator=generator, dtype=DTYPE), dim=-1)
    faces = torch.nn.functional.normalize(
        torch.randn(n_vertices, 2, 3, generator=generator, dtype=DTYPE), dim=-1)
    spin = haar(n_fragments, 13)

    _, report = criterion(
        haar(n_fragments, 14), haar(n_fragments, 15),
        normals=torch.einsum("nij,nj->ni", spin[fragment], normals),
        target_normals=normals,
        face_normals=torch.einsum("nij,nkj->nki", spin[fragment], faces),
        target_face_normals=faces,
        vertex_batch=fragment, edge_batch=fragment,
    )
    assert report["rotation_degrees"] == pytest.approx(126.48, abs=3.0)
    assert report["normal"] == pytest.approx(1.0, abs=0.06)
    assert report["face"] == pytest.approx(2.0, abs=0.12)


def test_the_geometric_terms_are_zero_for_a_perfect_prediction():
    """
    The geometric terms bottom out at zero. The embedding term does **not**,
    and that is worth knowing before reading a training curve.

    `correspondence_loss` is InfoNCE, whose minimum is not 0 but a small
    temperature-dependent floor: even a perfect embedding gives every negative
    a cosine above -1, so the denominator keeps a little mass. Reading "total"
    as "distance from perfect" is therefore wrong now -- the floor moved off
    zero when the embedding term stopped being minimisable by collapse.

    `match@1` is the number that does read absolutely: it is 1.0 exactly when
    every coincident vertex retrieves its true partner.
    """
    criterion = ReassemblyLoss()
    R = haar(20, 16)
    v = torch.randn(200, 3, dtype=DTYPE)
    n = torch.nn.functional.normalize(torch.randn(200, 3, dtype=DTYPE), dim=-1)
    fragment = torch.arange(20).repeat_interleave(10)
    z = torch.randn(200, 8, dtype=DTYPE)

    total, report = criterion(
        R, R,
        vertices=v, target_vertices=v, normals=n, target_normals=n,
        vertex_batch=fragment,
        embeddings=z[fragment], cluster=fragment, num_clusters=20,
    )
    for name in ("rotation", "position", "normal"):
        # `position` and `normal` floor at the `safe_norm` epsilon, 1e-8, so
        # this is "zero to the precision the guarded norms allow".
        assert abs(report[name]) < 1e-6, f"{name} = {report[name]} should be 0"

    # A perfect embedding retrieves every partner, and the residual loss is the
    # InfoNCE floor rather than an error.
    assert report["match@1"] == 1.0
    assert report["embedding"] < 0.2
    assert total.item() == pytest.approx(report["embedding"], abs=1e-6)


def test_the_face_term_uses_only_the_normal_channels():
    """
    Edges carry `(n1, n2, relative position)`. Only the first two are directions
    to be matched -- the third's *length* is its content, and a cosine loss
    would normalise it away and move chance from 2.0 to 3.0. The slice lives
    inside the loss because passing the whole `edge_attr` is the obvious mistake.
    """
    criterion = ReassemblyLoss()
    edges, fragments = 4000, 50
    batch = torch.arange(fragments).repeat_interleave(edges // fragments)
    generator = torch.Generator().manual_seed(21)
    three = torch.randn(edges, 3, 3, generator=generator, dtype=DTYPE)
    spin = haar(fragments, 22)

    _, report = criterion(
        haar(fragments, 23), haar(fragments, 24),
        face_normals=torch.einsum("nij,nkj->nki", spin[batch], three),
        target_face_normals=three, edge_batch=batch,
    )
    assert report["face"] == pytest.approx(2.0, abs=0.15), "chance is 2, not 3"

    two_only = three[:, :2, :]
    _, sliced = criterion(
        haar(fragments, 23), haar(fragments, 24),
        face_normals=torch.einsum("nij,nkj->nki", spin[batch], two_only),
        target_face_normals=two_only, edge_batch=batch,
    )
    assert report["face"] == pytest.approx(sliced["face"], abs=1e-12), (
        "passing three channels must give the same answer as passing two"
    )


# --------------------------------------------------------------------------
# The embedding head: why the agreement term alone was wrong
# --------------------------------------------------------------------------

def _clusters(pairs: int = 64, per: int = 2):
    return torch.arange(pairs).repeat_interleave(per), pairs


def test_the_agreement_term_alone_is_minimised_by_collapse():
    """
    The bug this replaced, stated as a test so it cannot come back.

    `embedding_consistency_loss` asks coincident vertices to agree, and the
    cheapest way to agree is for everything to agree. A constant embedding
    scores *exactly zero* -- better than any embedding that actually
    distinguishes anything. This is not a corner case: the first real training
    run found it in three epochs, reporting 0.0335 -> 0.0002 while the spread
    of the embeddings fell by 8x and their norm stayed put.
    """
    cluster, n = _clusters()
    constant = torch.ones(len(cluster), 16, dtype=DTYPE)
    assert embedding_consistency_loss(constant, cluster, n).item() == 0.0

    informative = torch.randn(n, 16, dtype=DTYPE).repeat_interleave(2, 0)
    informative = informative + 0.01 * torch.randn_like(informative)
    assert embedding_consistency_loss(informative, cluster, n).item() > 0.0


def test_the_contrastive_term_makes_collapse_the_worst_answer():
    """
    Collapse now scores `log(A - 1) - log(|pos|)` exactly -- the value of a
    uniform distribution over candidates, which is what a constant embedding
    is. Verified against the closed form rather than a magic number.
    """
    cluster, n = _clusters()
    size = len(cluster)
    constant = torch.ones(size, 16, dtype=DTYPE)
    collapsed = correspondence_loss(constant, cluster, n).item()
    assert collapsed == pytest.approx(math.log(size - 1) - math.log(1), abs=1e-9)

    perfect = torch.nn.functional.normalize(
        torch.randn(n, 16, dtype=DTYPE), dim=-1).repeat_interleave(2, 0)
    assert correspondence_loss(perfect, cluster, n).item() < 0.2 * collapsed


def test_match_at_1_reads_absolutely_where_the_loss_does_not():
    """
    The loss floor depends on temperature and on how many anchors there are, so
    it cannot be read without a per-batch reference. `match@1` can: it is the
    fraction of coincident vertices whose true partner is their own nearest
    neighbour -- chance near zero, perfect exactly 1.
    """
    cluster, n = _clusters()
    perfect = torch.nn.functional.normalize(
        torch.randn(n, 16, dtype=DTYPE), dim=-1).repeat_interleave(2, 0)
    _, accuracy = correspondence_loss(perfect, cluster, n, return_accuracy=True)
    assert accuracy.item() == 1.0

    # A collapsed embedding ties every distance, so its nearest neighbour is
    # arbitrary -- it cannot fake this number the way it faked the old loss.
    _, collapsed = correspondence_loss(
        torch.ones(len(cluster), 16, dtype=DTYPE), cluster, n, return_accuracy=True)
    assert collapsed.item() < 0.2


def test_the_contrastive_term_ignores_unclustered_vertices():
    """`-1` means "no coincidence partner" and must not become a cluster."""
    cluster = torch.tensor([0, 0, 1, 1, -1, -1, -1])
    z = torch.nn.functional.normalize(torch.randn(7, 16, dtype=DTYPE), dim=-1)
    z[4:] = z[0]                                  # unclustered copies of anchor 0
    with_noise = correspondence_loss(z, cluster, 2).item()
    z2 = z.clone()
    z2[4:] = torch.nn.functional.normalize(torch.randn(3, 16, dtype=DTYPE), dim=-1)
    assert correspondence_loss(z2, cluster, 2).item() == pytest.approx(with_noise)


def test_the_contrastive_term_caps_its_anchor_count():
    """
    The similarity matrix is quadratic in anchors and a scene can label
    thousands of vertices -- the same trap the cross-attention layer had. The
    cap keeps it bounded, and the loss must still be finite and differentiable.
    """
    cluster = torch.arange(3000).repeat_interleave(2)
    z = torch.randn(6000, 16, dtype=DTYPE, requires_grad=True)
    loss = correspondence_loss(z, cluster, 3000, max_anchors=256)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    # Only the sampled anchors receive gradient.
    assert 0 < int((z.grad.abs().sum(dim=-1) > 0).sum()) <= 256


def test_the_contrastive_term_survives_a_scene_with_no_clusters():
    empty = torch.zeros(0, dtype=torch.long)
    assert correspondence_loss(torch.zeros(0, 16, dtype=DTYPE), empty, 0).item() == 0.0
    lonely = torch.tensor([-1, -1, -1])
    assert correspondence_loss(torch.randn(3, 16, dtype=DTYPE), lonely, 0).item() == 0.0
