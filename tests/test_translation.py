"""Classical translation solver: exactness, robustness, gauge handling."""
import numpy as np
import pytest

from reassembly.assembly.matching import MatchSet, correspondence_components, match_scene
from reassembly.assembly.translation import solve_translations


def build_matches(num_fragments, pairs, t_true, rng, per_pair=25, noise=0.0):
    fa, fb, pa, pb, na, nb = [], [], [], [], [], []
    for (A, B) in pairs:
        shared = rng.normal(size=(per_pair, 3))
        n = rng.normal(size=(per_pair, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        pa.append(shared - t_true[A] + rng.normal(0, noise, shared.shape))
        pb.append(shared - t_true[B] + rng.normal(0, noise, shared.shape))
        na.append(n); nb.append(-n)
        fa += [A] * per_pair; fb += [B] * per_pair
    return MatchSet(
        np.array(fa), np.array(fb), np.concatenate(pa), np.concatenate(pb),
        np.concatenate(na), np.concatenate(nb), np.zeros(len(fa)),
    )


def test_exact_recovery_on_a_connected_graph(rng):
    F = 5
    t_true = rng.normal(0, 2, size=(F, 3)); t_true -= t_true[0]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]
    res = solve_translations(build_matches(F, pairs, t_true, rng), F)
    assert res.num_components == 1 and res.fully_constrained
    assert np.abs(res.translations - t_true).max() < 1e-8
    assert res.residual_rms < 1e-8


def test_noise_degrades_gracefully(rng):
    F = 4
    t_true = rng.normal(0, 2, size=(F, 3)); t_true -= t_true[0]
    pairs = [(0, 1), (1, 2), (2, 3)]
    errors = []
    for noise in (0.0, 0.01, 0.05):
        res = solve_translations(build_matches(F, pairs, t_true, rng, noise=noise), F)
        errors.append(np.abs(res.translations - t_true).max())
    assert errors == sorted(errors)
    assert errors[-1] < 0.15


def test_irls_beats_plain_least_squares_on_outliers(rng):
    """The reason the solver reweights instead of solving once. Mutual-NN
    matching on a learned embedding produces outliers; a pure quadratic fit is
    dominated by them."""
    F = 4
    t_true = rng.normal(0, 2, size=(F, 3)); t_true -= t_true[0]
    matches = build_matches(F, [(0, 1), (1, 2), (2, 3), (0, 3)], t_true, rng, per_pair=40)

    corrupt = rng.choice(len(matches), int(0.3 * len(matches)), replace=False)
    matches.point_b[corrupt] += rng.normal(0, 3.0, (len(corrupt), 3))

    plain = solve_translations(matches, F, irls_iterations=0)
    robust = solve_translations(matches, F, irls_iterations=20)

    err_plain = np.abs(plain.translations - t_true).max()
    err_robust = np.abs(robust.translations - t_true).max()
    assert err_robust < err_plain / 5, (err_plain, err_robust)
    assert err_robust < 0.1


def test_disconnected_graph_is_reported_not_silently_wrong(rng):
    F = 4
    t_true = rng.normal(0, 2, size=(F, 3)); t_true -= t_true[0]
    # fragment 3 touches nothing
    res = solve_translations(build_matches(F, [(0, 1), (1, 2)], t_true, rng), F)
    assert res.num_components == 2
    assert not res.fully_constrained


def test_no_matches_returns_a_defined_result():
    empty = MatchSet(*(np.empty(0, dtype=np.int64),) * 2,
                     *(np.empty((0, 3)),) * 4, np.empty(0))
    res = solve_translations(empty, 3)
    assert res.num_matches == 0
    assert res.translations.shape == (3, 3)
    assert res.num_components == 3


def test_gauge_is_fixed_per_component(rng):
    """Translations are determined only up to one offset per component; the
    solver must anchor each component, not leave the system singular."""
    F = 5
    t_true = rng.normal(0, 2, size=(F, 3)); t_true -= t_true[0]
    res = solve_translations(build_matches(F, [(0, 1), (2, 3)], t_true, rng), F)
    assert np.isfinite(res.translations).all()
    comps = correspondence_components(np.array([0, 2]), np.array([1, 3]), F)
    for c in np.unique(comps):
        members = np.flatnonzero(comps == c)
        assert np.allclose(res.translations[members[0]], 0.0)


def test_relative_translations_are_recovered_within_a_component(rng):
    F = 4
    t_true = rng.normal(0, 2, size=(F, 3))
    res = solve_translations(build_matches(F, [(0, 1), (1, 2), (2, 3)], t_true, rng), F)
    pred_rel = res.translations - res.translations.mean(0)
    true_rel = t_true - t_true.mean(0)
    assert np.abs(pred_rel - true_rel).max() < 1e-8


def test_mutual_nn_always_matches_something_so_filters_are_required(rng):
    """Documents the failure this project's filters exist to prevent: two
    unrelated fragments still produce mutual nearest neighbours."""
    from reassembly.assembly.matching import mutual_nearest_neighbors
    a = rng.normal(size=(50, 8))
    b = rng.normal(size=(50, 8)) + 1000.0          # nothing in common
    ia, ib, d = mutual_nearest_neighbors(a, b, ratio_threshold=None)
    assert len(ia) > 0, "mutual-NN is essentially never empty -- hence min_matches_per_pair"


def test_min_matches_per_pair_rejects_unrelated_fragments(rng):
    """With the filters on, two fragments that share nothing contribute no
    constraints, so the correspondence graph correctly reports them apart."""
    real = rng.normal(size=(40, 8))
    embeddings = [
        np.concatenate([real, rng.normal(size=(30, 8)) * 5]),
        np.concatenate([real + rng.normal(0, 1e-3, real.shape), rng.normal(size=(30, 8)) * 5]),
        rng.normal(size=(70, 8)) * 50 + 500.0,     # unrelated fragment
    ]
    pts = [rng.normal(size=(len(e), 3)) for e in embeddings]
    nrm = []
    for e in embeddings:
        n = rng.normal(size=(len(e), 3))
        nrm.append(n / np.linalg.norm(n, axis=1, keepdims=True))
    nrm[1] = -nrm[0][:len(embeddings[1])]          # fragments 0/1 face each other

    matches = match_scene(embeddings, pts, nrm, min_matches_per_pair=8)
    comps = correspondence_components(matches.frag_a, matches.frag_b, 3)
    assert comps[2] != comps[0], "unrelated fragment 2 should stay in its own component"
