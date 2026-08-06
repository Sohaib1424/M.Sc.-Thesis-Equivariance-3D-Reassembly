"""Vectorized resolve_duplicated_faces must match the reference loop exactly."""
import numpy as np
import pytest

from reassembly.data.scene_io import resolve_duplicated_faces


def reference(F1):
    """The original loop-based port, kept verbatim as the oracle."""
    n = F1.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=F1.dtype), np.empty((0,), dtype=np.int64)
    uF, IC = np.unique(np.sort(F1, axis=1), axis=0, return_inverse=True)
    IC = np.asarray(IC).reshape(-1)
    canonical = uF[IC]
    consistent = (
        np.all(F1 == canonical, axis=1)
        | ((F1[:, 0] == canonical[:, 1]) & (F1[:, 1] == canonical[:, 2]) & (F1[:, 2] == canonical[:, 0]))
        | ((F1[:, 0] == canonical[:, 2]) & (F1[:, 1] == canonical[:, 0]) & (F1[:, 2] == canonical[:, 1]))
    )
    m = uF.shape[0]
    bins = [[] for _ in range(m)]
    counts = np.zeros(m, dtype=np.int64)
    ucounts = np.zeros(m, dtype=np.int64)
    signed = (np.arange(n) + 1) * np.where(consistent, 1, -1)
    for i in range(n):
        u = IC[i]
        bins[u].append(signed[i])
        counts[u] += 1 if consistent[i] else -1
        ucounts[u] += 1
    kept = []
    for i in range(m):
        if ucounts[i] == 1:
            kept.append(abs(bins[i][0]) - 1); continue
        if counts[i] == 1:
            kept.append(abs(next(f for f in bins[i] if f > 0)) - 1)
        elif counts[i] == -1:
            kept.append(abs(next(f for f in bins[i] if f < 0)) - 1)
        elif counts[i] != 0 and bins[i]:
            kept.append(abs(bins[i][0]) - 1)
    J = np.array(kept, dtype=np.int64)
    return F1[J], J


CASES = {
    "empty": np.empty((0, 3), dtype=np.int64),
    "single": np.array([[0, 1, 2]]),
    "exact_duplicate": np.array([[0, 1, 2], [0, 1, 2]]),
    "reversed_duplicate": np.array([[0, 1, 2], [0, 2, 1]]),
    "triplicate_positive": np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1]]),
    "triplicate_negative": np.array([[0, 2, 1], [0, 2, 1], [0, 1, 2]]),
    "cyclic_permutations": np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
    "mixed": np.array([[0, 1, 2], [1, 2, 3], [0, 1, 2], [3, 2, 1], [4, 5, 6]]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_structural_cases_match_reference(name):
    F = CASES[name]
    F2r, Jr = reference(F)
    F2f, Jf = resolve_duplicated_faces(F)
    assert np.array_equal(Jr, Jf), name
    assert np.array_equal(F2r, F2f), name


def test_randomized_stress_matches_reference(rng):
    for _ in range(150):
        nf = int(rng.integers(1, 40))
        F = rng.integers(0, 8, size=(nf, 3))
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        if len(F) == 0:
            continue
        extra = F[rng.integers(0, len(F), size=int(rng.integers(0, len(F) + 1)))].copy()
        flip = rng.random(len(extra)) < 0.5
        extra[flip] = extra[flip][:, [0, 2, 1]]
        F = np.concatenate([F, extra], axis=0)
        assert np.array_equal(reference(F)[1], resolve_duplicated_faces(F)[1])


def test_output_is_a_subset_of_input():
    F = CASES["mixed"]
    F2, J = resolve_duplicated_faces(F)
    assert np.array_equal(F2, F[J])
