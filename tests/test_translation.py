"""The classical translation solver."""
from __future__ import annotations

import torch

from vngat.assembly.translation import (
    Matches, normal_compatibility, resolve_collisions, solve_translations,
)


def test_solver_recovers_known_translations_exactly():
    torch.manual_seed(0)
    num_frag = 4
    true_t = torch.randn(num_frag, 3)
    true_t = true_t - true_t.mean(0, keepdim=True)

    # interface points that should coincide once the translations are applied
    src_pts, dst_pts, a_ids, b_ids = [], [], [], []
    for a in range(num_frag):
        for b in range(a + 1, num_frag):
            meeting = torch.randn(12, 3)
            src_pts.append(meeting - true_t[a])
            dst_pts.append(meeting - true_t[b])
            a_ids.append(torch.full((12,), a))
            b_ids.append(torch.full((12,), b))

    points = torch.cat(src_pts + dst_pts)
    frag = torch.cat(a_ids + b_ids)
    n = len(torch.cat(a_ids))
    matches = Matches(
        src_idx=torch.arange(n), dst_idx=torch.arange(n, 2 * n), weight=torch.ones(n),
    )
    solved = solve_translations(points, frag, matches, num_frag, irls_iterations=3)
    assert torch.allclose(solved, true_t, atol=1e-4)


def test_solver_is_robust_to_outlier_matches():
    torch.manual_seed(1)
    num_frag = 3
    true_t = torch.randn(num_frag, 3)
    true_t = true_t - true_t.mean(0, keepdim=True)

    good = 60
    meeting = torch.randn(good, 3)
    src = meeting - true_t[0]
    dst = meeting - true_t[1]
    # 10 grossly wrong correspondences: a fixed, large, one-sided offset, so
    # the test measures IRLS rather than the luck of a random draw.
    bogus = torch.randn(10, 3)
    src = torch.cat([src, bogus])
    dst = torch.cat([dst, bogus + torch.tensor([25.0, -25.0, 25.0])])
    meeting2 = torch.randn(40, 3)
    src = torch.cat([src, meeting2 - true_t[1]])
    dst = torch.cat([dst, meeting2 - true_t[2]])

    points = torch.cat([src, dst])
    n = len(src)
    frag = torch.cat([
        torch.zeros(good + 10, dtype=torch.long), torch.ones(40, dtype=torch.long),
        torch.ones(good + 10, dtype=torch.long), torch.full((40,), 2, dtype=torch.long),
    ])
    matches = Matches(torch.arange(n), torch.arange(n, 2 * n), torch.ones(n))

    robust = solve_translations(points, frag, matches, num_frag, irls_iterations=8)
    plain = solve_translations(points, frag, matches, num_frag, irls_iterations=1,
                               huber_delta=1e9)
    assert (robust - true_t).norm() < (plain - true_t).norm()


def test_solution_is_gauge_fixed_to_zero_mean():
    torch.manual_seed(2)
    points = torch.randn(40, 3)
    frag = torch.randint(0, 3, (40,))
    matches = Matches(torch.arange(0, 20), torch.arange(20, 40), torch.ones(20))
    t = solve_translations(points, frag, matches, 3)
    assert torch.allclose(t.mean(0), torch.zeros(3), atol=1e-5)


def test_no_matches_yields_zero_translations():
    t = solve_translations(
        torch.randn(10, 3), torch.zeros(10, dtype=torch.long),
        Matches(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                torch.zeros(0)),
        num_fragments=3,
    )
    assert torch.equal(t, torch.zeros(3, 3))


def test_normal_compatibility_peaks_for_opposed_normals():
    n = torch.tensor([[0.0, 0.0, 1.0]])
    assert float(normal_compatibility(n, -n)) == 1.0
    assert float(normal_compatibility(n, n)) == 0.0


def test_collision_resolution_separates_overlapping_spheres():
    centroids = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    radii = torch.tensor([1.0, 1.0])
    t = resolve_collisions(torch.zeros(2, 3), centroids, radii, iterations=50)
    separation = (centroids + t)[0] - (centroids + t)[1]
    assert float(separation.norm()) > 1.5
