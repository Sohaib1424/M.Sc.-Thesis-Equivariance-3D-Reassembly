"""
Rigid perturbation of fragments.

Takes an assembled scene and scatters the pieces, producing the network's
input together with the ground-truth transforms that undo it.

Two changes from the original ``diffuse_fragments``:

* it draws from the global ``np.random`` state, which makes a run
  irreproducible and, under a multi-worker ``DataLoader``, gives every worker
  a correlated stream unless the seed is set per worker. An explicit
  ``Generator`` is now accepted.
* ``var_vec`` was passed as the ``scale`` argument of ``np.random.normal``,
  which is a standard *deviation*, not a variance. The name is corrected to
  ``translation_std`` so the units are not misread.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


def random_rotations(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """``(n, 3, 3)`` Haar-uniform rotations, via QR of a Gaussian matrix."""
    rng = rng if rng is not None else np.random.default_rng()
    a = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(a)
    d = np.sign(np.einsum("...ii->...i", r))
    d[d == 0] = 1.0
    q = q * d[:, None, :]
    flip = np.linalg.det(q) < 0
    q[flip, :, 0] *= -1.0                     # force det = +1, i.e. SO(3)
    return q


def diffuse_fragments(
    fragments: Sequence,
    translation_mean: Sequence[float] = (0.0, 0.0, 0.0),
    translation_std: Sequence[float] = (0.75, 0.75, 0.75),
    spread: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List, List[np.ndarray]]:
    """
    Apply an independent random SE(3) transform to each fragment.

    Returns ``(perturbed_fragments, matrices)`` with ``matrices[i]`` the 4x4
    homogeneous transform applied to fragment ``i``. The reassembly target is
    the inverse: a model predicting ``R_i`` should recover
    ``matrices[i][:3, :3].T``.

    ``spread`` scales an extra push proportional to the scene's largest
    extent, which keeps fragments from landing inside one another.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = len(fragments)
    if n == 0:
        return [], []

    max_dim = max(float(np.max(f.extents)) for f in fragments)
    rotations = random_rotations(n, rng)
    translations = (
        rng.normal(np.asarray(translation_mean, dtype=float),
                   np.asarray(translation_std, dtype=float), size=(n, 3))
        + rng.standard_normal((n, 3)) * max_dim * spread
    )

    perturbed, matrices = [], []
    for i, fragment in enumerate(fragments):
        matrix = np.eye(4)
        matrix[:3, :3] = rotations[i]
        matrix[:3, 3] = translations[i]
        moved = fragment.copy()
        moved.apply_transform(matrix)
        perturbed.append(moved)
        matrices.append(matrix)
    return perturbed, matrices


def centralize(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the centroid. Returns ``(centred_vertices, centroid)``."""
    centroid = vertices.mean(axis=0, keepdims=True)
    return vertices - centroid, centroid


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

class Normalized(NamedTuple):
    """
    Centred (and optionally rescaled) fragments, plus everything needed to undo it.

    ``centroid`` and ``divisor`` are kept because stage two has to put the
    fragments back in world coordinates: the network predicts rotation only, and
    the translation solver works in the original frame. A normalisation that
    cannot be inverted loses the centroids the solver is trying to recover.
    """
    vertices: List[np.ndarray]   # per fragment, centred and scaled
    centroid: np.ndarray         # (N, 3) world-frame centroid that was removed
    divisor: np.ndarray          # (N,)   scale factor that was divided out
    radius: np.ndarray           # (N,)   each fragment's own radius, world units


def normalize_fragments(
    fragments: Sequence,
    mode: str = "scene",
    rescale: bool = True,
    eps: float = 1e-12,
) -> Normalized:
    """
    Centre every fragment at its own centroid, then divide by a scale.

    ``mode="scene"`` (default)
        **One divisor for the whole scene** -- the largest fragment's radius. Every
        fragment keeps its size *relative to the others*, so the largest lands on
        the unit sphere and a chip a tenth its size stays a tenth its size.

    ``mode="fragment"``
        GARF's convention: each fragment is divided by its own radius, so every
        one lands on the unit sphere regardless of how big it really is.

    Why the default is ``"scene"``
    ------------------------------
    Two fracture surfaces that mate are **the same size in world units** -- they
    were one surface before the object broke. Per-fragment normalisation destroys
    that: a chip and the body it broke from arrive at the cross-fragment attention
    rescaled by different factors, so their surfaces no longer look like they fit.
    The scale feature carries enough information to undo it in principle, but that
    asks the network to learn a correction that per-scene normalisation simply
    never introduces.

    GARF normalises per fragment because its encoder processes fragments
    independently -- there is no cross-fragment geometry at that stage. This design
    does cross-fragment attention on raw coordinates, so it has a reason to differ.
    Both are here; ``mode`` is the switch.

    Rotation does not enter any of this: centroids and radii are
    rotation-equivariant and rotation-invariant respectively, so a rotated scene
    normalises to the rotated normalisation of the original.
    """
    if mode not in ("scene", "fragment"):
        raise ValueError(f"mode must be 'scene' or 'fragment', not {mode!r}")

    points = [np.asarray(getattr(f, "vertices", f), dtype=np.float64)
              for f in fragments]
    n = len(points)
    if n == 0:
        z = np.zeros(0, dtype=np.float64)
        return Normalized([], np.zeros((0, 3)), z, z)

    centroids = np.stack([p.mean(axis=0) if p.size else np.zeros(3) for p in points])
    centred = [p - c for p, c in zip(points, centroids)]
    radii = np.array([float(np.linalg.norm(c, axis=1).max()) if c.size else 0.0
                      for c in centred])

    if not rescale:
        divisor = np.ones(n, dtype=np.float64)
    elif mode == "scene":
        # One number for every fragment -- this is the whole point.
        divisor = np.full(n, max(float(radii.max()), eps), dtype=np.float64)
    else:
        divisor = np.maximum(radii, eps)

    scaled = [c / d for c, d in zip(centred, divisor)]
    return Normalized(scaled, centroids, divisor, radii)


def denormalize_fragments(norm: Normalized) -> List[np.ndarray]:
    """Invert :func:`normalize_fragments`, recovering world coordinates."""
    return [v * d + c for v, d, c in zip(norm.vertices, norm.divisor, norm.centroid)]
