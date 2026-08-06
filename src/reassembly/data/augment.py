"""
Synthetic "breaking and scattering": the random SE(3) perturbation whose
inverse rotation is what the network is trained to recover.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def diffuse_fragments(
    fragments: Sequence,
    translation_sigma: Tuple[float, float, float] = (0.75, 0.75, 0.75),
    translation_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    push_scale: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List, List[np.ndarray]]:
    """Apply an independent random SE(3) transform to every fragment.

    Returns ``(diffused_meshes, transform_matrices)`` where each transform is a
    4x4 homogeneous matrix taking the clean fragment to its scattered pose.
    The rotation block of that matrix is what
    :func:`reassembly.training.bridge.build_targets` inverts to get ``R_gt``.

    ``rng``: pass a seeded ``np.random.Generator`` for reproducible validation
    runs. The original used global ``np.random`` state, which is *shared
    across DataLoader worker processes on fork-based start methods* -- every
    worker inherits the same seed and then draws the same "random" rotations,
    silently reducing augmentation diversity. Threading an explicit generator
    through (seeded per worker, see ``dataset.py``) removes that failure mode.

    NOTE ON SCOPE: only the rotation block is used as supervision today. The
    translation is applied so the input genuinely looks scattered, but it is
    cancelled by per-fragment centralization during feature construction and
    is recovered at inference by the classical solver in
    ``reassembly.assembly.translation``, not by the network.
    """
    rng = np.random.default_rng() if rng is None else rng
    fragments = list(fragments)
    if not fragments:
        return [], []

    extents = [float(np.asarray(f.extents).max()) for f in fragments if len(f.vertices)]
    max_dim = max(extents) if extents else 1.0

    diffused, matrices = [], []
    for mesh in fragments:
        m = mesh.copy()

        # scipy's Rotation.random accepts a Generator via `random_state`.
        rotation = Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()

        translation = rng.normal(translation_mean, translation_sigma, size=3)
        translation = translation + rng.standard_normal(3) * max_dim * push_scale

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        m.apply_transform(matrix)

        diffused.append(m)
        matrices.append(matrix)

    return diffused, matrices
