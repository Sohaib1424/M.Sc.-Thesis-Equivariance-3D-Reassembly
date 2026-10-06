"""
Stage two: from predicted rotations to a placed assembly, and its score.

* :mod:`.translation` -- match fracture points across fragments in the
  network's invariant embedding and solve for the translations by weighted
  least squares.
* :mod:`.rotation` -- optionally, the rotations themselves from the same
  matches: fit each matched pair of fragments (Kabsch + RANSAC) and chain the
  fits outward from the anchor (``--evaluate --rotations matched``).
* :mod:`.scoring` -- run that per scene and report the benchmark's numbers in
  world units: translation RMSE, Chamfer distance, part accuracy.
"""
from .rotation import (  # noqa: F401
    PairRotation,
    chain_rotations,
    kabsch,
    match_rotations,
    pairwise_rotations,
    ransac_rotation,
)
from .scoring import ROTATION_SOURCES, mean_over_scenes, score_batch  # noqa: F401
from .translation import (  # noqa: F401
    Matches,
    assemble,
    mutual_nearest_neighbours,
    normal_compatibility,
    resolve_collisions,
    solve_translations,
    subsample_per_fragment,
)

__all__ = [
    "Matches",
    "PairRotation",
    "ROTATION_SOURCES",
    "assemble",
    "chain_rotations",
    "kabsch",
    "match_rotations",
    "mean_over_scenes",
    "mutual_nearest_neighbours",
    "normal_compatibility",
    "pairwise_rotations",
    "ransac_rotation",
    "resolve_collisions",
    "score_batch",
    "solve_translations",
    "subsample_per_fragment",
]
