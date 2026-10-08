"""
Stage two: from predicted rotations to a placed assembly, and its score.

* :mod:`.translation` -- match fracture points across fragments in the
  network's invariant embedding and solve for the translations by weighted
  least squares, over every match at once (``--placement global``).
* :mod:`.rotation` -- optionally, the rotations themselves from the same
  matches: fit each matched pair of fragments (Kabsch + RANSAC) and chain the
  fits outward from the anchor (``--evaluate --rotations matched``).
* :mod:`.placement` -- with those rotations, the translations from the pair
  fits the chain agrees with, the anchor held, instead of from every match
  (``--placement checked``, the default for matched rotations).
* :mod:`.scoring` -- run that per scene and report the benchmark's numbers in
  world units: translation RMSE, Chamfer distance, part accuracy.
"""
from .placement import (  # noqa: F401
    AGREEMENT_DEG,
    Placement,
    agreement_deg,
    place,
    solve_with_held,
    verified_pairs,
)
from .rotation import (  # noqa: F401
    PairRotation,
    chain_rotations,
    kabsch,
    match_rotations,
    pairwise_rotations,
    ransac_motion,
    ransac_rotation,
)
from .scoring import (  # noqa: F401
    PLACEMENTS,
    ROTATION_SOURCES,
    check_placement,
    default_placement,
    mean_over_scenes,
    score_batch,
)
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
    "AGREEMENT_DEG",
    "Matches",
    "PLACEMENTS",
    "PairRotation",
    "Placement",
    "ROTATION_SOURCES",
    "agreement_deg",
    "assemble",
    "chain_rotations",
    "check_placement",
    "default_placement",
    "kabsch",
    "match_rotations",
    "mean_over_scenes",
    "mutual_nearest_neighbours",
    "normal_compatibility",
    "pairwise_rotations",
    "place",
    "ransac_motion",
    "ransac_rotation",
    "resolve_collisions",
    "score_batch",
    "solve_translations",
    "solve_with_held",
    "subsample_per_fragment",
    "verified_pairs",
]
