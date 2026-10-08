"""
Stage two: from the network's per-vertex embedding to a placed assembly, and
its score.

* :mod:`.rotation` -- the rotations, from the embedding matches: fit each
  matched pair of fragments (Kabsch + RANSAC) and chain the fits outward from
  the anchor (:func:`match_batch`, for every scene of a batch). The network
  has had no rotation head since v7, so this is where every rotation comes from.
* :mod:`.placement` -- the translations from the pair fits the chain agrees
  with, the anchor held (``--placement checked``, the default).
* :mod:`.translation` -- match fracture points across fragments in the
  embedding and solve for the translations by weighted least squares, over
  every match at once (``--placement global``, v6's placement).
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
    MatchedRotations,
    PairRotation,
    chain_rotations,
    kabsch,
    match_batch,
    match_rotations,
    pairwise_rotations,
    ransac_motion,
    ransac_rotation,
)
from .scoring import (  # noqa: F401
    PLACEMENTS,
    check_placement,
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
    "MatchedRotations",
    "PLACEMENTS",
    "PairRotation",
    "Placement",
    "agreement_deg",
    "assemble",
    "chain_rotations",
    "check_placement",
    "kabsch",
    "match_batch",
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
