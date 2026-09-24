"""
Stage two: from predicted rotations to a placed assembly, and its score.

* :mod:`.translation` -- match fracture points across fragments in the
  network's invariant embedding and solve for the translations by weighted
  least squares.
* :mod:`.scoring` -- run that per scene and report the benchmark's numbers in
  world units: translation RMSE, Chamfer distance, part accuracy.
"""
from .scoring import mean_over_scenes, score_batch  # noqa: F401
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
    "assemble",
    "mean_over_scenes",
    "mutual_nearest_neighbours",
    "normal_compatibility",
    "resolve_collisions",
    "score_batch",
    "solve_translations",
    "subsample_per_fragment",
]
