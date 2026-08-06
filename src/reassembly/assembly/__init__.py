"""Stage 2: interface matching and the classical translation solver.

Pure numpy + scipy -- usable without torch.
"""
from .matching import MatchSet, correspondence_components, match_scene, mutual_nearest_neighbors
from .translation import (
    TranslationResult, assemble, refine_with_collision, solve_translations,
)

__all__ = [
    "MatchSet", "match_scene", "mutual_nearest_neighbors", "correspondence_components",
    "TranslationResult", "solve_translations", "refine_with_collision", "assemble",
]
