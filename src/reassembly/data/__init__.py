"""
Dataset traversal for Breaking Bad.

``paths``
    Scene discovery, subsets, official splits. ``find_scenes`` is
    depth-agnostic — it walks until it finds directories containing
    ``compressed_mesh.obj`` rather than assuming a fixed nesting per
    subset, because the nesting is not uniform.
``scene``
    ``SceneReader`` — decompresses fragments for a given fracture mode.
    Reads the intact mesh and cell matrix once and reuses them across all
    ~100 modes of a scene; that reuse is the single largest win on an
    exhaustive pass.
``transforms``
    SE(3) perturbation. ``diffuse_fragments`` takes an explicit
    ``numpy.random.Generator`` — drawing from global ``np.random`` gives
    correlated streams across ``DataLoader`` workers.
"""
from __future__ import annotations

__all__ = ["paths", "scene", "transforms"]
