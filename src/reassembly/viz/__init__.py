"""
Visualisation. Optional — no other subpackage imports this one.

``scene``
    ``build_scene`` returns a ``trimesh.Scene`` and never calls ``.show()``,
    so it works from a notebook or a headless machine. ``describe`` prints
    a per-fragment table, which is usually the faster way to find out why a
    view looks wrong.

``trimesh`` is imported inside the functions rather than at module scope,
so importing this package does not require it.
"""
from __future__ import annotations

__all__ = ["scene"]
