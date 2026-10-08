"""
Visualisation. Optional — no other subpackage imports this one.

``scene``
    ``build_scene`` returns a ``trimesh.Scene`` and never calls ``.show()``,
    so it works from a notebook or a headless machine. ``describe`` prints
    a per-fragment table, which is usually the faster way to find out why a
    view looks wrong.
``reassembly``
    Poses and frames for animating a prediction dumped by
    ``scripts/dump_prediction.py``.
``results``
    Training histories and evaluation results read back for figures and
    tables -- numpy only, so they can be read anywhere.
``figures``
    The figures ``scripts/make_figures.py`` writes, drawn from ``results``.

``trimesh`` and ``matplotlib`` are imported inside the functions, so
importing this package requires neither.
"""
from __future__ import annotations

__all__ = ["figures", "reassembly", "results", "scene"]
