"""
Breaking Bad reassembly — data, geometry and visualisation layer.

Layout
------
``reassembly.arrays``
    Low-level array primitives: int64 key encoding, grouping, index
    compaction. Everything below is built on these.
``reassembly.mesh``
    Pure-geometry operations on ``(V, F)`` arrays — topology, repair,
    fracture-surface extraction, cross-fragment correspondence.
``reassembly.data``
    Dataset traversal: scene discovery, fragment decompression, SE(3)
    perturbation.
``reassembly.nn``
    The network: Vector Neuron primitives, graph and cross-fragment
    attention, the losses.
``reassembly.training``
    Config, dataset, the epoch loop, checkpoints, evaluation, preflight.
``reassembly.distributed``
    One process per GPU; the per-step gradient all-reduce.
``reassembly.assembly``
    Stage two: the translation solver and the benchmark scores.
``reassembly.evaluation``
    Reported-only diagnostics and metrics.
``reassembly.viz``
    Optional rendering. Nothing else in the package depends on it.

Submodules are not imported eagerly: ``reassembly.data.scene`` needs
``igl`` and ``scipy``, and ``reassembly.viz`` needs ``trimesh``, none of
which should be required to use ``reassembly.arrays``. Import what you
need explicitly::

    from reassembly.mesh.fracture import extract_fracture_surface
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["arrays", "data", "mesh", "viz"]
