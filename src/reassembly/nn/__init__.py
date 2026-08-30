"""
The equivariant network.

Every module here is imported explicitly rather than eagerly: ``torch`` is a
heavy optional dependency, and nothing in ``reassembly.mesh`` or
``reassembly.data`` needs it. The geometry pipeline runs, and its tests pass,
on a machine with no torch installed at all.

    ``vn``       Vector Neuron primitives -- linear, nonlinearity, norm,
                 invariant readout, scale gate, Gram-Schmidt head.
    ``segment``  Scatter reductions over a flat index. No ``torch_scatter``.
    ``gat``      Intra-fragment attention along real mesh edges.
    ``cross``    Cross-fragment attention. Read its docstring before changing
                 it: what may cross between fragments is constrained, and the
                 obvious implementation is wrong in a way that trains fine.
    ``losses``   The composite objective, with verified chance values.
    ``model``    The backbone and the rotation convention.
"""
from __future__ import annotations

__all__ = ["cross", "gat", "losses", "model", "segment", "vn"]
