"""
Numbers that are reported and never optimised.

Kept out of :mod:`reassembly.nn.losses` deliberately. A loss has to be
differentiable, bounded and cheap, and those constraints have shaped several
choices there (``atan2`` over ``arccos``, a floored norm, a graph-connected
zero on an empty batch). A diagnostic has none of those constraints and a
different job: to say *why* a curve looks the way it does. Mixing them invites
a metric to acquire a gradient it should not have, and invites a loss to be
judged by whether it reads well.
"""
from .metrics import (  # noqa: F401
    chamfer_distance,
    head_collinearity,
    matrix_to_quaternion,
    part_accuracy,
    swing_twist_error,
)

__all__ = [
    "chamfer_distance",
    "head_collinearity",
    "matrix_to_quaternion",
    "part_accuracy",
    "swing_twist_error",
]
