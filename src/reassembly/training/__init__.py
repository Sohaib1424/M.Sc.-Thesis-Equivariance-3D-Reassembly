"""Training loop, losses, and the dataset/model bridge.

Members are imported lazily. `session.py`'s budget and history logic is
genuinely torch-free, and eagerly importing `bridge` (which is not) would make
it untestable without the full deep-learning stack -- for exactly the code a
run spanning a dozen capped sessions depends on most.
"""
from .session import SessionLimit, merge_history

__all__ = [
    "SessionLimit", "merge_history",
    "CompositeLoss", "cluster_consistency_loss",
    "build_model_inputs", "build_targets", "build_predictions",
    "apply_rotation_per_fragment", "select_input_graph",
    "fit", "run_epoch", "save_checkpoint", "load_checkpoint",
    "find_resume_checkpoint",
]

_LAZY = {
    "CompositeLoss": ("losses", "CompositeLoss"),
    "cluster_consistency_loss": ("losses", "cluster_consistency_loss"),
    "build_model_inputs": ("bridge", "build_model_inputs"),
    "build_targets": ("bridge", "build_targets"),
    "build_predictions": ("bridge", "build_predictions"),
    "apply_rotation_per_fragment": ("bridge", "apply_rotation_per_fragment"),
    "select_input_graph": ("bridge", "select_input_graph"),
    "fit": ("engine", "fit"),
    "run_epoch": ("engine", "run_epoch"),
    "save_checkpoint": ("engine", "save_checkpoint"),
    "load_checkpoint": ("engine", "load_checkpoint"),
    "find_resume_checkpoint": ("engine", "find_resume_checkpoint"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
