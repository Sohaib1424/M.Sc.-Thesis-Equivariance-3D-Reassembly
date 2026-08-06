"""Training loop, losses, and the dataset/model bridge."""
from .bridge import (
    apply_rotation_per_fragment, build_model_inputs, build_predictions,
    build_targets, select_input_graph,
)
from .engine import fit, load_checkpoint, run_epoch, save_checkpoint
from .losses import CompositeLoss, cluster_consistency_loss

__all__ = [
    "CompositeLoss", "cluster_consistency_loss",
    "build_model_inputs", "build_targets", "build_predictions",
    "apply_rotation_per_fragment", "select_input_graph",
    "fit", "run_epoch", "save_checkpoint", "load_checkpoint",
]
