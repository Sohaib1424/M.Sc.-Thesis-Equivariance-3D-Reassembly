"""Dataset, features, geometry preprocessing, and batching."""
from .splits import SceneIndex, assign_split, list_scene_directories, split_scene_directories

__all__ = [
    "SceneIndex", "assign_split", "list_scene_directories", "split_scene_directories",
    # torch-dependent names are re-exported lazily via __getattr__ below
    "BreakingBadDataset", "get_features", "merge_fragments", "collate_scenes",
    "breaking_bad_collate_fn",
]


def __getattr__(name):
    """Import torch-dependent members on first use.

    Keeps `from reassembly.data import SceneIndex` working in a torch-free
    environment (the assembly/evaluation stack is pure numpy and genuinely
    usable that way), while still exposing the dataset under one namespace.
    """
    if name in ("BreakingBadDataset",):
        from .dataset import BreakingBadDataset
        return BreakingBadDataset
    if name == "get_features":
        from .features import get_features
        return get_features
    if name in ("merge_fragments", "collate_scenes", "breaking_bad_collate_fn"):
        from . import collate
        return getattr(collate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
