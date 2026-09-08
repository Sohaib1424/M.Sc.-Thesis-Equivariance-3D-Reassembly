"""VN-GAT (TensorFlow): SO(3)-equivariant GNN for 3D fracture reassembly."""
from .config import Config, parse_config

__version__ = "1.0.0-tf"
__all__ = ["Config", "parse_config"]
