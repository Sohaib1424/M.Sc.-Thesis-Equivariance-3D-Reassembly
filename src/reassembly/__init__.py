"""
SO(3)-equivariant graph networks for 3D fracture reassembly on commodity GPUs.

Two-stage design:
  1. rotation  -- learned, equivariant (reassembly.models)
  2. translation -- classical geometric optimization (reassembly.assembly)

Submodules are imported lazily so that `import reassembly` works in an
environment without torch (e.g. to use the pure-numpy assembly and evaluation
code, or to read the config schema).
"""
__version__ = "0.2.0"

__all__ = ["config", "data", "models", "training", "assembly", "evaluation", "utils"]
