"""
Command-line entry points. Run as modules from the repository root::

    python -m scripts.extract_fracture_surfaces --root data --dry-run
    python -m scripts.visualize --root data --diffuse --show both --seed 0

Each script prepends ``src`` to ``sys.path`` at import, so they work
without installing the package. If you have run ``pip install -e .`` that
line is redundant but harmless.
"""
