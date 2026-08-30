"""
Loading fragments from a Breaking Bad scene directory.

A scene stores one intact fine mesh plus, per fracture mode, a label saying
which *cell* of that mesh belongs to which piece. Reconstructing a fragment
means selecting the triangles of one piece and cleaning up the resulting
sub-mesh.

Four things here that the original ``load_random_scene`` did differently, in
descending order of how much time they cost:

1. ``tri_labels`` was recomputed inside the per-piece loop even though it does
   not depend on the piece index. On a 12-piece fracture that is 12x the work
   for one array. It is hoisted.

2. ``trimesh.Trimesh(nv, nf)`` runs trimesh's ``process=True`` pipeline --
   merge vertices, drop degenerate faces -- on geometry that igl has just
   finished cleaning. Measured at ~17 ms per fragment on an 80k-face mesh
   against 0.26 ms with ``process=False``. Disabled by default, with a flag
   to turn it back on.

3. The intact mesh and the cell matrix were re-read for every fracture mode.
   :class:`SceneReader` reads them once and reuses them across all ~100 modes
   of a scene, which is the single biggest win for exhaustive passes.

4. Pieces that produce no triangles were skipped silently. They are now
   counted and reported, because a piece that vanishes every epoch is
   effectively deleted from the dataset and should be visible.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from scipy.sparse import load_npz

from ..mesh.repair import resolve_duplicated_faces
from .paths import CELL_MATRIX_FILE, FRACTURE_FILE, MESH_FILE


@dataclass
class FractureModeResult:
    """The fragments produced by one fracture mode, plus what went wrong."""
    mode: str
    fragments: List           # list of trimesh.Trimesh
    empty_pieces: int = 0     # piece labels that produced no triangles
    degenerate_faces: int = 0
    piece_ids: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fragments)


class SceneReader:
    """
    Reads one scene directory, holding the shared intact mesh in memory.

    The point is amortisation: a scene has one mesh and ~100 fracture modes,
    so parsing the ``.obj`` and the sparse matrix once instead of once per
    mode removes most of the I/O from an exhaustive pass.

        reader = SceneReader(scene_dir)
        for result in reader.iter_modes():
            ...

    Construct with ``lazy=True`` (the default) and nothing is read until the
    first access, so building readers for thousands of scenes is cheap.
    """

    def __init__(self, scene_dir: str | Path, lazy: bool = True, process: bool = False):
        self.scene_dir = Path(scene_dir)
        self.process = process
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None
        self._cell_matrix = None
        if not lazy:
            self._load_base()

    # ------------------------------------------------------------------ io --
    def _load_base(self) -> None:
        if self._vertices is not None:
            return
        import igl

        mesh_path = self.scene_dir / MESH_FILE
        cell_path = self.scene_dir / CELL_MATRIX_FILE
        if not mesh_path.is_file() or not cell_path.is_file():
            raise FileNotFoundError(f"{self.scene_dir} is not a scene directory")

        vertices, triangles = igl.read_triangle_mesh(str(mesh_path))
        self._vertices = np.ascontiguousarray(vertices)
        self._triangles = np.ascontiguousarray(triangles)
        self._cell_matrix = load_npz(cell_path)

    @property
    def vertices(self) -> np.ndarray:
        self._load_base()
        return self._vertices

    @property
    def triangles(self) -> np.ndarray:
        self._load_base()
        return self._triangles

    def mode_names(self) -> List[str]:
        """Fracture-mode directory names, sorted for deterministic iteration."""
        return sorted(
            d.name for d in self.scene_dir.iterdir()
            if d.is_dir() and (d / FRACTURE_FILE).is_file()
        )

    # -------------------------------------------------------------- pieces --
    def vertex_labels(self, mode: str) -> np.ndarray:
        """Per-fine-vertex piece label for one fracture mode."""
        self._load_base()
        cell_labels = np.load(self.scene_dir / mode / FRACTURE_FILE)
        return self._cell_matrix @ cell_labels

    def load_mode(self, mode: str) -> FractureModeResult:
        """All fragments of one fracture mode."""
        import trimesh

        vertex_labels = self.vertex_labels(mode)
        # Breaking Bad guarantees a triangle's vertices share a label, so the
        # first vertex decides the triangle. Computed once for the whole mode.
        triangle_labels = vertex_labels[self._triangles[:, 0]]
        n_pieces = int(triangle_labels.max()) + 1 if triangle_labels.size else 0

        fragments, piece_ids = [], []
        empty = 0
        for piece in range(n_pieces):
            selector = triangle_labels == piece
            if not selector.any():
                empty += 1
                continue
            vertices, faces = self._build_piece(self._triangles[selector])
            if faces.shape[0] == 0:
                empty += 1
                continue
            fragments.append(trimesh.Trimesh(vertices, faces, process=self.process))
            piece_ids.append(piece)

        return FractureModeResult(mode=mode, fragments=fragments,
                                  empty_pieces=empty, piece_ids=piece_ids)

    def _build_piece(self, faces: np.ndarray):
        """igl cleanup for one piece: unreference, dedupe vertices, resolve faces."""
        import igl

        vertices, faces = igl.remove_unreferenced(self._vertices, faces)[:2]
        merged, _, mapping, _ = igl.remove_duplicate_vertices(vertices, faces, 1e-10)
        faces = mapping[faces]
        faces, _ = resolve_duplicated_faces(faces)
        vertices, faces = igl.remove_unreferenced(merged, faces)[:2]
        return vertices, faces

    def iter_modes(self, modes: Optional[Sequence[str]] = None) -> Iterator[FractureModeResult]:
        """
        Yield every fracture mode in sorted order, one at a time.

        A generator rather than a list: a scene's ~100 modes times ~10
        fragments each is a lot of meshes to hold at once, and the original
        exhaustive script built all of them before returning.
        """
        for mode in (self.mode_names() if modes is None else modes):
            yield self.load_mode(mode)


def load_scene(scene_dir: str | Path, mode: Optional[str] = None,
               rng: Optional[np.random.Generator] = None,
               process: bool = False) -> List:
    """
    Fragments for one fracture mode of a scene.

    ``mode=None`` picks one at random; pass an explicit ``rng`` to make that
    choice reproducible. Returns a list of ``trimesh.Trimesh``.
    """
    reader = SceneReader(scene_dir, process=process)
    if mode is None:
        modes = reader.mode_names()
        if not modes:
            raise FileNotFoundError(f"no fracture modes under {scene_dir}")
        generator = rng if rng is not None else np.random.default_rng()
        mode = modes[int(generator.integers(len(modes)))]
    result = reader.load_mode(mode)
    if result.empty_pieces:
        warnings.warn(
            f"{Path(scene_dir).name}/{mode}: {result.empty_pieces} piece label(s) "
            f"produced no geometry",
            RuntimeWarning, stacklevel=2,
        )
    return result.fragments
