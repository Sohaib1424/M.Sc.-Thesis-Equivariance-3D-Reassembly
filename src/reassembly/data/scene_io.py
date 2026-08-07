"""
Reading one Breaking Bad scene off disk and turning it into fragment meshes.

A "scene" directory holds:
  compressed_mesh.obj    the unbroken object at the working resolution
  compressed_data.npz    sparse piece -> fine-vertex mapping
  <fracture_id>/compressed_fracture.npy   per-piece labels for one fracture

Each shape ships several fractures; one is drawn at random per call, which is
where most of this dataset's effective variety comes from.
"""
from __future__ import annotations

import fnmatch
import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # imported lazily-ish so `import reassembly` works without the geometry stack
    import igl
except ImportError:  # pragma: no cover
    igl = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

from scipy.sparse import load_npz


def _require_deps() -> None:
    if igl is None or trimesh is None:
        raise ImportError(
            "load_random_scene needs both `libigl` and `trimesh`. "
            "Install with: pip install libigl trimesh"
        )


def resolve_duplicated_faces(F1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``igl::resolve_duplicated_faces``.

    Resolves duplicated faces by directional winding balance:

    * a face appearing exactly once is always kept;
    * a group of duplicates whose signed count is +1 keeps the first
      *consistently* wound member;
    * a group whose signed count is -1 keeps the first *inconsistently* wound
      member;
    * any other balance (0, or |count| > 1) drops the group, except that
      libigl's implementation keeps the first member when the count is
      non-zero -- replicated here exactly.

    The original port was a Python ``for`` loop over every face plus a second
    loop over every unique face. On a fine mesh that is hundreds of thousands
    of interpreter iterations *per fragment, per scene draw*, which at ~100k
    scene loads per training run is a genuine slice of wall-clock time -- and
    this project's entire premise is reducing wall-clock time. This version is
    pure numpy and was verified to produce bit-identical ``F2``/``J`` output
    against the loop implementation across randomized stress cases (single
    faces, exact duplicates, reversed duplicates, triplicates, and mixtures).

    Returns ``(F2, J)`` where ``F2 = F1[J]``.
    """
    F1 = np.asarray(F1)
    n = F1.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=F1.dtype), np.empty((0,), dtype=np.int64)

    # Canonical (sorted-vertex) key per face, mirroring igl::unique_simplices.
    sorted_F1 = np.sort(F1, axis=1)
    uF, IC = np.unique(sorted_F1, axis=0, return_inverse=True)
    IC = np.asarray(IC).reshape(-1)
    num_unique = uF.shape[0]

    # Is this face wound the same way as its canonical representative
    # (i.e. equal up to a cyclic permutation)?
    canonical = uF[IC]
    consistent = (
        np.all(F1 == canonical, axis=1)
        | ((F1[:, 0] == canonical[:, 1]) & (F1[:, 1] == canonical[:, 2]) & (F1[:, 2] == canonical[:, 0]))
        | ((F1[:, 0] == canonical[:, 2]) & (F1[:, 1] == canonical[:, 0]) & (F1[:, 2] == canonical[:, 1]))
    )

    ucounts = np.bincount(IC, minlength=num_unique)                       # members per group
    counts = np.bincount(IC, weights=np.where(consistent, 1.0, -1.0),
                         minlength=num_unique).astype(np.int64)           # signed balance

    face_idx = np.arange(n)

    def _first_where(flag: np.ndarray) -> np.ndarray:
        """For each unique group, the smallest face index whose `flag` is True
        (or -1 if the group has no such member). ``np.minimum.at`` over a
        sentinel-filled array is the vectorized form of "break on first hit"."""
        out = np.full(num_unique, n, dtype=np.int64)
        sel = np.where(flag)[0]
        if sel.size:
            np.minimum.at(out, IC[sel], face_idx[sel])
        return np.where(out == n, -1, out)

    first_any = _first_where(np.ones(n, dtype=bool))
    first_pos = _first_where(consistent)
    first_neg = _first_where(~consistent)

    keep = np.full(num_unique, -1, dtype=np.int64)

    singleton = ucounts == 1
    keep[singleton] = first_any[singleton]

    multi = ~singleton
    pos = multi & (counts == 1)
    neg = multi & (counts == -1)
    other = multi & (counts != 1) & (counts != -1) & (counts != 0)

    if np.any(pos & (first_pos < 0)) or np.any(neg & (first_neg < 0)):
        raise AssertionError(
            "resolve_duplicated_faces: winding balance implies a member that does not "
            "exist. This means the input face list is malformed."
        )

    keep[pos] = first_pos[pos]
    keep[neg] = first_neg[neg]
    keep[other] = first_any[other]
    # counts == 0 with multiple members -> group dropped entirely (keep stays -1)

    # Order matters: the reference implementation emits kept faces in
    # *unique-group* order (groups are lexicographic in the sorted-vertex
    # key), not face order. `keep` is already indexed by group, so simply
    # dropping the -1 sentinels reproduces that order exactly.
    J = keep[keep >= 0].astype(np.int64)
    return F1[J], J


def load_scene(
    scene_dir: str,
    fracture_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    fracture_pattern: Optional[str] = None,
    base_mesh=None,
) -> List["trimesh.Trimesh"]:
    """Load one scene directory into a list of independent fragment meshes.

    ``fracture_id`` pins a specific fracture subdirectory (useful for
    reproducible evaluation); the default draws one at random, which is the
    training behaviour.

    ``base_mesh`` optionally supplies an already-parsed
    ``(vertices, faces, piece_to_fine_vertices)`` triple, skipping the two file
    reads. See ``data.cache.BaseMeshCache`` for why that is worth doing
    separately from caching the whole preprocessed scene.

    ``fracture_pattern`` restricts which subdirectories are eligible, e.g.
    ``"fractured_*"``. This matters and the default deliberately does NOT
    choose for you: a real scene directory was found to hold 80 ``fractured_*``
    fractures alongside 20 ``mode_*`` ones, so an unrestricted draw takes 20%
    of its samples from ``mode_*``. Those differ in kind -- one of them
    contained a single piece, i.e. an unbroken object with nothing to
    reassemble (rejected downstream by the two-fragment guard). Whether
    ``mode_*`` belongs in your training distribution is a dataset-protocol
    question; set this explicitly once you have decided.
    """
    _require_deps()

    mesh_path = os.path.join(scene_dir, "compressed_mesh.obj")
    data_path = os.path.join(scene_dir, "compressed_data.npz")
    if base_mesh is not None:
        # Supplied by the caller, already parsed. These two files are identical
        # across every fracture of a scene, and parsing them costs ~105 ms per
        # sample -- so on a randomly-drawn fracture the work is pure repetition.
        fine_vertices, fine_triangles, piece_to_fine_vertices = base_mesh
    else:
        fine_vertices, fine_triangles = igl.read_triangle_mesh(mesh_path)
        piece_to_fine_vertices = load_npz(data_path)

    if fracture_id is None:
        frac_dirs = sorted(
            d for d in os.listdir(scene_dir)
            if os.path.isdir(os.path.join(scene_dir, d))
            and (fracture_pattern is None or fnmatch.fnmatch(d, fracture_pattern))
        )
        if not frac_dirs:
            raise FileNotFoundError(
                f"No fracture subdirectories matching {fracture_pattern!r} "
                f"inside {scene_dir!r}"
            )
        chooser = rng.choice if rng is not None else random.choice
        fracture_id = chooser(frac_dirs)

    frac_path = os.path.join(scene_dir, fracture_id, "compressed_fracture.npy")
    piece_labels = np.load(frac_path)

    # Label dtype is not consistent across the dataset: on a real scene the
    # `fractured_*` files were int32 while the `mode_*` files were float64.
    # The float path still works (small integers compare exactly) but it makes
    # `fine_vertex_labels` float, and every downstream `== i` is then an
    # equality test on floats. Cast once, here, rather than relying on that.
    piece_labels = np.asarray(piece_labels)
    if piece_labels.dtype.kind == "f":
        rounded = np.rint(piece_labels)
        if not np.allclose(rounded, piece_labels, atol=1e-9):
            raise ValueError(
                f"{frac_path} holds non-integer piece labels; these index "
                f"fragments and must be whole numbers."
            )
        piece_labels = rounded
    piece_labels = piece_labels.astype(np.int64, copy=False)

    fine_vertex_labels = np.asarray(
        piece_to_fine_vertices @ piece_labels
    ).reshape(-1).astype(np.int64, copy=False)
    n_pieces = int(np.max(piece_labels) + 1)

    # A triangle belongs to the piece its first vertex belongs to. (Vertices on
    # the cut surface are duplicated per piece by the dataset, so this is
    # unambiguous in practice.)
    tri_labels = fine_vertex_labels[fine_triangles[:, 0]]

    meshes: List["trimesh.Trimesh"] = []
    for i in range(n_pieces):
        selected = tri_labels == i
        if not np.any(selected):
            continue

        vi, fi = igl.remove_unreferenced(fine_vertices, fine_triangles[selected, :])[:2]
        ui, _I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
        gi = J[fi]
        ffi, _ = resolve_duplicated_faces(gi)
        nv, nf, _, _ = igl.remove_unreferenced(ui, ffi)

        if len(nf) == 0 or len(nv) == 0:
            continue
        meshes.append(trimesh.Trimesh(nv, nf, process=False))

    return meshes


# Kept under the original name so existing notebooks/scripts keep working.
load_random_scene = load_scene


def scene_sizes(meshes: Sequence["trimesh.Trimesh"]) -> Tuple[int, int]:
    """(total vertices, total faces) across a list of fragment meshes."""
    return (
        int(sum(len(m.vertices) for m in meshes)),
        int(sum(len(m.faces) for m in meshes)),
    )
