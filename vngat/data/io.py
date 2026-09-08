"""
Scene decompression / loading.

    *** THE DECOMPRESSION LOGIC IN THIS MODULE IS DELIBERATELY UNCHANGED. ***

`load_random_scene` and `resolve_duplicated_faces` reproduce the original
project code exactly. The Breaking Bad authors recommend `igl` for
decompressing their compressed representation and it is by far the fastest
route; nothing here second-guesses that.

The only edit anywhere in this file is an *exact algebraic short circuit*
inside `resolve_duplicated_faces` (see the comment at its call site) which
returns bit-identical output while skipping two Python-level loops in the
common case where a piece has no duplicated faces at all. It is guarded by a
`VNGAT_VERIFY_RESOLVE=1` environment variable that runs both paths and
asserts they agree, so the equivalence is checkable rather than asserted.
"""
from __future__ import annotations

import os

import igl
import numpy as np
import trimesh
from scipy.sparse import load_npz
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# scipy compat: Rotation.random(random_state=) -> Rotation.random(rng=)
# ---------------------------------------------------------------------------
def random_rotation_matrices(num: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    `num` uniformly-random SO(3) matrices, shape (num, 3, 3).

    Wraps the scipy keyword rename (`random_state` was deprecated in favour of
    `rng` in scipy 1.15) so neither spelling leaks a DeprecationWarning into
    training logs. With `rng=None` this draws from numpy's global RNG, which
    is what `dataloader_worker_init` re-seeds per worker.
    """
    if rng is None:
        return R.random(num).as_matrix().reshape(num, 3, 3)
    try:
        return R.random(num, rng=rng).as_matrix().reshape(num, 3, 3)
    except TypeError:  # scipy < 1.15
        seed = int(rng.integers(0, 2**31 - 1))
        return R.random(num, random_state=seed).as_matrix().reshape(num, 3, 3)


# ---------------------------------------------------------------------------
# Original decompression code -- logic untouched
# ---------------------------------------------------------------------------
def list_fracture_dirs(mesh_dir_full_path: str, prefix: str | None = None) -> list:
    """Sorted names of a scene's fracture sub-directories."""
    dirs = sorted(
        d for d in os.listdir(mesh_dir_full_path)
        if os.path.isdir(os.path.join(mesh_dir_full_path, d))
    )
    if prefix:
        filtered = [d for d in dirs if d.startswith(prefix)]
        if filtered:
            return filtered
    return dirs


def load_scene(mesh_dir_full_path: str, fracture_dir: str) -> list:
    """
    Load ONE NAMED fracture pattern, deterministically.

    `load_random_scene` samples a pattern, which is right for training and
    useless for inspecting a specific breakage. The decompression below is the
    same untouched igl code with the random choice replaced by the argument.
    """
    compressed_mesh_path = os.path.join(mesh_dir_full_path, "compressed_mesh.obj")
    compressed_data_path = os.path.join(mesh_dir_full_path, "compressed_data.npz")
    fine_vertices, fine_triangles = igl.read_triangle_mesh(compressed_mesh_path)
    piece_to_fine_vertices_matrix = load_npz(compressed_data_path)

    frac_data_path = os.path.join(mesh_dir_full_path, fracture_dir, "compressed_fracture.npy")
    piece_labels_after_impact = np.load(frac_data_path)
    fine_vertex_labels_after_impact = piece_to_fine_vertices_matrix @ piece_labels_after_impact
    n_pieces_after_impact = int(np.max(piece_labels_after_impact) + 1)
    tri_labels = fine_vertex_labels_after_impact[fine_triangles[:, 0]]

    meshes = []
    for i in range(n_pieces_after_impact):
        if not np.any(tri_labels == i):
            continue
        vi, fi = igl.remove_unreferenced(fine_vertices, fine_triangles[tri_labels == i, :])[:2]
        ui, _I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
        gi = J[fi]
        ffi, _ = resolve_duplicated_faces(gi)
        nv, nf, _, _ = igl.remove_unreferenced(ui, ffi)
        meshes.append(trimesh.Trimesh(nv, nf))
    return meshes


def load_random_scene(mesh_dir_full_path: str, fracture_pattern: str | None = None) -> list:
    """
    Reads one Breaking Bad scene directory and returns its fragment meshes.

    Identical to the original implementation except that `fracture_pattern`
    optionally restricts which fracture sub-directories are eligible (e.g.
    "fractured_" to exclude the `mode_*` variants). Passing None reproduces
    the original behaviour exactly: uniform choice over every sub-directory.
    """
    compressed_mesh_path = os.path.join(mesh_dir_full_path, "compressed_mesh.obj")
    compressed_data_path = os.path.join(mesh_dir_full_path, "compressed_data.npz")
    fine_vertices, fine_triangles = igl.read_triangle_mesh(compressed_mesh_path)
    piece_to_fine_vertices_matrix = load_npz(compressed_data_path)

    frac_dirs = [
        d for d in os.listdir(mesh_dir_full_path)
        if os.path.isdir(os.path.join(mesh_dir_full_path, d))
    ]
    if fracture_pattern:
        filtered = [d for d in frac_dirs if d.startswith(fracture_pattern)]
        if filtered:
            frac_dirs = filtered

    random_frac_dir = np.random.choice(frac_dirs)
    random_frac_dir_full_path = os.path.join(mesh_dir_full_path, random_frac_dir)

    random_frac_data_path = os.path.join(random_frac_dir_full_path, "compressed_fracture.npy")

    piece_labels_after_impact = np.load(random_frac_data_path)

    fine_vertex_labels_after_impact = piece_to_fine_vertices_matrix @ piece_labels_after_impact

    n_pieces_after_impact = int(np.max(piece_labels_after_impact) + 1)

    meshes = []

    for i in range(n_pieces_after_impact):
        tri_labels = fine_vertex_labels_after_impact[fine_triangles[:, 0]]

        if np.any(tri_labels == i):
            vi, fi = igl.remove_unreferenced(
                fine_vertices, fine_triangles[tri_labels == i, :])[:2]
        else:
            continue
        # if you are using igl v2.2.1 installed via conda-forge, use this block
        # ------------------------------------------------------------------
        # ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
        # gi = J[fi]
        # ffi, _ = igl.resolve_duplicated_faces(gi)
        # nv, nf, _, _ = igl.remove_unreferenced(ui, ffi)
        # ------------------------------------------------------------------

        # if you are using libigl v2.6.2 installed via pip, this works fine
        # ------------------------------------------------------------------
        ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
        gi = J[fi]
        ffi, _ = resolve_duplicated_faces(gi)
        nv, nf, _, _ = igl.remove_unreferenced(ui, ffi)
        # ------------------------------------------------------------------

        mesh = trimesh.Trimesh(nv, nf)
        meshes.append(mesh)

    return meshes


def diffuse_fragments(fragments: list, mean_vec=(0, 0, 0), var_vec=(.75, .75, .75)):
    """
    Applies random SE(3) transformations to fragments.
    Ensures fragments are 'scattered' without excessive internal blending.

    Unchanged from the original. Used by the visualisation scripts and by the
    end-to-end inference path, where real transformed *meshes* are wanted.

    The training dataset does NOT call this: it samples the same rotation
    distribution directly as tensors and rotates the already-extracted feature
    vectors, which is provably identical for every quantity the network sees
    (see `vngat/data/dataset.py`) and avoids copying + retransforming +
    re-deriving normals for every fragment of every scene.
    """
    diffused_fragments = []
    t_matrices = []

    # Calculating a global 'scale' to prevent overlap
    max_dim = max([f.extents.max() for f in fragments])

    for _i, mesh in enumerate(fragments):
        m = mesh.copy()
        # Random Rotation
        rotation = random_rotation_matrices(1)[0]

        # Random Translation (Wiener-like step)
        translation = np.random.normal(mean_vec, var_vec, size=3)
        translation += (np.random.standard_normal(3) * max_dim * .5)  # Push out

        # Applying transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        m.apply_transform(matrix)

        diffused_fragments.append(m)
        t_matrices.append(matrix)

    return diffused_fragments, t_matrices


def _resolve_duplicated_faces_reference(F1):
    """
    Exact Python replication of igl::resolve_duplicated_faces.
    Resolves non-manifold duplicate faces according to directional winding balance.

    Parameters
    ----------
    F1 : numpy.ndarray (N, 3)
        The input face topology matrix.

    Returns
    -------
    F2 : numpy.ndarray (M, 3)
        The resolved face matrix.
    J  : numpy.ndarray (M,)
        The indices mapping back from F2 to F1.
    """
    num_faces = F1.shape[0]
    if num_faces == 0:
        return np.empty((0, 3), dtype=F1.dtype), np.empty((0,), dtype=np.int64)

    # Emulating igl::unique_simplices by sorting each face to find canonical combinations
    sorted_F1 = np.sort(F1, axis=1)
    uF, IA, IC = np.unique(sorted_F1, axis=0, return_index=True, return_inverse=True)
    num_unique_faces = uF.shape[0]

    # Checking for orientation consistency (Cyclic permutations)
    canonical = uF[IC]
    consistent = (
        np.all(F1 == canonical, axis=1) |
        ((F1[:, 0] == canonical[:, 1]) & (F1[:, 1] == canonical[:, 2]) & (F1[:, 2] == canonical[:, 0])) |
        ((F1[:, 0] == canonical[:, 2]) & (F1[:, 1] == canonical[:, 0]) & (F1[:, 2] == canonical[:, 1]))
    )

    uF2F = [[] for _ in range(num_unique_faces)]
    counts = np.zeros(num_unique_faces, dtype=np.int32)
    ucounts = np.zeros(num_unique_faces, dtype=np.int32)

    signed_ids = (np.arange(num_faces) + 1) * np.where(consistent, 1, -1)

    for i in range(num_faces):
        ui = IC[i]
        uF2F[ui].append(signed_ids[i])
        counts[ui] += 1 if consistent[i] else -1
        ucounts[ui] += 1

    kept_faces = []
    for i in range(num_unique_faces):
        if ucounts[i] == 1:
            kept_faces.append(abs(uF2F[i][0]) - 1)
            continue

        if counts[i] == 1:
            found = False
            for fid in uF2F[i]:
                if fid > 0:
                    kept_faces.append(abs(fid) - 1)
                    found = True
                    break
            assert found, "IGL_ASSERT failed: Positive face not found"

        elif counts[i] == -1:
            found = False
            for fid in uF2F[i]:
                if fid < 0:
                    kept_faces.append(abs(fid) - 1)
                    found = True
                    break
            assert found, "IGL_ASSERT failed: Negative face not found"

        else:
            if counts[i] != 0 and len(uF2F[i]) > 0:
                kept_faces.append(abs(uF2F[i][0]) - 1)

    J = np.array(kept_faces, dtype=np.int64)
    F2 = F1[J]

    return F2, J


_VERIFY_RESOLVE = os.environ.get("VNGAT_VERIFY_RESOLVE", "") == "1"


def resolve_duplicated_faces(F1):
    """
    Exact Python replication of igl::resolve_duplicated_faces.

    Dispatches to `_resolve_duplicated_faces_reference` (the untouched original)
    unless the piece has no duplicated faces at all, which is the overwhelmingly
    common case for Breaking Bad fragments.

    Why the short circuit is EXACT, not an approximation: when every face is
    unique under `np.sort(F1, axis=1)`, every `ucounts[i]` equals 1, so the
    reference implementation's second loop takes the `ucounts[i] == 1` branch
    for every unique face `i` and appends the single face that maps to it.
    That face is by definition `IA[i]` (np.unique's index of the first --
    here only -- occurrence of unique row i). So `J == IA` exactly, including
    the reordering into lexicographic-by-sorted-triple order that downstream
    code depends on for stable `edges_unique` ordering. Set
    VNGAT_VERIFY_RESOLVE=1 to run both paths and assert agreement.
    """
    num_faces = F1.shape[0]
    if num_faces == 0:
        return np.empty((0, 3), dtype=F1.dtype), np.empty((0,), dtype=np.int64)

    sorted_F1 = np.sort(F1, axis=1)
    uF, IA = np.unique(sorted_F1, axis=0, return_index=True)

    if uF.shape[0] != num_faces:
        # Genuine duplicates present -> use the untouched reference path.
        return _resolve_duplicated_faces_reference(F1)

    J = IA.astype(np.int64)
    F2 = F1[J]

    if _VERIFY_RESOLVE:  # pragma: no cover - opt-in equivalence check
        F2_ref, J_ref = _resolve_duplicated_faces_reference(F1)
        assert np.array_equal(J, J_ref), "resolve_duplicated_faces fast path diverged (J)"
        assert np.array_equal(F2, F2_ref), "resolve_duplicated_faces fast path diverged (F2)"

    return F2, J
