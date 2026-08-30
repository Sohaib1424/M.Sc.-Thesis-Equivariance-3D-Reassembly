"""
Fracture-surface extraction.

A fragment's boundary splits into two parts:

* **original surface** -- was on the outside of the intact object. Smooth,
  and shared with nobody.
* **fracture surface** -- created by the break. Jagged, and geometrically
  coincident with the fracture surface of whichever fragment used to sit
  against it.

Only the fracture surface carries the mating information that reassembly
needs, so isolating it is the point of this module.

Two independent criteria are implemented, and they answer different questions:

``method="dihedral"``
    A per-fragment heuristic. A face is fracture if it sits in a region of
    high dihedral variation. Needs a single fragment and nothing else, so it
    works at inference time on a fragment in isolation. This is the original
    ``extract_fractures`` algorithm, preserved exactly.

``method="coincidence"``
    Exact, but needs the whole scene in its assembled frame: a face is
    fracture if its vertices coincide with vertices of a *different*
    fragment. This is how GARF defines its ground truth -- "we directly
    extract shared surfaces between connected fragments from the mesh,
    defining them as fracture surfaces". It is ground truth, not a heuristic,
    and it is available for the whole Breaking Bad dataset because the
    fragments are stored pre-separation in a common frame.

The two are worth keeping side by side: ``coincidence`` gives labels to
supervise against, ``dihedral`` gives something computable when no
neighbours are known. :func:`fracture_agreement` scores one against the other.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from ..arrays import compact_indices, pair_key, unique_sorted
from .topology import face_adjacency, face_normals

DEFAULT_SHARP_THRESHOLD = 0.9


class FractureMask(NamedTuple):
    """Face-level fracture labelling plus the diagnostics behind it."""
    face_mask: np.ndarray        # (F,) bool -- faces on the fracture surface
    sharp_neighbours: np.ndarray # (F,) int  -- neighbours across a sharp dihedral
    total_neighbours: np.ndarray # (F,) int  -- edge-adjacent neighbours
    degenerate_faces: int        # zero-area faces found while computing normals


# Quantiles tried, in order, when an absolute threshold labels nothing. Each
# says "treat the roughest q of THIS fragment's dihedrals as sharp", which is
# free of both mesh resolution and object scale. q = 1.0 marks every adjacency
# sharp, so the whole fragment is labelled -- the correct answer for a small
# shard that really is fracture surface on every side.
ADAPTIVE_QUANTILES = (0.25, 0.5, 0.75, 1.0)


def _mask_at_threshold(faces, normals, src, dst, cosine, n_faces, n_vertices,
                       threshold):
    total_neighbours = np.bincount(src, minlength=n_faces)
    if src.size:
        sharp_neighbours = np.bincount(
            src, weights=np.abs(cosine) < threshold, minlength=n_faces)
    else:
        sharp_neighbours = np.zeros(n_faces, dtype=np.float64)

    touches_sharp = np.zeros(n_vertices, dtype=bool)
    sharp_faces = np.flatnonzero(sharp_neighbours > 0)
    if sharp_faces.size:
        # no np.unique needed -- writing True twice is the same as writing it once
        touches_sharp[faces[sharp_faces].ravel()] = True

    all_vertices_sharp = touches_sharp[faces].all(axis=1)
    majority_sharp = (total_neighbours // 2) <= sharp_neighbours
    return all_vertices_sharp & majority_sharp, sharp_neighbours, total_neighbours


def fracture_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    sharp_threshold: float = DEFAULT_SHARP_THRESHOLD,
    has_duplicate_faces: bool | None = None,
    min_faces: int = 0,
) -> FractureMask:
    """
    Dihedral-based fracture labelling.

    A neighbour pair is *sharp* when ``|n_i . n_j| < sharp_threshold`` --
    note the absolute value, so a fold back on itself (dot near -1) counts as
    smooth, not sharp. A face is kept when **both** hold:

    1. every one of its three vertices touches at least one face that has a
       sharp neighbour, and
    2. at least half of its own neighbours are sharp.

    Condition 2 is what stops the mask bleeding across the rim onto the smooth
    outer surface: a face merely *adjacent* to the jagged region satisfies (1)
    but not (2). It was present in the original as a comment and a computed
    variable, but was not applied to the output; it is applied here.

    Empty masks, and why ``min_faces`` exists
    -----------------------------------------
    Every fragment of a broken object touches at least one other, so a fragment
    with *no* fracture surface is impossible -- yet the full-dataset run reports
    ``frac_vertices`` min = 0. That is the heuristic failing, not a property of
    the data.

    The mechanism, measured rather than assumed: an absolute dihedral threshold
    asks whether adjacent faces differ by more than ~26 degrees, and on a small,
    gently curved shard *no* pair does. Every adjacency is then smooth, no face
    has a sharp neighbour, and the mask is empty. (A near-flat 6-face shard goes
    from 6/6 labelled to 0/6 as its tilt drops from 0.5 to 0.2.) The threshold
    is absolute while roughness is relative to triangle size, so the test is
    resolution-dependent in a way a fracture surface is not.

    Setting ``min_faces >= 1`` relaxes the threshold **for that fragment only**,
    to the quantiles in ``ADAPTIVE_QUANTILES``, until at least that many faces
    are labelled. Fragments that already produce a mask are untouched, so this
    cannot change results that were already sensible -- unlike lowering the
    global threshold, which the sweep in ``scripts.tune_sharp_threshold`` shows
    also floods smooth fragments with false positives.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    if n_faces == 0:
        z_i = np.zeros(0, dtype=np.int64)
        return FractureMask(np.zeros(0, dtype=bool), z_i, z_i, 0)

    src, dst = face_adjacency(faces, n_vertices, has_duplicate_faces=has_duplicate_faces)
    normals, n_degenerate = face_normals(vertices, faces, return_degenerate=True)
    cosine = (np.einsum("ij,ij->i", normals[src], normals[dst])
              if src.size else np.zeros(0))

    mask, sharp_neighbours, total_neighbours = _mask_at_threshold(
        faces, normals, src, dst, cosine, n_faces, n_vertices, sharp_threshold)

    if min_faces > 0 and int(mask.sum()) < min_faces and src.size:
        magnitude = np.abs(cosine)
        for quantile in ADAPTIVE_QUANTILES:
            # nextafter so the quantile value itself counts as sharp; without it
            # q = 1.0 leaves the single largest |cos| unmarked and a fragment
            # whose dihedrals are all identical stays empty.
            relaxed = np.nextafter(float(np.quantile(magnitude, quantile)), np.inf)
            mask, sharp_neighbours, total_neighbours = _mask_at_threshold(
                faces, normals, src, dst, cosine, n_faces, n_vertices, relaxed)
            if int(mask.sum()) >= min_faces:
                break

    return FractureMask(
        face_mask=mask,
        sharp_neighbours=sharp_neighbours.astype(np.int64),
        total_neighbours=total_neighbours.astype(np.int64),
        degenerate_faces=n_degenerate,
    )


def extract_fracture_surface(
    mesh,
    sharp_threshold: float = DEFAULT_SHARP_THRESHOLD,
    compact: bool = True,
    return_mask: bool = False,
    min_faces: int = 0,
):
    """
    The fracture-surface sub-mesh of a single fragment.

    ``compact=True`` drops vertices that no surviving face references, which
    is what you want for a standalone surface. ``compact=False`` keeps the
    parent vertex array intact so face and vertex indices stay comparable
    with the input mesh -- use it when the mask has to line up with
    per-vertex features computed on the full fragment.

    Returns a ``trimesh.Trimesh``, or ``(mesh, FractureMask)`` with
    ``return_mask=True``.
    """
    import trimesh

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    result = fracture_face_mask(vertices, faces, sharp_threshold,
                                min_faces=min_faces)
    kept_faces = faces[result.face_mask]

    if compact:
        kept, new_faces = compact_indices(kept_faces, vertices.shape[0])
        out = trimesh.Trimesh(vertices[kept], new_faces, process=False)
    else:
        out = trimesh.Trimesh(vertices, kept_faces, process=False)

    return (out, result) if return_mask else out


# --------------------------------------------------------------------------
# Exact labelling from cross-fragment coincidence
# --------------------------------------------------------------------------

def fracture_vertex_masks(
    fragments: Sequence,
    tol: float = 1e-6,
    round_decimals: int = 6,
) -> list[np.ndarray]:
    """
    Ground-truth fracture vertices, from coincidence between fragments.

    A vertex is on the fracture surface exactly when some *other* fragment has
    a vertex at the same location. Because Breaking Bad stores fragments in
    their assembled frame, this is decidable without any heuristic.

    ``fragments`` may be meshes or raw ``(N, 3)`` vertex arrays, and must all
    be in the same (unperturbed, un-centralised) coordinate frame. Returns one
    boolean array per fragment.

    Matching is done by rounding to ``round_decimals`` places and grouping,
    then -- for anything left unmatched -- a KD-tree query within ``tol``.
    The rounded pass catches the large majority at a fraction of the cost.
    """
    from scipy.spatial import cKDTree

    points = [
        np.asarray(getattr(f, "vertices", f), dtype=np.float64) for f in fragments
    ]
    sizes = np.array([len(p) for p in points], dtype=np.int64)
    if len(points) < 2 or sizes.sum() == 0:
        return [np.zeros(s, dtype=bool) for s in sizes]

    coords = np.concatenate(points, axis=0)
    owner = np.repeat(np.arange(len(points), dtype=np.int64), sizes)
    shared = np.zeros(coords.shape[0], dtype=bool)

    # pass 1 -- exact match on rounded coordinates
    rounded = np.round(coords, decimals=round_decimals)
    view = np.ascontiguousarray(rounded).view(
        np.dtype((np.void, rounded.dtype.itemsize * 3))
    ).ravel()
    _, group, counts = np.unique(view, return_inverse=True, return_counts=True)
    group = group.ravel()

    n_groups = counts.shape[0]
    lo = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(n_groups, -1, dtype=np.int64)
    np.minimum.at(lo, group, owner)
    np.maximum.at(hi, group, owner)
    multi_fragment = lo != hi                       # group spans >= 2 fragments
    shared |= multi_fragment[group]

    # pass 2 -- tolerance match for the rest
    rest = np.flatnonzero(~shared)
    if rest.size > 1:
        tree = cKDTree(coords[rest])
        pairs = tree.query_pairs(r=tol, output_type="ndarray")
        if pairs.size:
            a, b = rest[pairs[:, 0]], rest[pairs[:, 1]]
            cross = owner[a] != owner[b]
            shared[a[cross]] = True
            shared[b[cross]] = True

    return list(np.split(shared, np.cumsum(sizes)[:-1]))


def fracture_face_mask_from_vertices(
    faces: np.ndarray, vertex_mask: np.ndarray, require: str = "all"
) -> np.ndarray:
    """Lift a vertex-level mask to faces. ``require`` is ``"all"`` or ``"any"``."""
    if faces.size == 0:
        return np.zeros(0, dtype=bool)
    hit = vertex_mask[faces]
    return hit.all(axis=1) if require == "all" else hit.any(axis=1)


def fracture_agreement(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """
    Compare a predicted fracture mask against a ground-truth one.

    Returns precision / recall / F1 / IoU over the positive class. Use it to
    calibrate ``sharp_threshold``: run both methods over a sample of scenes
    and pick the threshold that maximises F1 rather than guessing.
    """
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "iou": iou}


def boundary_edges(faces: np.ndarray, num_vertices: int | None = None) -> np.ndarray:
    """
    Edges used by exactly one face -- the rim of an open surface.

    On an extracted fracture surface this traces the outline where the break
    met the original outer surface, which is a useful visual sanity check
    that the extraction did not bleed.
    """
    from .topology import half_edges

    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    n = int(faces.max()) + 1 if num_vertices is None else int(num_vertices)
    lo, hi = half_edges(faces)
    keys = pair_key(lo, hi, n)
    order = np.argsort(keys)
    ks = keys[order]
    new = np.empty(ks.shape[0], dtype=bool)
    new[0] = True
    if ks.shape[0] > 1:
        np.not_equal(ks[1:], ks[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, ks.shape[0]))
    single = ks[starts[sizes == 1]]
    n64 = np.int64(n)
    return np.stack([single // n64, single % n64], axis=1)


__all__ = [
    "DEFAULT_SHARP_THRESHOLD",
    "FractureMask",
    "boundary_edges",
    "extract_fracture_surface",
    "fracture_agreement",
    "fracture_face_mask",
    "fracture_face_mask_from_vertices",
    "fracture_vertex_masks",
]
