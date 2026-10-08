"""
Scene visualisation.

Replaces the five near-identical ``vis_*.py`` scripts. They differed only in
which of three independent switches were on -- perturb the fragments or not,
show full fragments or fracture surfaces, offset the two so they sit side by
side -- so they are switches here rather than files.

Rendering is optional: :func:`build_scene` returns a ``trimesh.Scene`` and
never calls ``.show()``, so it is usable from a notebook or a headless box.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Distinguishable colours; random per-call colours make two runs impossible to
# compare, which is the opposite of what a diagnostic view is for.
PALETTE = np.array([
    [ 66, 133, 244], [219,  68,  55], [244, 180,   0], [ 15, 157,  88],
    [156,  39, 176], [  0, 172, 193], [255, 112,  67], [ 92, 107, 192],
    [141, 110,  99], [ 38, 166, 154], [236,  64, 122], [124, 179,  66],
], dtype=np.uint8)


def fragment_colour(index: int, alpha: int = 255) -> np.ndarray:
    rgb = PALETTE[index % len(PALETTE)]
    return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8)


def build_scene(
    fragments: Sequence,
    show: str = "full",
    offset: Optional[Sequence[float]] = None,
    sharp_threshold: Optional[float] = None,
    alpha: int = 255,
    method: str = "dihedral",
    truth_masks=None,
):
    """
    Assemble a ``trimesh.Scene`` from a list of fragments.

    ``show`` selects what geometry to draw:

    ``"full"``
        the fragments as loaded.
    ``"fracture"``
        only the extracted fracture surfaces.
    ``"both"``
        both, with the fracture surfaces translated by ``offset`` so they sit
        beside the fragments instead of inside them. ``offset=None`` picks a
        sensible one from the scene's bounding box, rather than the original's
        hard-coded ``[5.5, 5.5, 5.5]`` which is wrong for any object not at
        Breaking Bad's default scale.

    ``method`` selects the labelling:

    ``"dihedral"``
        the per-fragment heuristic, computable at inference.
    ``"coincidence"``
        ground truth -- a vertex is fracture exactly when another fragment has
        a vertex at the same point. Needs the assembled scene, so it is
        unavailable at inference, but it is exact.

    Rendering both is the fastest way to see how far apart they are: on
    thin-walled Everyday objects the heuristic also lights up handles, rims and
    feet, which are sharp because the object was *designed* that way.

    Each fragment keeps one colour across both copies, so a piece can be
    matched to its own fracture surface by eye.
    """
    import trimesh

    from ..mesh.fracture import (
        extract_fracture_surface,
        fracture_face_mask_from_vertices,
        fracture_vertex_masks,
    )

    scene = trimesh.Scene()
    if not fragments:
        return scene

    if show in ("fracture", "both"):
        if method == "coincidence":
            # One pass over the whole scene: coincidence is a relation between
            # fragments, so it cannot be computed one fragment at a time.
            #
            # `truth_masks` must be supplied whenever the fragments have been
            # perturbed. Coincidence is defined in the ASSEMBLED frame; once
            # `diffuse_fragments` has scattered the pieces nothing coincides any
            # more and recomputing here would silently return an all-false mask
            # -- an empty fracture surface that looks like a labelling result
            # rather than the bug it is.
            masks = (truth_masks if truth_masks is not None
                     else fracture_vertex_masks(fragments))
            surfaces = []
            for fragment, vertex_mask in zip(fragments, masks):
                faces = np.asarray(fragment.faces)
                keep = fracture_face_mask_from_vertices(faces, vertex_mask)
                surfaces.append(trimesh.Trimesh(
                    np.asarray(fragment.vertices), faces[keep], process=False))
        else:
            kwargs = {} if sharp_threshold is None else {"sharp_threshold": sharp_threshold}
            surfaces = [extract_fracture_surface(f, **kwargs) for f in fragments]
    else:
        surfaces = None

    if show == "both" and offset is None:
        extents = np.max([f.extents for f in fragments], axis=0)
        offset = np.array([extents.max() * 2.5, 0.0, 0.0])

    for index, fragment in enumerate(fragments):
        colour = fragment_colour(index, alpha)
        if show in ("full", "both"):
            piece = fragment.copy()
            piece.visual.face_colors = colour
            scene.add_geometry(piece)
        if surfaces is not None:
            surface = surfaces[index]
            if len(surface.faces) == 0:
                continue
            if show == "both":
                surface = surface.copy()
                surface.apply_translation(np.asarray(offset, dtype=float))
            surface.visual.face_colors = colour
            scene.add_geometry(surface)
    return scene


def describe(fragments: Sequence, sharp_threshold: Optional[float] = None,
             truth_masks=None) -> str:
    """
    Per-fragment table comparing both labellings.

    Both are shown because on real Breaking Bad fragments they disagree a lot:
    measured over the full dataset, the dihedral mask is ~3.9x larger than the
    true fracture surface at the default threshold. Printing only one hides
    that.
    """
    from ..mesh.fracture import (
        DEFAULT_SHARP_THRESHOLD,
        fracture_face_mask,
        fracture_face_mask_from_vertices,
        fracture_vertex_masks,
    )

    threshold = sharp_threshold or DEFAULT_SHARP_THRESHOLD
    if truth_masks is None and len(fragments) > 1:
        truth_masks = fracture_vertex_masks(fragments)

    lines = [f"{'piece':>5}  {'verts':>7} {'faces':>7}  "
             f"{'dihedral':>9} {'true':>7}  {'ratio':>6}"]
    for index, fragment in enumerate(fragments):
        vertices = np.asarray(fragment.vertices)
        faces = np.asarray(fragment.faces)
        mask = fracture_face_mask(vertices, faces, threshold).face_mask
        n_pred = int(mask.sum())
        if truth_masks is None:
            lines.append(f"{index:>5}  {len(vertices):>7} {len(faces):>7}  "
                         f"{n_pred:>9} {'-':>7}  {'-':>6}")
            continue
        truth = fracture_face_mask_from_vertices(faces, truth_masks[index])
        n_true = int(truth.sum())
        ratio = f"{n_pred / n_true:.2f}x" if n_true else "-"
        lines.append(f"{index:>5}  {len(vertices):>7} {len(faces):>7}  "
                     f"{n_pred:>9} {n_true:>7}  {ratio:>6}")
    return "\n".join(lines)
