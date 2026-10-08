"""
Score a predicted assembly the way Breaking Bad's tables do.

Translation RMSE, Chamfer distance and part accuracy need an *assembly*, which
is what :func:`score_batch` builds: it takes each fragment's rotation from the
embedding matches (:mod:`reassembly.assembly.rotation`), places the turned
fragments (:mod:`reassembly.assembly.placement`, or the global solve of
:mod:`reassembly.assembly.translation`) scene by scene, and compares the result
with the ground-truth assembly -- all in **world units**, which is where the
0.01 part-accuracy threshold is defined. (The network sees per-scene normalised
coordinates; the batch carries the divisor and centroids that undo that.)

Conventions, stated because published numbers depend on them:

* **Anchor** (the default, ``anchor=True``). An assembly is only defined up to
  one global rotation and translation, so -- as in GARF and PuzzleFusion++ --
  each scene's largest fragment is set to its true pose and every other
  fragment is scored relative to it: the predicted rotations are aligned so the
  anchor's is exact (:mod:`reassembly.nn.anchor`), and translations are
  measured from the anchor's. The anchor itself is left out of every
  per-fragment number (rotation, RMSE(T), part Chamfer, part accuracy), where
  it would be a free perfect score; the whole-shape Chamfer includes it, since
  it is part of the shape.
* **Mean gauge** (``anchor=False``, the earlier convention). Rotations as
  predicted; both the prediction and the truth centred on the mean of their
  fragment positions before they are compared.
* **RMSE(T)** is per fragment -- the root mean square over x, y, z -- then
  averaged over the scene's scored fragments.
* **Chamfer** is the mean of *squared* nearest-neighbour distances, both ways
  (:func:`reassembly.evaluation.metrics.chamfer_distance`). ``chamfer`` is the
  whole assembled shape against the whole true shape, the benchmark's "CD";
  ``part_chamfer`` is the mean over fragments of each fragment against itself.
* **Part accuracy** is the fraction of scored fragments whose own Chamfer
  distance is below 0.01.
* Scenes are averaged **per scene, then over scenes** by the caller, as the
  benchmark does. Training logs are fragment-weighted instead; the two differ
  when scenes differ in fragment count.
* **Rotations** are fitted from the embedding matches and chained from the
  anchor (:func:`reassembly.assembly.rotation.match_batch`); the network has
  had no rotation head since v7. The placement and every score below use them.
* **Placement** is, by default, from the pair fits the chain agrees with, the
  anchor held (``placement="checked"``, :mod:`reassembly.assembly.placement`),
  or one least-squares solve over every embedding match (``placement="global"``,
  :mod:`reassembly.assembly.translation`; the only placement before v7). The
  global solve contracts many-piece assemblies onto their centre when many
  matches are wrong; that module says by how much.
* **Perturbed inputs** (``observed`` and ``drop``; ``--evaluate --jitter
  --drop``, :mod:`reassembly.evaluation.noise`): the method reads noisy
  coordinates and fewer break vertices, and the score still measures the
  predicted pose on the clean geometry.

Chamfer runs on an evenly strided subset of each fragment's vertices -- the
same subset for prediction and truth, which share their vertex order -- because
a fragment can have 83,000 vertices and the distance is quadratic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..evaluation.metrics import chamfer_distance, part_accuracy
from ..evaluation.noise import check_noise
from ..nn.losses import euler_rmse, geodesic_angle
from .placement import AGREEMENT_DEG, place
from .rotation import (INLIER_DISTANCE, MIN_MATCHES, RANSAC_ITERATIONS, MatchedRotations,
                       match_batch)
from .translation import assemble, subsample_per_fragment

PLACEMENTS = ("checked", "global")


def check_placement(placement: Optional[str]) -> str:
    """The placement to use -- ``"checked"`` when ``None`` -- or a
    ``ValueError`` naming the choices."""
    placement = "checked" if placement is None else placement
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    return placement


@torch.no_grad()
def score_batch(batch, prediction, *, threshold: float = 0.01,
                max_match_points: int = 2048, chamfer_points: int = 2048,
                iterations: int = 5, huber: float = 0.05,
                collision: bool = False, anchor: bool = True,
                min_matches: int = MIN_MATCHES,
                inlier_distance: float = INLIER_DISTANCE,
                ransac_iterations: int = RANSAC_ITERATIONS,
                seed: int = 0, placement: Optional[str] = None,
                agreement_deg: float = AGREEMENT_DEG, observed=None,
                drop: float = 0.0,
                matched: Optional[MatchedRotations] = None) -> List[Dict[str, float]]:
    """
    One dict per scene: ``geodesic_deg`` and ``euler_rmse_deg`` (the scene's
    own means), ``rmse_t``, ``chamfer``, ``part_chamfer``, ``part_accuracy``,
    ``matches``, ``fragments`` and ``matched_share`` (the scored fragments the
    chain reached; the rest keep a rotation unrelated to their true one) --
    plus, skipped by averaging, ``_part_chamfer`` (per fragment),
    ``_translation`` (the solved ``(F, 3)`` translations in world units:
    relative to the anchor's with ``anchor``, zero-mean without),
    ``_rotation`` (the ``(F, 3, 3)`` rotations the fragments were placed
    with), ``_scored_geodesic_deg`` (the scored fragments' angles, for
    fragment-weighted means), ``_reached`` (per fragment) and ``_anchor`` (the
    anchor's index within the scene, or -1). With ``placement="checked"``, also
    ``verified_matches``: the inliers of the verified pairs the reached
    fragments were placed by (``matches`` stays the count of every embedding
    match, as under the global solve).

    ``batch`` is a :class:`~reassembly.data.features.Batch` on any device and
    ``prediction`` the model's output for it. The rotations are ``matched`` --
    :func:`~reassembly.assembly.rotation.match_batch` on what the method was
    shown, as ``training._forward`` has already computed them -- or, when
    ``None``, matched here from ``prediction.vertex_embedding`` with
    ``min_matches``, ``inlier_distance``, ``ransac_iterations`` and ``seed``.
    ``anchor`` picks the convention (module docstring). ``placement`` picks
    the translation solve -- ``None`` is ``"checked"`` -- and ``agreement_deg``
    how far a pair's fit may be from the chained rotations and still place
    fragments (:mod:`reassembly.assembly.placement`).

    ``observed`` is the batch the method was shown, when that is not ``batch``
    -- ``batch`` with noise on its inputs (``--evaluate --jitter``,
    :mod:`reassembly.evaluation.noise`). The matching, the fits and the solve
    read the observed coordinates; the score applies the predicted pose to
    ``batch``'s clean ones, so the noise is in what the method saw and not in
    what it is measured against. ``drop`` leaves that share of the break
    vertices out of the matching (``--drop``), drawn per scene from ``seed``;
    it is ``matched``'s own when that is given.
    """
    from ..data.features import complete_batch
    from ..nn.anchor import anchor_alignment
    from ..nn.model import apply_rotation

    placement = check_placement(placement)
    check_noise(0.0, drop)
    checked = placement == "checked"
    batch = complete_batch(batch)
    # What the method reads; the score reads `batch`.
    seen = batch if observed is None else complete_batch(observed)
    if seen.node_features.shape != batch.node_features.shape:
        raise ValueError("observed must be the same scenes as batch: "
                         f"{tuple(seen.node_features.shape)} against "
                         f"{tuple(batch.node_features.shape)} input features")
    if matched is None:
        matched = match_batch(seen, prediction.vertex_embedding, seed=seed, drop=drop,
                              max_points=max_match_points, min_matches=min_matches,
                              tau=inlier_distance, iterations=ransac_iterations)
    fragment = batch.vertex_fragment
    unit = batch.unit.to(batch.node_features.dtype)[fragment, None]
    rotation = matched.rotation.to(batch.node_features.dtype)
    if anchor:
        # The root is its own frame's identity; the anchor protocol sets it to
        # its true pose, and every fragment chained from it moves with it.
        rotation, scored = anchor_alignment(batch, rotation)
    else:
        scored = torch.ones(rotation.shape[0], dtype=torch.bool, device=rotation.device)
    points = apply_rotation(seen.node_features[:, 0, :], rotation, fragment) * unit
    normals = apply_rotation(seen.node_features[:, 1, :], rotation, fragment)
    truth_local = batch.target_vertices * unit
    embedding = prediction.vertex_embedding

    scenes: List[Dict[str, float]] = []
    fragment_ptr = batch.fragment_ptr.tolist()
    vertex_ptr = batch.vertex_ptr.tolist()
    for scene in range(batch.num_scenes):
        f0, f1 = fragment_ptr[scene], fragment_ptr[scene + 1]
        v0, v1 = vertex_ptr[f0], vertex_ptr[f1]
        count = f1 - f0
        local = fragment[v0:v1] - f0
        reached = matched.reached[f0:f1]
        pairs = matched.pairs[scene]
        candidates = matched.candidates[scene]
        radii = None
        if collision:
            reach = points[v0:v1].norm(dim=-1)
            radii = torch.stack([reach[local == f].max() for f in range(count)])
        links = None
        if checked and count:
            # From the fits the chain agrees with, the root held -- not one
            # solve over every match (assembly/placement.py).
            placed_scene = place(
                points[v0:v1], normals[v0:v1], local, embedding[v0:v1], count, pairs,
                rotation[f0:f1], reached, matched.roots[scene], candidates=candidates,
                max_points=max_match_points, iterations=iterations, huber=huber,
                min_inliers=min_matches, agreement=agreement_deg, collision_radii=radii,
            )
            t_pred, matches, links = placed_scene
        else:
            t_pred, matches = assemble(
                points[v0:v1], normals[v0:v1], local, embedding[v0:v1], count,
                candidates=candidates,
                max_points=max_match_points, iterations=iterations, huber=huber,
                collision_radii=radii,
            )
        centroid = batch.centroid[f0:f1].double()
        t_pred = t_pred.double()
        own = scored[f0:f1]
        reference = -1
        if anchor and not bool(own.all()):
            # The anchor's translation is the gauge: it sits at its true
            # position, and every other fragment is placed relative to it.
            reference = int((~own).nonzero()[0])
            t_true = centroid - centroid[reference]
            t_pred = t_pred - t_pred[reference]
        else:
            t_true = centroid - centroid.mean(dim=0, keepdim=True)
            if links is not None:
                # The checked placement holds the root at 0; the global
                # solve's translations are zero-mean already.
                t_pred = t_pred - t_pred.mean(dim=0, keepdim=True)
        if observed is None:
            placed = points[v0:v1].double() + t_pred[local]
        else:
            # The predicted pose on the clean fragment: the noise was the input's.
            shape = apply_rotation(batch.node_features[v0:v1, 0, :], rotation[f0:f1],
                                   local) * unit[v0:v1]
            placed = shape.double() + t_pred[local]
        truth = truth_local[v0:v1].double() + t_true[local]

        sample = subsample_per_fragment(local, chamfer_points)
        per_part = torch.stack([
            chamfer_distance(placed[sample][local[sample] == f],
                             truth[sample][local[sample] == f])
            for f in range(count)
        ])
        predicted = rotation[f0:f1].double()
        target = batch.target_rotation[f0:f1].double()
        angle = torch.rad2deg(geodesic_angle(predicted, target))
        error_t = (t_pred - t_true).pow(2).mean(dim=-1).sqrt()
        parts = per_part[own]
        finite = torch.isfinite(parts)
        scored_any = bool(own.any())
        nothing = float("nan")
        entry = {
            # NaN when a scene has no scored fragment (a single fragment, under
            # the anchor): the per-scene averages skip it rather than count 0.
            "geodesic_deg": float(angle[own].mean()) if scored_any else nothing,
            "euler_rmse_deg": float(euler_rmse(predicted[own], target[own]))
            if scored_any else nothing,
            "rmse_t": float(error_t[own].mean()) if scored_any else nothing,
            "chamfer": float(chamfer_distance(placed[sample], truth[sample])),
            "part_chamfer": float(parts[finite].mean()) if bool(finite.any()) else nothing,
            "part_accuracy": float(part_accuracy(parts, threshold)) if scored_any else nothing,
            "matches": float(matches.source.numel()),
            "fragments": float(count),
            # Per fragment, for dumps and figures; underscored so averaging
            # and reporting code leaves them alone.
            "_part_chamfer": per_part.tolist(),
            "_translation": t_pred.tolist(),
            "_rotation": predicted.tolist(),
            "_scored_geodesic_deg": angle[own].tolist(),
            "_anchor": reference,
            "matched_share": (float(reached[own].double().mean()) if scored_any
                              else nothing),
            "_reached": reached.tolist(),
        }
        if links is not None:
            entry["verified_matches"] = float(links.source.numel())
        scenes.append(entry)
    return scenes


def mean_over_scenes(scenes: List[Dict[str, float]]) -> Dict[str, float]:
    """The benchmark's averaging: per scene, then over scenes, NaN-skipping."""
    if not scenes:
        return {}
    out: Dict[str, float] = {}
    for key in sorted({k for s in scenes for k in s if not k.startswith("_")}):
        values = [s[key] for s in scenes
                  if isinstance(s.get(key), (int, float)) and s[key] == s[key]]
        if values:
            out[key] = sum(values) / len(values)
    return out
