"""
Score a predicted assembly the way Breaking Bad's tables do.

Rotation metrics need only the network. Translation RMSE, Chamfer distance and
part accuracy need an *assembly*, which is what :func:`score_batch` builds: it
applies the predicted rotations, runs the translation solver
(:mod:`reassembly.assembly.translation`) scene by scene, and compares the result
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

Chamfer runs on an evenly strided subset of each fragment's vertices -- the
same subset for prediction and truth, which share their vertex order -- because
a fragment can have 83,000 vertices and the distance is quadratic.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..evaluation.metrics import chamfer_distance, part_accuracy
from ..nn.losses import euler_rmse, geodesic_angle
from .translation import assemble, subsample_per_fragment


@torch.no_grad()
def score_batch(batch, prediction, *, threshold: float = 0.01,
                max_match_points: int = 2048, chamfer_points: int = 2048,
                iterations: int = 5, huber: float = 0.05,
                collision: bool = False, anchor: bool = True) -> List[Dict[str, float]]:
    """
    One dict per scene: ``geodesic_deg`` and ``euler_rmse_deg`` (the scene's
    own means, which need no solver), ``rmse_t``, ``chamfer``,
    ``part_chamfer``, ``part_accuracy``, ``matches`` and ``fragments`` -- plus,
    skipped by averaging, ``_part_chamfer`` (per fragment), ``_translation``
    (the solved ``(F, 3)`` translations in world units: relative to the
    anchor's with ``anchor``, zero-mean without), ``_rotation`` (the ``(F, 3,
    3)`` rotations the fragments were placed with) and ``_anchor`` (the
    anchor's index within the scene, or -1).

    ``batch`` is a :class:`~reassembly.data.features.Batch` on any device and
    ``prediction`` the model's output for it. ``anchor`` picks the convention
    (module docstring).
    """
    from ..data.features import complete_batch
    from ..nn.anchor import anchor_alignment
    from ..nn.model import apply_rotation

    batch = complete_batch(batch)
    fragment = batch.vertex_fragment
    unit = batch.unit.to(batch.node_features.dtype)[fragment, None]
    rotation = prediction.rotation.to(batch.node_features.dtype)
    if anchor:
        rotation, scored = anchor_alignment(batch, rotation)
    else:
        scored = torch.ones(rotation.shape[0], dtype=torch.bool, device=rotation.device)
    points = apply_rotation(batch.node_features[:, 0, :], rotation, fragment) * unit
    normals = apply_rotation(batch.node_features[:, 1, :], rotation, fragment)
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
        radii = None
        if collision:
            reach = points[v0:v1].norm(dim=-1)
            radii = torch.stack([reach[local == f].max() for f in range(count)])
        t_pred, matches = assemble(
            points[v0:v1], normals[v0:v1], local, embedding[v0:v1], count,
            candidates=None if batch.fracture is None else batch.fracture[v0:v1],
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
        placed = points[v0:v1].double() + t_pred[local]
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
        scenes.append({
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
            "_anchor": reference,
        })
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
