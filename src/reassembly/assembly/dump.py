"""
One scored scene as the ``.npz`` dump every viewer reads -- the Visualizer
app, ``scripts/visualize_reassembly.py`` and ``scripts/render_gif.py``.

Written for one scene by ``scripts/dump_prediction.py`` and for every scene of
an evaluation by ``--evaluate --predictions`` (``training.evaluate``), from the
same function, so the two cannot drift apart.

Contents (plain arrays with offsets, so ``np.load`` needs no ``allow_pickle``)::

    vertices        (sum V, 3)  every fragment, assembled, world units
    faces           (sum F, 3)  indices local to each fragment
    vertex_offsets  (F+1,)      fragment f is vertices[o[f]:o[f+1]]
    face_offsets    (F+1,)
    centroids       (F, 3)      each fragment's true centroid
    A               (F, 3, 3)   the rotation the model was shown (the scatter)
    shift           (F, 3)      a display-only scatter offset
    R_pred, R_gt    (F, 3, 3)   predicted rotation, and the truth A^T
                                (R_pred with the scene's largest fragment set to
                                its true pose -- reassembly.nn.anchor)
    placement       (F, 3)      the solver's predicted centroid, world units,
                                measured from the anchor's true centroid
    geodesic_deg, tilt_deg, twist_deg, part_chamfer   (F,)
    rmse_t, chamfer, part_accuracy, matches           scalars for the scene
    scene, checkpoint, seed, epoch, rotations   (rotations: "matched" -- fitted
                                                 from the embedding matches)
    placement_method            checked or global

A fragment's vertex ``v`` with centroid ``c`` is shown scattered at
``A (v - c) + c + shift``, and reassembled at ``R_pred A (v - c) + placement``.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def dump_arrays(meshes: Sequence, scene: Dict, target_rotation, *, axis: str = "z",
                key: str = "", checkpoint: str = "", epoch: int = -1, seed: int = -1,
                placement: str = "checked") -> Dict[str, np.ndarray]:
    """
    The dump of one scene: ``meshes`` its fragments as loaded (assembled,
    world units; trimesh or anything with ``vertices`` and ``faces``), ``scene``
    its record from :func:`~reassembly.assembly.scoring.score_batch`, and
    ``target_rotation`` its ``(F, 3, 3)`` labels. ``seed`` is the scatter seed
    the scene was built with (-1: its own validation draw); it also seeds the
    display-only offsets. ``placement`` names the solve that placed it.
    """
    import torch

    from ..evaluation.metrics import swing_twist_error
    from ..nn.losses import geodesic_angle

    count = len(meshes)
    R_pred = torch.as_tensor(np.asarray(scene["_rotation"]), dtype=torch.float64)
    R_gt = torch.as_tensor(target_rotation).detach().double().cpu()
    A = R_gt.transpose(-1, -2)                     # the label is A^T
    geodesic = torch.rad2deg(geodesic_angle(R_pred, R_gt)).numpy()
    tilt, twist = swing_twist_error(R_pred, R_gt, axis=axis)

    centroids = np.stack([np.asarray(m.vertices, dtype=np.float64).mean(0) for m in meshes])
    # The solver's translations are relative to the anchor's, which sits at
    # its true centroid; without an anchor (one fragment) they are zero-mean
    # about the mean centroid.
    reference = int(scene["_anchor"])
    origin = centroids[reference] if reference >= 0 else centroids.mean(0)
    placed = origin + np.asarray(scene["_translation"])

    rng = np.random.default_rng(max(seed, 0) + 1)
    extent = float(max(np.ptp(np.asarray(m.vertices), axis=0).max() for m in meshes))
    shift = rng.standard_normal((count, 3)) * extent * 0.9

    vertices = [np.asarray(m.vertices, np.float32) for m in meshes]
    faces = [np.asarray(m.faces, np.int32) for m in meshes]
    return dict(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces),
        vertex_offsets=np.cumsum([0] + [len(v) for v in vertices]).astype(np.int64),
        face_offsets=np.cumsum([0] + [len(f) for f in faces]).astype(np.int64),
        centroids=centroids.astype(np.float32), A=A.numpy().astype(np.float32),
        shift=shift.astype(np.float32),
        R_pred=R_pred.numpy().astype(np.float32), R_gt=R_gt.numpy().astype(np.float32),
        placement=placed.astype(np.float32),
        geodesic_deg=geodesic.astype(np.float32),
        tilt_deg=tilt.numpy().astype(np.float32), twist_deg=twist.numpy().astype(np.float32),
        part_chamfer=np.asarray(scene["_part_chamfer"], np.float32),
        rmse_t=np.float32(scene["rmse_t"]), chamfer=np.float32(scene["chamfer"]),
        part_accuracy=np.float32(scene["part_accuracy"]), matches=np.float32(scene["matches"]),
        scene=np.array(key), checkpoint=np.array(checkpoint),
        seed=np.array(seed), epoch=np.array(epoch), rotations=np.array("matched"),
        placement_method=np.array(placement),
    )


def write_dump(target, arrays: Dict[str, np.ndarray]) -> None:
    """``arrays`` as a compressed ``.npz`` at ``target`` (a path or a file)."""
    if isinstance(target, (str, Path)):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def dump_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    """The ``.npz`` file :func:`write_dump` would write, as bytes."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()
