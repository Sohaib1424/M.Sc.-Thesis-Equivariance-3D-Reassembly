#!/usr/bin/env python3
"""
Dump one predicted reassembly -- rotations AND the solver's placement -- to a
small ``.npz`` that can be copied off Kaggle and animated anywhere.

    # which scenes are there?
    python -m scripts.dump_prediction --root_dir data --list --split val

    # dump one
    python -m scripts.dump_prediction --root_dir data --checkpoint runs/vgat/best.pt \\
        --scene everyday_compressed/Mug/<id>/fractured_3 --scatter-seed 0 --out pred.npz

    # then, on any machine with a clone:
    python -m scripts.visualize_reassembly --dump pred.npz
    python -m scripts.render_gif --dump pred.npz --out reassembly.gif

Deterministic given ``--scatter-seed``: the same command rebuilds the same
scatter, so a dump can be regenerated or compared across checkpoints.
``--scatter-seed -1`` uses the scene's own validation draw -- the perturbation
the model is scored on every epoch.

Contents (plain arrays with offsets, so ``np.load`` needs no ``allow_pickle``)::

    vertices        (sum V, 3)  every fragment, assembled, world units
    faces           (sum F, 3)  indices local to each fragment
    vertex_offsets  (F+1,)      fragment f is vertices[o[f]:o[f+1]]
    face_offsets    (F+1,)
    centroids       (F, 3)      each fragment's true centroid
    A               (F, 3, 3)   the rotation the model was shown (the scatter)
    shift           (F, 3)      a display-only scatter offset
    R_pred, R_gt    (F, 3, 3)   predicted rotation, and the truth A^T
    placement       (F, 3)      the solver's predicted centroid, world units
    geodesic_deg, tilt_deg, twist_deg, part_chamfer   (F,)
    rmse_t, chamfer, part_accuracy, matches           scalars for the scene
    scene, checkpoint, seed, epoch

A fragment's vertex ``v`` with centroid ``c`` is shown scattered at
``A (v - c) + c + shift``, and reassembled at ``R_pred A (v - c) + placement``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.config_flags import add_config_arguments, config_from_args  # noqa: E402


def dump(dataset, index, model, config, device, seed: int, out: Path,
         checkpoint: str = "", epoch: int = -1) -> dict:
    """Build the scene, predict, assemble, score, and write the dump."""
    import torch

    from reassembly.assembly import score_batch
    from reassembly.data.features import collate, complete_batch
    from reassembly.data.transforms import random_rotations
    from reassembly.evaluation.metrics import swing_twist_error
    from reassembly.nn.losses import geodesic_angle
    from reassembly.training import Skipped, _forward, _to_device, build_criterion

    meshes = dataset.meshes(index)
    count = len(meshes)
    if seed >= 0:
        scatter = random_rotations(count, np.random.default_rng(seed))
        sample = dataset.build(index, rotations=scatter)
    else:
        sample = dataset.build(index)
    if isinstance(sample, Skipped):
        raise SystemExit(f"{dataset.key(index)} is unusable: {sample.reason}")
    batch = complete_batch(_to_device(collate([sample]), device))
    keep = {}
    with torch.no_grad():
        _forward(model, batch, build_criterion(config), config, keep=keep)
    prediction = keep["prediction"]
    scene = score_batch(batch, prediction)[0]

    R_pred = prediction.rotation.double().cpu()
    R_gt = batch.target_rotation.double().cpu()
    A = R_gt.transpose(-1, -2)                     # the label is A^T
    geodesic = torch.rad2deg(geodesic_angle(R_pred, R_gt)).numpy()
    tilt, twist = swing_twist_error(R_pred, R_gt, axis=config.symmetry_axis)

    centroids = np.stack([np.asarray(m.vertices, dtype=np.float64).mean(0) for m in meshes])
    # The solver's translations are zero-mean; the truth's mean is the mean
    # centroid, so that is where the predicted assembly is placed.
    placement = centroids.mean(0) + np.asarray(scene["_translation"])
    part_chamfer = scene["_part_chamfer"]

    rng = np.random.default_rng(max(seed, 0) + 1)
    extent = float(max(np.ptp(np.asarray(m.vertices), axis=0).max() for m in meshes))
    shift = rng.standard_normal((count, 3)) * extent * 0.9

    vertices = [np.asarray(m.vertices, np.float32) for m in meshes]
    faces = [np.asarray(m.faces, np.int32) for m in meshes]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=np.concatenate(vertices), faces=np.concatenate(faces),
        vertex_offsets=np.cumsum([0] + [len(v) for v in vertices]).astype(np.int64),
        face_offsets=np.cumsum([0] + [len(f) for f in faces]).astype(np.int64),
        centroids=centroids.astype(np.float32), A=A.numpy().astype(np.float32),
        shift=shift.astype(np.float32),
        R_pred=R_pred.numpy().astype(np.float32), R_gt=R_gt.numpy().astype(np.float32),
        placement=placement.astype(np.float32),
        geodesic_deg=geodesic.astype(np.float32),
        tilt_deg=tilt.numpy().astype(np.float32), twist_deg=twist.numpy().astype(np.float32),
        part_chamfer=np.asarray(part_chamfer, np.float32),
        rmse_t=np.float32(scene["rmse_t"]), chamfer=np.float32(scene["chamfer"]),
        part_accuracy=np.float32(scene["part_accuracy"]), matches=np.float32(scene["matches"]),
        scene=np.array(dataset.key(index)), checkpoint=np.array(checkpoint),
        seed=np.array(seed), epoch=np.array(epoch),
    )
    return {"geodesic": geodesic, "tilt": tilt.numpy(), "twist": twist.numpy(),
            "scene": scene, "part_chamfer": part_chamfer}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--scene", default="", help="<object>/<mode>, as --list prints it")
    parser.add_argument("--scatter-seed", type=int, default=0,
                        help="seed of the scatter rotations; -1 = the scene's own "
                             "validation draw")
    parser.add_argument("--out", default="prediction.npz")
    parser.add_argument("--list", action="store_true", help="list the scenes of --split")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    import torch

    from reassembly.training import BreakingBadScenes, find_scene, load_checkpoint

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    state = None
    if args.checkpoint:
        model, config, state, _path = load_checkpoint(args.checkpoint, config, device)

    if args.list:
        import dataclasses

        listed = BreakingBadScenes(dataclasses.replace(config, modes_per_scene=None),
                                   args.split, epoch_seed=0)
        print(f"{len(listed)} scene(s) in the {args.split} split:")
        for i in range(len(listed)):
            print(f"  {listed.key(i)}")
        return 0
    if not args.checkpoint or not args.scene:
        print("need --checkpoint and --scene (or --list)")
        return 2

    try:
        dataset, index, split = find_scene(config, args.scene)
    except KeyError as error:
        print(error.args[0])
        return 2
    out = Path(args.out)
    result = dump(dataset, index, model, config, device, args.scatter_seed, out,
                  checkpoint=str(args.checkpoint), epoch=int(state.get("epoch", -1)))

    print(f"scene {args.scene}   ({split} split, scatter seed {args.scatter_seed})")
    print(f"  {'fragment':>8} {'geodesic':>9} {'tilt':>7} {'twist':>7} {'chamfer':>10}")
    for f, (g, t, w, c) in enumerate(zip(result["geodesic"], result["tilt"], result["twist"],
                                         result["part_chamfer"])):
        print(f"  {f:>8} {g:9.2f} {t:7.2f} {w:7.2f} {c:10.5f}")
    scene = result["scene"]
    print(f"  {'mean':>8} {np.mean(result['geodesic']):9.2f}   (chance 126.47)")
    print(f"  assembly: RMSE(T) {scene['rmse_t']:.4f}   Chamfer {scene['chamfer']:.5f}   "
          f"part accuracy {scene['part_accuracy']:.3f}   ({scene['matches']:.0f} matches)")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"view it:  python -m scripts.visualize_reassembly --dump {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
