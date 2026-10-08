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

For every scene of a split at once, ``python -m scripts.train --evaluate
--predictions FOLDER`` writes the same dumps into one zip as it scores them.

Deterministic given ``--scatter-seed``: the same command rebuilds the same
scatter, so a dump can be regenerated or compared across checkpoints.
``--scatter-seed -1`` uses the scene's own validation draw -- the perturbation
the model is scored on every epoch, and the one ``--evaluate`` scores.

The rotations are fitted from the embedding matches and chained from the
anchor (reassembly/assembly/rotation.py; the network has had no rotation head
since v7), and placed by ``--placement``: ``checked`` (the default) uses the
pair fits the chain agrees with and holds the anchor
(reassembly/assembly/placement.py); ``global`` (v6's only placement) is one
least-squares solve over every embedding match. The file's contents are listed
in ``reassembly/assembly/dump.py``.
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
         checkpoint: str = "", epoch: int = -1, rotations: str = "matched",
         placement=None) -> dict:
    """Build the scene, predict, fit the rotations from the matches, place,
    score, and write the dump. ``placement`` is the solve that places the
    turned fragments (``None``: checked). ``rotations`` is accepted only as
    ``"matched"``, so a call written before v7 removed the head fails rather
    than quietly scoring something else."""
    import torch

    from reassembly.assembly import check_placement, score_batch
    from reassembly.assembly.dump import dump_arrays, write_dump
    from reassembly.data.features import collate, complete_batch
    from reassembly.data.transforms import random_rotations
    from reassembly.training import Skipped, _forward, _to_device, build_criterion

    if rotations != "matched":
        raise ValueError(f"rotations {rotations!r}: since v7 every rotation is fitted from "
                         f"the embedding matches, so only 'matched' exists")
    method = check_placement(placement)
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
        _forward(model, batch, build_criterion(config), config, keep=keep, score=True)
    scene = score_batch(batch, keep["prediction"], seed=config.seed, placement=method,
                        matched=keep["matched"])[0]

    # Placed the way the scorer placed them: with the largest fragment set to
    # its true pose (reassembly.nn.anchor), so the anchor's error is 0 and the
    # other rows are errors relative to it.
    arrays = dump_arrays(meshes, scene, batch.target_rotation, axis=config.symmetry_axis,
                         key=dataset.key(index), checkpoint=checkpoint, epoch=epoch,
                         seed=seed, placement=method)
    write_dump(out, arrays)
    return {"geodesic": arrays["geodesic_deg"], "tilt": arrays["tilt_deg"],
            "twist": arrays["twist_deg"], "scene": scene,
            "part_chamfer": scene["_part_chamfer"], "placement": method}


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
    parser.add_argument("--rotations", default="matched", choices=["matched"],
                        help="kept so earlier commands still run: the rotations are always "
                             "fitted from the embedding matches (assembly/rotation.py)")
    parser.add_argument("--placement", default=None, choices=["checked", "global"],
                        help="checked (the default): place from the pair fits the chain "
                             "agrees with, the anchor held (reassembly/assembly/placement.py); "
                             "global (v6's only placement): one solve over every embedding "
                             "match")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    import torch

    from reassembly.assembly import check_placement
    from reassembly.training import BreakingBadScenes, find_scene, load_checkpoint

    placement = check_placement(args.placement)

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
                  checkpoint=str(args.checkpoint), epoch=int(state.get("epoch", -1)),
                  placement=placement)

    print(f"scene {args.scene}   ({split} split, scatter seed {args.scatter_seed}, "
          f"rotations from the matches, {placement} placement)")
    print(f"  {'fragment':>8} {'geodesic':>9} {'tilt':>7} {'twist':>7} {'chamfer':>10}")
    for f, (g, t, w, c) in enumerate(zip(result["geodesic"], result["tilt"], result["twist"],
                                         result["part_chamfer"])):
        print(f"  {f:>8} {g:9.2f} {t:7.2f} {w:7.2f} {c:10.5f}")
    scene = result["scene"]
    anchor = int(scene["_anchor"])
    others = [g for f, g in enumerate(result["geodesic"]) if f != anchor]
    if anchor >= 0:
        print(f"  fragment {anchor} is the anchor (the largest): set to its true pose, "
              f"so its error is 0 by construction")
    print(f"  {'mean':>8} {np.mean(others) if others else float('nan'):9.2f}   "
          f"(the other fragments; chance 126.47)")
    reached = scene.get("_reached") or []
    unplaced = [f for f, hit in enumerate(reached) if not hit]
    if unplaced:
        print(f"  not reached by the matching (rotation at chance): fragment(s) "
              f"{', '.join(map(str, unplaced))}")
    verified = (f", {scene['verified_matches']:.0f} in the verified pair fits"
                if "verified_matches" in scene else "")
    print(f"  assembly: RMSE(T) {scene['rmse_t']:.4f}   Chamfer {scene['chamfer']:.5f}   "
          f"part accuracy {scene['part_accuracy']:.3f}   ({scene['matches']:.0f} matches"
          f"{verified})")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"view it:  python -m scripts.visualize_reassembly --dump {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
