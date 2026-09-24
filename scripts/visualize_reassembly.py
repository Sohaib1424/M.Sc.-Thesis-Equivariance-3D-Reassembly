#!/usr/bin/env python3
"""
Watch a predicted reassembly, from a dump written by ``scripts.dump_prediction``.

    python -m scripts.visualize_reassembly --dump pred.npz                 # animate
    python -m scripts.visualize_reassembly --dump pred.npz --mode compare  # three copies
    python -m scripts.visualize_reassembly --dump pred.npz --at 1.0        # one frame
    python -m scripts.visualize_reassembly --dump pred.npz --export frames/

Needs numpy and trimesh, and pyglet (``pip install "pyglet<2"``) for a window;
no torch. Modes:

    reassemble  scattered -> the predicted rotations, placed by the solver
    rotation    scattered -> the predicted rotations at the TRUE centroids, so
                what is left misaligned is rotation error alone
    compare     scattered | predicted | ground truth, side by side
    truth       scattered -> ground truth: what a perfect model would show

Anything still misaligned at the end of ``reassemble`` is the model's error --
which is the point of watching it. See :mod:`reassembly.viz.reassembly` for
exactly what moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from reassembly.viz.reassembly import (build_trimesh_scene, describe,  # noqa: E402
                                       load_dump, world_bounds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--mode", default="reassemble",
                        choices=["reassemble", "rotation", "compare", "truth"])
    parser.add_argument("--steps", type=int, default=60,
                        help="frames from scattered to assembled")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--at", type=float, default=None,
                        help="show one frame at this s in [0, 1] instead of animating")
    parser.add_argument("--export", default="",
                        help="write one .glb per frame here instead of showing")
    args = parser.parse_args(argv)

    dump = load_dump(args.dump)
    print(describe(dump))
    geodesic = dump.get("geodesic_deg")
    if geodesic is not None and len(geodesic):
        worst = int(np.nanargmax(geodesic))
        print(f"worst fragment: {worst} at {geodesic[worst]:.2f} deg\n")
    span = 2.2 * world_bounds([dump], args.mode, samples=3)[1]

    if args.at is not None:
        build_trimesh_scene(dump, float(np.clip(args.at, 0, 1)), args.mode, span).show()
        return 0
    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.steps + 1):
            build_trimesh_scene(dump, i / args.steps, args.mode, span).export(
                out / f"frame_{i:04d}.glb")
        print(f"wrote {args.steps + 1} frames to {out}/")
        return 0

    try:
        import pyglet  # noqa: F401
    except ImportError:
        print('A window needs pyglet:  pip install "pyglet<2".  Showing three still '
              "frames instead: scattered, halfway, assembled.")
        for s in (0.0, 0.5, 1.0):
            build_trimesh_scene(dump, s, args.mode, span).show()
        return 0

    print("playing -- close the window to exit")
    scene = build_trimesh_scene(dump, 0.0, args.mode, span)
    state = {"frame": 0}

    def advance(viewer_scene):
        state["frame"] = (state["frame"] + 1) % (args.steps + 1)
        fresh = build_trimesh_scene(dump, state["frame"] / args.steps, args.mode, span)
        viewer_scene.geometry.clear()
        for name, geometry in fresh.geometry.items():
            viewer_scene.add_geometry(geometry, geom_name=name)

    scene.show(callback=advance, callback_period=1.0 / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
