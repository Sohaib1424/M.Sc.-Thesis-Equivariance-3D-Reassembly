#!/usr/bin/env python
"""Show only the extracted fracture surfaces of an assembled scene.

This is what `data.input_source: frac` feeds the model -- worth looking at
before running that experiment, since a bad extraction is invisible in a loss
curve.

    python scripts/visualize/vis_fractures.py --root-dir data

Fracture extraction is the slow part of the pipeline; expect a wait on scenes
with many fragments.
"""
import trimesh

from _common import base_parser, load, random_colors, show


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    _dir, meshes = load(args)

    from reassembly.data.mesh_ops import extract_fractures

    scene = trimesh.Scene()
    kept, total = 0, 0
    for mesh, color in zip(meshes, random_colors(len(meshes), seed=args.seed or 0)):
        frac = extract_fractures(mesh)
        total += len(mesh.faces)
        kept += len(frac.faces)
        frac.visual.face_colors = color
        scene.add_geometry(frac)

    print(f"fracture surface: {kept:,} of {total:,} faces "
          f"({100 * kept / max(total, 1):.1f}%)")
    show(scene, args, "fracture surfaces")


if __name__ == "__main__":
    main()
