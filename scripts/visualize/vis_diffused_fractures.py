#!/usr/bin/env python
"""Fracture surfaces of a scattered scene -- exactly what `input_source: frac`
sees at training time.

    python scripts/visualize/vis_diffused_fractures.py --root-dir data
"""
import trimesh

from _common import base_parser, load, random_colors, show


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    _dir, meshes = load(args)

    from reassembly.data.augment import diffuse_fragments
    from reassembly.data.mesh_ops import extract_fractures

    diffused, _transforms = diffuse_fragments(meshes)

    scene = trimesh.Scene()
    for mesh, color in zip(diffused, random_colors(len(diffused), seed=args.seed or 0)):
        frac = extract_fractures(mesh)
        frac.visual.face_colors = color
        scene.add_geometry(frac)
    show(scene, args, "diffused fracture surfaces")


if __name__ == "__main__":
    main()
