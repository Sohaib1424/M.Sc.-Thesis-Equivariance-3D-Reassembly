#!/usr/bin/env python
"""Show a scene's fragments in their assembled (ground-truth) positions.

    python scripts/visualize/vis_default.py --root-dir data
    python scripts/visualize/vis_default.py --root-dir data --save assembled.png
"""
import trimesh

from _common import base_parser, load, random_colors, show


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    _dir, meshes = load(args)

    scene = trimesh.Scene()
    for mesh, color in zip(meshes, random_colors(len(meshes), seed=args.seed or 0)):
        mesh.visual.face_colors = color
        scene.add_geometry(mesh)
    show(scene, args, "assembled")


if __name__ == "__main__":
    main()
