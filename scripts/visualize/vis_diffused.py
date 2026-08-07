#!/usr/bin/env python
"""Show a scene after the scattering transform -- the network's actual input.

    python scripts/visualize/vis_diffused.py --root-dir data
"""
import trimesh

from _common import base_parser, load, random_colors, show


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    _dir, meshes = load(args)

    from reassembly.data.augment import diffuse_fragments

    diffused, _transforms = diffuse_fragments(meshes)

    scene = trimesh.Scene()
    for mesh, color in zip(diffused, random_colors(len(diffused), seed=args.seed or 0)):
        mesh.visual.face_colors = color
        scene.add_geometry(mesh)
    show(scene, args, "diffused")


if __name__ == "__main__":
    main()
