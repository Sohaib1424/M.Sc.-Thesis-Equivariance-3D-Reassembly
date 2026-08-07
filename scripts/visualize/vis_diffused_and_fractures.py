#!/usr/bin/env python
"""Scattered fragments and their fracture surfaces, offset side by side.

    python scripts/visualize/vis_diffused_and_fractures.py --root-dir data --offset 5.5
"""
import numpy as np
import trimesh

from _common import base_parser, load, random_colors, show


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument("--offset", type=float, default=None,
                        help="Separation between the two copies; default scales "
                             "with the scene so it works on any object size")
    args = parser.parse_args(argv)
    _dir, meshes = load(args)

    from reassembly.data.augment import diffuse_fragments
    from reassembly.data.mesh_ops import extract_fractures

    diffused, _transforms = diffuse_fragments(meshes)

    # A fixed offset only looks right for one object scale. Breaking Bad
    # objects are normalised to roughly unit size, but a hard-coded 5.5 still
    # puts the two copies uncomfortably far apart on a small object and
    # overlapping on a large one.
    if args.offset is None:
        allv = np.concatenate([np.asarray(m.vertices) for m in diffused])
        args.offset = float((allv.max(0) - allv.min(0)).max() * 1.3)
    shift = np.array([args.offset, 0.0, 0.0])

    scene = trimesh.Scene()
    colors = random_colors(len(diffused), seed=args.seed or 0)
    for mesh, color in zip(diffused, colors):
        frac = extract_fractures(mesh)
        frac = trimesh.Trimesh(np.asarray(frac.vertices) + shift, frac.faces,
                               process=False)
        mesh.visual.face_colors = color
        frac.visual.face_colors = color
        scene.add_geometry(mesh)
        scene.add_geometry(frac)

    show(scene, args, "diffused + fracture surfaces")


if __name__ == "__main__":
    main()
