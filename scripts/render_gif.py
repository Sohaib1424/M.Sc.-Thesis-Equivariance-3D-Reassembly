#!/usr/bin/env python3
"""
Render a predicted reassembly to an animated GIF -- a figure, not a viewer.

    pip install trimesh numpy pillow matplotlib      # + "pyglet<2" for --backend trimesh

    python -m scripts.render_gif --dump pred.npz --out reassembly.gif
    python -m scripts.render_gif --dump pred.npz --mode compare --out compare.gif
    python -m scripts.render_gif --out scaling.gif \\
        --dump pred8.npz --dump pred32.npz --label "8 objects" --label "32 objects"
    python -m scripts.render_gif --dump pred.npz --still 1.0 --out final.png

A figure has requirements a viewer does not, and each is handled here:

* **A fixed camera.** An autofit camera re-frames every render, so the object
  seems to breathe as the fragments converge -- motion made by the renderer.
  The framing is computed once from every frame of every panel and held.
* **Stable colours.** Fragment k has the same colour in every frame, panel and
  figure (the package palette), so a piece can be followed.
* **A ping-pong loop with both ends held**, so the result is on screen long
  enough to be judged instead of snapping back.
* **Panels share one timeline**, so models are compared at equal progress.

``--backend trimesh`` needs an OpenGL context (usually absent on a headless
box); ``--backend matplotlib`` needs none and is the default for that reason.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from reassembly.viz.reassembly import (describe, frame_parameter, layouts,  # noqa: E402
                                       load_dump, pose_at, world_bounds)
from reassembly.viz.scene import fragment_colour  # noqa: E402


def _half_width(radius: float, mode: str, zoom: float) -> float:
    return radius * (3.4 if mode == "compare" else 1.25) / zoom


def render_matplotlib(dumps, labels, mode, s, centre, radius, size, elev, azim, zoom):
    """One frame with matplotlib -- no OpenGL, so it works anywhere."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    count = len(dumps)
    figure = plt.figure(figsize=(size[0] * count / 100, size[1] / 100), dpi=100)
    span = radius * 2.2
    width, height = _half_width(radius, mode, zoom), radius * 1.25 / zoom
    for panel, dump in enumerate(dumps):
        axes = figure.add_subplot(1, count, panel + 1, projection="3d")
        for i, fragment in enumerate(dump["fragments"]):
            colour = fragment_colour(i)[:3] / 255.0
            for dx, target, parameter in layouts(mode, s, span):
                M = pose_at(dump, i, parameter, target)
                v = fragment["vertices"] @ M[:3, :3].T + M[:3, 3] + np.array([dx, 0.0, 0.0])
                axes.add_collection3d(Poly3DCollection(
                    v[fragment["faces"]], facecolors=colour, edgecolors="none"))
        axes.set_xlim(centre[0] - width, centre[0] + width)
        axes.set_ylim(centre[1] - height, centre[1] + height)
        axes.set_zlim(centre[2] - height, centre[2] + height)
        axes.set_box_aspect((2 * width, 2 * height, 2 * height))
        axes.view_init(elev=elev, azim=azim)
        axes.set_axis_off()
        if labels and panel < len(labels):
            axes.set_title(labels[panel], fontsize=11, pad=0)
    figure.subplots_adjust(left=0, right=1, top=0.94, bottom=0, wspace=0)
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return image


def render_trimesh(dumps, labels, mode, s, centre, radius, size, elev, azim, zoom):
    """One frame with trimesh's offscreen renderer. Needs an OpenGL context."""
    import trimesh
    from PIL import Image

    panels = []
    span = radius * 2.2
    width = _half_width(radius, mode, zoom)
    for dump in dumps:
        scene = trimesh.Scene()
        for i, fragment in enumerate(dump["fragments"]):
            for dx, target, parameter in layouts(mode, s, span):
                mesh = trimesh.Trimesh(fragment["vertices"], fragment["faces"], process=False)
                mesh.apply_transform(pose_at(dump, i, parameter, target))
                if dx:
                    mesh.apply_translation([dx, 0.0, 0.0])
                mesh.visual.face_colors = fragment_colour(i)
                scene.add_geometry(mesh)
        scene.camera.resolution = size
        scene.camera_transform = scene.camera.look_at(
            points=np.array([centre - width, centre + width]),
            rotation=trimesh.transformations.euler_matrix(
                np.radians(elev), 0.0, np.radians(azim), "rxyz"))
        try:
            png = scene.save_image(resolution=size, visible=True)
        except Exception as error:                            # noqa: BLE001
            raise SystemExit(
                f"offscreen rendering failed ({type(error).__name__}: {error}). trimesh "
                f"needs an OpenGL context; without a display use --backend matplotlib."
            ) from None
        panels.append(Image.open(io.BytesIO(png)).convert("RGB"))
    return _side_by_side(panels, labels)


def _side_by_side(panels, labels):
    from PIL import Image, ImageDraw

    if len(panels) == 1 and not labels:
        return panels[0]
    pad = 26 if labels else 0
    out = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels) + pad),
                    "white")
    draw, x = ImageDraw.Draw(out), 0
    for i, panel in enumerate(panels):
        out.paste(panel, (x, pad))
        if labels and i < len(labels):
            draw.text((x + panel.width // 2 - 4 * len(labels[i]), 6), labels[i], fill="black")
        x += panel.width
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", action="append", required=True,
                        help="repeat for a multi-panel figure")
    parser.add_argument("--label", action="append", default=[],
                        help="caption per panel, in --dump order")
    parser.add_argument("--out", default="reassembly.gif")
    parser.add_argument("--mode", default="reassemble",
                        choices=["reassemble", "rotation", "compare"])
    parser.add_argument("--backend", default="matplotlib", choices=["matplotlib", "trimesh"])
    parser.add_argument("--steps", type=int, default=45, help="frames per sweep")
    parser.add_argument("--hold", type=int, default=12, help="frames held at each end")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--zoom", type=float, default=1.0,
                        help=">1 zooms in. The framing leaves room for the scattered "
                             "state; 1.2-1.5 frames the assembled object better")
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--still", type=float, default=None,
                        help="write one PNG at this s in [0, 1] instead of a GIF")
    args = parser.parse_args(argv)

    dumps = [load_dump(path) for path in args.dump]
    for path, dump in zip(args.dump, dumps):
        print(f"{Path(path).name}:\n{describe(dump)}")
    centre, radius = world_bounds(dumps, args.mode)
    size = (args.width, args.height)
    render = render_trimesh if args.backend == "trimesh" else render_matplotlib

    if args.still is not None:
        image = render(dumps, args.label, args.mode, float(np.clip(args.still, 0, 1)),
                       centre, radius, size, args.elev, args.azim, args.zoom)
        out = Path(args.out).with_suffix(".png")
        image.save(out)
        print(f"wrote {out}")
        return 0

    total = 2 * (args.hold + args.steps)
    frames = []
    for step in range(total):
        frames.append(render(dumps, args.label, args.mode,
                             frame_parameter(step, args.steps, args.hold),
                             centre, radius, size, args.elev, args.azim, args.zoom))
        if (step + 1) % 20 == 0 or step + 1 == total:
            print(f"  rendered {step + 1}/{total}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {total} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
