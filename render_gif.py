#!/usr/bin/env python
"""
Render a VN-GAT reassembly dump to an animated GIF, for a thesis figure.

    pip install trimesh numpy pillow "pyglet<2"

    # one model, scattered -> predicted
    python render_gif.py --dump pred8.npz --out fig_reassembly_8.gif

    # the figure you probably want: three models side by side
    python render_gif.py --out fig_scaling.gif \
        --dump pred8.npz --dump pred16.npz --dump pred32.npz \
        --label "8 objects" --label "16 objects" --label "32 objects"

    # scattered | predicted | ground truth, for one model
    python render_gif.py --dump pred8.npz --mode compare --out fig_compare_8.gif

Companion to `visualize_reassembly.py`, which is the interactive viewer. This
script exists separately because a FIGURE has requirements a viewer does not:

  * A FIXED CAMERA. trimesh re-frames the scene on every render, so an
    autofit camera would make the object appear to breathe as the fragments
    converge -- motion that is an artefact of the renderer, not of the model.
    The camera here is computed once, from the union of every frame, and held.
  * STABLE COLOURS. Fragment k is the same colour in every frame and in every
    panel, so the eye can follow a piece across the animation and across models.
  * A PING-PONG LOOP with a hold at each end, so a reader sees the final state
    long enough to judge it instead of it flashing past.
  * PANELS SHARE A TIMELINE, so three models are compared at equal progress.

RENDERING
---------
`trimesh.Scene.save_image` needs an OpenGL context (pyglet). That is fine on a
desktop and usually broken headless -- if it fails, the error says so and
suggests `--backend matplotlib`, which is uglier but needs no GL at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_reassembly import colours, load_dump, pose_at  # noqa: E402


# ---------------------------------------------------------------------------
def frame_parameter(step: int, steps: int, hold: int) -> float:
    """
    Animation parameter in [0, 1] for a ping-pong loop with held endpoints.

    A plain 0->1 loop snaps back to scattered the instant it finishes, which
    reads as a glitch and gives the reader no time on the result. Holding both
    ends and reversing makes the comparison legible.
    """
    period = hold + steps + hold + steps
    t = step % period
    if t < hold:
        return 0.0
    t -= hold
    if t < steps:
        return t / max(steps - 1, 1)
    t -= steps
    if t < hold:
        return 1.0
    return 1.0 - (t - hold) / max(steps - 1, 1)


def transformed_vertices(dump: dict, s: float, target: str = "pred") -> list:
    out = []
    for i, frag in enumerate(dump["fragments"]):
        M = pose_at(dump, i, s, target)
        v = frag["vertices"] @ M[:3, :3].T + M[:3, 3]
        out.append(v)
    return out


def world_bounds(dumps: list, mode: str, samples: int = 9) -> tuple:
    """
    Bounding box over EVERY frame of EVERY panel.

    Computed once and reused, so the camera never moves. An autofit camera
    would shrink as the fragments come together, which looks like the object is
    being pulled toward the viewer -- a renderer artefact that would mislead a
    reader about what the model did.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for dump in dumps:
        targets = ("pred", "gt") if mode == "compare" else ("pred",)
        for target in targets:
            for s in np.linspace(0.0, 1.0, samples):
                for v in transformed_vertices(dump, s, target):
                    lo = np.minimum(lo, v.min(0))
                    hi = np.maximum(hi, v.max(0))
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    return centre, max(radius, 1e-6)


# ---------------------------------------------------------------------------
def render_trimesh(dumps, labels, mode, s, centre, radius, size, elev, azim, zoom=1.0):
    """One frame, via trimesh's offscreen renderer. Needs an OpenGL context."""
    import trimesh
    from PIL import Image

    panels = []
    for dump in dumps:
        scene = trimesh.Scene()
        cols = colours(len(dump["fragments"]))
        span = radius * 2.2
        layouts = [(0.0, "pred", s)]
        if mode == "compare":
            layouts = [(-span, "pred", 0.0), (0.0, "pred", s), (span, "gt", 1.0)]
        for i, frag in enumerate(dump["fragments"]):
            for dx, target, param in layouts:
                mesh = trimesh.Trimesh(frag["vertices"], frag["faces"], process=False)
                mesh.apply_transform(pose_at(dump, i, param, target))
                if dx:
                    mesh.apply_translation([dx, 0, 0])
                mesh.visual.face_colors = cols[i]
                scene.add_geometry(mesh)

        # Fixed camera: same transform for every frame and every panel.
        width = radius * (3.4 if mode == "compare" else 1.25) / zoom
        scene.camera.resolution = size
        scene.camera_transform = scene.camera.look_at(
            points=np.array([centre - width, centre + width]),
            rotation=trimesh.transformations.euler_matrix(
                np.radians(elev), 0.0, np.radians(azim), "rxyz"))
        try:
            png = scene.save_image(resolution=size, visible=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Offscreen rendering failed ({type(exc).__name__}: {exc}).\n"
                f"trimesh needs an OpenGL context for save_image. On a desktop:\n"
                f"  pip install 'pyglet<2'\n"
                f"If this machine has no display, use --backend matplotlib instead."
            ) from None
        panels.append(Image.open(__import__("io").BytesIO(png)).convert("RGB"))
    return stack_panels(panels, labels)


def render_matplotlib(dumps, labels, mode, s, centre, radius, size, elev, azim, zoom=1.0):
    """One frame, via matplotlib. No OpenGL, so it works anywhere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    n = len(dumps)
    fig = plt.figure(figsize=(size[0] * n / 100, size[1] / 100), dpi=100)
    for panel, dump in enumerate(dumps):
        ax = fig.add_subplot(1, n, panel + 1, projection="3d")
        cols = colours(len(dump["fragments"])) / 255.0
        span = radius * 2.2
        layouts = [(0.0, "pred", s)]
        if mode == "compare":
            layouts = [(-span, "pred", 0.0), (0.0, "pred", s), (span, "gt", 1.0)]
        for i, frag in enumerate(dump["fragments"]):
            for dx, target, param in layouts:
                M = pose_at(dump, i, param, target)
                v = frag["vertices"] @ M[:3, :3].T + M[:3, 3] + np.array([dx, 0, 0])
                tris = v[frag["faces"]]
                ax.add_collection3d(Poly3DCollection(
                    tris, facecolors=cols[i][:3], edgecolors="none", alpha=1.0))
        width = radius * (3.4 if mode == "compare" else 1.25) / zoom
        height = radius * 1.25 / zoom
        ax.set_xlim(centre[0] - width, centre[0] + width)
        ax.set_ylim(centre[1] - height, centre[1] + height)
        ax.set_zlim(centre[2] - height, centre[2] + height)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((width * 2, height * 2, height * 2))
        if labels and panel < len(labels):
            ax.set_title(labels[panel], fontsize=11, pad=0)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0, wspace=0)
    fig.canvas.draw()
    img = Image.frombytes("RGB", fig.canvas.get_width_height(),
                          fig.canvas.tostring_rgb())
    plt.close(fig)
    return img


def stack_panels(panels, labels):
    """Join panels left-to-right, with captions if given."""
    from PIL import Image, ImageDraw

    if len(panels) == 1 and not labels:
        return panels[0]
    pad = 26 if labels else 0
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels) + pad
    out = Image.new("RGB", (w, h), "white")
    x = 0
    draw = ImageDraw.Draw(out)
    for i, p in enumerate(panels):
        out.paste(p, (x, pad))
        if labels and i < len(labels):
            draw.text((x + p.width // 2 - 4 * len(labels[i]), 6), labels[i], fill="black")
        x += p.width
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", action="append", required=True,
                   help="Repeat for a multi-panel figure.")
    p.add_argument("--label", action="append", default=[],
                   help="Caption per panel, in the same order as --dump.")
    p.add_argument("--out", default="reassembly.gif")
    p.add_argument("--mode", default="reassemble", choices=["reassemble", "compare"])
    p.add_argument("--backend", default="trimesh", choices=["trimesh", "matplotlib"])
    p.add_argument("--steps", type=int, default=45, help="Frames per sweep.")
    p.add_argument("--hold", type=int, default=12, help="Frames held at each end.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=520)
    p.add_argument("--height", type=int, default=520)
    p.add_argument("--zoom", type=float, default=1.0,
                   help="Magnification. >1 zooms in, <1 out. The framing is computed "
                        "from the union of every frame, which leaves headroom for the "
                        "scattered state; once the fragments converge that headroom is "
                        "wasted, so 1.2-1.5 usually frames the assembled object better. "
                        "Too high and the scattered pieces leave the frame.")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=35.0)
    p.add_argument("--still", type=float, default=None,
                   help="Write a single PNG at this s in [0,1] instead of a GIF.")
    args = p.parse_args(argv)

    dumps = [load_dump(d) for d in args.dump]
    for path, dump in zip(args.dump, dumps):
        geo = dump["geodesic_deg"]
        print(f"{Path(path).name}: {len(dump['fragments'])} fragments, "
              f"mean {geo.mean():.2f} deg, worst {geo.max():.2f}  "
              f"(chance 126.47)")

    centre, radius = world_bounds(dumps, args.mode)
    size = (args.width, args.height)
    render = render_trimesh if args.backend == "trimesh" else render_matplotlib

    if args.still is not None:
        img = render(dumps, args.label, args.mode, float(np.clip(args.still, 0, 1)),
                     centre, radius, size, args.elev, args.azim, args.zoom)
        out = Path(args.out).with_suffix(".png")
        img.save(out)
        print(f"wrote {out}")
        return 0

    total = 2 * (args.hold + args.steps)
    frames = []
    for i in range(total):
        s = frame_parameter(i, args.steps, args.hold)
        frames.append(render(dumps, args.label, args.mode, s, centre, radius,
                             size, args.elev, args.azim, args.zoom))
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"  rendered {i + 1}/{total}")

    out = Path(args.out)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {total} frames)")
    print("For the thesis, --still 1.0 also gives a print-ready PNG of the final state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
