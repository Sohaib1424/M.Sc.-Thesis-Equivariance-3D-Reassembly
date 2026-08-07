"""
Shared plumbing for the visualisation scripts.

The originals imported `src.utils` / `src.geometry` / `helpers` from the old
flat layout. Those modules moved:

    src.utils.load_random_scene   -> reassembly.data.scene_io.load_scene
    src.utils.diffuse_fragments   -> reassembly.data.augment.diffuse_fragments
    src.geometry.extract_fractures-> reassembly.data.mesh_ops.extract_fractures
    helpers.get_random_directory  -> reassembly.data.splits.SceneIndex

Everything here takes `--root-dir` rather than assuming a fixed data location,
and `--seed` so a figure can be reproduced.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reassembly.utils.console import quiet_third_party_warnings

quiet_third_party_warnings()


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root-dir", type=str, default="data",
                   help="Breaking Bad dataset root")
    p.add_argument("--scene", type=str, default=None,
                   help="A specific scene directory; default picks one at random")
    p.add_argument("--fracture", type=str, default=None,
                   help="A specific fracture subdirectory, e.g. fractured_3")
    p.add_argument("--fracture-pattern", type=str, default=None,
                   help="Restrict eligible fractures, e.g. 'fractured_*'")
    p.add_argument("--split", type=str, default=None,
                   choices=["train", "val", "test"],
                   help="Draw only from this split; default draws from all scenes")
    p.add_argument("--seed", type=int, default=None,
                   help="Reproduce a particular random choice")
    p.add_argument("--save", type=str, default=None,
                   help="Write a PNG instead of opening a window (headless-safe)")
    return p


def pick_scene(args) -> str:
    """A scene directory, either the one asked for or a random one."""
    if args.scene:
        return args.scene

    from reassembly.data.splits import SceneIndex

    try:
        index = SceneIndex(args.root_dir, split=getattr(args, "split", None))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    return str(index.sample(random.Random(args.seed)))


def load(args):
    """Load one scene's fragments, honouring the CLI options."""
    from reassembly.data.scene_io import load_scene

    scene = pick_scene(args)
    rng = random.Random(args.seed)
    meshes = load_scene(scene, fracture_id=args.fracture, rng=rng,
                        fracture_pattern=args.fracture_pattern)
    print(f"scene    : {scene}")
    print(f"fragments: {len(meshes)}")
    return scene, meshes


def show(scene, args, title: str = "") -> None:
    """Open a viewer, or save a PNG when --save is given.

    Kaggle and other headless environments have no display, so `scene.show()`
    raises there. `--save` renders offscreen instead, which is what makes these
    usable from a notebook.
    """
    if args.save:
        png = scene.save_image(resolution=(1600, 1200), visible=True)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "wb") as fh:
            fh.write(png)
        print(f"wrote {args.save}")
    else:
        try:
            scene.show(caption=title or None)
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"Could not open a viewer ({type(exc).__name__}: {exc}).\n"
                f"On a headless machine (Kaggle, Colab, a server) pass "
                f"--save out.png instead."
            )


def random_colors(n: int, seed: int = 0):
    """Distinguishable per-fragment colours.

    trimesh.visual.random_color() is uniform over RGB, so on a 50-fragment
    scene several fragments come out nearly the same shade. Even spacing
    around the hue circle keeps them apart, which matters when the point of
    the figure is telling fragments apart.
    """
    import colorsys

    import numpy as np

    rng = random.Random(seed)
    hues = [(i / max(n, 1) + rng.random() * 0.02) % 1.0 for i in range(n)]
    rng.shuffle(hues)
    out = []
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.55 + 0.35 * ((i % 3) / 2), 0.95)
        out.append(np.array([int(r * 255), int(g * 255), int(b * 255), 255],
                            dtype=np.uint8))
    return out
