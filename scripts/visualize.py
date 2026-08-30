#!/usr/bin/env python3
"""
View a Breaking Bad scene.

Replaces vis_default / vis_diffused / vis_fractures / vis_diffused_fractures /
vis_diffused_default_and_fractures. Those were the same script with different
switches, so they are switches::

    python -m scripts.visualize --root data                       # vis_default
    python -m scripts.visualize --root data --diffuse             # vis_diffused
    python -m scripts.visualize --root data --show fracture       # vis_fractures
    python -m scripts.visualize --root data --diffuse --show fracture
    python -m scripts.visualize --root data --diffuse --show both

Each run shows a different random scene, and prints the seed that produced it.
Passing that seed back with ``--seed`` reproduces the view exactly -- so unlike
the originals, a view can be described to somebody else and reproduced, without
having to give up random browsing to get it. ``--scene`` and ``--mode`` pin them
explicitly.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.data.paths import find_scenes  # noqa: E402
from reassembly.data.scene import SceneReader  # noqa: E402
from reassembly.data.transforms import diffuse_fragments  # noqa: E402
from reassembly.viz.scene import build_scene, describe  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render one Breaking Bad scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="data")
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--scene", default=None,
                        help="scene directory name (default: pick one at random)")
    parser.add_argument("--mode", default=None,
                        help="fracture mode directory (default: pick one at random)")
    parser.add_argument("--show", choices=["full", "fracture", "both"], default="full")
    parser.add_argument("--method", choices=["dihedral", "coincidence"],
                        default="dihedral",
                        help="which labelling to draw. 'coincidence' is ground "
                             "truth (needs the assembled scene); 'dihedral' is "
                             "the heuristic available at inference")
    parser.add_argument("--diffuse", action="store_true",
                        help="apply a random SE(3) transform to each fragment")
    parser.add_argument("--sharp-threshold", type=float, default=None)
    parser.add_argument("--seed", default="random", metavar="N|random",
                        help="fixes which scene, which fracture mode, and the "
                             "perturbation. The default draws a fresh one each run "
                             "and prints it, so any view can still be reproduced")
    parser.add_argument("--alpha", type=int, default=255, help="face opacity 0-255")
    parser.add_argument("--export", default=None,
                        help="write the scene to a file (.glb/.obj/.ply) instead of showing it")
    parser.add_argument("--no-show", action="store_true",
                        help="print the fragment table and exit without rendering")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="re-load the mode N times and report best/median, so the "
                             "number is not one cold-cache sample")
    args = parser.parse_args(argv)

    scenes = find_scenes(args.root, args.subsets)
    if not scenes:
        parser.error(f"no scene directories found under {args.root!r}")

    # Random by default, but never irreproducibly so: the drawn seed is printed
    # below, and passing it back with --seed reproduces the view exactly. A fixed
    # default meant every run without --seed showed the same scene, which is the
    # opposite of what browsing the dataset wants.
    if isinstance(args.seed, str) and args.seed.lower() in ("random", "rand"):
        seed = random.SystemRandom().randrange(2 ** 31)
    else:
        try:
            seed = int(args.seed)
        except (TypeError, ValueError):
            parser.error(f"--seed takes an integer or 'random', not {args.seed!r}")

    rng = random.Random(seed)
    generator = np.random.default_rng(seed)

    if args.scene:
        matches = [s for s in scenes if s.name == args.scene]
        if not matches:
            parser.error(f"no scene named {args.scene!r} (of {len(scenes)} found)")
        scene = matches[0]
    else:
        scene = rng.choice(scenes)

    reader = SceneReader(scene.path)

    # Import igl and trimesh BEFORE the clock starts. Both are imported lazily
    # inside SceneReader, so the first load in a process otherwise pays for
    # `import trimesh` -- measured here at ~500 ms against ~2 ms for the load
    # itself, a 240x distortion. A DataLoader worker pays that once at start-up
    # and then loads thousands of samples, so charging it to the first sample
    # would answer a question nobody is asking.
    import igl        # noqa: F401
    import trimesh    # noqa: F401

    # Timed in two parts on purpose. SceneReader parses the .obj and the sparse
    # cell matrix once and reuses them for every mode, so the base read is paid
    # once per scene and amortised over its ~100 modes, while the fragment build
    # is paid per training sample. Reporting a single total would hide which of
    # the two a slow batch loader is actually spending its time in.
    t0 = time.perf_counter()
    _ = reader.vertices                     # forces the .obj + .npz read
    base_seconds = time.perf_counter() - t0

    modes = reader.mode_names()
    if not modes:
        parser.error(f"{scene.path} has no fracture modes")
    mode = args.mode or rng.choice(modes)
    if mode not in modes:
        parser.error(f"{scene.name} has no mode {mode!r}; available: {', '.join(modes[:8])}...")

    # Repeats reuse the cached base mesh, so they time the per-sample cost only.
    samples = []
    for _ in range(max(1, args.repeat)):
        t1 = time.perf_counter()
        result = reader.load_mode(mode)
        samples.append(time.perf_counter() - t1)
    mode_seconds = min(samples)

    fragments = result.fragments
    print(f"scene     {scene.subset}/{scene.category}/{scene.name}".replace("//", "/"))
    print(f"mode      {mode}  ({len(modes)} available)")
    print(f"seed      {seed}   (--seed {seed} to see this exact scene again)")
    print(f"fragments {len(fragments)}"
          + (f"   [{result.empty_pieces} empty piece(s) skipped]" if result.empty_pieces else ""))

    total_ms = (base_seconds + mode_seconds) * 1e3
    per_fragment = f"{mode_seconds * 1e3 / len(fragments):.1f} ms/fragment" if fragments else "-"
    spread = ""
    if len(samples) > 1:
        spread = (f"  best of {len(samples)}"
                  f", median {statistics.median(samples) * 1e3:.1f} ms")
    print(f"load      mesh+matrix {base_seconds * 1e3:8.1f} ms   "
          f"(once per scene, amortised over {len(modes)} modes)")
    print(f"          fragments   {mode_seconds * 1e3:8.1f} ms   "
          f"(per sample; {per_fragment}){spread}")
    print(f"          total       {total_ms:8.1f} ms")

    # Coincidence is defined in the assembled frame, so it must be computed
    # before any perturbation. Doing it after --diffuse would find no shared
    # vertices at all and report an empty ground truth.
    truth_masks = None
    if len(fragments) > 1:
        from reassembly.mesh.fracture import fracture_vertex_masks
        truth_masks = fracture_vertex_masks(fragments)

    if args.diffuse:
        fragments, _ = diffuse_fragments(fragments, rng=generator)

    print()
    print(describe(fragments, args.sharp_threshold, truth_masks=truth_masks))

    if args.no_show and not args.export:
        return 0

    built = build_scene(fragments, show=args.show, sharp_threshold=args.sharp_threshold,
                        alpha=args.alpha, method=args.method, truth_masks=truth_masks)
    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        built.export(args.export)
        print(f"\nwrote {args.export}")
        return 0

    built.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
