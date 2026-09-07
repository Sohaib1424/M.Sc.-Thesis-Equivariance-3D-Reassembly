#!/usr/bin/env python
"""
Dump a trained model's predicted rotations for ONE specific breakage, in a form
that can be copied off Kaggle and animated locally.

    # 1. which objects did the N=8 run actually train on?
    python -m scripts.dump_prediction --list_pool --max_scenes 8

    # 2. which fracture patterns does one of them have?
    python -m scripts.dump_prediction --list_fractures --scene <path>

    # 3. dump the prediction
    python -m scripts.dump_prediction \
        --checkpoint /kaggle/working/overfit8/best.pt \
        --scene <path> --fracture fractured_3 --seed 0 \
        --out /kaggle/working/prediction.npz

Everything is deterministic given `--seed`: the same command reproduces the same
scatter, so a dump can be regenerated or compared across checkpoints.

WHAT THE OUTPUT CONTAINS

Geometry is stored flat with offsets rather than as ragged object arrays, so the
file loads with a plain `np.load` and no `allow_pickle`.

    vertices        (sum V_f, 3)  original, in the assembled frame
    faces           (sum F_f, 3)  indices LOCAL to each fragment
    vertex_offsets  (F+1,)        fragment f is vertices[o[f]:o[f+1]]
    face_offsets    (F+1,)
    centroids       (F, 3)        per-fragment centroid of the original mesh
    A               (F, 3, 3)     the scatter rotation applied to fragment f
    t               (F, 3)        the scatter translation
    R_pred          (F, 3, 3)     what the model predicts
    R_gt            (F, 3, 3)     the target, = A^T
    geodesic_deg    (F,)          per-fragment error
    tilt_deg        (F,)          residual off the symmetry axis
    twist_deg       (F,)          residual about it

HOW TO USE THE MATRICES

Fragment f's vertices, centred at its own centroid, are `x = vertices_f - c_f`.

    scattered   : A_f @ x        + c_f + t_f
    reassembled : R_pred_f @ A_f @ x + c_f
    ground truth:              x + c_f

A perfect prediction has R_pred = A^T, so `R_pred @ A = I` and the reassembled
geometry coincides with the original. The animation in
`visualize_reassembly.py` interpolates between the first two.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.utils.env import configure_warnings, limit_blas_threads  # noqa: E402

limit_blas_threads(1)
configure_warnings()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vngat.config import Config  # noqa: E402
from vngat.data.correspondence import compute_scene_correspondence  # noqa: E402
from vngat.data.features import get_features  # noqa: E402
from vngat.data.graph import merge_fragments  # noqa: E402
from vngat.data.io import list_fracture_dirs, load_scene  # noqa: E402
from vngat.data.mesh_ops import extract_fractures  # noqa: E402
from vngat.data.splits import scene_pool  # noqa: E402
from vngat.evaluation.metrics import geodesic_angle, swing_twist_error  # noqa: E402
from vngat.models.vn_gat import VNGATModel  # noqa: E402
from vngat.training.bridge import build_model_inputs, ground_truth_rotation  # noqa: E402


def load_model(path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(state["config"]) if state.get("config") else Config()
    model = VNGATModel(
        hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
        num_vn_slots=cfg.num_vn_slots, heads=cfg.heads, embed_dim=cfg.embed_dim,
        gram_bottleneck=cfg.gram_bottleneck, norm=cfg.norm,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, state


def random_rotations(num: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform via unit quaternions, seeded so a dump is reproducible."""
    q = rng.normal(size=(num, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--scene", default="")
    p.add_argument("--fracture", default="", help="e.g. fractured_3. Empty = the first one.")
    p.add_argument("--out", default="prediction.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--symmetry_axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--input_source", default="full", choices=["full", "frac"])
    # pool listing
    p.add_argument("--list_pool", action="store_true")
    p.add_argument("--list_fractures", action="store_true")
    p.add_argument("--root_dir", default="data")
    p.add_argument("--data_subsets", default="everyday_compressed")
    p.add_argument("--max_scenes", type=int, default=8)
    p.add_argument("--split_source", default="hash")
    p.add_argument("--split", default="train")
    args = p.parse_args(argv)

    if args.list_pool:
        # Reproduces exactly the subset --max_scenes selected during training.
        pool = scene_pool(
            args.root_dir, args.split, subsets=[s for s in args.data_subsets.split(",") if s],
            max_scenes=args.max_scenes, split_source=args.split_source,
        )
        print(f"{len(pool)} scene(s) in the {args.split} pool with --max_scenes {args.max_scenes}:")
        for i, s in enumerate(pool):
            print(f"  [{i}] {s}")
        return 0

    if args.list_fractures:
        if not args.scene:
            print("--list_fractures needs --scene")
            return 1
        dirs = list_fracture_dirs(args.scene)
        print(f"{len(dirs)} fracture pattern(s) in {args.scene}:")
        print("  " + ", ".join(dirs[:20]) + (" ..." if len(dirs) > 20 else ""))
        return 0

    if not args.checkpoint or not args.scene:
        print("need --checkpoint and --scene (or --list_pool / --list_fractures)")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, state = load_model(args.checkpoint, device)
    print(f"checkpoint epoch {state.get('epoch')} | train {state.get('train_loss'):.4f} "
          f"| val {state.get('val_loss'):.4f}")

    fracture = args.fracture or list_fracture_dirs(args.scene)[0]
    meshes = load_scene(args.scene, fracture)
    print(f"scene    : {args.scene}")
    print(f"fracture : {fracture}  ->  {len(meshes)} fragments")

    input_meshes = meshes
    if args.input_source == "frac":
        input_meshes = []
        for m in meshes:
            fm = extract_fractures(m)
            input_meshes.append(fm if (len(fm.faces) and len(fm.vertices) >= 3) else m)

    v_clu, e_clu = compute_scene_correspondence(input_meshes)
    graph = merge_fragments(
        [get_features(m, vc, ec) for m, vc, ec in zip(input_meshes, v_clu, e_clu)]
    ).to(device)

    rng = np.random.default_rng(args.seed)
    num_frag = graph.num_fragments
    A = random_rotations(num_frag, rng).astype(np.float32)
    extents = np.stack([m.vertices.max(0) - m.vertices.min(0) for m in meshes])
    max_dim = float(extents.max())
    t = (rng.normal(0.0, 0.75, size=(num_frag, 3))
         + rng.standard_normal((num_frag, 3)) * max_dim * 0.5).astype(np.float32)

    A_t = torch.from_numpy(A).to(device)
    with torch.no_grad():
        out = model(**build_model_inputs(graph.rotate_per_fragment(A_t)))
    R_pred = out["R_pred"].float().cpu()
    R_gt = ground_truth_rotation(A_t).cpu()

    geo = geodesic_angle(R_pred, R_gt).numpy()
    tilt, twist = swing_twist_error(R_pred, R_gt, axis=args.symmetry_axis)

    print(f"\nper-fragment geodesic error (deg), chance = 126.47:")
    for i, g in enumerate(geo):
        print(f"  fragment {i:>2}: {g:8.2f}    tilt {float(tilt[i]):7.2f}  twist {float(twist[i]):7.2f}")
    print(f"  {'mean':>11}: {geo.mean():8.2f}    tilt {float(tilt.mean()):7.2f}  twist {float(twist.mean()):7.2f}")

    verts = [np.asarray(m.vertices, np.float32) for m in meshes]
    faces = [np.asarray(m.faces, np.int32) for m in meshes]
    v_off = np.cumsum([0] + [len(v) for v in verts]).astype(np.int64)
    f_off = np.cumsum([0] + [len(f) for f in faces]).astype(np.int64)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vertices=np.concatenate(verts, 0),
        faces=np.concatenate(faces, 0),
        vertex_offsets=v_off,
        face_offsets=f_off,
        centroids=np.stack([v.mean(0) for v in verts]).astype(np.float32),
        A=A, t=t,
        R_pred=R_pred.numpy().astype(np.float32),
        R_gt=R_gt.numpy().astype(np.float32),
        geodesic_deg=geo.astype(np.float32),
        tilt_deg=np.asarray(tilt, np.float32),
        twist_deg=np.asarray(twist, np.float32),
        scene=np.array(str(args.scene)), fracture=np.array(fracture),
        seed=np.array(args.seed), epoch=np.array(int(state.get("epoch", -1))),
    )
    size = Path(args.out).stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size:.1f} MB)")
    print("Download it, then locally:  python visualize_reassembly.py --dump prediction.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
