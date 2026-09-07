#!/usr/bin/env python
"""
Find where non-finite values enter, for one named scene.

    python -m scripts.check_scene \
        --scene data/everyday_compressed/everyday_compressed/Cup/fc035f7d732166b97b6fd5468f603b31

Training logs which scenes produce NaN losses, but not WHY. The value can enter
at three places, and the fix differs at each:

  1. THE MESH -- trimesh divides a cross product by its own length to get a
     normal, so a zero-area triangle yields 0/0 = NaN. Repaired (and counted)
     by `vngat.data.features._sanitise`.
  2. THE MODEL IN FLOAT32 -- a genuine numerical bug, and the serious case.
  3. THE MODEL UNDER AMP ONLY -- fp16 overflow on this scene's size or
     fragment count. Mitigated by lowering the learning rate, or by --amp false.

This walks every fracture pattern of the scene through all three and reports
the first stage that goes non-finite.
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

from vngat.data.correspondence import compute_scene_correspondence  # noqa: E402
from vngat.data.features import get_features  # noqa: E402
from vngat.data.graph import merge_fragments  # noqa: E402
from vngat.data.io import load_random_scene, random_rotation_matrices  # noqa: E402
from vngat.losses.composite import CompositeLoss  # noqa: E402
from vngat.models.vn_gat import VNGATModel  # noqa: E402
from vngat.training.bridge import (  # noqa: E402
    build_model_inputs, build_predictions, build_targets,
)


def _bad(array) -> int:
    a = array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)
    return int((~np.isfinite(a)).sum())


def check_once(scene_dir: str, model, loss_fn, device, amp: bool, verbose: bool) -> dict:
    meshes = load_random_scene(scene_dir, fracture_pattern="fractured_")
    report = {"fragments": len(meshes), "stage": "ok"}
    if len(meshes) < 2:
        report["stage"] = "too few fragments"
        return report

    # -- stage 1: the raw mesh, BEFORE any repair --------------------------
    raw_bad = 0
    tiny = 0
    for m in meshes:
        raw_bad += _bad(np.asarray(m.vertices))
        if len(m.vertices):
            raw_bad += _bad(np.asarray(m.vertex_normals))
        if len(m.vertices) < 3 or len(m.faces) == 0:
            tiny += 1
    report["raw_non_finite"] = raw_bad
    report["degenerate_fragments"] = tiny

    # -- stage 2: features, AFTER repair -----------------------------------
    v_clu, e_clu = compute_scene_correspondence(meshes)
    frags = [get_features(m, vc, ec) for m, vc, ec in zip(meshes, v_clu, e_clu)]
    report["repaired"] = sum(getattr(f, "num_repaired", 0) for f in frags)
    graph = merge_fragments(frags).to(device)
    report["nodes"] = graph.num_nodes
    feat_bad = _bad(graph.node_vec) + _bad(graph.edge_vec) + _bad(graph.edge_len)
    report["feature_non_finite"] = feat_bad
    if feat_bad:
        report["stage"] = "FEATURES (repair did not cover this)"
        return report

    # -- stage 3: the model ------------------------------------------------
    rot = torch.from_numpy(
        random_rotation_matrices(graph.num_fragments).astype(np.float32)).to(device)
    diffused = graph.rotate_per_fragment(rot)

    def forward(use_amp: bool):
        ctx = (torch.autocast("cuda", dtype=torch.float16)
               if (use_amp and device.type == "cuda") else _Null())
        with ctx, torch.no_grad():
            out = model(**build_model_inputs(diffused))
            merged = dict(R_pred=out["R_pred"],
                          vertex_embedding=out["vertex_embedding"],
                          edge_embedding=out["edge_embedding"],
                          **build_predictions(diffused, out["R_pred"]))
            losses = loss_fn(merged, build_targets(graph, rot, diffused))
        return out, losses

    out32, loss32 = forward(False)
    report["model_fp32_non_finite"] = _bad(out32["R_pred"]) + _bad(out32["node_features"])
    report["loss_fp32_non_finite"] = sum(_bad(v) for v in loss32.values())
    if report["model_fp32_non_finite"] or report["loss_fp32_non_finite"]:
        report["stage"] = "MODEL in float32 -- a real numerical bug"
        return report

    if amp and device.type == "cuda":
        out16, loss16 = forward(True)
        report["model_amp_non_finite"] = _bad(out16["R_pred"]) + _bad(out16["node_features"])
        report["loss_amp_non_finite"] = sum(_bad(v) for v in loss16.values())
        if report["model_amp_non_finite"] or report["loss_amp_non_finite"]:
            report["stage"] = "MODEL under AMP only -- fp16 overflow"
    return report


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _locate(args, model, device) -> int:
    """
    Name the first module whose output goes non-finite, and show how activation
    magnitude grows through the network on the way there.

    Static reasoning found one overflow site and missed others; a forward hook
    on every module answers the question directly instead.
    """
    from vngat.data.io import random_rotation_matrices

    records: list = []

    def hook(name):
        def fn(_module, inputs, output):
            tensors = [t for t in (output if isinstance(output, tuple) else (output,))
                       if torch.is_tensor(t)]
            if not tensors:
                return
            worst = max(float(t.detach().abs().max()) if t.numel() else 0.0 for t in tensors)
            finite = all(bool(torch.isfinite(t).all()) for t in tensors)
            in_max = max((float(t.detach().abs().max())
                          for t in inputs if torch.is_tensor(t) and t.numel()), default=0.0)
            records.append((name, type(_module).__name__, in_max, worst, finite))
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]

    print(f"scene : {args.scene}")
    print(f"trials: {args.trials}\n")
    found = False
    for trial in range(args.trials):
        records.clear()
        meshes = load_random_scene(args.scene, fracture_pattern="fractured_")
        if len(meshes) < 2:
            continue
        v_clu, e_clu = compute_scene_correspondence(meshes)
        graph = merge_fragments(
            [get_features(m, vc, ec) for m, vc, ec in zip(meshes, v_clu, e_clu)]).to(device)
        rot = torch.from_numpy(
            random_rotation_matrices(graph.num_fragments).astype(np.float32)).to(device)
        diffused = graph.rotate_per_fragment(rot)

        ctx = (torch.autocast("cuda", dtype=torch.float16)
               if (args.amp and device.type == "cuda") else _Null())
        with ctx, torch.no_grad():
            out = model(**build_model_inputs(diffused))

        bad = [r for r in records if not r[4]]
        if not bad:
            peak = max(r[3] for r in records)
            print(f"  trial {trial:>3}: ok    frags={graph.num_fragments:>3} "
                  f"nodes={graph.num_nodes:>6} peak|activation|={peak:>12.1f}")
            continue

        found = True
        first = bad[0]
        idx = records.index(first)
        print(f"\n  trial {trial}: NON-FINITE  frags={graph.num_fragments} "
              f"nodes={graph.num_nodes}")
        print(f"  first non-finite module: {first[0]}  ({first[1]})")
        print(f"    input max |x| = {first[2]:.1f}   output max |y| = {first[3]:.1f}")
        print("\n  the ten modules leading up to it:")
        print(f"    {'module':<44}{'type':<18}{'in max':>12}{'out max':>12}")
        for name, kind, in_max, out_max, ok in records[max(0, idx - 10):idx + 1]:
            flag = "" if ok else "   <-- FIRST NON-FINITE"
            print(f"    {name:<44}{kind:<18}{in_max:>12.1f}{out_max:>12.1f}{flag}")
        break

    for h in handles:
        h.remove()

    if not found:
        print("\nNo non-finite output in any trial. If training still fails, the weights "
              "used here are not the ones that fail -- pass the checkpoint saved closest "
              "to the failing epoch.")
        return 0
    print("""
  READ:
    output max is far larger than input max  -> that module is the amplifier
    inputs already enormous (>10000)         -> the growth is upstream; the
                                                magnitude column shows where
    inputs modest but output non-finite      -> a genuine bug in that module
""")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to one scene directory.")
    p.add_argument("--trials", type=int, default=20,
                   help="Fracture patterns to try; the dataset draws one at random per call, "
                        "so a scene can fail on some patterns and not others.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="",
                   help="Trained weights. WITHOUT THIS the check is close to meaningless "
                        "for an overflow bug: a freshly initialised model has activations "
                        "of order 1 and cannot overflow float16 no matter what the geometry "
                        "is. The failure only appears once training has grown the weights.")
    p.add_argument("--locate", action="store_true",
                   help="Install forward hooks and report the FIRST module whose output "
                        "goes non-finite, with the activation magnitude profile leading "
                        "up to it.")
    args = p.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = state.get("config") or {}
        model = VNGATModel(
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 4),
            num_vn_slots=cfg.get("num_vn_slots", 8),
            heads=cfg.get("heads", 4),
            embed_dim=cfg.get("embed_dim", 32),
            gram_bottleneck=cfg.get("gram_bottleneck", 16),
            norm=cfg.get("norm", "layer"),
        ).to(device)
        model.load_state_dict(state["model"])
        print(f"loaded checkpoint from epoch {state.get('epoch')}")
        param_max = max(float(p.abs().max()) for p in model.parameters())
        print(f"largest parameter magnitude: {param_max:.2f}")
    else:
        model = VNGATModel().to(device)
        print("WARNING: no --checkpoint. An untrained model has O(1) activations and "
              "cannot reproduce an overflow. Pass --checkpoint to make this meaningful.")
    model.eval()
    loss_fn = CompositeLoss()

    if args.locate:
        return _locate(args, model, device)

    print(f"scene : {args.scene}")
    print(f"device: {device} | amp: {args.amp} | trials: {args.trials}\n")

    failures = []
    for i in range(args.trials):
        try:
            r = check_once(args.scene, model, loss_fn, device, args.amp, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  trial {i:>3}: raised {type(exc).__name__}: {exc}")
            continue
        flag = "ok" if r["stage"] == "ok" else "FAIL"
        print(f"  trial {i:>3}: {flag:<5} frags={r['fragments']:>3} nodes={r.get('nodes', 0):>6} "
              f"raw_nan={r.get('raw_non_finite', 0):>5} repaired={r.get('repaired', 0):>5} "
              f"degenerate_frags={r.get('degenerate_fragments', 0):>3}"
              + ("" if r["stage"] == "ok" else f"  <- {r['stage']}"))
        if r["stage"] != "ok":
            failures.append(r)

    print(f"\n{len(failures)} of {args.trials} trials went non-finite.")
    if not failures:
        print("This scene is clean under the current code. If training still reports NaN on it,\n"
              "the copy being trained is out of date -- run `python -m scripts.check_version`.")
        return 0

    stages = {f["stage"] for f in failures}
    print(f"failing stage(s): {sorted(stages)}")
    if any("FEATURES" in s for s in stages):
        print("\n-> Non-finite values survive the repair in vngat/data/features.py. Report the\n"
              "   counts above; the repair needs to cover whichever field is still bad.")
    if any("float32" in s for s in stages):
        print("\n-> A real numerical bug: the model produces NaN in full precision on this\n"
              "   geometry. This is worth fixing rather than skipping.")
    if any("AMP" in s for s in stages):
        print("\n-> fp16 overflow only. Lower --lr, or run this scene's size with --amp false.\n"
              "   Training already skips these steps, so the run is not corrupted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
