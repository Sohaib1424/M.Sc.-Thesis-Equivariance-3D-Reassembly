#!/usr/bin/env python3
"""
Find where a scene goes wrong: the data, the model in fp32, the model under
AMP, or the gradient.

    python -m scripts.check_scene --root_dir data --scene everyday_compressed/Mug/<id>/fractured_3
    python -m scripts.check_scene --root_dir data --scene <key> --checkpoint runs/vgat/last.pt
    python -m scripts.check_scene --root_dir data --scene <key> --checkpoint ... --locate

``<key>`` is the name training prints for a failed batch and lists in
``<out-dir>/offenders.json`` -- ``<object>/<mode>``. A path to a scene directory
works too (with ``--mode``).

Training drops a batch whose loss or gradient is not finite, and names it; it
cannot say *why*. The value can enter at four places and the fix differs at
each, so this walks the scene through all four, under ``--trials`` different
perturbations (a scene can fail under some rotations and not others):

1. **The mesh.** Non-finite coordinates (the dataset skips such a scene by
   name), zero-area faces and vertices no face reaches (both repaired to a zero
   normal, and counted -- a zero trains silently).
2. **The model in float32** -- a real numerical bug, and the serious case.
3. **The model under AMP only** -- fp16 overflow on this scene's size.
4. **The gradient** -- a finite loss whose backward is not finite (a norm of a
   zero vector, a Gram-Schmidt step on parallel axes). The loop drops these
   too; ``--locate`` names the parameters that receive it.

Pass ``--checkpoint``: an untrained model has activations of order 1 and
cannot reproduce an overflow that only appears once training has grown the
weights. ``--locate`` hooks every module and reports the first whose output
goes non-finite, with the magnitude profile leading up to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.config_flags import add_config_arguments, config_from_args  # noqa: E402


def _bad(value) -> int:
    import torch

    if value is None:
        return 0
    array = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return int((~np.isfinite(array)).sum()) if array.dtype.kind == "f" else 0


def mesh_report(meshes) -> dict:
    """Stage 1: what is wrong with the geometry itself, before any repair."""
    from reassembly.mesh.topology import face_normals, vertex_normals

    report = {"fragments": len(meshes), "nonfinite_coords": 0, "degenerate_faces": 0,
              "unreferenced_vertices": 0, "tiny_fragments": 0}
    for mesh in meshes:
        v, f = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
        report["nonfinite_coords"] += _bad(v)
        if len(v) < 4 or len(f) == 0:
            report["tiny_fragments"] += 1
        if len(f) and np.isfinite(v).all():
            _, degenerate = face_normals(v, f, return_degenerate=True)
            report["degenerate_faces"] += int(degenerate)
            normals = vertex_normals(v, f)
            report["unreferenced_vertices"] += int((np.abs(normals).sum(1) == 0).sum())
    return report


def check_trial(dataset, index, model, criterion, config, device, seed, amp) -> dict:
    """Stages 2-4 for one perturbation."""
    import torch

    from reassembly.data.features import collate, complete_batch
    from reassembly.training import Skipped, _forward, _to_device

    sample = dataset.build(index, seed=seed)
    if isinstance(sample, Skipped):
        return {"stage": f"UNUSABLE: {sample.reason}"}
    report = {"stage": "ok", "vertices": sum(len(f.target_vertices) for f in sample.fragments),
              "repaired": sample.repairs.total}
    batch = complete_batch(_to_device(collate([sample]), device))
    report["feature_nonfinite"] = sum(_bad(getattr(batch, name)) for name in (
        "node_features", "edge_attr", "target_vertices", "target_normals",
        "target_edge_normals", "log_scale", "target_rotation"))
    if report["feature_nonfinite"]:
        report["stage"] = "FEATURES -- non-finite values survived the repair"
        return report

    with torch.no_grad():
        keep = {}
        loss, terms, _ = _forward(model, batch, criterion, config, keep=keep)
    prediction = keep["prediction"]
    report["loss"] = float(loss)
    report["model_nonfinite"] = (_bad(prediction.rotation) + _bad(prediction.vertex_embedding)
                                 + _bad(prediction.vertex_features))
    if report["model_nonfinite"] or not np.isfinite(report["loss"]):
        report["stage"] = "MODEL in float32 -- a real numerical bug"
        return report

    if amp and str(device).startswith("cuda"):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            loss16, _, R16 = _forward(model, batch, criterion, config)
        if not np.isfinite(float(loss16)) or _bad(R16):
            report["stage"] = "MODEL under AMP only -- fp16 overflow"
            return report

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss, _, _ = _forward(model, batch, criterion, config)
        loss.backward()
    bad = [name for name, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    model.zero_grad(set_to_none=True)
    report["gradient_nonfinite"] = len(bad)
    if bad:
        report["stage"] = f"GRADIENT -- finite loss, non-finite gradient in {bad[0]}" + (
            f" (+{len(bad) - 1} more)" if len(bad) > 1 else "")
    return report


def locate(dataset, index, model, criterion, config, device, trials, amp) -> int:
    """The first module whose forward output goes non-finite, and how big the
    activations were on the way there."""
    import contextlib

    import torch

    from reassembly.data.features import collate, complete_batch
    from reassembly.training import Skipped, _forward, _to_device

    records = []

    def hook(name):
        def record(module, inputs, output):
            tensors = [t for t in (output if isinstance(output, tuple) else (output,))
                       if torch.is_tensor(t) and t.is_floating_point()]
            if not tensors:
                return
            in_max = max((float(t.detach().abs().max()) for t in inputs
                          if torch.is_tensor(t) and t.is_floating_point() and t.numel()),
                         default=0.0)
            out_max = max(float(t.detach().abs().max()) if t.numel() else 0.0 for t in tensors)
            finite = all(bool(torch.isfinite(t).all()) for t in tensors)
            records.append((name, type(module).__name__, in_max, out_max, finite))
        return record

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        for trial in range(trials):
            sample = dataset.build(index, seed=trial)
            if isinstance(sample, Skipped):
                print(f"  unusable: {sample.reason}")
                return 1
            batch = complete_batch(_to_device(collate([sample]), device))
            records.clear()
            precision = (torch.autocast("cuda", dtype=torch.float16)
                         if amp and str(device).startswith("cuda") else contextlib.nullcontext())
            with torch.no_grad(), precision:
                _forward(model, batch, criterion, config)
            broken = [r for r in records if not r[4]]
            if not broken:
                peak = max((r[3] for r in records), default=0.0)
                print(f"  trial {trial:>3}: ok   peak |activation| {peak:12.1f}")
                continue
            first = records.index(broken[0])
            print(f"\n  trial {trial}: NON-FINITE -- first module: {broken[0][0]} "
                  f"({broken[0][1]})")
            print(f"    {'module':<46}{'type':<26}{'in max':>12}{'out max':>12}")
            for name, kind, in_max, out_max, ok in records[max(0, first - 10):first + 1]:
                print(f"    {name:<46}{kind:<26}{in_max:>12.1f}{out_max:>12.1f}"
                      + ("" if ok else "   <-- first non-finite"))
            print("\n  output max far above input max -> that module amplifies;"
                  "\n  inputs already huge (>1e4)     -> the growth is upstream;"
                  "\n  modest inputs, non-finite out  -> a bug in that module.")
            return 1
    finally:
        for handle in handles:
            handle.remove()
    print("\nNo non-finite forward output in any trial (the gradient is checked without "
          "--locate).")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True,
                        help="<object>/<mode> as training names it, or a scene directory")
    parser.add_argument("--mode", default=None,
                        help="fracture mode, when --scene is a directory")
    parser.add_argument("--checkpoint", default="",
                        help="trained weights (strongly recommended, see above)")
    parser.add_argument("--trials", type=int, default=8,
                        help="perturbations to try; failures can depend on the rotation")
    parser.add_argument("--locate", action="store_true",
                        help="report the first module whose output goes non-finite")
    parser.add_argument("--device", default=None)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)

    import torch

    from reassembly.training import (build_criterion, build_model, find_scene,
                                     load_checkpoint)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        model, config, state, path = load_checkpoint(args.checkpoint, config, device)
        print(f"checkpoint {path} (epoch {state.get('epoch')}), largest |weight| "
              f"{max(float(p.detach().abs().max()) for p in model.parameters()):.2f}")
    else:
        model = build_model(config).to(device).eval()
        print("WARNING: no --checkpoint. An untrained model has O(1) activations and "
              "cannot reproduce an overflow that training grew into.")
    criterion = build_criterion(config)

    key = args.scene
    if Path(key).is_dir():
        from reassembly.data.catalog import build_catalog
        from reassembly.data.paths import find_scenes

        directory = Path(key).resolve()
        entry = next((e for e in build_catalog(find_scenes(config.root, config.subsets)).objects
                      if any(Path(d).resolve() == directory for d, _ in e.modes)), None)
        if entry is None:
            print(f"{key} is not a scene under --root_dir {config.root}")
            return 2
        modes = [m for d, m in entry.modes if Path(d).resolve() == directory]
        key = f"{entry.key}/{args.mode or modes[0]}"
    try:
        dataset, index, split = find_scene(config, key)
    except KeyError as error:
        print(error.args[0])
        return 2
    print(f"scene  {key}   ({split} split)\ndevice {device}   amp {config.amp}   "
          f"trials {args.trials}\n")

    report = mesh_report(dataset.meshes(index))
    print("mesh   " + "   ".join(f"{k.replace('_', ' ')} {v}" for k, v in report.items()))
    if report["nonfinite_coords"]:
        print("  -> non-finite coordinates: the dataset skips this scene every time it is "
              "drawn. The mesh file itself is broken.")
        return 1

    if args.locate:
        return locate(dataset, index, model, criterion, config, device, args.trials, config.amp)

    failures = []
    for trial in range(args.trials):
        try:
            result = check_trial(dataset, index, model, criterion, config, device,
                                 seed=trial, amp=config.amp)
        except Exception as error:                          # noqa: BLE001
            print(f"  trial {trial:>3}: raised {type(error).__name__}: {error}")
            failures.append({"stage": f"RAISED {type(error).__name__}"})
            continue
        flag = "ok  " if result["stage"] == "ok" else "FAIL"
        print(f"  trial {trial:>3}: {flag} vertices {result.get('vertices', 0):>7,}  "
              f"repaired {result.get('repaired', 0):>4}  loss {result.get('loss', float('nan')):9.4f}"
              + ("" if result["stage"] == "ok" else f"   <- {result['stage']}"))
        if result["stage"] != "ok":
            failures.append(result)

    print(f"\n{len(failures)} of {args.trials} trials failed.")
    if not failures:
        print("Clean under this code and these weights. If training still names this scene,\n"
              "the copy being trained may be out of date: python -m scripts.check_version")
        return 0
    stages = {f["stage"].split(" --")[0] for f in failures}
    print(f"failing stage(s): {sorted(stages)}")
    if any(s.startswith("MODEL in float32") for s in stages):
        print("-> a real numerical bug in full precision on this geometry: worth fixing,\n"
              "   not skipping. --locate names the module.")
    if any(s.startswith("MODEL under AMP") for s in stages):
        print("-> fp16 overflow only: train with --amp False (the default), or lower --lr.")
    if any(s.startswith("GRADIENT") for s in stages):
        print("-> the forward is fine and the backward is not: look at the named parameter's\n"
              "   module for a norm or normalisation of a vector that can be zero.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
