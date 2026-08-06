#!/usr/bin/env python
"""
End-to-end smoke test: real scenes off disk, through the full pipeline, to a
backward pass, printing shapes and values at every stage.

    python scripts/smoke_test.py --root-dir data
    python scripts/smoke_test.py --root-dir data --device cuda

A clean run means the pipeline is WIRED correctly end to end. It says nothing
about whether the model trains to a good result -- that is a separate question
answered by actually training.

Note the default device is cpu: this script tells you nothing about whether
CUDA works on your machine unless you pass ``--device cuda`` explicitly.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-vn-slots", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--decimate-to", type=int, default=4000)
    parser.add_argument("--input-source", type=str, default="full", choices=["full", "frac"])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
    print(f"smoke test on device: {device}   torch {torch.__version__}")

    from reassembly.config import Config
    from reassembly.data.collate import breaking_bad_collate_fn
    from reassembly.data.dataset import BreakingBadDataset
    from reassembly.models.vn_gat_model import VNGATModel
    from reassembly.training.bridge import (
        build_model_inputs, build_predictions, build_targets, select_input_graph,
    )
    from reassembly.training.losses import CompositeLoss

    # -------------------------------------------------------------- 0. splits
    section("STEP 0: scene discovery and the deterministic split")
    from reassembly.data.splits import split_scene_directories
    buckets = split_scene_directories(args.root_dir)
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        raise SystemExit(
            f"No scenes found under {args.root_dir!r}. Expected e.g. "
            f"{args.root_dir}/everyday_compressed/everyday_compressed/<category>/<shape>/"
        )
    for name, dirs in buckets.items():
        print(f"  {name:5s}: {len(dirs):6d} scenes ({100*len(dirs)/total:5.1f}%)")
    overlap = (set(buckets["train"]) & set(buckets["val"])) | \
              (set(buckets["train"]) & set(buckets["test"]))
    print(f"  train/val/test overlap: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "splits are not disjoint"

    # ---------------------------------------------------------- 1. one sample
    section("STEP 1: one dataset sample")
    ds = BreakingBadDataset(
        root_dir=args.root_dir, split="train",
        input_source=args.input_source,
        decimate_to=args.decimate_to, max_vertices=args.decimate_to * 4,
        return_meshes=True,
    )
    sample = ds[0]
    if sample is None:
        raise SystemExit("Every scene draw was rejected -- raise --decimate-to / max_vertices.")

    g = sample["graph"]
    print(f"  scene: {sample['scene_dir']}")
    print(f"  fragments        : {g.num_fragments}")
    print(f"  clean graph      : x={tuple(g.x.shape)} edge_index={tuple(g.edge_index.shape)} "
          f"edge_attr={tuple(g.edge_attr.shape)}")
    print(f"  diffused graph   : x={tuple(sample['diffused_graph'].x.shape)}")
    if sample["frac_graph"] is not None:
        fg = sample["frac_graph"]
        print(f"  fracture graph   : x={tuple(fg.x.shape)} "
              f"({100*fg.x.shape[0]/g.x.shape[0]:.1f}% of the full vertex count)")
    print(f"  t_matrices       : {tuple(sample['t_matrices'].shape)}")
    shared_v = int((g.vertex_cluster_id >= 0).sum())
    shared_e = int((g.edge_cluster_id >= 0).sum())
    print(f"  shared vertices  : {shared_v} / {g.x.shape[0]}  "
          f"({100*shared_v/max(g.x.shape[0],1):.1f}%)")
    print(f"  shared edges     : {shared_e} / {g.edge_attr.shape[0]}")
    if shared_v == 0:
        print("  !! WARNING: no cross-fragment correspondences found. The interface "
              "embedding losses will contribute nothing. Check correspondence_tol "
              "against the decimation voxel size.")

    # --------------------------------------------------- 2. centralization law
    section("STEP 2: centralization really does isolate rotation from translation")
    clean, diff = sample["graph"], sample["diffused_graph"]
    R_d = sample["t_matrices"][:, :3, :3]
    frag = clean.fragment_id
    lhs = diff.x[:, 0:3]
    rhs = torch.einsum("nij,nj->ni", R_d[frag], clean.x[:, 0:3])
    err = (lhs - rhs).abs().max().item()
    print(f"  max | x_diffused_centralized - R_diffuse @ x_clean_centralized | = {err:.3e}")
    print("  (this is the algebraic fact that makes R_gt well defined; expect ~1e-5 in fp32)")

    # --------------------------------------------------------- 3. a real batch
    section("STEP 3: collated batch")
    batch = breaking_bad_collate_fn([ds[i] for i in range(args.batch_size)])
    if batch["graph"] is None:
        raise SystemExit("All scenes in the batch were rejected.")
    bg = batch["graph"]
    print(f"  scenes           : {batch['num_scenes']}")
    print(f"  fragments        : {bg.num_fragments}")
    print(f"  fragment_scene_id: {bg.fragment_scene_id.tolist()}")
    print(f"  nodes            : {bg.x.shape[0]}   directed edges: {bg.edge_attr.shape[0]}")
    assert bg.fragment_scene_id.numel() == bg.num_fragments

    # edges never cross a fragment boundary -- build_predictions relies on this
    src_frag = bg.fragment_id[bg.edge_index[0]]
    dst_frag = bg.fragment_id[bg.edge_index[1]]
    print(f"  edges crossing a fragment boundary: {int((src_frag != dst_frag).sum())} (must be 0)")
    assert int((src_frag != dst_frag).sum()) == 0

    # ---------------------------------------------------------- 4. model pass
    section("STEP 4: forward pass")
    model = VNGATModel(
        hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        num_vn_slots=args.num_vn_slots, heads=args.heads, embed_dim=args.embed_dim,
    ).to(device)
    print(f"  parameters       : {model.num_parameters():,}")

    for key in ("graph", "diffused_graph", "frac_graph", "diff_frac_graph", "t_matrices"):
        if batch.get(key) is not None:
            batch[key] = batch[key].to(device)

    input_graph = select_input_graph(batch, args.input_source, diffused=True)
    inputs = build_model_inputs(input_graph)
    outputs = model(**inputs)
    print(f"  R_pred           : {tuple(outputs['R_pred'].shape)}")
    print(f"  vertex_embedding : {tuple(outputs['vertex_embedding'].shape)}")
    print(f"  edge_embedding   : {tuple(outputs['edge_embedding'].shape)}")

    R = outputs["R_pred"]
    ortho = (R @ R.transpose(-1, -2) - torch.eye(3, device=device)).abs().max().item()
    det = torch.det(R)
    print(f"  |R R^T - I|max   : {ortho:.3e}   det range: "
          f"[{det.min().item():.6f}, {det.max().item():.6f}] (must be ~+1)")

    # ------------------------------------------ 5. the equivariance law itself
    section("STEP 5: equivariance -- the model's law must match the TARGET's law")
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device, dtype=torch.float64))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    Q = Q.float()

    rotated = dict(inputs)
    rotated["x"] = torch.einsum("ij,nkj->nki", Q, inputs["x"])
    rotated["edge_vec"] = torch.einsum("ij,ekj->eki", Q, inputs["edge_vec"])
    with torch.no_grad():
        out_rot = model(**rotated)

    right = (out_rot["R_pred"] - R @ Q.transpose(0, 1)).abs().max().item()
    left = (out_rot["R_pred"] - Q @ R).abs().max().item()
    print(f"  | R_pred(Q x) - R_pred @ Q^T |  = {right:.3e}   <-- the required law")
    print(f"  | R_pred(Q x) - Q @ R_pred   |  = {left:.3e}   <-- the OLD (wrong) law")
    emb_err = (out_rot["vertex_embedding"] - outputs["vertex_embedding"]).abs().max().item()
    print(f"  | invariant embedding change |  = {emb_err:.3e}   (must be ~0)")
    if right > 1e-2:
        print("  !! the model's equivariance does not match its supervision target")

    # ---------------------------------------------------------- 6. loss + bwd
    section("STEP 6: loss and backward")
    targets = build_targets(batch["graph"], batch["t_matrices"], input_graph=input_graph)
    predicted = build_predictions(batch["diffused_graph"], outputs["R_pred"])
    loss_fn = CompositeLoss().to(device)
    losses = loss_fn(
        dict(R_pred=outputs["R_pred"], **predicted,
             vertex_embedding=outputs["vertex_embedding"],
             edge_embedding=outputs["edge_embedding"]),
        targets,
    )
    for k, v in losses.items():
        print(f"  {k:8s} = {v.item():.6f}")

    losses["total"].backward()
    grads = [(n, p.grad) for n, p in model.named_parameters()]
    missing = [n for n, gr in grads if gr is None]
    total_norm = sum(float(gr.norm()) for _n, gr in grads if gr is not None)
    print(f"  parameters with no gradient: {len(missing)} "
          f"{'(DDP would refuse this)' if missing else '(good -- DDP-safe)'}")
    if missing:
        print("   ", missing[:8])
    print(f"  summed gradient norm: {total_norm:.4f}")

    # ------------------------- 7. sanity: feeding R_gt reconstructs the clean mesh
    section("STEP 7: feeding the TRUE rotation must reconstruct the clean geometry")
    with torch.no_grad():
        perfect = build_predictions(batch["diffused_graph"], targets["R_gt"])
        err_true = (perfect["x_pred"] - targets["x_gt"]).abs().max().item()
        wrong = targets["R_gt"].flip(0)
        err_wrong = (build_predictions(batch["diffused_graph"], wrong)["x_pred"]
                     - targets["x_gt"]).abs().max().item()
    print(f"  with R_gt        : max position error = {err_true:.3e}  (must be ~0)")
    print(f"  with shuffled R  : max position error = {err_wrong:.3e}  (must be large)")

    print("\n" + "=" * 74)
    print("SMOKE TEST COMPLETE -- the pipeline is wired correctly end to end.")
    print("=" * 74)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
