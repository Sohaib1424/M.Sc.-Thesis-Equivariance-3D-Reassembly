#!/usr/bin/env python
"""
Measure real peak GPU memory for a configuration, without touching the dataset.

    python scripts/profile_memory.py --nodes 15000 --hidden-channels 32
    python scripts/profile_memory.py --sweep

This exists because the intermittent OOM that motivated this whole rework was
never reproducible on demand: it depended on which scene the sampler drew.
Feeding synthetic graphs of a chosen size makes peak memory a measurement
rather than an anecdote, and lets you find the largest scene a given config can
actually take BEFORE committing to a long run.

--sweep prints the four levers side by side (checkpointing, AMP, head_dim,
hidden width) so the trade-offs are visible in one table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from reassembly.models.vn_gat_model import VNGATModel
from reassembly.utils.memory import estimate_activation_bytes


def synthetic_scene(num_nodes, num_fragments, device, avg_degree=6, seed=0):
    """A graph with realistic mesh statistics: no cross-fragment edges, ~6
    directed edges per vertex, bidirectional with a forward mask."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    per_frag = max(num_nodes // num_fragments, 4)
    num_nodes = per_frag * num_fragments

    fragment_id = torch.arange(num_fragments).repeat_interleave(per_frag)
    src_list, dst_list = [], []
    for f in range(num_fragments):
        base = f * per_frag
        e = per_frag * avg_degree // 2
        s = torch.randint(0, per_frag, (e,), generator=g) + base
        d = torch.randint(0, per_frag, (e,), generator=g) + base
        keep = s != d
        src_list.append(s[keep]); dst_list.append(d[keep])
    src = torch.cat(src_list); dst = torch.cat(dst_list)

    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    E = src.numel()
    is_forward = torch.cat([torch.ones(E, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)])

    return dict(
        x=torch.randn(num_nodes, 2, 3, generator=g).to(device),
        edge_index=edge_index.to(device),
        edge_scalar=torch.rand(2 * E, 1, generator=g).to(device),
        edge_vec=torch.randn(2 * E, 3, 3, generator=g).to(device),
        fragment_id=fragment_id.to(device),
        num_fragments=num_fragments,
        fragment_scene_id=torch.zeros(num_fragments, dtype=torch.long, device=device),
        forward_edge_mask=is_forward.to(device),
    )


def measure(device, nodes, fragments, hidden, layers, heads, head_dim,
            checkpointing, amp, slots=8):
    model = VNGATModel(
        hidden_channels=hidden, num_layers=layers, num_vn_slots=slots,
        heads=heads, head_dim=head_dim, gradient_checkpointing=checkpointing,
    ).to(device).train()
    batch = synthetic_scene(nodes, fragments, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(**batch)
        loss = out["R_pred"].square().sum() + out["vertex_embedding"].square().sum()
        loss.backward()
        peak = (torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else float("nan"))
        status = "ok"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        peak, status = float("nan"), "OOM"

    del model, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return peak, status


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=15000)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)

    if device.type != "cuda":
        print("!! Not on CUDA: peak-memory numbers will be NaN. "
              "Run this on the target GPU for the numbers that matter.")

    if not args.sweep:
        peak, status = measure(device, args.nodes, args.fragments, args.hidden_channels,
                               args.num_layers, args.heads, args.head_dim,
                               args.checkpointing, args.amp, args.slots)
        est = estimate_activation_bytes(
            args.nodes, 6 * args.nodes, args.hidden_channels, args.num_layers,
            args.slots, amp=args.amp, gradient_checkpointing=args.checkpointing)
        print(f"nodes={args.nodes} hidden={args.hidden_channels} layers={args.num_layers} "
              f"ckpt={args.checkpointing} amp={args.amp}")
        print(f"  measured peak : {peak:.3f} GiB  [{status}]")
        print(f"  estimate      : {est.gib:.3f} GiB  ({est.dominant_term})")
        return

    print(f"{'nodes':>8} {'hidden':>7} {'head_dim':>9} {'ckpt':>5} {'amp':>5} "
          f"{'peak GiB':>10} {'status':>7}")
    print("-" * 60)
    for nodes in (5000, 15000, 40000):
        for hidden in (32,):
            for head_dim, label in ((None, "h//heads"), (hidden, "full")):
                for ckpt in (False, True):
                    for amp in (False, True):
                        peak, status = measure(device, nodes, args.fragments, hidden,
                                               args.num_layers, args.heads, head_dim,
                                               ckpt, amp, args.slots)
                        print(f"{nodes:>8} {hidden:>7} {label:>9} {str(ckpt):>5} "
                              f"{str(amp):>5} {peak:>10.3f} {status:>7}")
    print("\n'full' head_dim reproduces the original sizing (every head at the full\n"
          "width); 'h//heads' is standard multi-head splitting. The gap between the\n"
          "two rows is pure overhead the original paid for no representational gain.")


if __name__ == "__main__":
    main()
