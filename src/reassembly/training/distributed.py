"""
Distributed training helpers.

WHY DistributedDataParallel AND NOT DataParallel
------------------------------------------------
``nn.DataParallel`` splits every tensor argument along dim 0 to divide a batch
across GPUs. It has no concept of a graph: ``edge_index``'s dim-0 size of 2
means "source/target", not "batch", which is where the original
``Expected edge_index to have size 2, got 1`` came from. Past that,
``num_fragments`` is a Python int and cannot be split at all, and
``fragment_id`` / ``fragment_scene_id`` cannot be repartitioned by naive row
slicing -- they need recomputed offsets, which is exactly what collation
already does correctly for a whole batch and which DataParallel has no way to
redo per shard.

DDP avoids the entire problem: one process per GPU, each with its own
COMPLETE, self-consistent batch; nothing is split mid-forward, and gradients
are all-reduced after backward.

THE HANG-ON-PARTIAL-FAILURE PROBLEM, AND THE FIX
------------------------------------------------
Vanilla DDP has a well-known failure mode: if one rank raises (say, it drew an
unusually large scene and hit OOM) while the others did not, the failing rank
dies and the survivors block forever at their next collective. The run appears
to hang rather than fail.

``all_ranks_agree`` fixes it for the case that actually happens here. Every
rank votes on whether its own batch succeeded, the votes are all-reduced, and
every rank then takes the same branch. A rank that OOMs makes ALL ranks skip
that step together, so the collectives stay in lockstep and the run continues
instead of hanging. That turns the single most likely cause of a mid-epoch
hang into a logged, skipped batch.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(rank: int, world_size: int, master_port: int = 12355,
                      backend: Optional[str] = None) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def cleanup_distributed() -> None:
    if is_distributed():
        # Barrier first, so a non-main rank cannot tear the process group down
        # (or exit the script) while rank 0 is still writing its checkpoint.
        dist.barrier()
        dist.destroy_process_group()


def all_ranks_agree(local_ok: bool, device: torch.device) -> bool:
    """Collective AND: True only if EVERY rank reports success.

    All ranks must call this at the same point, unconditionally -- that is the
    whole mechanism. Skipping the call on a rank that failed reintroduces the
    desynchronization it exists to prevent.
    """
    if not is_distributed():
        return local_ok
    flag = torch.tensor([1.0 if local_ok else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


def reduce_metrics(metrics: dict, device: torch.device) -> dict:
    """Average a metrics dict across ranks so printed numbers describe the
    whole run, not whatever slice one process happened to see."""
    if not is_distributed():
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= get_world_size()
    return {k: float(v) for k, v in zip(keys, values)}


@contextmanager
def main_process_first():
    """Let rank 0 run a block before the others (e.g. creating directories)."""
    if is_distributed() and not is_main_process():
        dist.barrier()
    try:
        yield
    finally:
        if is_distributed() and is_main_process():
            dist.barrier()
