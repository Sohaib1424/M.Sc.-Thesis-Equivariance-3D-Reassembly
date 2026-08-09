"""
DistributedDataParallel helpers.

WHY DDP AND NOT nn.DataParallel
-------------------------------
`nn.DataParallel` splits every tensor argument along dim 0 to divide a batch
across GPUs. It has no concept of a graph: `edge_index`'s dim-0 size of 2 means
source/destination, not batch, so it fails immediately with
"expected size 2, got 1". Past that, `num_fragments` is a plain int and cannot
be split at all, and `node_frag`/`frag_scene` cannot be repartitioned by naive
row slicing -- they need recomputed offsets, which is exactly what collation
already does for a whole batch and which DataParallel cannot redo per shard.

DDP sidesteps all of it: one process per GPU, each building its own complete,
self-consistent batch; nothing is split mid-forward; gradients are all-reduced
after backward.

TWO DEADLOCK HAZARDS, HANDLED
-----------------------------
1. Collectives must be entered by every rank in the same order the same number
   of times. Anything that makes one rank skip a step -- a scene that fails to
   load, a batch that OOMs, an early `break` -- hangs the others at their next
   collective until NCCL times out. Every early exit in this project is
   therefore VOTED ON: `all_ranks_agree` runs an all-reduce so all ranks make
   the same decision, and the vote happens BEFORE any backward pass, never
   between one rank's backward and another's.

2. `no_sync()` wraps EVERY real micro-batch, and the all-reduce is fired
   afterwards by a fixed, tiny, zero-loss backward pass (see
   `trainer._sync_gradients`). Syncing on the last real micro-batch instead
   would mean that an out-of-memory failure part way through that backward had
   already fired some bucket reductions -- and the retry would fire them again,
   leaving this rank with more collectives than its peers. Deferring the
   reduction to a step that cannot fail keeps the collective count per
   optimiser step constant no matter what happens to the data.

   `no_sync()` is incompatible with `static_graph=True` (which assumes an
   identical autograd graph, including reduction points, every iteration), so
   static_graph is deliberately not set.

`find_unused_parameters` is left OFF: the loss is built so every parameter
always receives a gradient (see `cluster_consistency_loss`), which is both
faster and stricter than paying for a graph traversal every step.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.distributed as dist


def setup(rank: int, world_size: int, master_port: int = 12355, timeout_minutes: int = 30) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    import datetime

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:  # pragma: no cover
        device = torch.device("cpu")
    return device


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def all_ranks_agree(local_flag: bool, device: torch.device) -> bool:
    """
    Collective AND across ranks: True only if every rank passed True.

    Use this for any decision that changes control flow (stop early, skip the
    rest of an epoch, end on the time budget). Deciding locally is what turns
    "one rank ran out of time" into a silent hang.
    """
    if not is_distributed():
        return local_flag
    flag = torch.tensor([1.0 if local_flag else 0.0], device=device, dtype=torch.float32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


def reduce_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """
    Mean of a metric dict across ranks.

    Keys are SORTED before packing into the reduced tensor. Ranks can otherwise
    end up with differently-ordered (or, worse, differently-sized) dicts -- for
    example if one rank's epoch contained a batch with no shared vertices and
    another's did not -- and an all_reduce over mismatched tensors either
    corrupts values silently or hangs. Sorting plus an explicit size check
    makes the mismatch an immediate, readable error instead.
    """
    if not is_distributed():
        return metrics
    keys = sorted(metrics.keys())
    count = torch.tensor([len(keys)], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(count) for _ in range(world_size())]
    dist.all_gather(gathered, count)
    if len({int(c.item()) for c in gathered}) != 1:
        raise RuntimeError(
            f"reduce_metrics: ranks disagree on the number of metrics "
            f"{[int(c.item()) for c in gathered]}. Every rank must produce the same keys."
        )
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size()
    return {k: float(v) for k, v in zip(keys, values.tolist())}
