"""
Several GPUs: one process per GPU, and gradients averaged by one explicit
all-reduce per optimizer step.

    from reassembly import distributed
    distributed.setup(rank, world)          # inside each spawned process
    ...
    distributed.sum_gradients(model.parameters())   # once per optimizer step

Every function here is the identity when no process group is running, so the
training loop calls them unconditionally and a single-GPU or CPU run takes
exactly the same path as a multi-GPU one.

Why not DistributedDataParallel
-------------------------------
DDP all-reduces *inside* ``backward()``, bucket by bucket, for every backward
that runs outside ``no_sync()``. That ties each rank's backward calls to every
other rank's: a rank that skips a batch -- too many vertices, nothing usable
in it, a non-finite loss -- or loses one to an out-of-memory error leaves its
peers blocked in a collective that never completes. The run does not crash; it
stops making progress and says nothing. That was this project's multi-GPU
path, twice over: the skips were rank-local, and the one guard against the OOM
case was to stop the whole run.

Here the ranks meet only at points every rank reaches by construction:

* once per optimizer step -- one tiny all-reduce of two numbers (how many
  fragments contributed, whether anyone wants to stop), then one all-reduce of
  the gradient itself;
* once per epoch -- the summaries are gathered, so the logged numbers cover
  every rank's data rather than rank 0's share of it.

Between those points a rank may skip, retry or drop whatever it likes, and the
others are never waiting on it.

What it costs: DDP overlaps the gradient all-reduce with the backward pass.
This model has about 0.9M parameters -- 3.7 MB of fp32 gradient -- so the one
all-reduce per step is a few milliseconds against steps of seconds. There is
nothing to recover.

What it relies on, and checks
-----------------------------
Replicas stay identical only if they start identical and then apply identical
updates. The first is enforced (:func:`broadcast_parameters`, because each
rank initialises from its own seed); the second follows from every rank
receiving the bit-identical reduced gradient and running the same optimizer
arithmetic. ``tests/test_distributed.py`` runs two and three real processes
and compares the replicas bit for bit.
"""
from __future__ import annotations

import os
import socket
from contextlib import closing
from typing import Iterable, List, Optional, Sequence

DEFAULT_PORT = 29513


def active() -> bool:
    """True inside a running process group."""
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if active() else 1


def rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if active() else 0


def free_port() -> int:
    """A port nothing is listening on, for a process group on this machine."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def setup(rank: int, world: int, cuda: bool) -> str:
    """
    Join the process group and return this rank's device.

    NCCL on GPUs, gloo otherwise -- gloo is what lets the multi-process path be
    tested on a CPU-only machine, where a bug that needs two ranks to appear
    would otherwise first show itself in a paid GPU session.
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(DEFAULT_PORT))
    if cuda:
        # Before init_process_group: NCCL binds each rank to the current device.
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    return f"cuda:{rank}" if cuda else "cpu"


def teardown() -> None:
    import torch.distributed as dist

    if active():
        dist.barrier()
        dist.destroy_process_group()


def sum_scalars(values: Sequence[float], device="cpu") -> List[float]:
    """
    Sum a handful of numbers across ranks, in float64.

    Float64 because these are counts and running totals: at float32 a sum of
    fragment counts is exact only up to 2^24.
    """
    if not active():
        return [float(v) for v in values]
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64,
                          device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def any_rank(flag: bool, device="cpu") -> bool:
    """Whether ``flag`` is set on at least one rank. Every rank gets the same answer."""
    return sum_scalars([1.0 if flag else 0.0], device)[0] > 0.0


def gather(value) -> list:
    """
    Every rank's ``value``, in rank order, on every rank.

    For the per-epoch summaries -- predictions, categories, dropped-sample
    names -- which are small, irregular and only needed once an epoch, so
    pickling them is the right trade against a hand-built tensor protocol.
    """
    if not active():
        return [value]
    import torch.distributed as dist

    out: list = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return out


def broadcast_parameters(module, source: int = 0) -> None:
    """
    Make every rank's weights (and buffers) rank ``source``'s.

    Each rank seeds itself differently -- so their *data* streams differ -- and
    that makes their freshly initialised weights differ too. DDP hides this by
    broadcasting in its constructor; without DDP it has to be done here, or the
    replicas start apart and never meet.
    """
    if not active():
        return
    import torch
    import torch.distributed as dist

    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=source)


def sum_gradients(parameters: Iterable) -> None:
    """
    Replace every rank's gradient with the sum over ranks, in one all-reduce.

    A parameter for which **no** rank produced a gradient keeps ``grad=None``,
    so the optimizer leaves it untouched exactly as it would on one GPU --
    weight decay and Adam's moment decay included. A parameter for which *some*
    rank did gets the sum on every rank, with the ranks that had nothing
    contributing zero. That keeps the ``None`` pattern identical across ranks,
    which the optimizer's per-parameter state depends on: a rank that stepped a
    parameter its peer skipped would drift from it for the rest of the run.

    The presence flags travel in the same buffer as the gradients, so this is
    one collective, not two.
    """
    if not active():
        return
    import torch
    import torch.distributed as dist

    params = [p for p in parameters if p.requires_grad]
    if not params:
        return
    reference = next((p.grad for p in params if p.grad is not None), params[0])
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    sizes = [p.numel() for p in params]
    flat = torch.zeros(sum(sizes) + len(params), dtype=dtype,
                       device=params[0].device)
    offset = 0
    flags = flat[sum(sizes):]
    for index, (p, size) in enumerate(zip(params, sizes)):
        if p.grad is not None:
            flat[offset:offset + size].copy_(p.grad.reshape(-1))
            flags[index] = 1.0
        offset += size

    dist.all_reduce(flat, op=dist.ReduceOp.SUM)

    offset = 0
    present = (flags > 0).tolist()
    for p, size, has in zip(params, sizes, present):
        if has:
            chunk = flat[offset:offset + size].view_as(p)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
        else:
            p.grad = None
        offset += size


def parameters_differ(module, atol: float = 0.0) -> Optional[str]:
    """
    Name the first parameter whose value is not the same on every rank, or
    ``None`` when all agree. For tests and for the paranoid; it gathers the
    whole model.
    """
    if not active():
        return None
    import torch

    for name, parameter in module.named_parameters():
        values = gather(parameter.detach().cpu())
        for other in values[1:]:
            if not torch.allclose(values[0], other, atol=atol, rtol=0.0):
                return name
    return None
