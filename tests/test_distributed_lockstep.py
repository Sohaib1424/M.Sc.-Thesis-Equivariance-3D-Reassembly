"""
Collective ordering under DDP.

These test the property that broke a real run: NCCL matches collectives by
ORDER, so every rank must issue the same sequence. A vote placed AFTER
backward() cannot fix a divergence that backward() already caused.

They run on gloo/CPU with real processes, so they exercise the actual
collective machinery without needing two GPUs.
"""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _run(body: str, world_size: int = 2, timeout: int = 90):
    """Run `body` in `world_size` real processes under gloo."""
    import subprocess
    import tempfile
    import textwrap

    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {SRC!r})
        import torch, torch.distributed as dist
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("gloo", rank=rank, world_size=world)
        device = torch.device("cpu")
        from reassembly.training.distributed import all_ranks_agree, reduce_metrics
        try:
{textwrap.indent(textwrap.dedent(body), " " * 12)}
        finally:
            dist.destroy_process_group()
        print("OK", rank)
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name

    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
               WORLD_SIZE=str(world_size))
    procs = []
    for rank in range(world_size):
        procs.append(subprocess.Popen(
            [sys.executable, path], env=dict(env, RANK=str(rank)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for p in procs:
        try:
            outs.append(p.communicate(timeout=timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            outs.append(("", "TIMEOUT"))
    return outs


@pytest.mark.slow
def test_vote_before_work_keeps_ranks_in_lockstep():
    """The correct ordering: vote first, then all ranks take the same branch."""
    outs = _run("""
        for step in range(5):
            # rank 1 has an unusable batch on step 2
            local_ok = not (rank == 1 and step == 2)
            if not all_ranks_agree(local_ok, device):
                continue                     # every rank skips together
            work = torch.ones(64) * (step + 1)
            dist.all_reduce(work)            # stands in for the DDP gradient
    """)
    for out, err in outs:
        assert "OK" in out, f"rank did not finish cleanly: {err[-400:]}"


@pytest.mark.slow
def test_vote_after_work_deadlocks():
    """The bug, reproduced. This is what the old loop did, and it hangs --
    which is why it surfaced as a 600-second NCCL watchdog timeout rather
    than as an error at the point of divergence."""
    outs = _run("""
        for step in range(5):
            local_ok = not (rank == 1 and step == 2)
            if local_ok:
                work = torch.ones(64) * (step + 1)
                dist.all_reduce(work)        # rank 0 issues this, rank 1 does not
            all_ranks_agree(local_ok, device)   # too late
    """, timeout=20)
    assert any("TIMEOUT" in err or "OK" not in out for out, err in outs), \
        "voting after the collective should deadlock; it did not"


@pytest.mark.slow
def test_reduce_metrics_survives_a_rank_with_no_batches():
    """A rank that processed zero usable batches reports NaN for a FIXED key
    list. Deriving the key list from each rank's own dict would give the
    all-reduce different sizes on different ranks."""
    outs = _run("""
        KEYS = ("total", "rot", "pos")
        if rank == 0:
            metrics = {"total": 1.0, "rot": 2.0, "pos": 3.0}
        else:
            metrics = {k: float("nan") for k in KEYS}
        out = reduce_metrics(metrics, device, keys=KEYS)
        assert set(out) == set(KEYS), out
    """)
    for out, err in outs:
        assert "OK" in out, f"rank did not finish cleanly: {err[-400:]}"


@pytest.mark.slow
def test_metrics_average_across_ranks():
    outs = _run("""
        KEYS = ("total",)
        out = reduce_metrics({"total": float(rank)}, device, keys=KEYS)
        assert abs(out["total"] - 0.5) < 1e-9, out     # mean of 0 and 1
    """)
    for out, err in outs:
        assert "OK" in out, f"rank did not finish cleanly: {err[-400:]}"
