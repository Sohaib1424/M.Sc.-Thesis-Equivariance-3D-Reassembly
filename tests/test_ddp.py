"""
Two and three real DDP processes (gloo, CPU): the replicas must stay identical,
and the step's gradient must be the exact fragment-weighted mean over EVERY
rank.

This is the property a rank-local rescale breaks silently: DDP all-reduces at
the sync backward, so anything a rank multiplies in AFTER that makes the two
replicas take different steps -- and nothing in DDP ever re-synchronises
parameters, so they drift apart for the rest of the run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_scene  # noqa: E402

# Rank -> the scenes it draws, by fragment layout. Ranks see different fragment
# totals (5, 4 and 6) and different numbers of micro-batches, which is exactly
# when equal rank weighting is wrong. A run with `world` ranks uses the first
# `world` entries.
LAYOUTS = {
    0: [((10, 21), (7, 15), (13, 29)), ((9, 17), (6, 11))],
    1: [((11, 23), (7, 15), (19, 31), (8, 14))],
    2: [((8, 16), (9, 18)), ((6, 12), (7, 13)), ((12, 25), (10, 20))],
}
TIMEOUT = 240.0


def _micro(layout, seed):
    g = torch.Generator().manual_seed(seed)
    scene = make_scene(layout, seed=seed)
    q, _ = torch.linalg.qr(torch.randn(scene.num_fragments, 3, 3, generator=g, dtype=torch.float64))
    q[:, :, 0] *= torch.linalg.det(q).sign().unsqueeze(-1)
    return {"target": scene, "input": None, "rot": q.float(), "scene_dirs": [f"s{seed}"],
            "categories": ["C"], "num_scenes": 1}


def _all_micros(world=len(LAYOUTS)):
    return {rank: [_micro(layout, 100 * rank + i) for i, layout in enumerate(layouts)]
            for rank, layouts in LAYOUTS.items() if rank < world}


def _model():
    from vngat.models.vn_gat import VNGATModel

    torch.manual_seed(0)
    return VNGATModel(hidden_channels=8, num_layers=2, num_vn_slots=3, heads=2,
                      head_dim=4, embed_dim=4)


def _worker(rank, world, init_file, out_dir, drop_rank):
    from vngat.config import Config
    from vngat.losses.composite import CompositeLoss
    from vngat.training import trainer as T

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world)
    try:
        model = _model()
        ddp = T._wrap_ddp(model, torch.device("cpu"), rank, sync_buffers=False)
        cfg = Config(amp=False, device="cpu", grad_clip=0.0)
        scaler = T.make_scaler(torch.device("cpu"), False)
        loss_fn = CompositeLoss()
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)

        micros = _all_micros(world)[rank]
        frags = [m["target"].num_fragments for m in micros]
        rank_frags = float(sum(frags))
        contributed = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (micro, nf) in enumerate(zip(micros, frags)):
            if rank == drop_rank and i == 0:
                continue                                  # a dropped micro-batch
            result = T._run_micro_step(ddp, model, loss_fn, micro, torch.device("cpu"), cfg,
                                       True, scaler, ddp.no_sync, scale=nf / rank_frags)
            assert result.status == "ok"
            contributed += nf

        outcome, _ = T._finish_step(ddp, loss_fn, optimizer, scaler, torch.device("cpu"),
                                    cfg, rank_frags, contributed)
        assert outcome == "stepped"
        # SGD leaves the reduced gradient in place, and clipping is off.
        grads = [p.grad.detach().clone() for p in model.parameters()]
        torch.save({"grads": grads, "params": [p.detach().clone() for p in model.parameters()]},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _reference(drop_rank, world=2):
    """d/dtheta of the fragment-weighted mean over every contributed scene."""
    from vngat.losses.composite import CompositeLoss
    from vngat.training import trainer as T

    model = _model()
    loss_fn = CompositeLoss()
    pieces = []
    for rank, micros in _all_micros(world).items():
        for i, micro in enumerate(micros):
            if rank == drop_rank and i == 0:
                continue
            pieces.append(micro)
    total_frags = sum(m["target"].num_fragments for m in pieces)
    loss = 0.0
    for micro in pieces:
        scene = T.prepare_scene(micro, torch.device("cpu"))
        out = T._forward_loss(model, loss_fn, scene, torch.device("cpu"), False)
        loss = loss + out["total"] * (micro["target"].num_fragments / total_frags)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def _spawn(world, *args):
    """Run `world` ranks; fail -- not hang -- if they have not finished in time."""
    import time

    context = mp.spawn(_worker, args=(world, *args), nprocs=world, join=False)
    deadline = time.time() + TIMEOUT
    while not context.join(timeout=1.0):
        if time.time() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(f"{world} ranks did not finish in {TIMEOUT:.0f}s: a rank is "
                        f"waiting on a collective its peer never entered")


@pytest.mark.parametrize("world,drop_rank", [(2, -1), (2, 0), (3, -1), (3, 0), (3, 2)])
def test_ddp_step_is_the_global_fragment_mean_and_replicas_agree(tmp_path, world, drop_rank):
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    init_file = tmp_path / "init"
    _spawn(world, str(init_file), str(tmp_path), drop_rank)
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(world)]
    reference = _reference(drop_rank, world)
    for index, ref in enumerate(reference):
        first = ranks[0]["grads"][index]
        for other in ranks[1:]:
            assert torch.equal(first, other["grads"][index]), "ranks disagree on the reduced gradient"
        assert torch.allclose(first, ref, rtol=1e-4, atol=1e-6)
    for index in range(len(ranks[0]["params"])):
        for other in ranks[1:]:
            assert torch.equal(ranks[0]["params"][index], other["params"][index]), (
                "replicas drifted apart after one step")
