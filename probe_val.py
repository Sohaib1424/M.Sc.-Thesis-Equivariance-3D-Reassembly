#!/usr/bin/env python3
"""
Read-only probe of a trained checkpoint on the scenes it was validated on.

Changes nothing in the repository. It imports the package, rebuilds the run's
own validation set from the Config stored in the checkpoint (same scenes, same
perturbations), predicts, and writes per-fragment and per-pair numbers that the
history cannot show.

    cd "E:\\Thesis v6"
    python path\\to\\probe_val.py --checkpoint path\\to\\W10\\best.pt --root_dir <dataset> --out probe\\W10

Outputs <out>/fragments.csv, <out>/pairs.csv and <out>/summary.txt, and prints:

  token reach   share of vertices within k mesh hops of a token, and how many
                hops each cross layer's output travels before the pooling
  (a)           anchor-aligned error by quartile of fragment size, fracture
                share, tokens and token reach
  (b)           relative-rotation error of touching vs non-touching pairs, and
                the error through the anchor by contact hops from it
  (c)           tilt/twist of the error in the OBJECT frame, E = (C R_hat) R^T,
                per category, about x, y and z -- beside the logged values,
                which use R_hat^T R (the input frame)

--split train scores training scenes the same way, for a train-vs-val
comparison of the same tables. --print_flags prints the scripts.train command
that reproduces the checkpoint's run, and exits. Run from the repository root,
or pass --repo <repository root>.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z")
HOPS = 4
CHANCE_DEG = 126.48


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="e.g. runs/W10/best.pt")
    p.add_argument("--root_dir", default=None,
                   help="dataset root, when it is not where the run was trained")
    p.add_argument("--split", default="val", choices=("val", "train", "test"))
    p.add_argument("--limit", type=int, default=-1,
                   help="-1: the run's own validation subset (--val_steps x --batch_size); "
                        "0: the whole split; N: N scenes, strided like --val_steps")
    p.add_argument("--min_contact", type=int, default=10,
                   help="shared coincident vertices for two fragments to count as touching")
    p.add_argument("--num_workers", type=int, default=None,
                   help="data-loader workers; default: the run's own")
    p.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    p.add_argument("--repo", default=".",
                   help="repository root, the folder holding src/ (default: current folder)")
    p.add_argument("--out", default="probe", help="output folder")
    p.add_argument("--print_flags", action="store_true",
                   help="print the scripts.train command that reproduces the checkpoint's "
                        "run (its stored Config), and exit")
    return p.parse_args(argv)


def config_from_checkpoint(state, args):
    """The run's own Config, with only the data location, workers and scene count changed."""
    from reassembly.training import Config

    names = {f.name for f in dataclasses.fields(Config)}
    values = {k: v for k, v in (state.get("config") or {}).items() if k in names}
    if values.get("schedule") is not None:
        values["schedule"] = tuple(values["schedule"])
    if args.root_dir:
        values["root"] = args.root_dir
    if args.num_workers is not None:
        values["workers"] = args.num_workers
    key = "limit_train" if args.split == "train" else "limit_val"
    if args.limit == 0:
        values[key] = None
    elif args.limit > 0:
        values[key] = args.limit
    elif args.split == "train":
        values[key] = values.get("limit_val")       # as many scenes as validation scores
    return Config(**values)


def cross_hops(schedule):
    """(layer number, intra layers after it) for every cross layer."""
    schedule = list(schedule)
    return [(i + 1, sum(k == "intra" for k in schedule[i + 1:]))
            for i, kind in enumerate(schedule) if kind == "cross"]


def contact_matrix(cluster, vertex_fragment, count):
    """(F, F) number of coincidence clusters each pair of fragments shares."""
    import torch

    contact = np.zeros((count, count), np.int64)
    if cluster is None or cluster.numel() != vertex_fragment.numel():
        return contact
    mask = cluster >= 0
    if not bool(mask.any()):
        return contact
    pairs = torch.unique(torch.stack([cluster[mask], vertex_fragment[mask]]), dim=1)
    c, f = pairs.cpu().numpy()
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    sizes = np.diff(np.r_[starts, len(c)])
    two = starts[sizes == 2]
    np.add.at(contact, (f[two], f[two + 1]), 1)
    for s, n in zip(starts[sizes > 2], sizes[sizes > 2]):
        members = f[s:s + n]
        for x in range(n):
            for y in range(x + 1, n):
                contact[members[x], members[y]] += 1
    return contact + contact.T


def analyse(batch, prediction, config, min_contact):
    """Per-fragment rows, per-pair rows and reach counts for one batch."""
    import torch
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    from reassembly.evaluation.metrics import swing_twist_error
    from reassembly.nn.anchor import align_to_anchor, anchor_fragments
    from reassembly.nn.losses import geodesic_angle

    deg = 180.0 / math.pi
    F, S = int(batch.num_fragments), int(batch.num_scenes)
    scene = batch.fragment_scene.long()
    vf = batch.vertex_fragment.long()
    N = vf.numel()
    dev = vf.device

    R_hat = prediction.rotation.double()
    T = batch.target_rotation.double()
    anchor = anchor_fragments(batch.log_scale, scene, S)
    aligned, _ = align_to_anchor(R_hat, T, scene, anchor)
    err = (geodesic_angle(aligned, T) * deg).cpu().numpy()
    absolute = (geodesic_angle(R_hat, T) * deg).cpu().numpy()
    # Object frame: residual = aligned @ T^T (swing_twist_error uses pred^T @ target).
    obj = {a: [t.cpu().numpy() for t in swing_twist_error(
        aligned.transpose(-1, -2), T.transpose(-1, -2), axis=a)] for a in AXES}
    logged = [t.cpu().numpy() for t in swing_twist_error(aligned, T, axis=config.symmetry_axis)]

    ones = torch.ones(N, dtype=torch.float64, device=dev)
    count = torch.zeros(F, dtype=torch.float64, device=dev).index_add_(0, vf, ones)

    def share(mask):
        total = torch.zeros(F, dtype=torch.float64, device=dev).index_add_(0, vf, mask.double())
        return (total / count.clamp(min=1)).cpu().numpy()

    nan = np.full(F, np.nan)
    fracture = share(batch.fracture) if (batch.fracture is not None
                                         and batch.fracture.numel() == N) else nan
    has_cluster = batch.cluster is not None and batch.cluster.numel() == N
    coincident = share(batch.cluster >= 0) if has_cluster else nan
    tokens = torch.bincount(vf[batch.token_index], minlength=F).cpu().numpy()

    # Mesh hops from the nearest token: what each intra layer after a cross layer spreads.
    dist = torch.full((N,), HOPS + 1, dtype=torch.long, device=dev)
    reached = torch.zeros(N, dtype=torch.bool, device=dev)
    if batch.token_index.numel():
        reached[batch.token_index] = True
        dist[batch.token_index] = 0
    src, dst = batch.edge_index[0], batch.edge_index[1]
    for hop in range(1, HOPS + 1):
        nxt = torch.zeros_like(reached)
        nxt[dst[reached[src]]] = True
        new = nxt & ~reached
        dist[new] = hop
        reached |= new
    reach = np.stack([share(dist <= h) for h in range(HOPS + 1)], axis=1)
    reach_counts = np.array([float((dist <= h).sum()) for h in range(HOPS + 1)])

    contact = contact_matrix(batch.cluster if has_cluster else None, vf, F)
    touch = contact >= min_contact
    np.fill_diagonal(touch, False)

    anchors = anchor.cpu().numpy()
    scene_np = scene.cpu().numpy()
    hops = np.full(F, np.inf)
    present = np.flatnonzero(anchors < F)
    if len(present):
        d = shortest_path(csr_matrix(touch.astype(np.int8)), directed=False, unweighted=True,
                          indices=anchors[present])
        row = np.full(S, -1)
        row[present] = np.arange(len(present))
        ok = row[scene_np] >= 0
        hops[ok] = d[row[scene_np[ok]], np.flatnonzero(ok)]

    ptr = batch.fragment_ptr.cpu().numpy()
    I, J = [], []
    for s in range(S):
        lo, hi = int(ptr[s]), int(ptr[s + 1])
        ii, jj = np.triu_indices(hi - lo, k=1)
        I.append(ii + lo)
        J.append(jj + lo)
    I = np.concatenate(I) if I else np.zeros(0, int)
    J = np.concatenate(J) if J else np.zeros(0, int)
    rel_matrix = np.full((F, F), np.nan)
    rel = np.zeros(0)
    if len(I):
        It = torch.as_tensor(I, device=R_hat.device)
        Jt = torch.as_tensor(J, device=R_hat.device)
        P = R_hat[Jt].transpose(-1, -2) @ R_hat[It]      # predicted i -> j
        Q = T[Jt].transpose(-1, -2) @ T[It]              # true i -> j
        rel = (geodesic_angle(P, Q) * deg).cpu().numpy()
        rel_matrix[I, J] = rel
        rel_matrix[J, I] = rel

    log_scale = batch.log_scale.reshape(-1).double().cpu().numpy()
    counts = count.cpu().numpy()
    scene_vertices = np.bincount(scene_np, weights=counts, minlength=S)

    rows = []
    for f in range(F):
        s = int(scene_np[f])
        a = int(anchors[s])
        neighbours = np.flatnonzero(touch[f])
        nbr = rel_matrix[f, neighbours]
        row = dict(
            scene=batch.scene_keys[s] if batch.scene_keys else str(s),
            category=batch.categories[s] if batch.categories else "",
            fragment=f - int(ptr[s]), anchor=int(f == a),
            vertices=int(counts[f]), vertex_share=counts[f] / max(scene_vertices[s], 1),
            radius_to_anchor=math.exp(log_scale[f] - log_scale[a]),
            fracture_share=fracture[f], contact_share=coincident[f], tokens=int(tokens[f]),
            **{f"reach{h}": reach[f, h] for h in range(HOPS + 1)},
            neighbours=len(neighbours), contact_with_anchor=int(contact[f, a]) if f != a else 0,
            hops_to_anchor=hops[f],
            err_anchor_deg=err[f], err_absolute_deg=absolute[f],
            err_nbr_mean_deg=float(nbr.mean()) if len(nbr) else np.nan,
            err_nbr_min_deg=float(nbr.min()) if len(nbr) else np.nan,
            **{f"tilt_{x}": obj[x][0][f] for x in AXES},
            **{f"twist_{x}": obj[x][1][f] for x in AXES},
            tilt_logged=logged[0][f], twist_logged=logged[1][f],
        )
        rows.append(row)

    pairs = [dict(scene=rows[i]["scene"], category=rows[i]["category"],
                  i=rows[i]["fragment"], j=rows[j]["fragment"],
                  with_anchor=int(rows[i]["anchor"] or rows[j]["anchor"]),
                  contact=int(contact[i, j]), touching=int(touch[i, j]),
                  vertices_i=rows[i]["vertices"], vertices_j=rows[j]["vertices"],
                  rel_err_deg=float(r))
             for i, j, r in zip(I, J, rel)]
    return rows, pairs, reach_counts, N


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column(rows, name):
    return np.array([r[name] for r in rows], dtype=float)


def quartiles(x, e):
    """Mean of e in four equal-count bins of x, each with the median x, and Spearman rho."""
    from scipy.stats import rankdata, spearmanr

    ok = np.isfinite(x) & np.isfinite(e)
    x, e = x[ok], e[ok]
    if len(x) < 8:
        return None
    bins = ((rankdata(x, method="ordinal") - 1) * 4) // len(x)
    cells = [(e[bins == k].mean(), np.median(x[bins == k])) for k in range(4)]
    rho = spearmanr(x, e)[0] if np.ptp(x) > 0 else float("nan")
    return cells, rho


def summarise(rows, pairs, reach_counts, vertex_total, config, state, args, seconds):
    out = []
    say = out.append
    scored = column(rows, "anchor") == 0
    err = column(rows, "err_anchor_deg")
    scenes = len({r["scene"] for r in rows})
    say(f"{args.split}: {scenes} scenes, {int(scored.sum())} scored fragments "
        f"({len(rows) - int(scored.sum())} anchors), {seconds:.0f} s")
    say(f"  mean anchor-aligned error {err[scored].mean():.2f} deg, median "
        f"{np.median(err[scored]):.2f} (chance {CHANCE_DEG})")
    history = state.get("history") or []
    logged = [r for r in history if r.get("epoch") == state.get("epoch")] or history[-1:]
    if logged and args.split == "val" and "val_geodesic_deg" in logged[0]:
        say(f"  the run logged val_geodesic_deg {logged[0]['val_geodesic_deg']:.2f} at epoch "
            f"{logged[0].get('epoch')} -- the two should agree (same scenes, same draws)")

    # -- token reach (check 3) ----------------------------------------------
    share = reach_counts / max(vertex_total, 1)
    say("")
    say("token reach (vertex-weighted): tokens " + f"{100 * share[0]:.0f}%  " +
        "  ".join(f"+{h} {100 * share[h]:.0f}%" for h in range(1, HOPS + 1)))
    layers = cross_hops(config.schedule)
    if layers:
        say(f"  schedule {' '.join(config.schedule)}")
        say("  " + ", ".join(f"cross layer {n} -> +{h}" for n, h in layers))
        first = max(h for _, h in layers)
        last = layers[-1][1]
        say(f"  pooled by the head: {100 * share[min(first, HOPS)]:.0f}% of vertices carry "
            f"some cross-fragment information, {100 * share[min(last, HOPS)]:.0f}% carry the "
            f"last round")
    else:
        first = last = 0

    # -- (a) error vs fragment properties -------------------------------------
    sub = [r for r, k in zip(rows, scored) if k]
    e = err[scored]
    say("")
    say("(a) anchor-aligned error by quartile, scored fragments; in brackets the "
        "quartile's median value")
    say(f"    {'':22}{'Q1 (lowest)':>17}{'Q2':>17}{'Q3':>17}{'Q4 (highest)':>17}{'Spearman':>10}")
    props = [("vertex share", "vertex_share"), ("radius / anchor", "radius_to_anchor"),
             ("fracture share", "fracture_share"), ("contact share", "contact_share"),
             ("tokens", "tokens"), (f"reach +{last}", f"reach{min(last, HOPS)}"),
             (f"reach +{first}", f"reach{min(first, HOPS)}"),
             ("touching neighbours", "neighbours")]
    for label, name in props:
        result = quartiles(column(sub, name), e)
        if result is None:
            continue
        cells, rho = result
        text = "".join(f"{m:8.1f} ({v:6.3g})" for m, v in cells)
        say(f"    {label:22}{text}{rho:10.2f}")

    # -- (b) pairs and hops ---------------------------------------------------
    say("")
    say(f"(b) relative-rotation error between fragments of one scene, deg "
        f"(touching = at least {args.min_contact} shared coincident vertices)")
    rel = column(pairs, "rel_err_deg")
    touching = column(pairs, "touching") == 1
    with_anchor = column(pairs, "with_anchor") == 1
    say(f"    {'pairs':34}{'n':>7}{'mean':>8}{'median':>8}")

    def line(label, mask, values):
        if mask.any():
            say(f"    {label:34}{int(mask.sum()):7d}{values[mask].mean():8.1f}"
                f"{np.median(values[mask]):8.1f}")
        else:
            say(f"    {label:34}{0:7d}")

    line("touching, with the anchor", touching & with_anchor, rel)
    line("touching, without the anchor", touching & ~with_anchor, rel)
    line("not touching, with the anchor", ~touching & with_anchor, rel)
    line("not touching, without the anchor", ~touching & ~with_anchor, rel)
    if touching.sum() >= 8:
        cells, rho = quartiles(column(pairs, "contact")[touching], rel[touching])
        say("    touching pairs by contact size:  " +
            "  ".join(f"Q{k + 1} {m:.1f} ({v:.0f})" for k, (m, v) in enumerate(cells)) +
            f"   Spearman {rho:.2f}")
    hops = column(sub, "hops_to_anchor")
    say(f"    {'error through the anchor, by hops':34}{'n':>7}{'mean':>8}{'median':>8}")
    line("  1 (touches the anchor)", hops == 1, e)
    line("  2", hops == 2, e)
    line("  3 or more", np.isfinite(hops) & (hops >= 3), e)
    line("  not connected to it", ~np.isfinite(hops), e)
    nbr = column(sub, "err_nbr_mean_deg")
    has = np.isfinite(nbr)
    if has.any():
        say(f"    same {int(has.sum())} fragments that touch something: error to the anchor "
            f"{e[has].mean():.1f}, mean error to their touching neighbours {nbr[has].mean():.1f}")

    # -- (c) tilt / twist in the object frame ---------------------------------
    say("")
    say("(c) tilt / twist of the anchor-aligned error, deg (chance 90 / 90). Object frame "
        "E = (C R_hat) R^T,")
    say(f"    about the object's x, y, z; 'logged' is R_hat^T R about "
        f"{config.symmetry_axis!r}, as val_tilt_deg / val_twist_deg")
    say(f"    {'category':18}{'n':>6}{'geo':>7}   {'x tilt/twist':>13}   {'y tilt/twist':>13}"
        f"   {'z tilt/twist':>13}   {'logged':>13}")
    categories = np.array([r["category"] for r in sub])
    groups = sorted(set(categories)) + ["ALL"]
    for name in groups:
        mask = np.ones(len(sub), bool) if name == "ALL" else categories == name
        if not mask.any():
            continue
        cells = "".join(
            f"   {column(sub, f'tilt_{x}')[mask].mean():6.1f}/{column(sub, f'twist_{x}')[mask].mean():6.1f}"
            for x in AXES)
        cells += (f"   {column(sub, 'tilt_logged')[mask].mean():6.1f}/"
                  f"{column(sub, 'twist_logged')[mask].mean():6.1f}")
        say(f"    {name[:18]:18}{int(mask.sum()):6d}{e[mask].mean():7.1f}{cells}")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    if not (repo / "src" / "reassembly").is_dir():
        print(f"no src/reassembly under {repo}: run from the repository root or pass --repo")
        return 2
    for path in (repo, repo / "src"):
        sys.path.insert(0, str(path))

    import torch

    from reassembly.training import (BreakingBadScenes, _forward, _loader, _to_device,
                                     build_criterion, build_model)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(state, args)
    if args.print_flags:
        from scripts.config_flags import config_flags

        from reassembly.training import Config, flag

        flags = config_flags(config)
        # config_flags drops a None ("no limit") whose default is not None, so
        # --modes_per_scene 0 would come back as the default 8: say it explicitly.
        default = Config()
        for name in ("modes_per_scene", "limit_train", "max_objects", "max_fragments"):
            if getattr(config, name) is None and getattr(default, name) is not None:
                flags += [flag(name), "0"]
        quote = (lambda s: f'"{s}"' if not s or any(c.isspace() for c in s) else s)
        print("python -m scripts.train " + " ".join(quote(str(f)) for f in flags))
        return 0
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = BreakingBadScenes(config, args.split, epoch_seed=0)
    loader = _loader(dataset, config, False, 0, 1, 0,
                     pairs_in_worker=not str(device).startswith("cuda"))
    criterion = build_criterion(config)
    print(f"checkpoint {args.checkpoint} (epoch {state.get('epoch')}), {len(dataset)} "
          f"{args.split} scenes, schedule {' '.join(config.schedule)}, device {device}")

    rows, pairs = [], []
    reach_counts = np.zeros(HOPS + 1)
    vertex_total = dropped = 0
    began = time.time()
    with torch.no_grad():
        for index, (batch, skipped) in enumerate(loader):
            dropped += len(skipped)
            if batch is None:
                continue
            keep = {}
            _forward(model, _to_device(batch, device), criterion, config, keep=keep)
            r, p, counts, n = analyse(keep["batch"], keep["prediction"], config,
                                      args.min_contact)
            rows += r
            pairs += p
            reach_counts += counts
            vertex_total += n
            if (index + 1) % 10 == 0:
                print(f"  {index + 1} batches, {len(rows)} fragments", flush=True)
    if not rows:
        print("no scenes could be built")
        return 1
    if dropped:
        print(f"  {dropped} scene(s) skipped by the dataset (single fragment or unreadable)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fragments.csv", rows)
    write_csv(out / "pairs.csv", pairs)
    lines = summarise(rows, pairs, reach_counts, vertex_total, config, state, args,
                      time.time() - began)
    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n")
    print(f"\nwrote {out / 'fragments.csv'}, {out / 'pairs.csv'}, {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
