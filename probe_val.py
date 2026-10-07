#!/usr/bin/env python3
"""
Read-only probe of a trained checkpoint on the scenes it was validated on.

Changes nothing in the repository. It imports the package, rebuilds the run's
own validation set from the Config stored in the checkpoint (same scenes, same
perturbations), predicts, and writes per-fragment and per-pair numbers that the
history cannot show.

    cd "E:\\Thesis v6"
    python probe_val.py --checkpoint path\\to\\W10\\best.pt --out probe\\W10

Outputs <out>/fragments.csv, <out>/pairs.csv and <out>/summary.txt, and prints:

  token reach   share of vertices within k mesh hops of a token, and how many
                hops each cross layer's output travels before the pooling
  (a)           anchor-aligned error by quartile of fragment size, fracture
                share, tokens and token reach
  (b)           relative-rotation error of touching vs non-touching pairs, and
                the error through the anchor by contact hops from it
  (c)           tilt/twist of the error in the OBJECT frame, E = (C R_hat) R^T,
                per category, about x, y and z -- beside what the installed code
                logs (the input frame, R_hat^T R, before metrics.swing_twist_error
                was fixed; the object frame after)
  (d)           with --procrustes: the relative rotation of each pair fitted
                from the model's own embedding matches, against the network's

Two tests
---------
--cross none   The cross layers get no partners: every fragment is predicted
               as if it were alone in its scene.
--cross swap   Each scene's tokens attend to the tokens of a DIFFERENT object's
               scene instead of their own. The cross layers get realistic input
               that carries no information about this scene. Run with fewer
               scenes per batch (4 by default) to bound memory.
               If the validation error barely moves under these, the cross
               layers are not contributing to the rotation.

--procrustes   For every pair of fragments, match fracture-surface points by
               their embeddings (mutual nearest neighbours, as the assembly
               stage does), fit the rotation that lines the matched points up
               (Kabsch with RANSAC), and compare its error with the network's.
               Then chain those fitted rotations outward from the anchor along
               the best-matched pairs -- no ground-truth contacts used -- and
               report the per-fragment error that would give.

Is the matched route using anything it should not?
---------------------------------------------------
--hide_truth            Every ground-truth field of the batch (target rotations,
                        positions, normals, centroids, coincidence clusters) is
                        replaced by NaN / none before the network and the solver
                        run; the truth is used only afterwards, to score. If
                        anything read it, the numbers would change or turn NaN.
--untrained             Random weights, same architecture: fingerprints that
                        learned nothing. The matched route should fail.
--shuffle_fingerprints  The trained fingerprints, dealt out at random among each
                        fragment's fracture-surface vertices. Should fail too.
--jitter S --drop P     Noise on every input vertex (S, in largest-fragment
                        radii) and P of the fracture vertices left out of the
                        matching: the two sides of a break no longer coincide
                        and many partners are missing. Measures how much the
                        result owes to Breaking Bad's shared break vertices.

--split train scores training scenes the same way. --print_flags prints the
scripts.train command that reproduces the checkpoint's run, and exits. Run from
the repository root, or pass --repo <repository root>.
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
    p.add_argument("--cross", default="normal", choices=("normal", "none", "swap"),
                   help="what the cross layers see: the scene's own fragments (normal), "
                        "nothing (none), or another object's scene (swap)")
    p.add_argument("--batch_scenes", type=int, default=None,
                   help="scenes per forward pass; default: the run's own (at most 4 with "
                        "--cross swap)")
    p.add_argument("--procrustes", action="store_true",
                   help="fit each pair's rotation from the embedding matches (table d)")
    p.add_argument("--min_matches", type=int, default=6,
                   help="--procrustes: matches (and RANSAC inliers) a pair needs")
    p.add_argument("--ransac_tau", type=float, default=0.05,
                   help="--procrustes: inlier distance, in units of the largest fragment's "
                        "radius")
    p.add_argument("--ransac_iters", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="RANSAC draws and the swap order")
    # -- tests of what the matched route depends on ------------------------
    p.add_argument("--hide_truth", action="store_true",
                   help="replace every ground-truth field of the batch (target rotations, "
                        "positions, normals, centroids, coincidence clusters) with NaN / none "
                        "before the network and the solver run; the truth is used only "
                        "afterwards, to score. The numbers must not change")
    p.add_argument("--untrained", action="store_true",
                   help="the same architecture with random weights instead of the checkpoint's")
    p.add_argument("--shuffle_fingerprints", action="store_true",
                   help="--procrustes: shuffle the embeddings among each fragment's "
                        "fracture-surface vertices before matching")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="Gaussian noise on every input vertex, in units of the scene's largest "
                        "fragment's radius, so the two sides of a break no longer coincide")
    p.add_argument("--drop", type=float, default=0.0,
                   help="--procrustes: drop this share of each fragment's fracture-surface "
                        "vertices from the matching at random, so many partners are missing")
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


def tests(args) -> str:
    """The active tests, for the headers; '' when none."""
    active = [name for name in ("hide_truth", "untrained", "shuffle_fingerprints")
              if getattr(args, name)]
    active += [f"{name}={getattr(args, name):g}" for name in ("jitter", "drop")
               if getattr(args, name) > 0]
    return ", ".join(active)


def cross_hops(schedule):
    """(layer number, intra layers after it) for every cross layer."""
    schedule = list(schedule)
    return [(i + 1, sum(k == "intra" for k in schedule[i + 1:]))
            for i, kind in enumerate(schedule) if kind == "cross"]


# ---------------------------------------------------------------------------
# --cross: what the cross layers are allowed to see
# ---------------------------------------------------------------------------

def rewire(batch, mode, counters):
    """The batch with its cross-fragment pair lists replaced, per ``mode``."""
    import torch

    if mode == "normal":
        return batch
    device = batch.token_index.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    S = int(batch.num_scenes)
    if mode == "none" or S < 2:
        if mode == "swap":
            counters["swap_alone"] += S
        return batch._replace(token_query=empty, token_key=empty)

    # swap: tokens are stored scene by scene, so each scene's are one block.
    token_scene = batch.fragment_scene[batch.vertex_fragment[batch.token_index]]
    counts = torch.bincount(token_scene, minlength=S).tolist()
    starts = np.cumsum([0] + counts[:-1]).tolist()
    objects = [key.rsplit("/", 1)[0] for key in batch.scene_keys] if batch.scene_keys \
        else [str(s) for s in range(S)]
    queries, keys = [], []
    for s in range(S):
        if counts[s] == 0:
            continue
        order = [(s + k) % S for k in range(1, S)]
        partner = next((p for p in order if counts[p] and objects[p] != objects[s]), None)
        if partner is None:
            partner = next((p for p in order if counts[p]), None)
            counters["swap_same_object"] += 1
        if partner is None:
            counters["swap_alone"] += 1
            continue
        q = torch.arange(starts[s], starts[s] + counts[s], device=device)
        k = torch.arange(starts[partner], starts[partner] + counts[partner], device=device)
        queries.append(q.repeat_interleave(k.numel()))
        keys.append(k.repeat(q.numel()))
    if not queries:
        return batch._replace(token_query=empty, token_key=empty)
    return batch._replace(token_query=torch.cat(queries), token_key=torch.cat(keys))


# ---------------------------------------------------------------------------
# --hide_truth, --jitter: what the network and the solver are allowed to see
# ---------------------------------------------------------------------------

TRUTH_FIELDS = ("target_rotation", "target_vertices", "target_normals",
                "target_edge_normals", "centroid")


def hide_truth(batch):
    """
    The batch with every ground-truth field gone: the target rotations,
    positions, normals and edge normals and the true centroids become NaN, and
    the coincidence clusters (which vertices of two fragments were one point)
    become "none". What is left is what the model is given at inference -- the
    scattered input, the mesh graph, the fracture mask, the tokens, the scale.
    Anything that still read a hidden field would turn NaN or find nothing.
    """
    import torch

    hidden = {name: torch.full_like(getattr(batch, name), float("nan"))
              for name in TRUTH_FIELDS if getattr(batch, name) is not None}
    hidden["cluster"] = torch.full_like(batch.cluster, -1)
    hidden["num_clusters"] = 0
    return batch._replace(**hidden)


def jitter_inputs(batch, sigma, generator):
    """
    Independent Gaussian noise on every input vertex, so the two sides of a
    break no longer share exact positions; each edge's relative-position
    feature (``p_source - p_destination``) is recomputed to match.
    """
    import torch

    node = batch.node_features.clone()
    node[:, 0, :] += sigma * torch.randn(node[:, 0, :].shape, generator=generator,
                                         device=node.device, dtype=node.dtype)
    edge = batch.edge_attr.clone()
    edge[:, 2, :] = node[batch.edge_index[0], 0, :] - node[batch.edge_index[1], 0, :]
    return batch._replace(node_features=node, edge_attr=edge)


# ---------------------------------------------------------------------------
# --procrustes: rotations fitted from the embedding matches
# ---------------------------------------------------------------------------

def kabsch(P, Q):
    """``(R, t)`` minimising ``sum ||R p + t - q||^2`` over the last-but-one axis."""
    import torch

    cp, cq = P.mean(-2), Q.mean(-2)
    H = (P - cp.unsqueeze(-2)).transpose(-1, -2) @ (Q - cq.unsqueeze(-2))
    U, _, Vt = torch.linalg.svd(H)
    V, Ut = Vt.transpose(-1, -2), U.transpose(-1, -2)
    d = torch.sign(torch.det(V @ Ut))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = V @ D @ Ut
    t = cq - (R @ cp.unsqueeze(-1)).squeeze(-1)
    return R, t


def ransac_rotation(P, Q, iterations, tau, generator):
    """Best rotation mapping ``P`` onto ``Q`` despite wrong matches, and its inlier count."""
    import torch

    m = P.shape[0]
    weights = torch.ones(iterations, m, device=P.device)
    sample = torch.multinomial(weights, 3, replacement=False, generator=generator)
    R, t = kabsch(P[sample], Q[sample])                            # (K, 3, 3), (K, 3)
    residual = (torch.einsum("kij,mj->kmi", R, P) + t[:, None, :] - Q[None]).norm(dim=-1)
    inliers = residual < tau
    best = int(inliers.sum(1).argmax())
    mask = inliers[best]
    for _ in range(2):                                             # refit on the inliers
        if int(mask.sum()) < 3:
            break
        R_fit, t_fit = kabsch(P[mask], Q[mask])
        mask = (P @ R_fit.T + t_fit - Q).norm(dim=-1) < tau
        R[best], t[best] = R_fit, t_fit
    return R[best], int(mask.sum())


def fitted_rotations(batch, prediction, args, generator, noise=None):
    """``{(i, j): (R_ij, matches, inliers)}``, global fragment indices ``i < j``;
    ``R_ij`` maps fragment i's input frame onto fragment j's.

    Reads only the network's input (scattered coordinates, which fragment each
    vertex is in, the fracture mask, the normalisation scale) and its output
    (the embeddings) -- never a ground-truth field; ``--hide_truth`` checks it.
    ``noise`` is the generator for ``--drop`` and ``--shuffle_fingerprints``,
    kept apart from the RANSAC one so those draws are unchanged without them."""
    import torch

    from reassembly.assembly.translation import mutual_nearest_neighbours, subsample_per_fragment

    fitted = {}
    points = batch.node_features[:, 0, :].double()
    unit = batch.unit.double()
    fragment = batch.vertex_fragment.long()
    fragment_ptr = batch.fragment_ptr.tolist()
    vertex_ptr = batch.vertex_ptr.tolist()
    embedding = prediction.vertex_embedding
    for s in range(int(batch.num_scenes)):
        f0, f1 = fragment_ptr[s], fragment_ptr[s + 1]
        v0, v1 = vertex_ptr[f0], vertex_ptr[f1]
        if f1 - f0 < 2:
            continue
        local = fragment[v0:v1] - f0
        # In units of the scene's largest fragment, whatever the normalisation mode.
        scale = unit[f0:f1] / unit[f0:f1].max()
        p = points[v0:v1] * scale[local, None]
        candidates = None if batch.fracture is None else batch.fracture[v0:v1].clone()
        if args.drop > 0 and candidates is not None:
            candidates &= torch.rand(candidates.shape, generator=noise,
                                     device=candidates.device) >= args.drop
        scene_embedding = embedding[v0:v1]
        if args.shuffle_fingerprints:
            # Same fingerprints, each fragment's dealt out among its own
            # fracture-surface vertices at random: what they encode about the
            # piece survives, which vertex carries which does not.
            scene_embedding = scene_embedding.clone()
            pool = candidates if candidates is not None else torch.ones_like(local, dtype=torch.bool)
            for f in range(f1 - f0):
                index = torch.nonzero((local == f) & pool).flatten()
                if index.numel() > 1:
                    order = torch.randperm(index.numel(), generator=noise, device=index.device)
                    scene_embedding[index] = scene_embedding[index[order]]
        keep = subsample_per_fragment(local, 2048, candidates)
        source, target = mutual_nearest_neighbours(scene_embedding[keep], local[keep])
        source, target = keep[source], keep[target]
        if source.numel() == 0:
            continue
        a, b = local[source], local[target]                       # a < b by construction
        pair_id = a * (f1 - f0) + b
        for pid in torch.unique(pair_id).tolist():
            chosen = pair_id == pid
            count = int(chosen.sum())
            if count < args.min_matches:
                continue
            i, j = divmod(pid, f1 - f0)
            R, inliers = ransac_rotation(p[source[chosen]], p[target[chosen]],
                                         args.ransac_iters, args.ransac_tau, generator)
            fitted[(f0 + i, f0 + j)] = (R, count, inliers)
    return fitted


def chain_from_anchor(fitted, anchors, fragment_ptr, roots, min_inliers):
    """Fitted rotations composed outward from each scene's anchor along a maximum
    spanning tree of inlier counts. ``{fragment: (R, parent)}`` for reached fragments.

    ``roots`` are the NETWORK's rotations: each tree starts from the anchor's
    predicted rotation, so nothing here touches the truth. Scoring then sets the
    anchor to its true pose (the benchmark protocol), which turns ``R`` into
    ``T_a R_hat_a^T R`` -- exactly as if the tree had started from ``T_a``."""
    reached = {}
    for s, a in enumerate(anchors):
        f0, f1 = fragment_ptr[s], fragment_ptr[s + 1]
        if a >= f1 or a < f0:
            continue
        edges = {}
        for (i, j), (R, _, inliers) in fitted.items():
            if f0 <= i < f1 and inliers >= min_inliers:
                edges[(i, j)] = (inliers, R)
        pose = {a: roots[a]}
        reached[a] = (roots[a], -1)
        while True:
            best = None
            for (i, j), (w, R) in edges.items():
                if (i in pose) != (j in pose) and (best is None or w > best[0]):
                    best = (w, i, j, R)
            if best is None:
                break
            _, i, j, R = best
            if i in pose:            # parent i, child j: R maps i's input onto j's
                pose[j] = pose[i] @ R.transpose(-1, -2)
                reached[j] = (pose[j], i)
            else:                    # parent j, child i
                pose[i] = pose[j] @ R
                reached[i] = (pose[i], j)
    return reached


# ---------------------------------------------------------------------------
# One batch
# ---------------------------------------------------------------------------

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


def analyse(batch, prediction, config, args, generator=None, seen=None, noise=None):
    """Per-fragment rows, per-pair rows and reach counts for one batch.

    ``batch`` carries the truth and is used to SCORE; ``seen`` is what the
    solver is given (the same batch, or with ``--hide_truth`` one whose
    ground-truth fields are NaN)."""
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
    # Object frame, E = aligned @ T^T. Computed with the identity as the
    # "prediction", so it does not depend on which residual the installed
    # swing_twist_error takes: both conventions then give E or E^T, whose
    # tilt/twist are the same. 'logged' is whatever the installed code writes.
    E = aligned @ T.transpose(-1, -2)
    eye = torch.eye(3, dtype=E.dtype, device=E.device).expand_as(E)
    obj = {a: [t.cpu().numpy() for t in swing_twist_error(eye, E, axis=a)] for a in AXES}
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
    touch = contact >= args.min_contact
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

    fitted, chained = {}, {}
    if args.procrustes:
        # The solver gets `seen` -- with --hide_truth, a batch whose ground-truth
        # fields are NaN -- and starts each tree from the network's own rotation.
        fitted = fitted_rotations(seen if seen is not None else batch, prediction, args,
                                  generator, noise)
        chained = chain_from_anchor(fitted, anchors.tolist(), ptr.tolist(), R_hat,
                                    args.min_matches)
        # Scoring only: the benchmark protocol sets each anchor to its true pose.
        for f, (R, parent) in list(chained.items()):
            a = int(anchors[int(scene_np[f])])
            chained[f] = (T[a] @ R_hat[a].transpose(-1, -2) @ R, parent)

    def fit_error(R, i, j):
        truth = T[j].transpose(-1, -2) @ T[i]
        return float(geodesic_angle(R[None].to(T.dtype), truth[None])[0] * deg)

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
        if args.procrustes:
            pose = chained.get(f)
            parent = pose[1] if pose else -1
            row.update(
                chain_reached=int(pose is not None and f != a),
                chain_err_deg=(float(geodesic_angle(pose[0][None], T[f][None])[0] * deg)
                               if pose is not None and f != a else np.nan),
                chain_edge_touching=(int(touch[f, parent]) if pose is not None and f != a
                                     else -1))
        rows.append(row)

    pairs = []
    for i, j, r in zip(I, J, rel):
        row = dict(scene=rows[i]["scene"], category=rows[i]["category"],
                   i=rows[i]["fragment"], j=rows[j]["fragment"],
                   with_anchor=int(rows[i]["anchor"] or rows[j]["anchor"]),
                   contact=int(contact[i, j]), touching=int(touch[i, j]),
                   vertices_i=rows[i]["vertices"], vertices_j=rows[j]["vertices"],
                   rel_err_deg=float(r))
        if args.procrustes:
            hit = fitted.get((int(i), int(j)))
            row.update(matches=hit[1] if hit else 0, inliers=hit[2] if hit else 0,
                       fit_err_deg=fit_error(hit[0], int(i), int(j)) if hit else np.nan)
        pairs.append(row)
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


def summarise(rows, pairs, reach_counts, vertex_total, config, state, args, seconds, counters):
    out = []
    say = out.append
    scored = column(rows, "anchor") == 0
    err = column(rows, "err_anchor_deg")
    scenes = len({r["scene"] for r in rows})
    say(f"{args.split}: {scenes} scenes, {int(scored.sum())} scored fragments "
        f"({len(rows) - int(scored.sum())} anchors), {seconds:.0f} s, cross layers: {args.cross}")
    if tests(args):
        say(f"  tests: {tests(args)}")
    say(f"  mean anchor-aligned error {err[scored].mean():.2f} deg, median "
        f"{np.median(err[scored]):.2f} (chance {CHANCE_DEG})")
    history = state.get("history") or []
    logged = [r for r in history if r.get("epoch") == state.get("epoch")] or history[-1:]
    unchanged_model = not (args.untrained or args.jitter > 0)
    if logged and args.split == "val" and "val_geodesic_deg" in logged[0]:
        if args.cross == "normal" and unchanged_model:
            say(f"  the run logged val_geodesic_deg {logged[0]['val_geodesic_deg']:.2f} at epoch "
                f"{logged[0].get('epoch')} -- the two should agree (same scenes, same draws)")
        else:
            say(f"  the run logged val_geodesic_deg {logged[0]['val_geodesic_deg']:.2f} at epoch "
                f"{logged[0].get('epoch')} with the trained model, clean input and the cross "
                f"layers working normally")
    if args.cross == "swap":
        say(f"  swap partners: {counters['swap_same_object']} scene(s) had to take a scene of "
            f"the same object, {counters['swap_alone']} had no partner (left with none)")
    if counters["oom"]:
        say(f"  {counters['oom']} batch(es) ran out of memory and were skipped "
            f"(lower --batch_scenes)")

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
    say(f"    about the object's x, y, z; 'logged' is what this code version writes to "
        f"val_tilt_deg / val_twist_deg, about {config.symmetry_axis!r}")
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

    # -- (d) rotations fitted from the embedding matches ----------------------
    if args.procrustes:
        say("")
        say("(d) relative rotation fitted from the embedding matches (mutual nearest neighbours "
            "on fracture-surface")
        say(f"    points, Kabsch + RANSAC, inliers within {args.ransac_tau:g} of the largest "
            f"fragment's radius, >= {args.min_matches} needed)")
        fit = column(pairs, "fit_err_deg")
        has_fit = np.isfinite(fit)

        def compare(label, mask):
            n = int(mask.sum())
            if not n:
                say(f"    {label}: none")
                return
            net, fitted_ = rel[mask], fit[mask]
            say(f"    {label}: {n} pairs")
            say(f"      {'':16}{'network':>16}{'fitted':>16}")
            say(f"      {'mean / median':16}{net.mean():8.1f}/{np.median(net):6.1f}"
                f"{fitted_.mean():9.1f}/{np.median(fitted_):6.1f}")
            for threshold in (5, 15, 30):
                say(f"      {f'< {threshold} deg':16}{100 * (net < threshold).mean():15.0f}%"
                    f"{100 * (fitted_ < threshold).mean():15.0f}%")
            say(f"      fitted closer to the truth in {100 * (fitted_ < net).mean():.0f}% "
                f"of them")

        say(f"    touching pairs with a fit: {int((touching & has_fit).sum())} of "
            f"{int(touching.sum())}")
        compare("touching", touching & has_fit)
        compare("touching, with the anchor", touching & has_fit & with_anchor)
        say(f"    pairs that do NOT touch but still got a fit (false matches): "
            f"{int((~touching & has_fit).sum())} of {int((~touching).sum())}, fitted error "
            + (f"{np.median(fit[~touching & has_fit]):.1f} median" if (~touching & has_fit).any()
               else "-"))
        chain = column(sub, "chain_err_deg")
        got = np.isfinite(chain)
        say("    chained outward from the anchor along the best-matched pairs (no ground-truth "
            "contacts used):")
        if got.any():
            edge = column(sub, "chain_edge_touching")
            say(f"      reached {int(got.sum())} of {len(sub)} scored fragments "
                f"({100 * got.mean():.0f}%); tree edges between pieces that really touch: "
                f"{100 * (edge[got] == 1).mean():.0f}%")
            say(f"      on those fragments: network {e[got].mean():.1f} / {np.median(e[got]):.1f}"
                f"   chained {chain[got].mean():.1f} / {np.median(chain[got]):.1f}  "
                f"(mean / median)")
            combined = np.where(got, chain, e)
            say(f"      all scored fragments, chained where reached and the network elsewhere: "
                f"{combined.mean():.1f} mean, {np.median(combined):.1f} median "
                f"(network alone {e.mean():.1f})")
        else:
            say("      no fragment reached")
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

    from reassembly.data.features import complete_batch
    from reassembly.training import (BreakingBadScenes, _is_oom, _loader, _to_device,
                                     build_model)

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
    if args.untrained:
        torch.manual_seed(args.seed)               # random weights, same architecture
    model = build_model(config).to(device)
    if not args.untrained:
        model.load_state_dict(state["model"])
    model.eval()

    scenes_per_batch = args.batch_scenes or (min(config.batch_size, 4) if args.cross == "swap"
                                             else config.batch_size)
    config = dataclasses.replace(config, batch_size=scenes_per_batch)
    dataset = BreakingBadScenes(config, args.split, epoch_seed=0)
    if args.cross == "swap":
        # Neighbouring scenes are often break patterns of the same object; mixing
        # the order puts different objects in each batch to swap between.
        order = np.random.default_rng(args.seed).permutation(len(dataset.items))
        dataset.items = [dataset.items[i] for i in order]
    loader = _loader(dataset, config, False, 0, 1, 0,
                     pairs_in_worker=not str(device).startswith("cuda"))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # A separate stream for the tests, so RANSAC draws exactly as it does without them.
    noise = torch.Generator(device=device).manual_seed(args.seed + 1)
    print(f"checkpoint {args.checkpoint} (epoch {state.get('epoch')}), {len(dataset)} "
          f"{args.split} scenes, schedule {' '.join(config.schedule)}, cross layers: "
          f"{args.cross}, {scenes_per_batch} scenes per batch, device {device}")
    if tests(args):
        print(f"tests: {tests(args)}")

    rows, pairs = [], []
    reach_counts = np.zeros(HOPS + 1)
    vertex_total = dropped = 0
    counters = {"swap_same_object": 0, "swap_alone": 0, "oom": 0}
    began = time.time()
    with torch.no_grad():
        for index, (batch, skipped) in enumerate(loader):
            dropped += len(skipped)
            if batch is None:
                continue
            # The scattered input is made from the truth here, as the data
            # loader always does; after this point the truth is only for scoring.
            batch = rewire(complete_batch(_to_device(batch, device)), args.cross, counters)
            if args.jitter > 0:
                batch = jitter_inputs(batch, args.jitter, noise)
            seen = hide_truth(batch) if args.hide_truth else batch
            try:
                # Exactly the inputs `training._forward` passes the model.
                prediction = model(
                    seen.node_features, seen.edge_index, seen.edge_attr,
                    seen.vertex_fragment, seen.num_fragments,
                    log_scale=seen.log_scale, token_index=seen.token_index,
                    token_query=seen.token_query, token_key=seen.token_key)
            except Exception as error:                      # noqa: BLE001
                if not _is_oom(error):
                    raise
                counters["oom"] += 1
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            r, p, counts, n = analyse(batch, prediction, config, args, generator,
                                      seen=seen, noise=noise)
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
                      time.time() - began, counters)
    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n")
    print(f"\nwrote {out / 'fragments.csv'}, {out / 'pairs.csv'}, {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
