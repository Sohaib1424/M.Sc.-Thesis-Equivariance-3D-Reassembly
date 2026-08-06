"""
Translating a collated batch into (a) the tensors the model consumes and
(b) the ground truth the loss consumes.

Kept as its own module rather than inlined in the training loop because the
index bookkeeping here is exactly the kind of thing that is easy to get subtly
wrong once and then train on for a week without noticing.

THE INPUT-SOURCE SPLIT
----------------------
With ``input_source='frac'`` the model sees the pruned fracture-surface graph
while every geometry loss is still evaluated against the FULL mesh. That is
the experiment the thesis wants: "can a model that only ever looks at the
interface skeleton still reassemble the whole fragment?", which is a direct
accuracy-vs-compute measurement.

It is sound because centralization makes the rotation relationship identical on
both meshes. For any vertex subset S of a fragment, the diffused subset's own
centroid is ``R_d c_S + t``, so its centralized coordinates are
``R_d (v - c_S)`` -- a pure rotation, with the same ``R_d``, independent of
which subset was chosen. Both graphs therefore share one ``R_gt``.

Routing, explicitly:
    model input        -> diffused INPUT graph (frac or full)
    geometry targets   -> clean FULL graph
    geometry predicted -> diffused FULL graph, rotated by R_pred
    embedding targets  -> INPUT graph's cluster ids (the embeddings live on
                          the input graph's nodes/edges, so their supervision
                          must come from the same graph)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _num_fragments(graph) -> int:
    """Fragment count, preferring the explicit attribute.

    ``fragment_id.max() + 1`` is only correct when every fragment contributed
    at least one node. The pruned fracture-surface variant can produce an empty
    fragment, which never appears in ``fragment_id`` -- and then every
    per-fragment tensor (R_pred, t_matrices, fragment_scene_id) silently
    misaligns, with no shape error to catch it.
    """
    explicit = getattr(graph, "num_fragments", None)
    if explicit is not None:
        return int(explicit)
    if graph.fragment_id.numel() == 0:
        return 0
    return int(graph.fragment_id.max().item()) + 1


def build_model_inputs(input_graph) -> Dict:
    """The scattered scene is the network's input."""
    x = input_graph.x                                     # (N, 6) = [pos(3), norm(3)]
    x_vec = torch.stack([x[:, 0:3], x[:, 3:6]], dim=1)    # (N, 2, 3)

    edge_attr = input_graph.edge_attr                     # (2E, 10)
    edge_scalar = edge_attr[:, 0:1]                       # (2E, 1) length
    edge_vec = edge_attr[:, 1:10].reshape(-1, 3, 3)       # (2E, 3, 3) = [mid, n1, n2]

    return dict(
        x=x_vec,
        edge_index=input_graph.edge_index,
        edge_scalar=edge_scalar,
        edge_vec=edge_vec,
        fragment_id=input_graph.fragment_id,
        num_fragments=_num_fragments(input_graph),
        fragment_scene_id=getattr(input_graph, "fragment_scene_id", None),
        forward_edge_mask=input_graph.is_forward_edge,
    )


def build_targets(
    clean_graph,
    t_matrices: torch.Tensor,
    input_graph=None,
) -> Dict:
    """Ground truth for the composite loss.

    ``clean_graph``  the unperturbed FULL scene -- what a correctly reassembled
                     fragment's local geometry should look like.
    ``t_matrices``   ``(F, 4, 4)`` per-fragment diffusion transforms.
    ``input_graph``  whichever graph the model consumed; its cluster ids
                     supervise the embeddings. Defaults to ``clean_graph``.

    ``R_gt = R_diffuse^T``. Justification: feature construction centralizes each
    fragment and diffusion applies ``v' = R v + t``, so centralization cancels
    the translation exactly and ``x~_diffused = R x~_clean``. Undoing that needs
    ``R^-1 = R^T`` (a rotation is orthogonal).
    """
    if input_graph is None:
        input_graph = clean_graph

    x_full = clean_graph.x
    forward = clean_graph.is_forward_edge
    edge_attr = clean_graph.edge_attr

    R_diffuse = t_matrices[:, :3, :3]
    targets = dict(
        R_gt=R_diffuse.transpose(-1, -2),
        x_gt=x_full[:, 0:3],
        n_gt=x_full[:, 3:6],
        mid_gt=edge_attr[forward, 1:4],
        n1_gt=edge_attr[forward, 4:7],
        n2_gt=edge_attr[forward, 7:10],
    )

    vcid = getattr(input_graph, "vertex_cluster_id", None)
    targets["vertex_cluster_id"] = (
        vcid if vcid is not None
        else torch.full((input_graph.x.size(0),), -1, dtype=torch.long,
                        device=input_graph.x.device)
    )

    ecid = getattr(input_graph, "edge_cluster_id", None)
    if ecid is not None:
        # edge_cluster_id was doubled (forward + backward) during feature
        # construction; the model's edge embeddings are forward-only, so slice
        # with the SAME explicit mask, never a positional half-slice.
        targets["edge_cluster_id"] = ecid[input_graph.is_forward_edge]
    else:
        n_fwd = int(input_graph.is_forward_edge.sum().item())
        targets["edge_cluster_id"] = torch.full(
            (n_fwd,), -1, dtype=torch.long, device=input_graph.x.device
        )
    return targets


def apply_rotation_per_fragment(
    vectors: torch.Tensor, R_pred: torch.Tensor, entity_fragment_id: torch.Tensor
) -> torch.Tensor:
    """``R_pred[frag(i)] @ vectors[i]`` for every i. ``vectors: (M, 3)``."""
    return torch.einsum("mij,mj->mi", R_pred[entity_fragment_id], vectors)


def build_predictions(full_diffused_graph, R_pred: torch.Tensor) -> Dict:
    """Rotate the scattered FULL geometry by the predicted rotation.

    These are a deterministic function of ``R_pred`` and known input geometry,
    not separate learned outputs.
    """
    x = full_diffused_graph.x
    fragment_id = full_diffused_graph.fragment_id

    x_pred = apply_rotation_per_fragment(x[:, 0:3], R_pred, fragment_id)
    n_pred = apply_rotation_per_fragment(x[:, 3:6], R_pred, fragment_id)

    edge_attr = full_diffused_graph.edge_attr
    fwd = full_diffused_graph.is_forward_edge
    # Mesh edges never cross a fragment boundary, so either endpoint's
    # fragment id is the edge's own -- no separate per-edge fragment tensor is
    # needed. (Guarded by tests/test_collate.py.)
    edge_fragment_id = fragment_id[full_diffused_graph.edge_index[0][fwd]]

    return dict(
        x_pred=x_pred,
        n_pred=n_pred,
        mid_pred=apply_rotation_per_fragment(edge_attr[fwd, 1:4], R_pred, edge_fragment_id),
        n1_pred=apply_rotation_per_fragment(edge_attr[fwd, 4:7], R_pred, edge_fragment_id),
        n2_pred=apply_rotation_per_fragment(edge_attr[fwd, 7:10], R_pred, edge_fragment_id),
    )


def select_input_graph(batch: Dict, input_source: str, diffused: bool = True) -> Optional[object]:
    """Pick the graph the model consumes, per ``input_source``."""
    if input_source == "frac":
        key = "diff_frac_graph" if diffused else "frac_graph"
    else:
        key = "diffused_graph" if diffused else "graph"
    graph = batch.get(key)
    if graph is None:
        raise KeyError(
            f"input_source={input_source!r} needs batch[{key!r}], which is None. "
            f"Build the dataset with input_source={input_source!r} (or "
            f"build_frac_graph=True) so that graph is produced."
        )
    return graph
