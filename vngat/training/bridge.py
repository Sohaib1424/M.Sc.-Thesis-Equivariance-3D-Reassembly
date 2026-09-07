"""
The single place where "which graph is the input", "which is the target", and
"what exactly is R_gt" are decided.

Kept isolated because this index/convention bookkeeping is precisely the kind
of thing that is easy to get subtly wrong once and then train on for days
without any visible symptom.
"""
from __future__ import annotations

from typing import Dict

import torch

from ..data.graph import SceneBatch


def build_model_inputs(graph: SceneBatch) -> Dict:
    """Keyword arguments for `VNGATModel.forward`."""
    return dict(
        node_vec=graph.node_vec,
        edge_index=graph.edge_index,
        edge_len=graph.edge_len,
        edge_vec=graph.edge_vec,
        node_frag=graph.node_frag,
        num_fragments=graph.num_fragments,
        frag_scene=graph.frag_scene,
    )


def ground_truth_rotation(diffusion_rot: torch.Tensor) -> torch.Tensor:
    """
    R_gt = A^T, where A is the rotation that scattered the fragment.

    Per-fragment centralisation makes the diffusion translation cancel exactly
    (x~' = A x~), so undoing the scattering means applying A^{-1}, and A^{-1}
    is A^T because rotations are orthogonal.
    """
    return diffusion_rot.transpose(-1, -2)


def build_targets(clean: SceneBatch, diffusion_rot: torch.Tensor, input_graph: SceneBatch) -> Dict:
    """
    Ground truth for the composite loss.

    Geometry targets come from the CLEAN FULL graph regardless of what the
    network was fed, so a model trained on the pruned fracture surface is
    still scored on reconstructing the whole fragment.

    Cluster ids come from `input_graph`, because the embeddings being made
    consistent are the ones the network computed on the graph it actually saw.
    """
    return dict(
        R_gt=ground_truth_rotation(diffusion_rot),
        x_gt=clean.node_vec[:, 0],
        n_gt=clean.node_vec[:, 1],
        mid_gt=clean.edge_vec[:, 0],
        n1_gt=clean.edge_vec[:, 1],
        n2_gt=clean.edge_vec[:, 2],
        vertex_cluster_id=input_graph.vertex_cluster_id,
        edge_cluster_id=input_graph.edge_cluster_id,
    )


def apply_rotation_per_entity(
    vectors: torch.Tensor, rot: torch.Tensor, entity_frag: torch.Tensor
) -> torch.Tensor:
    """(M, 3) row vectors rotated by their fragment's (F, 3, 3) rotation."""
    return torch.einsum('mij,mj->mi', rot.index_select(0, entity_frag), vectors)


def build_predictions(diffused: SceneBatch, R_pred: torch.Tensor) -> Dict:
    """
    The "predicted" geometry the local losses compare against ground truth.

    These are NOT separate network outputs. They are the network's predicted
    rotation applied to the raw diffused input geometry, per fragment -- a
    deterministic function of R_pred and known geometry, which is what turns a
    single global geodesic term into a spatially distributed gradient signal.

    (An earlier version read "position" and "normal" off fixed channel indices
    of the learned hidden features. That is meaningless the moment any
    VNLinear mixes channels: nothing keeps hidden channel 0 semantically
    "position" once training starts moving weights.)
    """
    edge_frag = diffused.edge_frag
    return dict(
        x_pred=apply_rotation_per_entity(diffused.node_vec[:, 0], R_pred, diffused.node_frag),
        n_pred=apply_rotation_per_entity(diffused.node_vec[:, 1], R_pred, diffused.node_frag),
        mid_pred=apply_rotation_per_entity(diffused.edge_vec[:, 0], R_pred, edge_frag),
        n1_pred=apply_rotation_per_entity(diffused.edge_vec[:, 1], R_pred, edge_frag),
        n2_pred=apply_rotation_per_entity(diffused.edge_vec[:, 2], R_pred, edge_frag),
    )


def prepare_scene(batch: Dict, device: torch.device, non_blocking: bool = True) -> Dict:
    """
    Move one (micro-)batch to the device and derive the diffused views there.

    The diffused geometry is produced by ROTATING the clean feature vectors on
    the GPU rather than by transforming meshes and re-extracting features on
    the CPU. For a rigid transform the two are identical for every quantity the
    network consumes (centralised positions, vertex normals, edge lengths,
    midpoints and adjacent face normals all either rotate or are invariant,
    and topology is untouched), so this is a pure cost saving: it removes one
    full `get_features` pass -- the single most expensive CPU step -- from
    every sample, and it does the arithmetic where it is free.
    """
    target = batch["target"].to(device, non_blocking=non_blocking)
    model_input = batch["input"]
    model_input = target if model_input is None else model_input.to(device, non_blocking=non_blocking)
    rot = batch["rot"].to(device, non_blocking=non_blocking)

    diffused_target = target.rotate_per_fragment(rot)
    diffused_input = diffused_target if model_input is target else model_input.rotate_per_fragment(rot)

    return {
        "clean_target": target,
        "diffused_target": diffused_target,
        "diffused_input": diffused_input,
        "rot": rot,
        "trans": batch["trans"].to(device, non_blocking=non_blocking),
    }
