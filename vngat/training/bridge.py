"""
The single place where "which graph is the input", "which is the target", and
"what exactly is R_gt" are decided, plus the numpy -> tf boundary.

Kept isolated because this index/convention bookkeeping is precisely the kind of
thing that is easy to get subtly wrong once and then train on for days with no
visible symptom.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from ..data.graph import SceneBatch
from ..data.padding import pad_scene_batch


def to_tensors(graph: SceneBatch, dtype=tf.float32) -> Dict:
    """numpy SceneBatch -> the keyword arguments `VNGATModel.call` expects."""
    return dict(
        node_vec=tf.constant(graph.node_vec, dtype),
        edge_index=tf.constant(graph.edge_index, tf.int32),
        edge_len=tf.constant(graph.edge_len, dtype),
        edge_vec=tf.constant(graph.edge_vec, dtype),
        node_frag=tf.constant(graph.node_frag, tf.int32),
        num_fragments=int(graph.num_fragments),
        frag_scene=tf.constant(graph.frag_scene, tf.int32),
    )


def ground_truth_rotation(diffusion_rot):
    """
    R_gt = A^T, where A is the rotation that scattered the fragment.

    Per-fragment centralisation makes the diffusion translation cancel exactly
    (x~' = A x~), so undoing the scattering means applying A^{-1} = A^T.
    """
    return tf.linalg.matrix_transpose(diffusion_rot)


def apply_rotation_per_entity(vectors, rot, entity_frag):
    """(M, 3) row vectors rotated by their fragment's (F, 3, 3) rotation."""
    return tf.einsum('mij,mj->mi', tf.gather(rot, entity_frag), vectors)


def build_predictions(diffused: SceneBatch, R_pred, dtype=tf.float32) -> Dict:
    """
    The "predicted" geometry the local losses compare against ground truth.

    NOT separate network outputs: the predicted rotation applied to the raw
    diffused geometry, per fragment. That is what turns a single global geodesic
    term into a spatially distributed gradient signal.

    (Reading "position" and "normal" off fixed channel indices of the learned
    features would be meaningless the moment any VNLinear mixes channels --
    nothing keeps hidden channel 0 semantically "position" once training
    starts.)
    """
    node_frag = tf.constant(diffused.node_frag, tf.int32)
    edge_frag = tf.constant(diffused.edge_frag, tf.int32)
    nv = tf.constant(diffused.node_vec, dtype)
    ev = tf.constant(diffused.edge_vec, dtype)
    return dict(
        x_pred=apply_rotation_per_entity(nv[:, 0], R_pred, node_frag),
        n_pred=apply_rotation_per_entity(nv[:, 1], R_pred, node_frag),
        mid_pred=apply_rotation_per_entity(ev[:, 0], R_pred, edge_frag),
        n1_pred=apply_rotation_per_entity(ev[:, 1], R_pred, edge_frag),
        n2_pred=apply_rotation_per_entity(ev[:, 2], R_pred, edge_frag),
    )


def build_targets(clean: SceneBatch, diffusion_rot, input_graph: SceneBatch,
                  dtype=tf.float32) -> Dict:
    """
    Ground truth for the composite loss.

    Geometry targets come from the CLEAN FULL graph regardless of what the
    network was fed, so a model trained on the pruned fracture surface is still
    scored on reconstructing the whole fragment. Cluster ids come from
    `input_graph`, because the embeddings being made consistent are the ones the
    network computed on the graph it actually saw.
    """
    nv = tf.constant(clean.node_vec, dtype)
    ev = tf.constant(clean.edge_vec, dtype)
    return dict(
        R_gt=ground_truth_rotation(diffusion_rot),
        x_gt=nv[:, 0], n_gt=nv[:, 1],
        mid_gt=ev[:, 0], n1_gt=ev[:, 1], n2_gt=ev[:, 2],
        vertex_cluster_id=tf.constant(input_graph.vertex_cluster_id, tf.int32),
        edge_cluster_id=tf.constant(input_graph.edge_cluster_id, tf.int32),
    )


def prepare_scene(batch: Dict, dtype=tf.float32, pad: bool = False) -> Dict:
    """
    Derive the diffused views and convert to tensors.

    The diffused geometry is produced by ROTATING the clean feature vectors
    rather than transforming meshes and re-extracting features. For a rigid
    transform the two are identical for every quantity the network consumes, so
    this is a pure cost saving -- it removes one full `get_features` pass, the
    single most expensive CPU step, from every sample.

    `pad=True` pads to bucket shapes and returns the masks, which XLA needs so
    it compiles a handful of programs instead of one per scene.
    """
    target = batch["target"]
    model_input = batch["input"] if batch.get("input") is not None else target
    # Match the requested precision instead of hardcoding float32. An
    # unconditional cast silently downcasts a float64 verification run and
    # drops its residual from ~1e-16 to ~1e-8, which is the difference between
    # a meaningful equivariance check and a meaningless one.
    rot_np = np.asarray(batch["rot"], dtype.as_numpy_dtype)

    masks = None
    if pad:
        padded = pad_scene_batch(target)
        padded_in = padded if model_input is target else pad_scene_batch(model_input)
        f_pad = padded.graph.num_fragments
        if rot_np.shape[0] < f_pad:
            pad_rot = np.tile(np.eye(3, dtype=rot_np.dtype), (f_pad - rot_np.shape[0], 1, 1))
            rot_np = np.concatenate([rot_np, pad_rot], 0)
        masks = dict(node_mask=tf.constant(padded.node_mask),
                     edge_mask=tf.constant(padded.edge_mask),
                     frag_mask=tf.constant(padded.frag_mask))
        target, model_input = padded.graph, padded_in.graph

    diffused_target = target.rotate_per_fragment(rot_np)
    diffused_input = (diffused_target if model_input is target
                      else model_input.rotate_per_fragment(rot_np))
    out = {
        "clean_target": target,
        "diffused_target": diffused_target,
        "diffused_input": diffused_input,
        "rot": tf.constant(rot_np, dtype),
    }
    if masks:
        out.update(masks)
    return out
