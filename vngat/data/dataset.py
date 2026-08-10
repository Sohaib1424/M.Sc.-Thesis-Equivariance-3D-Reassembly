"""
`BreakingBadDataset`: one item == one fully-built scene graph.

WHAT THIS COSTS AND WHERE IT GOES
---------------------------------
Every item does genuine geometry work -- there is no cache and nothing is
precomputed (a deliberate constraint of this project). The costs, in order:

  1. `load_random_scene`  : igl decompression.  UNCHANGED, untouched.
  2. `get_features`       : trimesh normals + unique edges + face-normal lookup.
  3. correspondence       : cross-fragment interface matching.

Three things keep this off the critical path without touching the data itself:

  * `get_features` runs ONCE per fragment, on the clean mesh only. The diffused
    view the network consumes is derived by rotating the extracted feature
    vectors (see `SceneBatch.rotate_per_fragment`), which is exact for a rigid
    transform and happens on the GPU. The original pipeline called
    `get_features` two to four times per sample -- on the clean mesh, the
    diffused mesh, and optionally both fracture variants -- each time paying
    for a full `mesh.vertex_normals` recomputation on geometry that differs
    only by a rotation.

  * correspondence is computed on the mesh variant the model actually consumes,
    and only that one.

  * edges are stored undirected (halving every per-edge tensor); the model
    symmetrises them on device.

None of this changes a single vertex, edge, or face the model sees.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.progress import write
from .correspondence import compute_scene_correspondence
from .features import get_features
from .graph import SceneBatch, collate_scenes, merge_fragments
from .io import load_random_scene, random_rotation_matrices
from .mesh_ops import extract_fractures
from .splits import get_random_directory, scene_pool


class BreakingBadDataset(Dataset):
    """
    `__getitem__` ignores its index and draws a fresh random scene from the
    split's pool (which is why the split has to be hash-based rather than
    index-range-based -- see `vngat/data/splits.py`). `__len__` is therefore a
    *nominal* epoch length, set by `steps_per_epoch * batch_size` from the
    training config.
    """

    def __init__(
        self,
        root_dir: str = "data",
        split: Optional[str] = "train",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int = 0,
        max_scenes: int = 0,
        subsets: Optional[List[str]] = None,
        fracture_pattern: Optional[str] = None,
        input_source: str = "full",
        with_correspondence: bool = True,
        correspondence_tol: float = 1e-5,
        nominal_length: int = 10_000,
        min_fragments: int = 2,
        max_retries: int = 8,
        keep_meshes: bool = False,
    ) -> None:
        if input_source not in ("full", "frac"):
            raise ValueError(f"input_source must be 'full' or 'frac', got {input_source!r}")
        self.root_dir = root_dir
        self.split = split
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed
        self.max_scenes = max_scenes
        self.subsets = subsets
        self.fracture_pattern = fracture_pattern
        self.input_source = input_source
        self.with_correspondence = with_correspondence
        self.correspondence_tol = correspondence_tol
        self.nominal_length = nominal_length
        self.min_fragments = min_fragments
        self.max_retries = max_retries
        self.keep_meshes = keep_meshes

    def __len__(self) -> int:
        return self.nominal_length

    @property
    def pool_size(self) -> int:
        return len(scene_pool(
            self.root_dir, self.split, self.val_frac, self.test_frac,
            self.split_seed, self.max_scenes, self.subsets,
        ))

    # ------------------------------------------------------------------
    def __getitem__(self, _idx: int) -> Dict:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            scene_dir = get_random_directory(
                self.root_dir, split=self.split, val_frac=self.val_frac,
                test_frac=self.test_frac, split_seed=self.split_seed,
                max_scenes=self.max_scenes, subsets=self.subsets,
            )
            try:
                sample = self._build_sample(scene_dir)
            except Exception as exc:  # noqa: BLE001 - a bad scene must not kill a 12h run
                last_error = exc
                if attempt == 0:
                    write(f"  [dataset] skipping {scene_dir.name}: {type(exc).__name__}: {exc}")
                continue
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Could not build a usable scene after {self.max_retries} attempts "
            f"(split={self.split!r}). Last error: {last_error!r}"
        )

    # ------------------------------------------------------------------
    def _build_sample(self, scene_dir) -> Optional[Dict]:
        t0 = time.perf_counter()
        meshes = load_random_scene(str(scene_dir), fracture_pattern=self.fracture_pattern)
        if len(meshes) < self.min_fragments:
            return None

        input_meshes = meshes
        frac_fallbacks = 0
        if self.input_source == "frac":
            input_meshes = []
            for m in meshes:
                fm = extract_fractures(m)
                # Degenerate fallback: a fragment whose fracture surface prunes
                # to nothing still has to appear, or the fragment count of the
                # input graph would silently diverge from the target graph and
                # the per-fragment rotations would address the wrong fragments.
                if len(fm.faces) == 0 or len(fm.vertices) < 3:
                    fm = m
                    frac_fallbacks += 1
                input_meshes.append(fm)

        v_clusters: List[Optional[np.ndarray]] = [None] * len(input_meshes)
        e_clusters: List[Optional[np.ndarray]] = [None] * len(input_meshes)
        if self.with_correspondence:
            v_clusters, e_clusters = compute_scene_correspondence(
                input_meshes, tol=self.correspondence_tol
            )

        # Target graph: ALWAYS the full fragment mesh. Even when the network is
        # fed the pruned fracture surface, the geometric losses and every
        # reported metric are evaluated on the whole fragment, so that a model
        # trained on the "skeleton" is still judged on how well it
        # reconstructs the complete object.
        if self.input_source == "full":
            # Model input and target are the same meshes, so correspondence is
            # attached directly and `get_features` runs exactly once per
            # fragment for the whole sample.
            target_frags = [
                get_features(m, vertex_cluster_ids=vc, edge_cluster_ids=ec)
                for m, vc, ec in zip(meshes, v_clusters, e_clusters)
            ]
            target_graph = merge_fragments(target_frags)
            input_graph = None                      # None means "same as target"
        else:
            target_frags = [get_features(m) for m in meshes]
            target_graph = merge_fragments(target_frags)
            input_frags = [
                get_features(m, vertex_cluster_ids=vc, edge_cluster_ids=ec)
                for m, vc, ec in zip(input_meshes, v_clusters, e_clusters)
            ]
            input_graph = merge_fragments(input_frags)
            if input_graph.num_fragments != target_graph.num_fragments:
                raise RuntimeError(
                    "input/target fragment count mismatch "
                    f"({input_graph.num_fragments} vs {target_graph.num_fragments})"
                )

        num_frags = target_graph.num_fragments
        rot = torch.from_numpy(random_rotation_matrices(num_frags).astype(np.float32))

        # Diffusion translation. The rotation network never sees it -- per
        # fragment centralisation cancels it exactly -- but it is recorded so
        # the translation solver and the assembly metrics have the ground
        # truth they need without a second pass over the data.
        # `amax` raises on a zero-length reduction, so a fragment that somehow
        # ends up with no vertices must not reach it.
        spans = [
            (g.node_vec[:, 0].amax(0) - g.node_vec[:, 0].amin(0))
            for g in target_frags if g.num_nodes > 0
        ]
        max_dim = float(torch.stack(spans).max()) if spans else 1.0
        trans = torch.from_numpy(
            (np.random.normal(0.0, 0.75, size=(num_frags, 3))
             + np.random.standard_normal((num_frags, 3)) * max_dim * 0.5).astype(np.float32)
        )

        return {
            "target": target_graph,
            "input": input_graph,
            "rot": rot,
            "trans": trans,
            "scene_dir": str(scene_dir),
            "load_seconds": time.perf_counter() - t0,
            "frac_fallbacks": frac_fallbacks,
            "meshes": meshes if self.keep_meshes else None,
        }


def collate_fn(samples: List[Dict]) -> Dict:
    """Batch several scenes. Every per-scene index space is offset here."""
    samples = [s for s in samples if s is not None]
    if not samples:
        raise RuntimeError("collate_fn received no usable samples")

    target = collate_scenes([s["target"] for s in samples])
    has_separate_input = any(s["input"] is not None for s in samples)
    if has_separate_input:
        inputs = [s["input"] if s["input"] is not None else s["target"] for s in samples]
        model_input = collate_scenes(inputs)
    else:
        model_input = None

    return {
        "target": target,
        "input": model_input,
        "rot": torch.cat([s["rot"] for s in samples], 0),
        "trans": torch.cat([s["trans"] for s in samples], 0),
        "scene_dirs": [s["scene_dir"] for s in samples],
        "load_seconds": float(sum(s["load_seconds"] for s in samples)),
        "frac_fallbacks": int(sum(s["frac_fallbacks"] for s in samples)),
        "num_scenes": len(samples),
        "meshes": [s["meshes"] for s in samples] if samples[0]["meshes"] is not None else None,
    }


def split_batch_by_scene(batch: Dict) -> List[Dict]:
    """
    Re-split a collated batch into single-scene batches.

    Used by the training loop's micro-batching: peak activation memory then
    depends only on the LARGEST SINGLE SCENE rather than on `batch_size`,
    while gradient accumulation keeps the effective batch size unchanged.
    That is what makes a 16 GB T4 able to train at any batch size.
    """
    target: SceneBatch = batch["target"]
    model_input: Optional[SceneBatch] = batch["input"]
    out: List[Dict] = []

    node_start = edge_start = frag_start = 0
    in_node_start = in_edge_start = 0
    for scene_idx in range(target.num_scenes):
        frag_mask = target.frag_scene == scene_idx
        num_frags = int(frag_mask.sum())
        node_mask = (target.node_frag >= frag_start) & (target.node_frag < frag_start + num_frags)
        num_nodes = int(node_mask.sum())
        edge_mask = node_mask[target.edge_index[0]]
        num_edges = int(edge_mask.sum())

        scene_target = _slice_scene(target, node_start, num_nodes, edge_start, num_edges,
                                    frag_start, num_frags)
        if model_input is None:
            scene_input = None
        else:
            in_frag_mask = model_input.frag_scene == scene_idx
            in_num_frags = int(in_frag_mask.sum())
            in_node_mask = ((model_input.node_frag >= frag_start)
                            & (model_input.node_frag < frag_start + in_num_frags))
            in_num_nodes = int(in_node_mask.sum())
            in_num_edges = int(in_node_mask[model_input.edge_index[0]].sum())
            scene_input = _slice_scene(model_input, in_node_start, in_num_nodes,
                                       in_edge_start, in_num_edges, frag_start, in_num_frags)
            in_node_start += in_num_nodes
            in_edge_start += in_num_edges

        out.append({
            "target": scene_target,
            "input": scene_input,
            "rot": batch["rot"][frag_start:frag_start + num_frags],
            "trans": batch["trans"][frag_start:frag_start + num_frags],
            "scene_dirs": [batch["scene_dirs"][scene_idx]],
            "num_scenes": 1,
        })
        node_start += num_nodes
        edge_start += num_edges
        frag_start += num_frags
    return out


def _slice_scene(g: SceneBatch, n0: int, nn: int, e0: int, ne: int, f0: int, nf: int) -> SceneBatch:
    """Contiguous slice of one scene out of a collated batch.

    Relies on collation being a plain concatenation in scene order, which
    makes every scene's nodes/edges/fragments a contiguous run.
    """
    return SceneBatch(
        node_vec=g.node_vec[n0:n0 + nn],
        edge_index=g.edge_index[:, e0:e0 + ne] - n0,
        edge_len=g.edge_len[e0:e0 + ne],
        edge_vec=g.edge_vec[e0:e0 + ne],
        node_frag=g.node_frag[n0:n0 + nn] - f0,
        frag_scene=torch.zeros(nf, dtype=torch.long),
        frag_centroid=g.frag_centroid[f0:f0 + nf],
        vertex_cluster_id=g.vertex_cluster_id[n0:n0 + nn],
        edge_cluster_id=g.edge_cluster_id[e0:e0 + ne],
        num_fragments=nf,
        num_scenes=1,
    )
