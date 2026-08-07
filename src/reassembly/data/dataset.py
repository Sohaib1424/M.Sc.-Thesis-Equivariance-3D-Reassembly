"""
The torch Dataset.

Design note carried over from the original code, and worth restating because
it drives several decisions elsewhere: ``__getitem__`` IGNORES its index and
draws a fresh random scene every call. That is a legitimate choice for this
dataset (a "sample" is a (shape, fracture, random SE(3)) triple, so the space
is effectively continuous), but it has consequences:

* index-range splitting does nothing -- hence the hash-based split in
  ``splits.py``;
* ``__len__`` is a nominal number, and "one epoch" means "``steps_per_epoch``
  draws", not "one pass over the data";
* a ``DistributedSampler`` has no index partitioning to do, so each DDP rank
  just draws independently.

WHAT IS NEW HERE
----------------
* ``max_vertices`` / ``decimate_to`` -- a hard vertex budget per scene, which
  is what turns peak GPU memory from a property of the dataset into a number
  you choose. See ``decimate.py``.
* ``input_source`` -- feed the model the full mesh or the pruned
  fracture-surface mesh, while *always* keeping the full mesh for supervision,
  so a model trained on the interface skeleton can still be scored on how well
  it reassembles the whole fragment.
* per-worker RNG seeding, so DataLoader workers do not all replay the same
  "random" augmentations.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .augment import diffuse_fragments
from .collate import merge_fragments
from .correspondence import (
    compute_scene_correspondence,
    derive_edge_clusters_for_scene,
    transfer_vertex_clusters,
)
from .cache import BaseMeshCache, ScenePreprocessCache
from .decimate import decimate_scene, suggested_correspondence_tol
from .features import get_features
from .mesh_ops import extract_fractures_with_map
from .scene_io import load_scene
from .splits import SceneIndex

INPUT_SOURCES = ("full", "frac")


class BreakingBadDataset(Dataset):
    """One item = one scene: clean graph, diffused graph, and the transforms.

    Parameters
    ----------
    root_dir
        Dataset root (contains ``everyday_compressed/`` / ``artifact_compressed/``).
    split
        ``'train'`` / ``'val'`` / ``'test'``, or ``None`` for "all scenes".
    max_vertices
        Reject any scene whose total vertex count exceeds this, redrawing up to
        ``max_retries`` times. ``None`` disables the check. Use together with
        ``decimate_to`` as a backstop for scenes decimation cannot shrink
        enough.
    decimate_to
        Target total vertices per scene, via shared-grid voxel clustering.
        ``None`` disables decimation.
    input_source
        ``'full'`` (default) or ``'frac'``. With ``'frac'`` the model consumes
        the pruned fracture-surface mesh while losses are still evaluated
        against the full mesh.
    nominal_length
        What ``__len__`` reports. Only affects how samplers size themselves.
    """

    def __init__(
        self,
        root_dir: str = "data",
        split: Optional[str] = None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int = 0,
        return_meshes: bool = False,
        diffuse: bool = True,
        input_source: str = "full",
        build_frac_graph: bool = False,
        correspondence_tol: float = 1e-5,
        max_vertices: Optional[int] = None,
        decimate_to: Optional[int] = None,
        min_vertices_per_fragment: int = 32,
        fracture_pattern: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_max_gib: float = 8.0,
        max_retries: int = 8,
        nominal_length: int = 10_000,
        seed: int = 0,
    ):
        if input_source not in INPUT_SOURCES:
            raise ValueError(f"input_source must be one of {INPUT_SOURCES}, got {input_source!r}")

        self.root_dir = root_dir
        self.return_meshes = return_meshes
        self.diffuse = diffuse
        self.input_source = input_source
        # The fracture graph is always built when it is the model input.
        self.build_frac_graph = build_frac_graph or input_source == "frac"
        self.correspondence_tol = correspondence_tol
        self.max_vertices = max_vertices
        self.decimate_to = decimate_to
        self.min_vertices_per_fragment = min_vertices_per_fragment
        self.fracture_pattern = fracture_pattern
        self.cache = ScenePreprocessCache(
            cache_dir, max_bytes=int(cache_max_gib * 1024 ** 3))
        # Keyed by scene alone, so it hits from the first repeat -- unlike the
        # (scene, fracture) cache, which with ~100 fractures per scene rarely
        # sees the same pair twice early in training.
        self.base_cache = BaseMeshCache(cache_dir)
        self.max_retries = max_retries
        self.nominal_length = nominal_length
        self.seed = seed

        self.scene_index = SceneIndex(
            root=root_dir, split=split, val_frac=val_frac,
            test_frac=test_frac, seed=split_seed,
        )

        self._rng: Optional[np.random.Generator] = None
        self._py_rng: Optional[random.Random] = None

        # Diagnostics, read by the training loop for its per-epoch summary.
        self.stats = {"scenes_loaded": 0, "scenes_rejected": 0, "decimations": 0,
                      "scenes_over_budget": 0}

    # -- RNG -----------------------------------------------------------------
    def _rngs(self):
        """Lazily build per-worker generators.

        Built lazily (not in ``__init__``) because worker identity only exists
        inside the worker process. On a fork start method every worker would
        otherwise inherit the parent's generator state and draw identical
        "random" scenes and rotations.
        """
        if self._rng is None:
            info = get_worker_info()
            worker_id = info.id if info is not None else 0
            base = (self.seed * 100_003) + worker_id * 7919 + torch.initial_seed() % 2**31
            self._rng = np.random.default_rng(base)
            self._py_rng = random.Random(base)
        return self._rng, self._py_rng

    def __len__(self) -> int:
        return self.nominal_length

    # -- main path -----------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[Dict]:
        rng, py_rng = self._rngs()

        for _attempt in range(max(1, self.max_retries)):
            scene_dir = self.scene_index.sample(py_rng)

            # Pick the fracture here rather than inside load_scene, so it can
            # go into the cache key. Everything from here to correspondence
            # detection is a pure function of (scene, fracture, settings).
            fracture_id = self._choose_fracture(scene_dir, py_rng)
            if fracture_id is None:
                self.stats["scenes_rejected"] += 1
                continue

            cache_key = self.cache.key(
                str(scene_dir), fracture_id,
                decimate_to=self.decimate_to,
                min_vertices_per_fragment=self.min_vertices_per_fragment,
                correspondence_tol=self.correspondence_tol,
                input_source=self.input_source,
            )
            cached = self.cache.load(cache_key)
            if cached is not None:
                self.stats["scenes_loaded"] += 1
                return self._build_sample_from_arrays(cached, scene_dir, rng)

            base = self.base_cache.load(str(scene_dir))
            try:
                meshes = load_scene(str(scene_dir), fracture_id=fracture_id,
                                    rng=py_rng,
                                    fracture_pattern=self.fracture_pattern,
                                    base_mesh=base)
            except Exception:
                self.stats["scenes_rejected"] += 1
                continue

            if base is None:
                # Populate on the way past. Reading it back costs 2.3 ms
                # against 105 ms to reparse.
                try:
                    import igl
                    from scipy.sparse import load_npz as _load_npz
                    v, f = igl.read_triangle_mesh(
                        os.path.join(str(scene_dir), "compressed_mesh.obj"))
                    self.base_cache.store(
                        str(scene_dir), v, f,
                        _load_npz(os.path.join(str(scene_dir), "compressed_data.npz")))
                except Exception:                            # noqa: BLE001
                    pass                                     # caching is best-effort

            if len(meshes) < 2:
                # A "scene" with one fragment has no cross-fragment structure
                # to learn from and no correspondence to supervise on.
                self.stats["scenes_rejected"] += 1
                continue

            tol = self.correspondence_tol
            if self.decimate_to is not None:
                meshes, info = decimate_scene(
                    meshes,
                    target_vertices=self.decimate_to,
                    min_vertices_per_fragment=self.min_vertices_per_fragment,
                )
                if info.reduced:
                    self.stats["decimations"] += 1
                    # Gated on `reduced`, not `applied`. A search that ran to
                    # completion without decimating anything still reports a
                    # voxel size, and on a many-fragment scene that size can
                    # exceed the object itself -- widening the correspondence
                    # tolerance until every vertex matches every other vertex.
                    # Observed on a real 53-fragment scene: voxel 0.527 on an
                    # object 1.0 across, with zero vertices removed.
                    tol = suggested_correspondence_tol(info.voxel_size, base_tol=tol)
                elif info.applied and not info.target_met:
                    # The budget is unreachable for this scene: too many small
                    # fragments, all of them pinned by min_vertices_per_fragment.
                    # Left alone it would sail past the vertex budget and OOM
                    # exactly the way the original code did.
                    self.stats["scenes_over_budget"] += 1

            total_vertices = sum(len(m.vertices) for m in meshes)
            if self.max_vertices is not None and total_vertices > self.max_vertices:
                # Over budget even after decimation: redraw rather than risk an
                # OOM mid-epoch. Counted so the rejection rate is visible.
                self.stats["scenes_rejected"] += 1
                continue

            self.stats["scenes_loaded"] += 1
            return self._build_sample(meshes, scene_dir, tol, rng,
                                      cache_key=cache_key)

        return None  # collate_fn drops Nones

    def _choose_fracture(self, scene_dir, py_rng) -> Optional[str]:
        """Pick a fracture subdirectory, so the choice can enter the cache key."""
        import fnmatch
        import os

        try:
            names = sorted(
                d for d in os.listdir(str(scene_dir))
                if os.path.isdir(os.path.join(str(scene_dir), d))
                and (self.fracture_pattern is None
                     or fnmatch.fnmatch(d, self.fracture_pattern))
            )
        except OSError:
            return None
        return py_rng.choice(names) if names else None

    def _build_sample_from_arrays(self, cached: Dict, scene_dir, rng) -> Dict:
        """Rebuild a sample from cached geometry.

        The scattering transform is applied here, NOT read from the cache --
        it must stay random per epoch or every visit to a scene would present
        the model with the same rotation.
        """
        import trimesh

        meshes = [
            trimesh.Trimesh(np.asarray(v, dtype=np.float64),
                            np.asarray(f, dtype=np.int64), process=False)
            for v, f in zip(cached["vertices"], cached["faces"])
        ]
        return self._assemble(meshes, cached["vertex_cluster_ids"],
                              cached["edge_cluster_ids"], scene_dir, rng)

    def _build_sample(self, meshes, scene_dir, tol, rng, cache_key=None) -> Dict:
        vertex_ids, edge_ids = compute_scene_correspondence(meshes, tol=tol)

        if cache_key is not None:
            self.cache.store(
                cache_key,
                [np.asarray(m.vertices) for m in meshes],
                [np.asarray(m.faces) for m in meshes],
                vertex_ids, edge_ids, tol,
            )

        return self._assemble(meshes, vertex_ids, edge_ids, scene_dir, rng)

    def _assemble(self, meshes, vertex_ids, edge_ids, scene_dir, rng) -> Dict:

        clean_graph = merge_fragments([
            get_features(m, vcid, ecid) for m, vcid, ecid in zip(meshes, vertex_ids, edge_ids)
        ])

        frac_graph = None
        frac_meshes: List = []
        frac_vertex_ids: List[np.ndarray] = []
        if self.build_frac_graph:
            frac_meshes, frac_maps = [], []
            for m in meshes:
                fm, vmap = extract_fractures_with_map(m)
                frac_meshes.append(fm)
                frac_maps.append(vmap)

            # Transfer vertex cluster ids through the pruning map, then
            # re-derive edge cluster ids from them. This is what makes the
            # interface losses meaningful on the fracture mesh: its
            # `edges_unique` is rebuilt from a face subset and has a totally
            # different ordering, so the full mesh's edge ids cannot be reused.
            frac_vertex_ids = [
                transfer_vertex_clusters(vcid, vmap, len(fm.vertices))
                for vcid, vmap, fm in zip(vertex_ids, frac_maps, frac_meshes)
            ]
            frac_edge_ids = derive_edge_clusters_for_scene(frac_meshes, frac_vertex_ids)
            frac_graph = merge_fragments([
                get_features(m, vcid, ecid)
                for m, vcid, ecid in zip(frac_meshes, frac_vertex_ids, frac_edge_ids)
            ])

        diffused_graph = None
        diff_frac_graph = None
        t_tensor = None
        diffused_meshes = None

        if self.diffuse:
            diffused_meshes, t_matrices = diffuse_fragments(meshes, rng=rng)
            # A rigid transform changes where things sit, never which things
            # coincide, so the cluster ids computed above carry over unchanged.
            diffused_graph = merge_fragments([
                get_features(m, vcid, ecid)
                for m, vcid, ecid in zip(diffused_meshes, vertex_ids, edge_ids)
            ])
            t_tensor = torch.stack([torch.from_numpy(m).float() for m in t_matrices])

            if self.build_frac_graph:
                diff_frac_meshes = [
                    fm.copy().apply_transform(t) for fm, t in zip(frac_meshes, t_matrices)
                ]
                diff_frac_edge_ids = derive_edge_clusters_for_scene(
                    diff_frac_meshes, frac_vertex_ids
                )
                diff_frac_graph = merge_fragments([
                    get_features(m, vcid, ecid)
                    for m, vcid, ecid in zip(diff_frac_meshes, frac_vertex_ids, diff_frac_edge_ids)
                ])

        # Consistency guard. Every per-fragment tensor downstream (R_pred,
        # t_matrices, fragment_scene_id, and the rotation applied to the FULL
        # mesh for the losses) is indexed by fragment position, so a mismatch
        # between the graph the model consumes and the graph the losses use
        # would misalign all of them -- silently, with no shape error, because
        # both are just long concatenated tensors.
        for name, g in (("frac_graph", frac_graph), ("diffused_graph", diffused_graph),
                        ("diff_frac_graph", diff_frac_graph)):
            if g is not None and g.num_fragments != clean_graph.num_fragments:
                raise RuntimeError(
                    f"fragment-count mismatch in scene {scene_dir}: clean graph has "
                    f"{clean_graph.num_fragments} fragments but {name} has "
                    f"{g.num_fragments}. Every per-fragment tensor is indexed by "
                    f"position, so this would misalign rotations and transforms."
                )

        return {
            "graph": clean_graph,
            "diffused_graph": diffused_graph,
            "frac_graph": frac_graph,
            "diff_frac_graph": diff_frac_graph,
            "t_matrices": t_tensor,
            "meshes": meshes if self.return_meshes else None,
            "diffused_meshes": diffused_meshes if self.return_meshes else None,
            "scene_dir": str(scene_dir),
        }
