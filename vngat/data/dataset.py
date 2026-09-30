"""
`BreakingBadDataset`: one item == one fully built scene graph.

WHAT THIS COSTS AND WHERE IT GOES
---------------------------------
Every item does genuine geometry work -- there is no cache and nothing is
precomputed (a deliberate constraint of this project). The costs, in order:

  1. `load_scene`         : igl decompression. The decompression code itself
                            is untouched (see `vngat/data/io.py`).
  2. `get_features`       : trimesh normals, unique edges, canonical face-normal
                            order, relative positions.
  3. correspondence       : cross-fragment interface matching.

Three things keep this off the critical path without touching the data:

  * `get_features` runs ONCE per fragment, on the clean mesh only. The diffused
    view the network consumes is derived by rotating the extracted vectors on
    the GPU (`SceneBatch.rotate_per_fragment`), which is exact for a rigid
    transform.
  * correspondence is computed on the mesh variant the model consumes, and only
    that one.
  * edges are stored undirected (halving every per-edge tensor); the model
    builds the reverse copies on device.

WHAT IS DRAWN
-------------
The catalogue (`vngat.data.catalog`) decides which objects and break patterns
belong to this split. Training items ignore their index and draw a fresh
object -- uniformly, or category-balanced -- then a uniformly random break
pattern of it, and fresh random rotations. `__len__` is then a NOMINAL epoch
length, `steps_per_epoch * batch_size`.

A FIXED item list (`fixed=True`, the default for validation) instead maps
index i to the same (object, break pattern, rotations) every epoch, so
validation numbers from different epochs measure the same thing.

With `max_fragments` set, the catalogue holds only the break patterns of
`min_fragments` to `max_fragments` pieces (`vngat.data.catalog.limit_fragments`),
so both kinds of item are drawn from those alone.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.progress import write
from .catalog import (
    bounded_subset, build_catalog, fixed_items, limit_fragments, object_weights, split_objects,
)
from .correspondence import compute_scene_correspondence
from .features import get_features
from .graph import SceneBatch, collate_scenes, merge_fragments, split_scenes
from .io import load_scene, random_rotation_matrices
from .mesh_ops import extract_fractures


class BreakingBadDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data",
        split: Optional[str] = "train",
        *,
        subsets: Optional[List[str]] = None,
        fracture_pattern: Optional[str] = None,
        split_source: str = "hash",
        split_by: str = "object",
        fracture_pool: str = "train",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int = 0,
        max_objects: int = 0,
        balance: str = "none",
        balance_temperature: float = 1.0,
        fixed: bool = False,
        fixed_count: int = 0,
        input_source: str = "full",
        normalize_mode: str = "scene",
        with_correspondence: bool = True,
        correspondence_tol: float = 1e-5,
        nominal_length: int = 10_000,
        min_fragments: int = 2,
        max_fragments: int = 0,
        max_retries: int = 8,
        keep_meshes: bool = False,
    ) -> None:
        if input_source not in ("full", "frac"):
            raise ValueError(f"input_source must be 'full' or 'frac', got {input_source!r}")
        self.root_dir = root_dir
        self.split = split
        self.split_seed = split_seed
        self.input_source = input_source
        self.normalize_mode = normalize_mode
        self.with_correspondence = with_correspondence
        self.correspondence_tol = correspondence_tol
        self.nominal_length = nominal_length
        self.min_fragments = min_fragments
        self.max_fragments = max_fragments
        self.max_retries = max_retries
        self.keep_meshes = keep_meshes
        self.fixed = fixed

        catalog = build_catalog(root_dir, subsets, fracture_pattern)
        objects = split_objects(
            catalog, split, root=root_dir, split_by=split_by, split_source=split_source,
            fracture_pool=fracture_pool, val_frac=val_frac, test_frac=test_frac,
            seed=split_seed,
        )
        # The fragment limit comes AFTER the split, so it only ever removes
        # break patterns -- none moves between train and validation because of
        # it -- and BEFORE the subset, so `max_objects` counts objects that
        # still have a pattern to draw.
        objects, self.fragment_limit = limit_fragments(objects, min_fragments, max_fragments)
        if not objects:
            raise ValueError(
                f"split={split!r} has no break pattern of {min_fragments} to {max_fragments} "
                f"pieces. Raise --max_fragments, or set it to 0 for no limit."
            )
        self.objects = bounded_subset(objects, max_objects)
        self.weights = object_weights(self.objects, balance, balance_temperature)
        self._probs = (None if self.weights is None
                       else np.asarray(self.weights, dtype=np.float64) / float(sum(self.weights)))
        self.items = fixed_items(self.objects, fixed_count, split_seed) if fixed else []

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items) if self.fixed else self.nominal_length

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def num_patterns(self) -> int:
        return sum(len(e) for e in self.objects)

    @property
    def categories(self) -> List[str]:
        """Every category in this split, sorted -- identical on every rank."""
        return sorted({e.category for e in self.objects})

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        if self.fixed:
            return self._fixed_item(idx)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            index = (int(np.random.choice(len(self.objects), p=self._probs))
                     if self._probs is not None else int(np.random.randint(len(self.objects))))
            entry = self.objects[index]
            scene_dir, mode = entry.modes[int(np.random.randint(len(entry.modes)))]
            try:
                sample = self._build_sample(scene_dir, mode, entry.category, rng=None)
            except Exception as exc:  # noqa: BLE001 - a bad scene must not kill a 12h run
                last_error = exc
                if attempt == 0:
                    write(f"  [dataset] skipping {scene_dir}/{mode}: {type(exc).__name__}: {exc}")
                continue
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Could not build a usable scene after {self.max_retries} attempts "
            f"(split={self.split!r}). Last error: {last_error!r}"
        )

    def _fixed_item(self, idx: int) -> Dict:
        """
        The same scene and the same rotations for index `idx`, every time.

        Should that pattern be unusable (it fails to load, or its piece count
        is outside `min_fragments`..`max_fragments`), the object's other
        patterns are tried in a fixed order -- so the substitute is
        deterministic too.
        """
        obj_index, scene_dir, mode = self.items[idx]
        entry = self.objects[obj_index]
        candidates = [(scene_dir, mode)] + [m for m in entry.modes if m != (scene_dir, mode)]
        last_error: Optional[Exception] = None
        for scene, pattern in candidates[:self.max_retries]:
            rng = np.random.default_rng([int(self.split_seed), 7919, int(idx)])
            try:
                sample = self._build_sample(scene, pattern, entry.category, rng=rng)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if sample is not None:
                return sample
        raise RuntimeError(f"validation item {idx} ({entry.key}) is unusable. "
                           f"Last error: {last_error!r}")

    # ------------------------------------------------------------------
    def _build_sample(self, scene_dir: str, mode: str, category: str,
                      rng: Optional[np.random.Generator]) -> Optional[Dict]:
        t0 = time.perf_counter()
        meshes = load_scene(str(scene_dir), mode)
        # The catalogue has already dropped patterns above `max_fragments`;
        # this checks the loaded count, which can only be lower than the
        # labelled one, so it holds even for a pattern the count could not read.
        if len(meshes) < self.min_fragments or 0 < self.max_fragments < len(meshes):
            return None
        target_graph, input_graph, info = build_graphs(
            meshes, normalize_mode=self.normalize_mode, input_source=self.input_source,
            with_correspondence=self.with_correspondence,
            correspondence_tol=self.correspondence_tol,
        )
        rot = torch.from_numpy(
            random_rotation_matrices(target_graph.num_fragments, rng).astype(np.float32))
        if info["num_repaired"]:
            write(f"  [data] repaired {info['num_repaired']} non-finite feature value(s) in "
                  f"{scene_dir}/{mode} (degenerate triangles in the source mesh)")
        return {
            "target": target_graph,
            "input": input_graph,
            "rot": rot,
            "scene_dir": str(scene_dir),
            "mode": mode,
            "category": category,
            "load_seconds": time.perf_counter() - t0,
            "frac_fallbacks": info["frac_fallbacks"],
            "num_repaired": info["num_repaired"],
            "meshes": meshes if self.keep_meshes else None,
        }


def build_graphs(meshes: Sequence, *, normalize_mode: str = "scene", input_source: str = "full",
                 with_correspondence: bool = True, correspondence_tol: float = 1e-5):
    """
    `(target, model_input, info)` for one scene's loaded fragment meshes --
    exactly what the dataset builds, for scripts that load a scene themselves.

    `model_input` is None when the network sees the full meshes (it is then the
    target itself). `info` counts the repairs and the fracture-surface
    fallbacks.
    """
    input_meshes = list(meshes)
    frac_fallbacks = 0
    if input_source == "frac":
        input_meshes = []
        for m in meshes:
            fm = extract_fractures(m)
            # A fragment whose fracture surface prunes to nothing still has to
            # appear, or the input's fragment count would diverge from the
            # target's and the per-fragment rotations would address the wrong
            # fragments.
            if len(fm.faces) == 0 or len(fm.vertices) < 3:
                fm = m
                frac_fallbacks += 1
            input_meshes.append(fm)

    v_clusters: Sequence[Optional[np.ndarray]] = [None] * len(input_meshes)
    e_clusters: Sequence[Optional[np.ndarray]] = [None] * len(input_meshes)
    if with_correspondence:
        v_clusters, e_clusters = compute_scene_correspondence(input_meshes, tol=correspondence_tol)

    # Target graph: ALWAYS the full fragment mesh. Even when the network is fed
    # the fracture surface, the geometric losses and every reported metric are
    # evaluated on the whole fragment.
    if input_source == "full":
        target_frags = [get_features(m, vertex_cluster_ids=vc, edge_cluster_ids=ec)
                        for m, vc, ec in zip(meshes, v_clusters, e_clusters)]
        target_graph = merge_fragments(target_frags, normalize_mode)
        input_graph = None
    else:
        target_frags = [get_features(m) for m in meshes]
        target_graph = merge_fragments(target_frags, normalize_mode)
        input_frags = [get_features(m, vertex_cluster_ids=vc, edge_cluster_ids=ec)
                       for m, vc, ec in zip(input_meshes, v_clusters, e_clusters)]
        # The FULL fragments' radii: input and target share one divisor and one
        # scale feature, or prediction and target live in different units.
        input_graph = merge_fragments(input_frags, normalize_mode,
                                      radii=[g.radius for g in target_frags])
        if input_graph.num_fragments != target_graph.num_fragments:
            raise RuntimeError("input/target fragment count mismatch "
                               f"({input_graph.num_fragments} vs {target_graph.num_fragments})")
    info = {"frac_fallbacks": frac_fallbacks,
            "num_repaired": sum(getattr(g, "num_repaired", 0) for g in target_frags)}
    return target_graph, input_graph, info


def draw_scene(cfg, split: Optional[str] = "val", rng: Optional[np.random.Generator] = None):
    """
    One random `(scene_dir, pattern, category)` from a split, under a config's
    data definition -- for scripts that want "some held-out scene". The same
    catalogue and split the training run used, so it cannot pick a training
    object by accident.
    """
    rng = rng or np.random.default_rng()
    objects = BreakingBadDataset(split=split, **dataset_kwargs(cfg)).objects
    entry = objects[int(rng.integers(len(objects)))]
    scene_dir, mode = entry.modes[int(rng.integers(len(entry.modes)))]
    return scene_dir, mode, entry.category


def collate_fn(samples: List[Dict]) -> Dict:
    """Batch several scenes. Every per-scene index space is offset here."""
    samples = [s for s in samples if s is not None]
    if not samples:
        raise RuntimeError("collate_fn received no usable samples")

    target = collate_scenes([s["target"] for s in samples])
    if any(s["input"] is not None for s in samples):
        model_input = collate_scenes(
            [s["input"] if s["input"] is not None else s["target"] for s in samples])
    else:
        model_input = None

    return {
        "target": target,
        "input": model_input,
        "rot": torch.cat([s["rot"] for s in samples], 0),
        "scene_dirs": [s["scene_dir"] for s in samples],
        "modes": [s.get("mode", "") for s in samples],
        "categories": [s.get("category", "") for s in samples],
        "load_seconds": float(sum(s["load_seconds"] for s in samples)),
        "frac_fallbacks": int(sum(s["frac_fallbacks"] for s in samples)),
        "num_repaired": int(sum(s.get("num_repaired", 0) for s in samples)),
        "num_scenes": len(samples),
        "meshes": [s["meshes"] for s in samples] if samples[0]["meshes"] is not None else None,
    }


def split_batch_by_scene(batch: Dict) -> List[Dict]:
    """
    Re-split a collated batch into single-scene batches.

    Used by the training loop's micro-batching: peak activation memory then
    depends only on the LARGEST SINGLE SCENE rather than on `batch_size`.
    """
    targets: List[SceneBatch] = split_scenes(batch["target"])
    inputs = (split_scenes(batch["input"]) if batch["input"] is not None
              else [None] * len(targets))
    out: List[Dict] = []
    f0 = 0
    for i, (target, model_input) in enumerate(zip(targets, inputs)):
        nf = target.num_fragments
        out.append({
            "target": target,
            "input": model_input,
            "rot": batch["rot"][f0:f0 + nf],
            "scene_dirs": [batch["scene_dirs"][i]],
            "modes": [batch.get("modes", [""] * len(targets))[i]],
            "categories": [batch.get("categories", [""] * len(targets))[i]],
            "num_scenes": 1,
        })
        f0 += nf
    return out


def merge_micro_batches(chunk: List[Dict]) -> Dict:
    """The inverse of `split_batch_by_scene` for a group of single-scene batches."""
    if len(chunk) == 1:
        return chunk[0]
    return {
        "target": collate_scenes([c["target"] for c in chunk]),
        "input": (collate_scenes([c["input"] for c in chunk])
                  if chunk[0]["input"] is not None else None),
        "rot": torch.cat([c["rot"] for c in chunk], 0),
        "scene_dirs": [d for c in chunk for d in c["scene_dirs"]],
        "modes": [m for c in chunk for m in c.get("modes", [""] * c["num_scenes"])],
        "categories": [k for c in chunk for k in c.get("categories", [""] * c["num_scenes"])],
        "num_scenes": sum(c["num_scenes"] for c in chunk),
    }


def dataset_kwargs(cfg) -> Dict:
    """
    The dataset arguments that DEFINE the data, read from a `Config`.

    One function for the trainer, the evaluator and every script, so a split
    can never be built from a subset of the settings. (The evaluator used to
    omit `split_source` and `data_subsets`, and so scored an officially-split
    model on the HASH split's objects -- some of which were official training
    objects.)
    """
    return dict(
        root_dir=cfg.root_dir,
        subsets=[x.strip() for x in cfg.data_subsets.split(",") if x.strip()] or None,
        fracture_pattern=cfg.fracture_pattern or None,
        split_source=cfg.split_source,
        split_by=cfg.split_by,
        fracture_pool=cfg.fracture_pool,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        split_seed=cfg.split_seed,
        max_objects=cfg.max_scenes,
        input_source=cfg.input_source,
        normalize_mode=cfg.normalize_mode,
        with_correspondence=cfg.correspondence,
        correspondence_tol=cfg.correspondence_tol,
        min_fragments=cfg.min_fragments,
        max_fragments=cfg.max_fragments,
    )
