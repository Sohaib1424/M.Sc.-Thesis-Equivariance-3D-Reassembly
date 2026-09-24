"""
From meshes to the tensors the network consumes.

Two stages, deliberately split. :func:`build_scene` is pure numpy and does the
geometry, so it runs in ``DataLoader`` workers; :func:`collate` is the only
part that touches torch and does nothing but concatenate and index-shift. The
split keeps the expensive per-scene work parallel and the batch assembly cheap.

The two copies
--------------
Every sample carries the same fragments twice: perturbed (the input) and
assembled (the target). They are built from **one** topology and differ only
in coordinates, which is not a convenience -- it is what makes the losses
meaningful. Position, normal and face-normal terms all match rows *by index*,
so two independently constructed copies would silently compare unrelated
vertices. ``repair.resolve_duplicated_faces`` reorders faces lexicographically,
so "build both and hope" really does produce two different orderings.
:func:`build_scene` therefore derives the perturbed copy from the assembled one
by rotating coordinates, and never re-runs topology.

What survives normalisation
---------------------------
Centring removes the perturbation's translation, and the per-scene divisor is
the largest fragment radius -- a quantity unchanged by rotation and by
translation-then-centring. So the two copies share a divisor, and the exact
relation ``v_perturbed = v_assembled @ Q.T`` holds after normalisation as well
as before. The rotation label is therefore ``Q.T`` in normalised coordinates
too, with no rescaling of the target. ``tests/test_features.py`` asserts this
rather than trusting it.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..mesh.orientation import canonical_edge_normals, directed_edge_normals
from ..mesh.patches import fracture_patches, scene_tokens
from ..mesh.topology import edge_topology, face_normals, vertex_normals
from .transforms import normalize_fragments, random_rotations

# One number, not two. The per-scene total alone bounds the cross-fragment pair
# count: for a budget T split any way at all, the undirected pair count is
# (T^2 - sum t_i^2)/2 < T^2/2, so T = 2048 can never exceed ~2.1 million pairs
# -- 0.03x GARF's six-layer stack -- however lopsided the split.
#
# There is deliberately no per-fragment cap. See `scene_tokens` for why the
# earlier one was removed: it did not tighten that bound, and it destroyed the
# area proportionality that is the whole point of splitting by area.
DEFAULT_TOKENS_PER_SCENE = 2048


class FragmentArrays(NamedTuple):
    """
    One fragment, both poses, sharing a single topology.

    The perturbed copy (``vertices``, ``normals``, ``edge_normals``) is ``None``
    when the scene was built with ``perturb=False``: it is then derived on the
    device by :func:`perturb_on_device`, from the assembled copy and the label.
    """
    vertices: Optional[np.ndarray]      # (V, 3) perturbed, centred, normalised
    target_vertices: np.ndarray         # (V, 3) assembled, centred, normalised
    normals: Optional[np.ndarray]       # (V, 3) perturbed vertex normals
    target_normals: np.ndarray          # (V, 3) assembled vertex normals
    edge_index: np.ndarray              # (2, 2E) directed, both copies
    edge_normals: Optional[np.ndarray]  # (2E, 3, 3) perturbed: n1, n2, p_src - p_dst
    target_edge_normals: np.ndarray
    rotation: np.ndarray          # (3, 3) label: maps perturbed -> assembled
    radius: float                 # world units, before normalisation
    token_vertices: np.ndarray    # (T,) local vertex indices for cross-attention
    # What undoes the normalisation, so an assembly can be scored in WORLD
    # units: ``world = normalised * divisor + centroid``. Kept because the
    # translation solver and the part-accuracy threshold (0.01) are both defined
    # there; a Chamfer distance in normalised units is a different number.
    centroid: Optional[np.ndarray] = None   # (3,) world-frame centroid removed
    divisor: float = 1.0                    # scale divided out
    fracture: Optional[np.ndarray] = None   # (V,) bool, the fracture mask


class Repairs(NamedTuple):
    """
    What :func:`build_scene` had to repair in one scene, counted.

    Nothing here is NaN by the time it reaches a tensor -- ``face_normals`` and
    ``vertex_normals`` are NaN-free by construction -- and that is exactly why
    the counts matter: a repaired value is a *zero*, which trains quietly, and a
    scene full of them is a scene the model is learning little from. trimesh
    would have returned NaN for every one of these; an earlier run of the
    previous design lost three objects that way, every epoch, for forty epochs.
    """
    degenerate_faces: int = 0     # zero-area triangles: face normal set to 0
    zero_normals: int = 0         # vertices whose normal came out 0 (all faces degenerate, or none)

    @property
    def total(self) -> int:
        return int(self.degenerate_faces) + int(self.zero_normals)


class SceneSample(NamedTuple):
    """One scene: its fragments, plus the coincidence clusters spanning them."""
    fragments: List[FragmentArrays]
    cluster: np.ndarray           # (sum V,) coincidence id, -1 where none
    category: str = ""
    """
    The object category this scene came from ("Bottle", "Mug", ...), carried
    only so metrics can be broken down by it.

    It travels with the sample rather than being recovered afterwards from the
    loader's position, because samples are DROPPED -- a single-fragment mode
    returns ``Skipped`` -- so position in the output stream does not index the
    dataset's item list. A breakdown built on that assumption silently
    attributes each sample to the wrong category from the first drop onward,
    and looks perfectly plausible.
    """
    key: str = ""
    """``<object>/<mode>`` -- so a batch that fails can say which scenes were in it."""
    repairs: Repairs = Repairs()


def _scene_token_sets(vertices, faces, masks, *, mode, metric, total, max_per_fragment):
    """
    Which vertices of each fragment become cross-fragment tokens.

    Thin wrapper over :func:`~reassembly.mesh.patches.scene_tokens`, whose job
    is the per-scene budget split; this converts its ``PatchAssignment`` output
    into the vertex-index arrays the collate wants, and handles the case where
    the whole scene has no fracture surface at all.

    ``total=None`` disables the scene budget and falls back to the per-fragment
    cap alone -- the unreduced ``mode="vertex"`` baseline. Note what that costs
    at the tail before using it: a 99-fragment scene at 128 tokens each is 79
    million cross-fragment pairs, more than GARF's entire six-layer stack.
    """
    n = len(vertices)
    if n == 0:
        return []
    if not any(m.any() for m in masks):
        return [np.zeros(0, np.int64) for _ in range(n)]

    pairs = list(zip(vertices, faces))
    if total is None:
        assignments = [
            fracture_patches(v, np.asarray(f), m, mode=mode, metric=metric,
                             max_patches=int(max_per_fragment or max(int(m.sum()), 1)))
            if m.any() else None
            for (v, f), m in zip(pairs, masks)
        ]
    else:
        assignments = scene_tokens(
            pairs, masks, int(total), mode=mode, metric=metric,
            max_per_fragment=None if max_per_fragment is None else int(max_per_fragment),
        )
    return [
        np.zeros(0, np.int64) if a is None or not m.any()
        else np.unique(a.vertex_index)
        for a, m in zip(assignments, masks)
    ]


def _fragment_geometry(vertices: np.ndarray, faces: np.ndarray):
    """
    Topology and the three equivariant edge features, computed once per
    fragment and reused for both poses.

    An edge carries ``(n1, n2, delta)``: its two canonically ordered adjacent
    face normals, and ``p_source - p_destination``.

    ``delta`` is the relative position of the neighbour, and it earns its slot.
    A message in :class:`~reassembly.nn.gat.VNGraphAttention` is a linear map of
    the *source* node's features and the edge's, so without it the message
    cannot say **which way** the neighbour lies -- only what it looks like. The
    destination's own coordinate enters the layer through a separate self-loop
    term, so a difference is expressible in the first layer if the weights
    conspire, but after one layer the channels are mixed abstractions and
    relative position stops being cleanly recoverable. Every standard
    equivariant message-passing construction supplies it explicitly for this
    reason.

    The raw difference rather than the unit direction ``r_ij``: it is equally
    equivariant, translation-invariant either way, and the length is an
    invariant the network can read off with a norm. Normalising throws that
    away for nothing.
    """
    topology = edge_topology(faces, vertices.shape[0])
    face_norms, degenerate = face_normals(vertices, faces, return_degenerate=True)
    canonical = canonical_edge_normals(vertices, faces, topology, face_norms)
    edge_index, n1, n2 = directed_edge_normals(canonical)
    # Taken from `edge_index`, so the reverse copy of each undirected edge gets
    # the negated vector automatically rather than by a second convention.
    delta = vertices[edge_index[0]] - vertices[edge_index[1]]
    return topology, edge_index, np.stack([n1, n2, delta], axis=1), degenerate


def build_scene(
    vertices: Sequence[np.ndarray],
    faces: Sequence[np.ndarray],
    fracture_masks: Sequence[np.ndarray],
    *,
    rotations: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    normalize_mode: str = "scene",
    token_mode: str = "sample",
    token_metric: str = "geodesic",
    tokens_per_scene: Optional[int] = DEFAULT_TOKENS_PER_SCENE,
    max_tokens_per_fragment: Optional[int] = None,
    cluster: Optional[np.ndarray] = None,
    category: str = "",
    key: str = "",
    perturb: bool = True,
) -> SceneSample:
    """
    Build one scene's arrays.

    ``vertices`` and ``faces`` are the fragments **in their assembled frame** --
    which is also the only frame where the coincidence labels mean anything, so
    if ``cluster`` is being computed it must already have happened. ``rotations``
    supplies the per-fragment perturbation; omitted, it is drawn from ``rng``.

    ``perturb=False`` leaves the perturbed copy out: the rotation that makes it
    is carried as the label anyway, so :func:`perturb_on_device` can produce it
    on the GPU with three batched matrix products instead of shipping a second
    copy of every position, normal and edge feature through the data loader.
    The label, the tokens and everything else are identical either way.

    Tokens: a fixed number per *scene*
    ----------------------------------
    All vertices are kept as nodes and every mesh edge as an intra-fragment
    edge. The fracture mask decides only which vertices are *eligible* to become
    cross-fragment tokens, and ``scene_tokens`` then divides one per-scene
    budget among the fragments in proportion to their fracture-surface area,
    sampling within each fragment's own eligible set.

    ``tokens_per_scene`` is that budget, and it is the *only* limit by default.
    It alone bounds the graph -- the undirected pair count for a budget ``T`` is
    ``(T^2 - sum t_i^2)/2 < T^2/2`` however the split falls -- so a fragment with
    four times the fracture surface of its neighbours draws four times the
    tokens, which is the behaviour the area weighting exists to produce.
    ``max_tokens_per_fragment`` is available but defaults to ``None``; see
    ``patches.scene_tokens`` for why capping is usually wrong.

    Sampling is per fragment, never over the pooled scene. At input time the
    fragments are in arbitrary perturbed poses, so one global farthest-point
    pass would pick points according to where the pieces happened to land -- a
    different sample for every perturbation of the same scene, and no
    equivariance left.

    ``token_mode`` may be ``"sample"`` or ``"vertex"``, both of which *select*
    real mesh vertices, so a token is a vertex index and the cross-fragment
    layer can read and write the same feature array the mesh graph uses.
    ``"patch"`` is rejected: its tokens are centroids of groups of vertices,
    which are not vertices, so they cannot be an index at all. Supporting it
    means a separate pool-attend-scatter path, not a different argument here --
    and silently falling back would ignore the budget entirely while appearing
    to have worked.
    """
    if token_mode not in ("sample", "vertex"):
        raise ValueError(
            f"token_mode={token_mode!r}: build_scene supports 'sample' and "
            "'vertex', which select real mesh vertices. 'patch' pools vertices "
            "into centroids, which are not vertex indices and need a separate "
            "path through the model."
        )
    n = len(vertices)
    if n == 0:
        return SceneSample([], np.zeros(0, np.int64), category, key)
    if rotations is None:
        rotations = random_rotations(n, rng)
    rotations = np.asarray(rotations, dtype=np.float64)

    # Normalise the assembled copy. The perturbed copy inherits the same
    # divisor by construction, because rotation does not change a radius.
    class _Holder:
        def __init__(self, v):
            self.vertices = v

    normalized = normalize_fragments(
        [_Holder(np.asarray(v, dtype=np.float64)) for v in vertices],
        mode=normalize_mode,
    )

    # One budget for the whole scene, split by fracture-surface area, sampled
    # inside each fragment. Done here, before the per-fragment loop, because the
    # split is a property of the scene: a fragment's share depends on how much
    # fracture surface the *others* have.
    masks = [np.asarray(m, dtype=bool) for m in fracture_masks]
    token_sets = _scene_token_sets(
        normalized.vertices, faces, masks,
        mode=token_mode, metric=token_metric, total=tokens_per_scene,
        max_per_fragment=max_tokens_per_fragment,
    )

    fragments: List[FragmentArrays] = []
    degenerate = zero_normals = 0
    for i in range(n):
        f = np.asarray(faces[i])
        target_v = normalized.vertices[i]
        _, edge_index, target_edge_n, bad_faces = _fragment_geometry(target_v, f)
        target_n = vertex_normals(target_v, f)
        degenerate += int(bad_faces)
        zero_normals += int((np.abs(target_n).sum(axis=1) == 0.0).sum())

        Q = rotations[i]
        if perturb:
            moved_v = target_v @ Q.T
            # Normals and edge normals are rotated rather than recomputed: they
            # are the *same* geometry, and recomputing invites the two copies to
            # differ by round-off in a quantity the loss compares directly.
            moved_n = target_n @ Q.T
            moved_edge_n = target_edge_n @ Q.T
        else:
            moved_v = moved_n = moved_edge_n = None

        tokens = token_sets[i]
        assert tokens.size == 0 or tokens.max() < target_v.shape[0]

        fragments.append(FragmentArrays(
            vertices=moved_v, target_vertices=target_v,
            normals=moved_n, target_normals=target_n,
            edge_index=edge_index,
            edge_normals=moved_edge_n, target_edge_normals=target_edge_n,
            # The label is the rotation that undoes the perturbation. Deriving
            # it here rather than in the training loop keeps the one transpose
            # that matters in a single place.
            rotation=Q.T,
            radius=float(normalized.radius[i]),
            token_vertices=tokens,
            centroid=np.asarray(normalized.centroid[i], dtype=np.float64),
            divisor=float(normalized.divisor[i]),
            fracture=masks[i] if i < len(masks) else None,
        ))

    total = sum(len(v) for v in vertices)
    if cluster is None:
        cluster = np.full(total, -1, np.int64)
    return SceneSample(fragments, np.asarray(cluster, dtype=np.int64), category,
                       key, Repairs(degenerate, zero_normals))


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

class Batch(NamedTuple):
    """
    A batch of scenes, flattened.

    Fragments are numbered globally and vertices are numbered globally, so
    every index tensor here is already shifted -- nothing downstream needs to
    know where one scene ends and the next begins except through
    ``fragment_scene`` and the ``ptr`` vectors.

    ``node_features`` and ``edge_attr`` -- the perturbed copy the network reads
    -- are ``None`` when the scenes were built with ``perturb=False``;
    :func:`perturb_on_device` fills them in from the assembled copy and the
    label, on whatever device the batch has been moved to.
    """
    node_features: "object"       # (N, 2, 3) centred coordinate, vertex normal
    edge_index: "object"          # (2, E) global vertex indices
    edge_attr: "object"           # (E, 3, 3) n1, n2, relative position
    vertex_fragment: "object"     # (N,)
    fragment_scene: "object"      # (F,)
    vertex_ptr: "object"          # (F+1,) GARF's cu(l) over vertices
    fragment_ptr: "object"        # (S+1,) over fragments
    log_scale: "object"           # (F, 1) invariant scale feature
    token_index: "object"         # (T,) global vertex indices
    token_query: "object"         # (P,) cross-fragment pair list
    token_key: "object"           # (P,)
    target_rotation: "object"     # (F, 3, 3)
    target_vertices: "object"     # (N, 3)
    target_normals: "object"      # (N, 3)
    target_edge_normals: "object"  # (E, 3, 3)
    cluster: "object"             # (N,) global coincidence id, -1 where none
    num_clusters: int
    num_fragments: int
    num_scenes: int
    categories: Tuple[str, ...] = ()   # (S,) object category per scene
    # -- for assembling and scoring in world units ---------------------------
    unit: "object" = None         # (F,) divisor: world = normalised * unit + centroid
    centroid: "object" = None     # (F, 3) world-frame centroid removed by centring
    fracture: "object" = None     # (N,) bool: the fracture-surface mask
    # -- for naming what failed ------------------------------------------------
    scene_keys: Tuple[str, ...] = ()      # (S,) "<object>/<mode>"
    repairs: Tuple[Repairs, ...] = ()     # (S,) what build_scene had to repair


def collate(samples: Sequence[SceneSample], device=None, dtype=None,
            pairs: bool = True) -> Batch:
    """
    Concatenate scenes into one batch, shifting every index.

    Order is ``(scene, fragment)``, so each fragment's vertices are contiguous
    and each scene's fragments are contiguous. Variable-length attention
    kernels need that, and so does the ``ptr`` form.

    ``pairs=False`` leaves the cross-fragment pair lists out
    (``token_query``/``token_key`` are ``None``); :func:`complete_batch` builds
    them where the batch is used. They are a pure function of the token list
    and the fragment ids, and they are most of a batch: at 2,048 tokens a scene
    of four fragments has ~3.1 million ordered pairs, two int64 lists of them --
    89% of the bytes of a two-scene batch, measured, and ~0.6 s of one CPU
    thread to build. On a GPU they take milliseconds and never cross the bus.

    If any scene arrives without its perturbed copy (``perturb=False``), the
    batch ships without one for all of them and :func:`perturb_on_device`
    derives it later. Mixing the two paths inside one batch would buy nothing
    and make the arrays' provenance depend on which scenes met in a batch.
    """
    import torch

    dtype = torch.float32 if dtype is None else dtype
    perturbed = all(f.vertices is not None for s in samples for f in s.fragments)

    node, edges, edge_attr = [], [], []
    target_v, target_n, target_edge_n = [], [], []
    vertex_fragment, fragment_scene, rotations, radii = [], [], [], []
    tokens, clusters = [], []
    vertex_counts, fragment_counts = [], []
    units, centroids, fracture = [], [], []

    vertex_offset = 0
    fragment_offset = 0
    cluster_offset = 0

    for scene_id, sample in enumerate(samples):
        scene_start = vertex_offset
        for fragment in sample.fragments:
            count = fragment.target_vertices.shape[0]
            if perturbed:
                node.append(np.stack([fragment.vertices, fragment.normals], axis=1))
                edge_attr.append(fragment.edge_normals)
            edges.append(fragment.edge_index + vertex_offset)
            target_v.append(fragment.target_vertices)
            units.append(fragment.divisor)
            centroids.append(np.zeros(3) if fragment.centroid is None
                             else fragment.centroid)
            fracture.append(np.zeros(count, bool) if fragment.fracture is None
                            else np.asarray(fragment.fracture, dtype=bool))
            target_n.append(fragment.target_normals)
            target_edge_n.append(fragment.target_edge_normals)
            vertex_fragment.append(np.full(count, fragment_offset, np.int64))
            fragment_scene.append(scene_id)
            rotations.append(fragment.rotation)
            radii.append(fragment.radius)
            tokens.append(fragment.token_vertices + vertex_offset)
            vertex_counts.append(count)
            vertex_offset += count
            fragment_offset += 1

        # Coincidence ids are per scene; shift them so they stay disjoint
        # across the batch. -1 (no cluster) must stay -1.
        local = sample.cluster
        if local.size:
            shifted = np.where(local >= 0, local + cluster_offset, -1)
            clusters.append(shifted)
            if (local >= 0).any():
                cluster_offset += int(local.max()) + 1
        fragment_counts.append(len(sample.fragments))
        assert vertex_offset - scene_start == sum(
            f.target_vertices.shape[0] for f in sample.fragments
        )

    def cat(parts, kind=dtype):
        if not parts:
            return torch.empty(0, dtype=kind, device=device)
        return torch.as_tensor(np.concatenate(parts), dtype=kind, device=device)

    long = torch.long
    vertex_fragment_t = cat(vertex_fragment, long)
    fragment_scene_t = torch.as_tensor(fragment_scene, dtype=long, device=device)
    token_index = cat(tokens, long)

    query = key = None
    if pairs:
        from ..nn.cross import cross_fragment_index

        query, key = cross_fragment_index(
            vertex_fragment_t[token_index],
            fragment_scene_t[vertex_fragment_t[token_index]],
            len(samples),
        )

    radius = torch.as_tensor(radii, dtype=dtype, device=device).clamp(min=1e-12)
    cluster = cat(clusters, long) if clusters else torch.empty(0, dtype=long, device=device)

    if edges:
        edge_index = torch.as_tensor(
            np.concatenate(edges, axis=1), dtype=long, device=device
        )
    else:
        edge_index = torch.empty(2, 0, dtype=long, device=device)

    return Batch(
        node_features=cat(node) if perturbed else None,
        edge_index=edge_index,
        edge_attr=cat(edge_attr) if perturbed else None,
        vertex_fragment=vertex_fragment_t,
        fragment_scene=fragment_scene_t,
        vertex_ptr=_ptr(vertex_counts, device),
        fragment_ptr=_ptr(fragment_counts, device),
        # log, not raw: fragments span 4 to 83,039 vertices, so the raw scale is
        # heavy-tailed enough to dominate whatever it is concatenated with.
        log_scale=torch.log(radius).unsqueeze(-1),
        token_index=token_index,
        token_query=query,
        token_key=key,
        target_rotation=torch.as_tensor(np.asarray(rotations), dtype=dtype, device=device)
        if rotations else torch.empty(0, 3, 3, dtype=dtype, device=device),
        target_vertices=cat(target_v),
        target_normals=cat(target_n),
        target_edge_normals=cat(target_edge_n),
        cluster=cluster,
        num_clusters=cluster_offset,
        num_fragments=fragment_offset,
        num_scenes=len(samples),
        categories=tuple(sample.category for sample in samples),
        unit=torch.as_tensor(units, dtype=dtype, device=device),
        centroid=torch.as_tensor(np.asarray(centroids, dtype=np.float64).reshape(-1, 3),
                                 dtype=dtype, device=device),
        fracture=cat(fracture, torch.bool),
        scene_keys=tuple(sample.key for sample in samples),
        repairs=tuple(sample.repairs for sample in samples),
    )


def perturb_on_device(batch: Batch) -> Batch:
    """
    Fill in the perturbed copy from the assembled one and the label.

    The label is ``R = Q^T``, so the perturbation is ``Q = R^T`` and every
    geometric quantity moves by ``x -> x Q^T``: centred positions (the centroid
    co-moves), vertex normals, both face normals of an edge and its relative
    position. It is exact rather than an approximation -- ``build_scene``
    already makes the perturbed copy this way, by rotating instead of
    recomputing -- and the edge-normal slot order survives it, because the
    triple product that fixes the order is rotation-invariant.

    A no-op on a batch that already carries its perturbed copy.
    """
    import torch

    if batch.node_features is not None and batch.edge_attr is not None:
        return batch
    Q = batch.target_rotation.transpose(-1, -2)
    node = torch.stack([batch.target_vertices, batch.target_normals], dim=1)
    node = torch.einsum("nij,ncj->nci", Q[batch.vertex_fragment], node)
    edge_fragment = batch.vertex_fragment[batch.edge_index[0]]
    edge = torch.einsum("eij,ecj->eci", Q[edge_fragment], batch.target_edge_normals)
    return batch._replace(node_features=node, edge_attr=edge)


def pair_on_device(batch: Batch) -> Batch:
    """Build the cross-fragment pair lists where the batch is. A no-op if present."""
    if batch.token_query is not None and batch.token_key is not None:
        return batch
    from ..nn.cross import cross_fragment_index

    fragment = batch.vertex_fragment[batch.token_index]
    query, key = cross_fragment_index(fragment, batch.fragment_scene[fragment],
                                      batch.num_scenes)
    return batch._replace(token_query=query, token_key=key)


def complete_batch(batch: Batch) -> Batch:
    """
    Everything a batch may have left to the device: the perturbed copy
    (:func:`perturb_on_device`) and the pair lists (:func:`pair_on_device`).
    Idempotent, so any consumer can call it.
    """
    return pair_on_device(perturb_on_device(batch))


def _ptr(counts: Sequence[int], device=None):
    import torch

    tensor = torch.as_tensor(list(counts), dtype=torch.long, device=device)
    return torch.cat([tensor.new_zeros(1), torch.cumsum(tensor, 0)])


def clustered_vertices(cluster) -> Tuple["object", "object", int]:
    """
    Drop the unclustered vertices and renumber what is left contiguously.

    Returns ``(keep, renumbered, count)``: a boolean mask over vertices, the
    cluster id of each kept vertex in ``0..count-1``, and how many clusters
    there are.

    The embedding-consistency loss is only defined over coincident vertices,
    and most vertices are not coincident -- 89% of them, dataset-wide. Feeding
    the raw array with its ``-1`` markers to a segment reduction would index
    out of range, or, with a shift, quietly collect every unmatched vertex in
    the batch into one enormous cluster and ask the model to make them all
    agree.
    """
    import torch

    keep = cluster >= 0
    if not bool(keep.any()):
        empty = torch.empty(0, dtype=torch.long, device=cluster.device)
        return keep, empty, 0
    unique, renumbered = torch.unique(cluster[keep], return_inverse=True)
    return keep, renumbered, int(unique.numel())
