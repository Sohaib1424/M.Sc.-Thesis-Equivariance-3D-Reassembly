"""
Reducing a fragment's cross-attention token set, so the inter-fragment graph
stops scaling with mesh resolution.

Why
---
Connecting every fracture vertex of every fragment to every fracture vertex of
every other fragment costs

    E_cross = 1/2 [ (sum_i f_i)^2 - sum_i f_i^2 ]

which is quadratic in token count and therefore scales with **mesh resolution**.
GARF has no such term: it samples a fixed M points per object, so its attention
cost is constant however fine the mesh is. Measured on the full Breaking Bad run
(1,096,825 fragments), the median scene already costs about the same as GARF's
whole flow-matching stack and the p90 scene costs several times more -- see
``scripts.analyze_graph_cost``.

Three ways to bound it, all returning the same structure so they are
interchangeable and can be compared rather than assumed:

``mode="patch"``
    pool vertices into centroids. Every vertex contributes, but a token sits at
    a centroid, which is not a point on the surface.
``mode="sample"``
    farthest-point *selection* of real vertices -- the deterministic,
    rotation-invariant analogue of GARF's Poisson-disk sampling. Tokens are
    genuine surface points, but most vertices are dropped.
``mode="vertex"``
    the unreduced graph, kept as the baseline the other two are measured
    against.

``allocate_budget`` splits one fixed *per-scene* budget across fragments. A
per-fragment cap does not actually fix the cost -- a 20-fragment scene still
costs 100x a 2-fragment one -- so the per-scene budget is what makes the pair
count a constant of the configuration.

Gate or feature
---------------
``eligible_mask`` decides whether fracture-ness *gates* the graph (tokens drawn
only from the fracture surface) or rides along as a per-token *feature*
(``fracture_fraction``, tokens drawn from the whole surface). GARF does the
latter: its attention runs over all M points and fracture segmentation is only a
pretraining objective. The choice decides how much the labelling's precision
matters -- see ``fracture_patches`` and ``docs/design-notes.md`` section 4.

The partition must be rotation-invariant
----------------------------------------
If the patch assignment changed under rotation, pooled patch features would not
be equivariant and the whole backbone's guarantee would be void -- silently,
since every tensor would still have the right shape. Both mechanisms here are
invariant by construction:

* **Connected components** are purely topological: they depend on the mesh
  graph, not on coordinates, so no transform of any kind can change them.
* **Farthest-point sampling** selects by pairwise Euclidean distance, which is
  rotation-invariant, seeded from the vertex farthest from the fragment
  centroid -- also an invariant choice. The same *vertex indices* are therefore
  selected in the same order for a rotated copy.

`tests/test_patches.py` asserts both under random SO(3).

Components first, then a budget
-------------------------------
Components are used before any geometric clustering because they carry real
meaning: a fragment that broke away from three neighbours has, roughly, three
separate fracture regions, and those are exactly its connected components.
Splitting them geometrically first would blend surfaces that face different
neighbours. Components larger than their share of the budget are then
subdivided by farthest-point sampling, which bounds the cost.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .topology import unique_edges

# Matches the recommended ``max_per_fragment`` in ``scene_tokens``, so a fragment
# reduced on its own and a fragment reduced as part of a scene agree. 128 is
# deliberately above the median fragment's 125 coincidence fracture vertices: the
# default should not silently reduce a typical scene, only bound the tail. The
# earlier value of 32 was chosen for ``mode="patch"`` against the dihedral mask,
# which over-labels by ~3.9x, so it was ~4x tighter than it looked.
DEFAULT_MAX_PATCHES = 128

# How "spread out" is measured when selecting tokens.
#
#   "geodesic"  -- shortest path along the fracture surface itself (Dijkstra on
#                  the mesh edges whose endpoints are both on the break)
#   "euclidean" -- straight-line distance through space
#
# Geodesic is the default because a fracture surface is a thin strip wrapping a
# curved fragment, and straight-line distance measures *through the material*:
# two points on opposite arms of such a strip read as neighbours, so tokens
# bunch instead of spreading. Both metrics are rotation-invariant -- edge
# lengths do not change under rigid motion -- so this is a question of coverage
# quality, not of correctness.
DEFAULT_METRIC = "geodesic"


class PatchAssignment(NamedTuple):
    """Token membership for one fragment's cross-attention set."""
    vertex_index: np.ndarray      # (M,) mesh vertex indices, ascending
    patch_of_vertex: np.ndarray   # (M,) patch id in [0, num_patches)
    num_patches: int
    component_of_patch: np.ndarray  # (P,) originating connected component
    # (P,) fraction of each token's member vertices that are fracture surface.
    # 0.0 or 1.0 in "sample"/"vertex" mode; a genuine fraction when pooling.
    # This is the channel that lets fracture-ness be a *feature* rather than a
    # gate -- see `eligible_mask` in `fracture_patches`.
    fracture_fraction: np.ndarray


def _fracture_adjacency(edges: np.ndarray, mask: np.ndarray):
    """Edges with both endpoints on the fracture surface, reindexed locally."""
    keep = mask[edges[:, 0]] & mask[edges[:, 1]]
    return edges[keep]


def _components(local_edges: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if count == 0:
        return np.zeros(0, np.int64), 0
    if local_edges.size == 0:
        return np.arange(count, dtype=np.int64), count
    data = np.ones(local_edges.shape[0], dtype=np.int8)
    graph = coo_matrix((data, (local_edges[:, 0], local_edges[:, 1])),
                       shape=(count, count))
    n, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), int(n)


def _tie_tolerance(points: np.ndarray) -> float:
    """
    Absolute tolerance for treating two squared distances as equal.

    Rotating a fragment perturbs every coordinate in the last few bits, so two
    points at *exactly* the same distance before the rotation are at slightly
    different distances after it. A plain ``argmax`` then picks a different
    index and the whole partition changes -- which is the one thing this module
    may not do. Symmetric fragments make such exact ties common, and Breaking
    Bad's Everyday subset is full of surfaces of revolution.
    """
    if points.shape[0] == 0:
        return 0.0
    extent = float(np.abs(points).max())
    return 1e-9 * max(extent, 1.0) ** 2


def _argmax_stable(values: np.ndarray, tol: float) -> int:
    """Lowest index among the near-maximal values.

    Vertex indices do not change under any transform of the coordinates, so
    breaking ties by index is exactly invariant -- unlike breaking them by
    whichever float happened to come out largest.
    """
    return int(np.flatnonzero(values >= values.max() - tol)[0])


def _farthest_point_seeds(points: np.ndarray, k: int, tol: float) -> np.ndarray:
    """
    Indices of ``k`` farthest-point-sampled seeds, by straight-line distance.

    Seeded from the point farthest from the centroid rather than from index 0
    or a random draw: index 0 is an arbitrary artifact of mesh ordering, and a
    random draw would make the partition differ between the two copies of a
    scene the loader builds. Distance-to-centroid is rotation-invariant, and
    ties are broken by index, so a rotated fragment yields identical seeds.
    """
    n = points.shape[0]
    if k <= 0:
        # Reachable: `allocate_budget` hands a fragment 0 tokens when the scene
        # total is too small to give every fragment its minimum. Without this
        # guard the `k >= n` test below is False, and `seeds[0] = first` raises
        # on a zero-length array.
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    centroid = points.mean(axis=0)
    first = _argmax_stable(((points - centroid) ** 2).sum(axis=1), tol)

    seeds = np.empty(k, dtype=np.int64)
    seeds[0] = first
    best = ((points - points[first]) ** 2).sum(axis=1)
    for i in range(1, k):
        nxt = _argmax_stable(best, tol)
        seeds[i] = nxt
        np.minimum(best, ((points - points[nxt]) ** 2).sum(axis=1), out=best)
    return seeds


def _surface_graph(local_edges: np.ndarray, points: np.ndarray):
    """
    Weighted adjacency over the fracture surface: mesh edges, Euclidean lengths.

    Only edges with *both* endpoints on the fracture surface are kept, so a
    shortest path is confined to the break and never takes a shortcut across the
    original exterior. Edge lengths are rotation-invariant, so the resulting
    distances are too.
    """
    from scipy.sparse import coo_matrix

    n = points.shape[0]
    if local_edges.size == 0:
        return coo_matrix((n, n), dtype=np.float64).tocsr()
    lengths = np.linalg.norm(points[local_edges[:, 0]] - points[local_edges[:, 1]], axis=1)
    # Symmetric: an undirected edge as two directed ones, so Dijkstra can
    # traverse it either way without `directed=False` copying the matrix.
    rows = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
    cols = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
    data = np.concatenate([lengths, lengths])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _geodesic_seeds(points: np.ndarray, graph, k: int, tol: float) -> np.ndarray:
    """
    Farthest-point sampling under *surface* distance instead of straight-line.

    Why the metric matters here
    ---------------------------
    A fracture surface is typically a thin strip wrapping around a curved
    fragment -- look at any mug shard. Straight-line distance reads two points
    on opposite arms of such a strip as near neighbours because it measures
    *through the material*, so it declines to place tokens on both and the
    strip ends up unevenly covered. Surface distance measures along the break,
    which is where the mating actually happens, and spreads tokens the way the
    picture of the intended design shows them.

    Distances are Dijkstra on the mesh edge graph, which is the usual discrete
    stand-in for a true geodesic. It overestimates -- a path must follow edges
    rather than cross faces -- by a bounded factor that does not vary much
    across a mesh, and FPS only compares distances, so a uniform bias is
    harmless. Exact geodesics would need the heat method or MMP, neither of
    which is worth a dependency for a ranking.

    Disconnected pieces fall out for free. A fracture surface can be several
    separate patches (one per neighbouring fragment), and those are at infinite
    graph distance from each other, so ``argmax`` places a seed in every patch
    before refining any of them -- exactly the right priority, with no special
    case. Ties, including ties between infinities, go to the lowest vertex
    index, which is what keeps the choice rotation-invariant.
    """
    from scipy.sparse.csgraph import dijkstra

    n = points.shape[0]
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    centroid = points.mean(axis=0)
    first = _argmax_stable(((points - centroid) ** 2).sum(axis=1), _tie_tolerance(points))

    seeds = np.empty(k, dtype=np.int64)
    seeds[0] = first
    best = dijkstra(graph, indices=first, min_only=True)
    for i in range(1, k):
        nxt = _argmax_stable(best, tol)
        seeds[i] = nxt
        np.minimum(best, dijkstra(graph, indices=nxt, min_only=True), out=best)
    return seeds


def _linear_tolerance(points: np.ndarray) -> float:
    """Tie tolerance for *unsquared* distances -- see :func:`_tie_tolerance`."""
    if points.shape[0] == 0:
        return 0.0
    return 1e-9 * max(float(np.abs(points).max()), 1.0)


_DIST_BLOCK = 4096


def _squared_distances(points: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    """
    ``(n, k)`` squared distances, without an ``(n, k, 3)`` temporary.

    The direct form ``((points[:, None] - seeds[None]) ** 2).sum(-1)`` allocates
    ``n * k * 3`` floats before reducing them: 338 MB for a 10k-vertex fragment
    at k = 1024, which a batched loader turns into an OOM. Expanding
    ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` reduces that to one ``(n, k)``
    matmul, and blocking the rows caps the peak regardless of fragment size.

    The expansion is the form that caused a real bug elsewhere in this project
    (identical point clouds scoring 1.8e-7 in a Chamfer distance), so it is worth
    being explicit about why it is safe *here*: its absolute error is on the
    order of ``eps * extent^2`` ~ 1e-16 * extent^2, while the tie tolerance this
    feeds is 1e-9 * extent^2 -- seven orders of magnitude larger. The clamp keeps
    cancellation from producing a negative. `test_patches.py` asserts the two
    formulations give identical assignments.
    """
    n = points.shape[0]
    sq_p = np.einsum("ij,ij->i", points, points)
    sq_s = np.einsum("ij,ij->i", seed_points, seed_points)
    out = np.empty((n, seed_points.shape[0]), dtype=np.float64)
    for start in range(0, n, _DIST_BLOCK):
        stop = min(start + _DIST_BLOCK, n)
        block = out[start:stop]
        np.matmul(points[start:stop], seed_points.T, out=block)
        block *= -2.0
        block += sq_p[start:stop, None]
        block += sq_s[None, :]
    np.maximum(out, 0.0, out=out)
    return out


def _split_component(points: np.ndarray, k: int, tol: float) -> np.ndarray:
    """
    Assign each point to its nearest of ``k`` farthest-point seeds.

    Ties go to the lowest seed index, for the same reason as in the seeding:
    a point equidistant from two seeds must not change patch when the fragment
    is rotated.
    """
    if k <= 1 or points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.int64)
    seeds = _farthest_point_seeds(points, k, tol)
    d = _squared_distances(points, points[seeds])
    near = d <= (d.min(axis=1, keepdims=True) + tol)
    return np.argmax(near, axis=1).astype(np.int64)


def _budget(sizes: np.ndarray, max_patches: int) -> np.ndarray:
    """
    Split ``max_patches`` across components, proportional to size, >= 1 each.

    Largest-remainder rather than rounding, so the parts sum to exactly the
    budget; plain rounding silently over- or under-shoots and the resulting
    patch count then depends on the fragment.
    """
    n = sizes.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)
    if max_patches <= 0:
        # A zero budget means zero patches. Without this the `n >= max_patches`
        # test below is True and every component silently gets one, which
        # over-produces rather than raising.
        return np.zeros(n, dtype=np.int64)
    if n >= max_patches:
        return np.ones(n, dtype=np.int64)

    share = sizes / sizes.sum() * max_patches
    counts = np.maximum(1, np.floor(share)).astype(np.int64)
    counts = np.minimum(counts, sizes)          # never more patches than points

    spare = max_patches - int(counts.sum())
    if spare > 0:
        headroom = sizes - counts
        order = np.argsort(-(share - np.floor(share)))
        for i in order:
            if spare == 0:
                break
            add = min(spare, int(headroom[i]))
            counts[i] += add
            spare -= add
    return counts


def fracture_patches(
    vertices: np.ndarray,
    faces: np.ndarray,
    fracture_vertex_mask: np.ndarray,
    *,
    mode: str = "patch",
    max_patches: int = DEFAULT_MAX_PATCHES,
    eligible_mask: np.ndarray | None = None,
    metric: str = DEFAULT_METRIC,
) -> PatchAssignment:
    """
    Reduce a fragment's fracture-surface vertices to a bounded token set.

    Three modes, all producing the same structure so they can be swapped and
    benchmarked against each other:

    ``mode="patch"``
        connected components, subdivided by farthest-point sampling, then
        *pooled*. Each token is the mean of many vertices, so every vertex
        contributes.
    ``mode="sample"``
        farthest-point *selection* of at most ``max_patches`` real vertices;
        each token is one actual mesh vertex and the rest are dropped from
        cross-attention. This is the vertex-level analogue of GARF's
        Poisson-disk sampling: FPS on a fixed point set gives approximately
        blue-noise coverage, and unlike dart-throwing it is deterministic and
        index-tie-broken, hence exactly rotation-invariant.
    ``mode="vertex"``
        every fracture vertex is its own token -- the unreduced graph, kept as
        the baseline the other two are measured against.

    Pooling versus selection is a real trade, not a formality. ``patch`` keeps
    information from every vertex but its tokens sit at centroids, which are
    not points on the surface. ``sample`` keeps real surface points but throws
    most of them away. Which wins is an empirical question; that is why both
    are here.

    ``fracture_vertex_mask`` is a boolean over all mesh vertices; index-based,
    so it is unaffected by any rescaling or centring applied to ``vertices``.

    Gate or feature: ``eligible_mask``
    ----------------------------------
    ``eligible_mask`` says which vertices may become tokens. It defaults to the
    fracture mask, which **gates** the graph: only fracture vertices talk across
    fragments. Pass an all-true mask instead and tokens are drawn from the whole
    fragment surface, with ``fracture_fraction`` carrying fracture-ness as a
    per-token **feature**.

    The distinction decides how much the labelling's precision matters. As a
    gate, a wrong label is an edge that should not exist or one that is missing,
    and the model cannot undo either. As a feature, a wrong label is one noisy
    input channel among several, and attention can learn to discount it.
    Measured on the full dataset the dihedral mask is ~3.9x larger than the true
    fracture surface, which is damaging for a gate and largely survivable for a
    feature.

    GARF takes the feature route. Its encoder produces ``F = E(P)`` over **all**
    M sampled points and its global attention runs with ``l = M`` -- every point,
    not the fracture subset. Fracture segmentation is a *pretraining objective*
    that shapes those features, never a mask on the attention graph.

    Original sharp features carry real assembly information too: a rim broken
    across three fragments still has to close into one circle. Excluding them by
    construction throws that away.
    """
    if mode not in ("patch", "sample", "vertex"):
        raise ValueError(
            f"mode must be 'patch', 'sample' or 'vertex', not {mode!r}")

    vertices = np.asarray(vertices, dtype=np.float64)
    fracture = np.asarray(fracture_vertex_mask, dtype=bool)
    eligible = fracture if eligible_mask is None else np.asarray(eligible_mask, dtype=bool)
    index = np.flatnonzero(eligible)
    m = index.shape[0]

    if m == 0:
        z = np.zeros(0, np.int64)
        return PatchAssignment(z, z, 0, z, np.zeros(0, np.float64))

    is_fracture = fracture[index].astype(np.float64)

    if mode == "vertex":
        ids = np.arange(m, dtype=np.int64)
        return PatchAssignment(index, ids, m, ids, is_fracture)

    if mode != "vertex" and int(max_patches) <= 0:
        # A zero allocation means no tokens, not "one per component". Reachable
        # from `scene_tokens` when the scene total cannot cover every fragment.
        # `mode="vertex"` deliberately ignores `max_patches` -- it is the
        # unreduced baseline.
        z = np.zeros(0, np.int64)
        return PatchAssignment(z, z, 0, z, np.zeros(0, np.float64))

    if mode == "sample":
        points = vertices[index]
        k = min(int(max_patches), m)
        if metric == "geodesic":
            remap = np.full(eligible.shape[0], -1, dtype=np.int64)
            remap[index] = np.arange(m, dtype=np.int64)
            edges = unique_edges(np.asarray(faces), eligible.shape[0])
            local = remap[_fracture_adjacency(edges, eligible)]
            graph = _surface_graph(local, points)
            seeds = _geodesic_seeds(points, graph, k, _linear_tolerance(points))
        else:
            seeds = _farthest_point_seeds(points, k, _tie_tolerance(points))
        # Sorted so the token order is the mesh's own vertex order rather than
        # the order FPS happened to visit -- one less arbitrary convention for
        # downstream code to depend on.
        seeds = np.sort(seeds)
        ids = np.arange(seeds.shape[0], dtype=np.int64)
        return PatchAssignment(index[seeds], ids, seeds.shape[0], ids,
                               is_fracture[seeds])

    # local reindex: mesh vertex id -> position within `index`
    remap = np.full(eligible.shape[0], -1, dtype=np.int64)
    remap[index] = np.arange(m, dtype=np.int64)
    edges = unique_edges(np.asarray(faces), eligible.shape[0])
    local = remap[_fracture_adjacency(edges, eligible)]

    labels, n_components = _components(local, m)
    sizes = np.bincount(labels, minlength=n_components)
    counts = _budget(sizes, max_patches)

    patch_of_vertex = np.empty(m, dtype=np.int64)
    component_of_patch = []
    next_id = 0
    points = vertices[index]
    tol = _tie_tolerance(points)
    for comp in range(n_components):
        members = np.flatnonzero(labels == comp)
        k = int(counts[comp])
        sub = _split_component(points[members], k, tol)
        used = int(sub.max()) + 1 if sub.size else 0
        patch_of_vertex[members] = sub + next_id
        component_of_patch.extend([comp] * used)
        next_id += used

    # Per-token fracture fraction: the share of a pooled patch's vertices that
    # lie on the fracture surface. A patch straddling the boundary between a
    # break and the original exterior gets an intermediate value rather than
    # being forced to one side.
    sums = np.zeros(next_id, dtype=np.float64)
    np.add.at(sums, patch_of_vertex, is_fracture)
    sizes_per_patch = np.bincount(patch_of_vertex, minlength=next_id)
    fracture_fraction = sums / np.maximum(sizes_per_patch, 1)

    return PatchAssignment(index, patch_of_vertex, next_id,
                           np.asarray(component_of_patch, dtype=np.int64),
                           fracture_fraction)


def patch_centroids(vertices: np.ndarray, assignment: PatchAssignment) -> np.ndarray:
    """
    Mean position of each patch's vertices.

    Equivariant: a mean of positions rotates with them. Pooling with anything
    that is not a linear combination (a max, a norm-based selection) would
    break that, which is why this is a plain mean.
    """
    points = np.asarray(vertices, dtype=np.float64)[assignment.vertex_index]
    if assignment.num_patches == 0:
        return np.zeros((0, 3), dtype=np.float64)
    sums = np.zeros((assignment.num_patches, 3), dtype=np.float64)
    np.add.at(sums, assignment.patch_of_vertex, points)
    counts = np.bincount(assignment.patch_of_vertex,
                         minlength=assignment.num_patches).astype(np.float64)
    return sums / np.maximum(counts, 1.0)[:, None]


def allocate_budget(weights, total: int, minimum: int = 1,
                    capacity=None) -> np.ndarray:
    """
    Split a per-scene token budget across fragments, proportional to ``weights``.

    Per-*fragment* caps ("at most 32 each") do not fix the cross-attention cost:
    a 2-fragment scene and a 20-fragment scene then differ by 100x, and small
    fragments -- of which Breaking Bad has many, the median fragment carrying
    216 dihedral-labelled fracture vertices -- never reach their cap anyway.
    Allocating one per-*scene* total is what makes the count actually constant,
    and it is what GARF does: a fixed M = 5000 points per object, weighted by
    fragment surface area, not a per-fragment quota.

    ``weights`` should be fracture-surface **area** where available. Vertex
    counts are a workable stand-in but they measure tessellation density as much
    as surface size, so a finely meshed chip would be over-served.

    ``capacity`` caps each fragment at what it can actually supply -- its number
    of eligible vertices. Without it a fragment with **zero** fracture surface
    still receives its ``minimum`` share, which it cannot use, so the budget is
    silently under-spent and the fixed-count promise quietly fails. The real
    dataset contains such fragments (``frac_vertices`` min = 0), so this is not
    hypothetical. Anything a fragment cannot take is redistributed to those that
    can.

    Largest-remainder, so the parts sum to exactly ``total`` whenever the
    capacities allow it, rather than drifting with rounding.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n = weights.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)

    # A non-finite weight is a degenerate fragment, not a reason to produce
    # nonsense. Left in, a NaN makes `positive.sum()` NaN, every share NaN, and
    # `floor(NaN).astype(int64)` INT64_MIN -- a silently catastrophic count that
    # numpy reports only as a RuntimeWarning. `fracture_surface_area` returns
    # NaN for a face with a NaN vertex, which Breaking Bad meshes do contain.
    weights = np.where(np.isfinite(weights), weights, 0.0)

    total = max(int(total), 0)
    cap = (np.full(n, total, dtype=np.int64) if capacity is None
           else np.asarray(capacity, dtype=np.int64))
    cap = np.maximum(cap, 0)

    # A fragment with nothing to give takes no share of the minimum either.
    floor = np.where(cap > 0, minimum, 0).astype(np.int64)
    floor = np.minimum(floor, cap)

    if total <= int(floor.sum()):
        # Not enough budget for everyone's minimum. The total is the hard bound
        # -- the cost guarantee rests on `sum(t) <= total`, so handing out the
        # floor regardless would break it: `total=5, minimum=10` over two
        # fragments used to return `[5, 5]`, twice the budget it was given.
        #
        # So spend exactly `total`, largest fracture surface first, and let the
        # smallest fragments get nothing. They lose their cross-fragment edges,
        # which is a real cost -- but it is the honest consequence of a budget
        # too small for the scene, and the fix is a larger `total`, not a
        # quietly overspent one.
        counts = np.zeros(n, np.int64)
        order = np.lexsort((np.arange(n), -np.where(cap > 0, weights, -np.inf)))
        spare = total
        for i in order:
            if spare <= 0:
                break
            take = min(spare, int(floor[i]))
            counts[i] = take
            spare -= take
        return counts

    positive = np.where(cap > 0, np.maximum(weights, 0.0), 0.0)
    if positive.sum() <= 0:
        positive = (cap > 0).astype(np.float64)
    if positive.sum() <= 0:
        return np.zeros(n, np.int64)

    # Quantise the share itself, not just its fractional part. A weight computed
    # from geometry carries float noise around 1e-16, and shares land on exact
    # integers more often than intuition suggests -- fragments with radii in a
    # simple ratio give areas in a simple ratio. `floor(3.9999999999)` is 3 while
    # `floor(4.0000000001)` is 4, so without this the allocation flips between
    # two copies of the same scene and the inter-fragment graph changes with
    # pose, which is precisely what this module exists to prevent. 9 places is
    # orders of magnitude coarser than the noise and finer than any real
    # difference in fragment size.
    share = np.round(positive / positive.sum() * total, 9)
    counts = np.maximum(floor, np.floor(share)).astype(np.int64)
    counts = np.minimum(counts, cap)

    spare = total - int(counts.sum())
    if spare > 0:
        # Hand the remainder out by largest fractional part, skipping anyone
        # already at capacity, and repeating until nobody can take more.
        #
        # The fractional parts are rounded first, and ties broken by index. A
        # weight computed from geometry carries float noise around 1e-14; without
        # the rounding, that noise reorders this list and two copies of the same
        # scene get different token counts -- an inter-fragment graph that
        # changes with pose, which is precisely what this module exists to
        # prevent. Rounding to 9 places is far coarser than the noise and far
        # finer than any real difference in fragment size.
        fractional = np.round(share - np.floor(share), 9)
        order = np.lexsort((np.arange(n), -fractional))
        progress = True
        while spare > 0 and progress:
            progress = False
            for i in order:
                if spare == 0:
                    break
                room = int(cap[i] - counts[i])
                if room > 0:
                    take = min(spare, room)
                    counts[i] += take
                    spare -= take
                    progress = True
    elif spare < 0:
        order = np.argsort(-counts)
        for i in order:
            if spare == 0:
                break
            take = min(-spare, int(counts[i] - floor[i]))
            counts[i] -= take
            spare += take
    return counts


def cross_fragment_pairs(patch_counts) -> int:
    """
    Undirected cross-fragment attention pairs for one scene, given each
    fragment's patch count. The quantity ``scripts.analyze_graph_cost``
    measures, in the unit it reports.
    """
    counts = np.asarray(patch_counts, dtype=np.float64)
    return int(0.5 * (counts.sum() ** 2 - (counts ** 2).sum()))


def _as_vertices_faces(fragment):
    """Accept either a mesh-like object or a plain ``(vertices, faces)`` pair."""
    if hasattr(fragment, "vertices") and hasattr(fragment, "faces"):
        vertices, faces = fragment.vertices, fragment.faces
    else:
        vertices, faces = fragment
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces)


def fracture_surface_area(vertices, faces, vertex_mask) -> float:
    """
    Total area of the faces lying wholly inside ``vertex_mask``.

    The right weight for splitting a scene budget. Vertex *counts* measure
    tessellation density as much as surface size, so a finely meshed chip would
    be over-served at the expense of a coarsely meshed body with far more actual
    fracture surface. GARF weights its sampling by surface area for exactly this
    reason -- it is what makes point density comparable across fragments.

    Area is rotation- and translation-invariant mathematically. Computing it in
    float64 is *almost* invariant: a fragment sitting far from the origin has
    already lost low-order bits when its coordinates were stored, so a sphere
    translated by 2000 units reports an area differing in the 14th digit.
    Centring before differencing does **not** fix this -- the precision is gone
    at storage time, not at subtraction -- which is why this function does not
    bother trying.

    That residue is harmless as an area, but it feeds the budget weights, where a
    hair's difference could flip which fragment receives a remainder token and so
    change the graph between two copies of one scene. :func:`allocate_budget`
    quantises the weights for exactly that reason. In the real pipeline the
    fragments are centred by ``normalize_fragments`` before any of this runs, so
    the residue is small anyway.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    mask = np.asarray(vertex_mask, dtype=bool)
    if faces.size == 0:
        return 0.0
    keep = mask[faces].all(axis=1)
    if not keep.any():
        return 0.0
    tri = vertices[faces[keep]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def scene_tokens(
    fragments,
    vertex_masks,
    total: int,
    *,
    max_per_fragment: int | None = None,
    mode: str = "sample",
    weights=None,
    minimum: int = 1,
    eligible_masks=None,
    metric: str = DEFAULT_METRIC,
):
    """
    Split one per-scene token budget across a scene's fragments and build the
    tokens, in one call.

    This is the operation the loader actually needs, and the reason it exists as
    a function: ``allocate_budget`` and ``fracture_patches`` are separately
    useful but chaining them by hand invites getting the per-fragment share
    wrong, which fails silently as a wrong-sized graph.

    ``weights`` defaults to each fragment's **fracture-surface area** rather than
    its fracture-vertex count -- see :func:`fracture_surface_area`.

    One limit, not two: why there is no per-fragment cap
    ----------------------------------------------------
    ``total`` alone bounds the cost. For a budget ``T`` split into ``t_i``, the
    undirected cross-fragment pair count is

        (T^2 - sum_i t_i^2) / 2   <   T^2 / 2

    for *any* split, so ``total=2048`` can never exceed ~2.1 million pairs --
    0.03x GARF's six-layer stack -- however lopsided the allocation. A
    per-fragment cap does not tighten that ceiling by a single pair.

    What a cap *does* do is destroy the proportionality this function exists to
    provide. Take a head statue with both ears, the nose and a piece of hair
    broken off. The head carries four mating surfaces and each small piece
    carries one, so weighting by fracture-surface area gives:

    ======  ==================  ==================
    piece   proportional        after a 128 cap
    ======  ==================  ==================
    head            1024                     128
    ear              256                     128
    ear              256                     128
    nose             256                     128
    hair             256                     128
    ======  ==================  ==================

    The head has four times the fracture surface of an ear and ends up
    describing it with the same number of tokens. That is exactly backwards:
    the piece with the most interface to match gets the least resolution per
    unit of it.

    An earlier version of this docstring argued that a cap was needed because
    "a per-scene budget alone starves the tail -- 256 tokens across 99 fragments
    is 2 each". The observation is true and the conclusion does not follow: a
    cap cannot cure starvation, since capping at 128 when every fragment already
    receives 20 changes nothing. The fix for a starved tail is a larger
    ``total``, which is why the default is 2048 rather than 256.

    ``max_per_fragment`` is kept as an option, for an ablation or for a
    deployment that needs a hard per-fragment bound on memory. It defaults to
    ``None`` and should stay there without a specific reason.

    Why the sampling is per fragment, never global
    ----------------------------------------------
    An obvious-looking alternative is to pool every fracture vertex in the scene
    and run one farthest-point pass over the union. **That would be wrong here.**
    At input time the fragments are in arbitrary, randomly perturbed poses, so a
    global pass would select points according to where the pieces happen to have
    been thrown -- a different sample for every perturbation of the same scene,
    and no equivariance left. Allocating a quota per fragment and sampling
    *within* each one keeps every choice a function of that fragment's own shape,
    which is pose-invariant.

    Returns one ``PatchAssignment`` per fragment, in input order.
    """
    pairs = [_as_vertices_faces(f) for f in fragments]
    n = len(pairs)
    if n == 0:
        return []

    if weights is None:
        weights = [fracture_surface_area(v, f, m)
                   for (v, f), m in zip(pairs, vertex_masks)]
    weights = np.asarray(weights, dtype=np.float64)

    # Capacity is how many vertices each fragment can actually offer, so the
    # budget is never spent on a fragment that has nothing to sample.
    pools = vertex_masks if eligible_masks is None else eligible_masks
    capacity = [int(np.asarray(m, dtype=bool).sum()) for m in pools]
    if max_per_fragment is not None:
        capacity = [min(c, int(max_per_fragment)) for c in capacity]
    counts = allocate_budget(weights, total, minimum=minimum, capacity=capacity)
    return [
        fracture_patches(v, f, m, mode=mode, max_patches=int(k), metric=metric,
                         eligible_mask=(None if eligible_masks is None else eligible_masks[i]))
        for i, ((v, f), m, k) in enumerate(zip(pairs, vertex_masks, counts))
    ]
