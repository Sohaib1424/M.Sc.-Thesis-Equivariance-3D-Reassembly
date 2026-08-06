"""
Exercise the real data path on one scene directory, using the project's actual
numpy code.

    python validation/run_real_scene.py /path/to/<scene_hash>

`load_scene`'s three igl calls are replicated in numpy (remove_unreferenced,
remove_duplicate_vertices) and a minimal mesh stand-in is supplied, so this
runs without trimesh or igl. Everything else -- resolve_duplicated_faces,
decimate_scene, find_shared_points -- is imported from the project and runs
unmodified.

If trimesh and torch are available, scripts/smoke_test.py covers this ground
and continues through the model and a backward pass. See REAL_SCENE.md.
"""
import sys, os, glob, collections
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

# A minimal functional stand-in for trimesh.Trimesh: decimate_scene only ever
# reads .vertices/.faces and constructs new ones, so this exercises the real
# code without the real dependency.
import types
_tm = types.ModuleType("trimesh")
class _Mesh:
    def __init__(self, vertices=None, faces=None, **kw):
        self.vertices = np.asarray(vertices); self.faces = np.asarray(faces)
_tm.Trimesh = _Mesh
sys.modules.setdefault("trimesh", _tm)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reassembly.data.scene_io import resolve_duplicated_faces
from reassembly.data.decimate import decimate_scene, suggested_correspondence_tol
from reassembly.data.correspondence import find_shared_points

if len(sys.argv) > 1:
    SCENE = sys.argv[1]
else:
    raise SystemExit(
        "usage: python run_real_scene.py /path/to/<scene_hash>\n"
        "  expects compressed_mesh.obj, compressed_data.npz, and one or more\n"
        "  <fracture>/compressed_fracture.npy inside that directory."
    )

for _required in ("compressed_mesh.obj", "compressed_data.npz"):
    if not os.path.exists(os.path.join(SCENE, _required)):
        raise SystemExit(f"{SCENE!r} has no {_required} -- is this a scene directory?")


def read_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line[2:].split()])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line[2:].split()])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def remove_unreferenced(V, F):
    """numpy equivalent of igl.remove_unreferenced."""
    if len(F) == 0:
        return V[:0], F, np.full(len(V), -1, dtype=np.int64)
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F], remap


def remove_duplicate_vertices(V, F, eps=1e-10):
    """numpy equivalent of igl.remove_duplicate_vertices."""
    if len(V) == 0:
        return V, F, np.empty(0, dtype=np.int64)
    if eps > 0:
        key = np.round(V / eps) * eps
    else:
        key = V
    _uniq, first, inverse = np.unique(key, axis=0, return_index=True,
                                      return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return V[np.sort(first)], np.argsort(np.argsort(first))[inverse], inverse


def load_scene_numpy(scene_dir, fracture_id):
    """Faithful replica of scene_io.load_scene, minus trimesh/igl."""
    V, F = read_obj(os.path.join(scene_dir, "compressed_mesh.obj"))
    piece_to_fine = load_npz(os.path.join(scene_dir, "compressed_data.npz"))
    piece_labels = np.load(os.path.join(scene_dir, fracture_id,
                                        "compressed_fracture.npy"))

    fine_vertex_labels = piece_to_fine @ piece_labels
    n_pieces = int(np.max(piece_labels) + 1)
    tri_labels = fine_vertex_labels[F[:, 0]]

    frags, dropped = [], 0
    for i in range(n_pieces):
        sel = tri_labels == i
        if not np.any(sel):
            dropped += 1
            continue
        vi, fi, _ = remove_unreferenced(V, F[sel, :])
        ui, gi_map, _ = remove_duplicate_vertices(vi, fi, 1e-10)
        gi = gi_map[fi] if gi_map.ndim == 1 and len(gi_map) == len(vi) else fi
        ffi, _ = resolve_duplicated_faces(gi)
        nv, nf, _ = remove_unreferenced(ui, ffi)
        if len(nf) == 0 or len(nv) == 0:
            dropped += 1
            continue
        frags.append((nv, nf))
    return frags, n_pieces, dropped, piece_labels


def report(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- inventory
report("STEP 0: what the scene directory actually contains")
subdirs = sorted(d for d in os.listdir(SCENE) if os.path.isdir(os.path.join(SCENE, d)))
fractured = [d for d in subdirs if d.startswith("fractured_")]
modes = [d for d in subdirs if d.startswith("mode_")]
print(f"  subdirectories load_scene would choose from : {len(subdirs)}")
print(f"    fractured_* : {len(fractured)}")
print(f"    mode_*      : {len(modes)}")

dtypes, piececounts = collections.Counter(), {}
for d in subdirs:
    a = np.load(os.path.join(SCENE, d, "compressed_fracture.npy"))
    dtypes[(d.split("_")[0], str(a.dtype))] += 1
    piececounts[d] = int(a.max()) + 1
print(f"  label dtypes: {dict(dtypes)}")

singles = [d for d, n in piececounts.items() if n < 2]
print(f"  fractures yielding FEWER THAN 2 pieces: {len(singles)}  {singles}")

# ------------------------------------------------------------ load fragments
report("STEP 1: loading fragments through the real code path")
_sample = ([d for d in subdirs if d.startswith("fractured_")][:2] +
           [d for d in subdirs if d.startswith("mode_")][:2]) or subdirs[:4]
for frac in _sample:
    frags, n_pieces, dropped, labels = load_scene_numpy(SCENE, frac)
    nv = sum(len(v) for v, _ in frags)
    nf = sum(len(f) for _, f in frags)
    sizes = sorted(len(v) for v, _ in frags)
    print(f"  {frac:14s} labels->{n_pieces:3d} pieces | meshes built {len(frags):3d} "
          f"| dropped {dropped:2d} | verts {nv:6,} faces {nf:6,}")
    print(f"  {'':14s} fragment vertex counts: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")

# ------------------------------------------------- interface correspondences
report("STEP 2: do fragments actually share interface vertices?")
print("  (this is what the embedding losses are supervised by -- if it is zero,")
print("   two of the seven loss terms contribute nothing)\n")
_by_pieces = sorted(subdirs, key=lambda d: piececounts[d])
_probe = [d for d in (_by_pieces[0], _by_pieces[len(_by_pieces)//2],
                      _by_pieces[-1]) if piececounts[d] >= 2]
for frac in dict.fromkeys(_probe):
    frags, n_pieces, _, _ = load_scene_numpy(SCENE, frac)
    verts = [v for v, _ in frags]
    total = sum(len(v) for v in verts)
    clusters = find_shared_points(verts, tol=1e-5)
    shared = sum(len(c) for c in clusters)
    pairs = set()
    for c in clusters:
        fs = sorted({f for f, _ in c})
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                pairs.add((fs[a], fs[b]))
    print(f"  {frac:14s} {len(frags):3d} frags, {total:6,} verts -> "
          f"{len(clusters):5,} clusters covering {shared:6,} verts "
          f"({100*shared/max(total,1):5.1f}%), {len(pairs)} touching pairs")

# ----------------------------------------------------------- decimation path
report("STEP 3: decimation on the real scene")
for frac in dict.fromkeys(d for d in (_by_pieces[-1], _by_pieces[len(_by_pieces)//2])
                          if piececounts[d] >= 2):
    frags, _, _, _ = load_scene_numpy(SCENE, frac)
    meshes = [_Mesh(v, f) for v, f in frags]
    before = sum(len(m.vertices) for m in meshes)
    base = find_shared_points([m.vertices for m in meshes], tol=1e-5)
    print(f"\n  {frac}: {len(meshes)} fragments, {before:,} verts, "
          f"{len(base):,} interface clusters undecimated")
    for target in (8000, 4000, 2000):
        dm, info = decimate_scene(meshes, target_vertices=target)
        after = sum(len(m.vertices) for m in dm)
        tol = suggested_correspondence_tol(info.voxel_size) if info.applied else 1e-5
        cl = find_shared_points([m.vertices for m in dm], tol=tol)
        kept = 100 * len(cl) / max(len(base), 1)
        assert len(dm) == len(meshes), "FRAGMENT COUNT CHANGED"
        print(f"    target {target:5,} -> {after:5,} verts "
              f"(voxel {info.voxel_size:.5f}, tol {tol:.2e}) | "
              f"{len(cl):5,} clusters = {kept:5.1f}% of undecimated | "
              f"fragments {len(dm)} (unchanged)")
