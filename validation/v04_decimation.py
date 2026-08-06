"""Voxel-cluster decimation: validity, monotonicity, and interface survival.

decimate_arrays is pure numpy, so this is a real execution test.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from reassembly.data.decimate import decimate_arrays, voxel_cluster_ids, suggested_correspondence_tol

rng = np.random.default_rng(3)

def icosphere_like(n_theta=40, n_phi=80, radius=1.0, centre=(0,0,0)):
    """A UV sphere -- a closed manifold mesh with plenty of vertices."""
    th = np.linspace(0, np.pi, n_theta); ph = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij')
    V = np.stack([np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)], -1).reshape(-1,3)*radius
    V = V + np.asarray(centre)
    F = []
    for i in range(n_theta-1):
        for j in range(n_phi):
            a = i*n_phi + j; b = i*n_phi + (j+1)%n_phi
            c = (i+1)*n_phi + j; d = (i+1)*n_phi + (j+1)%n_phi
            F += [[a,b,c],[b,d,c]]
    return V, np.array(F)

def check_valid(V, F, name):
    problems = []
    if len(F):
        if F.min() < 0 or F.max() >= len(V): problems.append("face index out of range")
        deg = (F[:,0]==F[:,1])|(F[:,1]==F[:,2])|(F[:,0]==F[:,2])
        if deg.any(): problems.append(f"{deg.sum()} degenerate faces")
        keyed = np.sort(F, axis=1)
        if len(np.unique(keyed, axis=0)) != len(F): problems.append("duplicate faces")
        if len(np.unique(F)) != len(V): problems.append("unreferenced vertices")
    if not np.isfinite(V).all(): problems.append("non-finite vertices")
    print(f"  {name:34s} V={len(V):6d} F={len(F):6d}  {'OK' if not problems else 'FAIL: '+'; '.join(problems)}")
    return not problems

# ---- 1. validity across a range of voxel sizes ------------------------------
V, F = icosphere_like()
origin = V.min(0)
print(f"[1] validity of decimated output (source: V={len(V)}, F={len(F)})")
all_ok = True
counts = []
for h in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
    nv, nf, vmap = decimate_arrays(V, F, origin, h)
    all_ok &= check_valid(nv, nf, f"voxel={h}")
    counts.append(len(nv))
    assert vmap.shape == (len(V),)
    live = vmap >= 0
    assert vmap[live].max() < len(nv), "vertex_map points outside new vertices"

# ---- 2. monotonicity --------------------------------------------------------
mono = all(counts[i] >= counts[i+1] for i in range(len(counts)-1))
print(f"[2] vertex count non-increasing in voxel size: {counts}  {'OK' if mono else 'FAIL'}")

# ---- 3. shape preservation (Hausdorff-ish) ----------------------------------
from scipy.spatial import cKDTree
print("[3] geometric error vs voxel size (max distance from original vertex to nearest survivor)")
for h in [0.05, 0.1, 0.2]:
    nv, nf, _ = decimate_arrays(V, F, origin, h)
    d, _ = cKDTree(nv).query(V)
    print(f"      voxel={h:<5} -> {len(nv):5d} verts, max err={d.max():.4f}, mean err={d.mean():.4f}  "
          f"(voxel diag={h*np.sqrt(3):.4f})")

# ---- 4. THE critical one: does a shared interface survive? ------------------
# Two boxes meeting exactly at x=0; the shared face's vertices are duplicated,
# one copy per fragment, at identical coordinates -- exactly like a real fracture.
def box(x0, x1, n=14):
    g = np.linspace(0, 1, n)
    pts, faces = [], []
    # only the two x-faces matter for this test; make a dense grid on each
    for x in (x0, x1):
        Y, Z = np.meshgrid(g, g, indexing='ij')
        base = len(pts)
        pts += list(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], -1))
        for i in range(n-1):
            for j in range(n-1):
                a=base+i*n+j; b=base+i*n+j+1; c=base+(i+1)*n+j; d=base+(i+1)*n+j+1
                faces += [[a,b,c],[b,d,c]]
    return np.array(pts), np.array(faces)

VA, FA = box(-1.0, 0.0)     # fragment A: its x=0 face is the fracture surface
VB, FB = box( 0.0, 1.0)     # fragment B: its x=0 face is the same surface

shared_origin = np.minimum(VA.min(0), VB.min(0))
print("[4] cross-fragment interface survival, SHARED grid vs PER-FRAGMENT grids")
for h in [0.08, 0.15, 0.3]:
    # shared grid (what decimate_scene does)
    nvA, _, _ = decimate_arrays(VA, FA, shared_origin, h)
    nvB, _, _ = decimate_arrays(VB, FB, shared_origin, h)
    iA = nvA[np.abs(nvA[:,0]) < 1e-9]; iB = nvB[np.abs(nvB[:,0]) < 1e-9]
    d_shared = cKDTree(iB).query(iA)[0].max() if len(iA) and len(iB) else np.inf

    # per-fragment grids (the naive thing -- each anchored at its own bbox corner)
    pvA, _, _ = decimate_arrays(VA, FA, VA.min(0), h)
    pvB, _, _ = decimate_arrays(VB, FB, VB.min(0), h)
    jA = pvA[np.abs(pvA[:,0]) < 1e-9]; jB = pvB[np.abs(pvB[:,0]) < 1e-9]
    d_per = cKDTree(jB).query(jA)[0].max() if len(jA) and len(jB) else np.inf

    tol = suggested_correspondence_tol(h)
    print(f"      voxel={h:<5} shared-grid max twin gap={d_shared:.4f} (tol={tol:.3f}) -> "
          f"{'MATCHED' if d_shared <= tol else 'LOST'}   |   per-fragment grid gap={d_per:.4f} -> "
          f"{'matched' if d_per <= tol else 'LOST'}")

print(f"\noverall validity: {'OK' if all_ok and mono else 'FAIL'}")
