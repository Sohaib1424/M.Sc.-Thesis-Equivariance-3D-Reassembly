"""Harder interface-survival test: oblique fracture plane, asymmetric fragment
extents (so per-fragment grids are genuinely offset), and non-interface
vertices sharing cells with interface vertices (so cell means really differ).
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from reassembly.data.decimate import decimate_arrays, suggested_correspondence_tol
from scipy.spatial import cKDTree
rng = np.random.default_rng(11)

# --- build an oblique shared fracture surface, duplicated into both fragments -
n = 26
u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij')
e1 = np.array([0.9, 0.3, -0.2]); e2 = np.array([-0.15, 0.8, 0.55])   # oblique basis
origin_pt = np.array([0.137, -0.291, 0.443])                          # off-grid offset
interface = origin_pt + u.ravel()[:, None]*e1 + v.ravel()[:, None]*e2
nrm = np.cross(e1, e2); nrm /= np.linalg.norm(nrm)

def make_fragment(side, extra_pts, jitter=0.0):
    """Interface surface (shared, identical coords) + a bulk of other vertices."""
    bulk = interface + side * nrm * rng.uniform(0.05, 1.3, size=(len(interface), 1))
    bulk = bulk[rng.random(len(bulk)) < extra_pts]
    V = np.concatenate([interface, bulk], axis=0)
    if jitter: V = V + rng.normal(0, jitter, V.shape)
    # arbitrary triangulation just so faces exist; validity is checked elsewhere
    idx = np.arange(len(V)); rng.shuffle(idx)
    F = np.stack([idx[:-2], idx[1:-1], idx[2:]], axis=1)
    F = F[(F[:,0]!=F[:,1])&(F[:,1]!=F[:,2])&(F[:,0]!=F[:,2])]
    return V, F

VA, FA = make_fragment(+1, 0.9)     # fragment A is bulky
VB, FB = make_fragment(-1, 0.35)    # fragment B is a thin shard -> very different bbox
n_iface = len(interface)
print(f"fragment A: {len(VA)} verts   fragment B: {len(VB)} verts   shared interface: {n_iface} verts")
print(f"A bbox corner {VA.min(0).round(3)}   B bbox corner {VB.min(0).round(3)}   (differ -> grids offset)\n")

shared_origin = np.minimum(VA.min(0), VB.min(0))
print(f"{'voxel':>7} | {'shared grid: max twin gap':>26} {'matched?':>9} | {'per-frag grid: max gap':>23} {'matched?':>9}")
print("-"*90)
for h in [0.05, 0.1, 0.2, 0.35]:
    tol = suggested_correspondence_tol(h)

    # shared grid
    aV,_,amap = decimate_arrays(VA, FA, shared_origin, h)
    bV,_,bmap = decimate_arrays(VB, FB, shared_origin, h)
    # follow the ORIGINAL interface vertices through the map into the new meshes
    ai = amap[:n_iface]; bi = bmap[:n_iface]
    live = (ai >= 0) & (bi >= 0)
    gap_shared = np.linalg.norm(aV[ai[live]] - bV[bi[live]], axis=1).max() if live.any() else np.inf

    # per-fragment grids
    aV2,_,amap2 = decimate_arrays(VA, FA, VA.min(0), h)
    bV2,_,bmap2 = decimate_arrays(VB, FB, VB.min(0), h)
    ai2 = amap2[:n_iface]; bi2 = bmap2[:n_iface]
    live2 = (ai2 >= 0) & (bi2 >= 0)
    gap_per = np.linalg.norm(aV2[ai2[live2]] - bV2[bi2[live2]], axis=1).max() if live2.any() else np.inf

    print(f"{h:>7} | {gap_shared:>26.5f} {'YES' if gap_shared<=tol else 'NO':>9} | "
          f"{gap_per:>23.5f} {'yes' if gap_per<=tol else 'NO':>9}    (tol={tol:.3f})")

# --- what fraction of true twins would correspondence actually recover? -------
print("\nrecovered-twin fraction (KD-tree match within tol, as correspondence.py does):")
for h in [0.1, 0.2]:
    tol = suggested_correspondence_tol(h)
    for label, oA, oB in [("shared    ", shared_origin, shared_origin),
                          ("per-frag  ", VA.min(0),     VB.min(0))]:
        aV,_,amap = decimate_arrays(VA, FA, oA, h)
        bV,_,bmap = decimate_arrays(VB, FB, oB, h)
        ai = np.unique(amap[:n_iface][amap[:n_iface] >= 0])
        bi = np.unique(bmap[:n_iface][bmap[:n_iface] >= 0])
        if len(ai)==0 or len(bi)==0: print(f"  voxel={h} {label}: no interface left"); continue
        d,_ = cKDTree(bV[bi]).query(aV[ai])
        print(f"  voxel={h} {label}: {100*(d<=tol).mean():5.1f}% of A's interface reps find a B twin within tol")
