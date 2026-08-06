"""End-to-end stage-2 test on synthetic data, using the REAL modules
(assembly/ and evaluation/ are pure numpy+scipy, so this executes for real).

Builds a scene with known R_gt and t_gt, fakes an embedding that encodes true
interface identity plus noise, and checks the whole
match -> solve -> refine -> evaluate chain.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from reassembly.assembly.matching import match_scene, mutual_nearest_neighbors, correspondence_components
from reassembly.assembly.translation import solve_translations, assemble
from reassembly.evaluation.metrics import (evaluate_assembly, rotation_error_euler_deg,
                                           rotation_error_geodesic_deg, chamfer_distance)

rng = np.random.default_rng(21)
def rand_rot(r):
    U,_,Vt = np.linalg.svd(r.normal(size=(3,3))); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R

# ---------------------------------------------------------------- build scene
F = 4
D = 16                       # embedding dim
n_iface, n_bulk = 60, 200

# shared interface locations, one set per touching fragment pair
pairs = [(0,1),(1,2),(2,3),(0,2)]
iface_world = {p: rng.normal(size=(n_iface,3)) for p in pairs}
iface_id    = {p: rng.normal(size=(n_iface,D)) for p in pairs}   # "true" embedding

t_gt = rng.normal(0, 1.5, size=(F,3)); t_gt -= t_gt[0]
R_gt = np.stack([rand_rot(rng) for _ in range(F)])

verts_world, emb, nrm_world = [], [], []
for f in range(F):
    pts, es, ns = [], [], []
    for p in pairs:
        if f not in p: continue
        pts.append(iface_world[p])
        es.append(iface_id[p] + rng.normal(0, 0.05, (n_iface, D)))   # noisy embedding
        # facing normals: opposite for the two sides of the interface
        base = rng.normal(size=(n_iface,3)); base/=np.linalg.norm(base,axis=1,keepdims=True)
        ns.append(base if f == p[0] else -base)
    pts.append(rng.normal(0,3,(n_bulk,3)))                    # non-interface bulk
    es.append(rng.normal(0,3,(n_bulk,D)))
    b = rng.normal(size=(n_bulk,3)); ns.append(b/np.linalg.norm(b,axis=1,keepdims=True))
    verts_world.append(np.concatenate(pts)); emb.append(np.concatenate(es))
    nrm_world.append(np.concatenate(ns))

# fragment-local (centralized) coords: world = R_gt @ local + t_gt  => local = R^T (world - t)
verts_local = [ (verts_world[f] - t_gt[f]) @ R_gt[f] for f in range(F) ]
nrm_local   = [ nrm_world[f] @ R_gt[f] for f in range(F) ]

print(f"scene: {F} fragments, {sum(len(v) for v in verts_local)} points, "
      f"{len(pairs)} touching pairs")

# ------------------------------------------------- 1. matching + solve, PERFECT R
out = assemble(R_gt, verts_local, nrm_local, emb,
               match_kwargs=dict(ratio_threshold=0.9, normal_opposition=-0.3),
               solve_kwargs=dict(irls_iterations=15))
res = out["result"]
err = np.abs(out["translations"] - t_gt).max()
print(f"\n[1] perfect rotations -> matches={len(out['matches'])}, components={res.num_components}, "
      f"residual_rms={res.residual_rms:.2e}")
print(f"    max|t_hat - t_gt| = {err:.2e}   {'OK' if err < 1e-6 else 'FAIL'}")

# ------------------------------------------------- 2. imperfect rotations
print("\n[2] degrading the rotations and re-solving:")
for deg in [0.0, 1.0, 5.0, 15.0]:
    Rp = []
    for f in range(F):
        ax = rng.normal(size=3); ax/=np.linalg.norm(ax); th=np.deg2rad(deg)
        K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        Rp.append((np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K) @ R_gt[f])
    Rp = np.stack(Rp)
    o = assemble(Rp, verts_local, nrm_local, emb,
                 match_kwargs=dict(ratio_threshold=0.9, normal_opposition=-0.3))
    m = evaluate_assembly(Rp, R_gt, o["translations"], t_gt, verts_local,
                          pa_threshold=0.05, points_per_part=400)
    print(f"    rot perturbation {deg:5.1f} deg -> {m}")

# ------------------------------------------------- 3. metric sanity checks
print("\n[3] metric sanity")
m_perfect = evaluate_assembly(R_gt, R_gt, t_gt, t_gt, verts_local, points_per_part=300)
print(f"    identical assembly: RMSE_R={m_perfect.rmse_rotation_euler_deg:.2e} "
      f"RMSE_T={m_perfect.rmse_translation:.2e} PA={m_perfect.part_accuracy:.2f} "
      f"CD={m_perfect.chamfer_distance:.2e}  (expect ~0, PA=1)")

# euler vs geodesic really do differ
Rp = np.stack([rand_rot(rng) for _ in range(F)])
e_eu = np.sqrt(np.mean(rotation_error_euler_deg(Rp, R_gt)**2))
e_ge = np.sqrt(np.mean(rotation_error_geodesic_deg(Rp, R_gt)**2))
print(f"    random rotations: Euler-RMSE={e_eu:.2f} deg vs geodesic-RMSE={e_ge:.2f} deg "
      f"-> ratio {e_eu/e_ge:.2f}  (they are NOT interchangeable)")

# wraparound handled
Rz = lambda a: np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
w = rotation_error_euler_deg(Rz(np.deg2rad(1))[None], Rz(np.deg2rad(-1))[None])
print(f"    +1 deg vs -1 deg about z -> euler err {w.max():.4f} deg (expect 2, not 358)")

# ------------------------------------------------- 4. disconnected graph detection
print("\n[4] disconnected correspondence graph is DETECTED, not silently wrong")
emb_iso = [e.copy() for e in emb]
emb_iso[3] = rng.normal(0, 50, emb_iso[3].shape)      # fragment 3 matches nothing
o = assemble(R_gt, verts_local, nrm_local, emb_iso)
r = o["result"]
print(f"    components={r.num_components} fully_constrained={r.fully_constrained} "
      f"(fragment 3 isolated: {list(r.components)})")
