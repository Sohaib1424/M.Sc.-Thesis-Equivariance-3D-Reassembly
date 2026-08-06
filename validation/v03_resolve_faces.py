"""Vectorized resolve_duplicated_faces == original loop implementation?

Pure numpy on both sides, so this is a real execution test, not a prototype.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from reassembly.data.scene_io import resolve_duplicated_faces as fast


def reference(F1):
    """Verbatim transcription of the original loop-based port."""
    num_faces = F1.shape[0]
    if num_faces == 0:
        return np.empty((0, 3), dtype=F1.dtype), np.empty((0,), dtype=np.int64)
    sorted_F1 = np.sort(F1, axis=1)
    uF, IA, IC = np.unique(sorted_F1, axis=0, return_index=True, return_inverse=True)
    IC = np.asarray(IC).reshape(-1)
    num_unique_faces = uF.shape[0]
    canonical = uF[IC]
    consistent = (
        np.all(F1 == canonical, axis=1) |
        ((F1[:,0]==canonical[:,1]) & (F1[:,1]==canonical[:,2]) & (F1[:,2]==canonical[:,0])) |
        ((F1[:,0]==canonical[:,2]) & (F1[:,1]==canonical[:,0]) & (F1[:,2]==canonical[:,1]))
    )
    uF2F = [[] for _ in range(num_unique_faces)]
    counts = np.zeros(num_unique_faces, dtype=np.int64)
    ucounts = np.zeros(num_unique_faces, dtype=np.int64)
    signed_ids = (np.arange(num_faces) + 1) * np.where(consistent, 1, -1)
    for i in range(num_faces):
        ui = IC[i]
        uF2F[ui].append(signed_ids[i]); counts[ui] += 1 if consistent[i] else -1; ucounts[ui] += 1
    kept = []
    for i in range(num_unique_faces):
        if ucounts[i] == 1:
            kept.append(abs(uF2F[i][0]) - 1); continue
        if counts[i] == 1:
            for fid in uF2F[i]:
                if fid > 0: kept.append(abs(fid)-1); break
        elif counts[i] == -1:
            for fid in uF2F[i]:
                if fid < 0: kept.append(abs(fid)-1); break
        else:
            if counts[i] != 0 and len(uF2F[i]) > 0:
                kept.append(abs(uF2F[i][0]) - 1)
    J = np.array(kept, dtype=np.int64)
    return F1[J], J


rng = np.random.default_rng(7)
cases = {}

# hand-built structural cases
cases['empty']        = np.empty((0,3), dtype=np.int64)
cases['single']       = np.array([[0,1,2]])
cases['exact_dup']    = np.array([[0,1,2],[0,1,2]])                       # count=+2 -> drop
cases['reversed_dup'] = np.array([[0,1,2],[0,2,1]])                       # count=0  -> drop
cases['triplicate']   = np.array([[0,1,2],[0,1,2],[0,2,1]])               # count=+1 -> keep first pos
cases['triple_neg']   = np.array([[0,2,1],[0,2,1],[0,1,2]])               # count=-1 -> keep first neg
cases['cyclic']       = np.array([[0,1,2],[1,2,0],[2,0,1]])               # all consistent, count=+3
cases['mixed_mesh']   = np.array([[0,1,2],[1,2,3],[0,1,2],[3,2,1],[4,5,6]])

# randomized stress: small vertex pool -> many collisions, plus random flips
for t in range(400):
    nf = rng.integers(1, 40)
    F = rng.integers(0, 8, size=(nf, 3))
    F = F[(F[:,0]!=F[:,1]) & (F[:,1]!=F[:,2]) & (F[:,0]!=F[:,2])]        # drop degenerates
    if len(F) == 0: continue
    dup = rng.integers(0, len(F), size=rng.integers(0, len(F)+1))
    extra = F[dup].copy()
    flip = rng.random(len(extra)) < 0.5
    extra[flip] = extra[flip][:, [0,2,1]]
    cases[f'rand{t}'] = np.concatenate([F, extra], axis=0)

bad = 0
for name, F in cases.items():
    F2r, Jr = reference(F)
    F2f, Jf = fast(F)
    ok = np.array_equal(Jr, Jf) and np.array_equal(F2r, F2f)
    if not ok:
        bad += 1
        if bad <= 3:
            print(f"  MISMATCH {name}\n    ref J={Jr}\n    new J={Jf}\n    F=\n{F}")
print(f"resolve_duplicated_faces: {len(cases)-bad}/{len(cases)} cases bit-identical to the reference loop")

# speed
big = rng.integers(0, 3000, size=(60000, 3))
big = big[(big[:,0]!=big[:,1]) & (big[:,1]!=big[:,2]) & (big[:,0]!=big[:,2])]
import time
t0=time.perf_counter(); reference(big); t_ref=time.perf_counter()-t0
t0=time.perf_counter(); fast(big);      t_new=time.perf_counter()-t0
print(f"speed on {len(big):,} faces: loop {t_ref*1000:.0f} ms  ->  vectorized {t_new*1000:.0f} ms  ({t_ref/t_new:.0f}x)")
