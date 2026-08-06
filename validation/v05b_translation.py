import numpy as np
rng = np.random.default_rng(5)

def solve(fa, fb, pa, pb, w, F, anchor=0):
    L = np.zeros((F,F)); c = np.zeros((F,3)); d = pa - pb
    np.add.at(L, (fa,fa), w); np.add.at(L, (fb,fb), w)
    np.add.at(L, (fa,fb), -w); np.add.at(L, (fb,fa), -w)
    np.add.at(c, fa, -w[:,None]*d); np.add.at(c, fb, w[:,None]*d)
    free = [i for i in range(F) if i != anchor]
    t = np.zeros((F,3))
    if free: t[free] = np.linalg.lstsq(L[np.ix_(free,free)], c[free], rcond=None)[0]
    return t

def components(fa, fb, F):
    """Connected components of the fragment correspondence graph."""
    parent = list(range(F))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for a,b in zip(fa,fb):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    return np.array([find(i) for i in range(F)])

def build_scene(F, pair_prob, seed):
    r = np.random.default_rng(seed)
    t_true = r.normal(0,2.0,size=(F,3)); t_true -= t_true[0]
    fa,fb,pa,pb = [],[],[],[]
    for A in range(F):
        for B in range(A+1,F):
            if r.random() > pair_prob: continue
            m = r.integers(8,30); shared = r.normal(size=(m,3))
            pa.append(shared - t_true[A]); pb.append(shared - t_true[B])
            fa += [A]*m; fb += [B]*m
    if not pa: return None
    return (np.array(fa), np.array(fb), np.concatenate(pa), np.concatenate(pb), t_true)

print("[B] does the solve recover known translations, when the graph is CONNECTED?")
for pair_prob in [1.0, 0.8, 0.5, 0.3]:
    sc = build_scene(6, pair_prob, seed=12)
    if sc is None: continue
    fa,fb,pa,pb,t_true = sc
    comp = components(fa,fb,6)
    connected = len(set(comp.tolist()))==1
    t = solve(fa,fb,pa,pb,np.ones(len(pa)),6)
    print(f"  pair_prob={pair_prob:<4} components={len(set(comp.tolist()))} "
          f"{'CONNECTED' if connected else 'DISCONNECTED'}  max err={np.abs(t-t_true).max():.2e}")

print("\n  -> a fragment in its own component has NO constraint tying it to the rest;")
print("     its translation is genuinely undetermined, not badly estimated.\n")

# connected scene from here on
fa,fb,pa,pb,t_true = build_scene(6, 1.0, seed=12)
F=6
print("[C] noise on correspondences (connected graph)")
for sigma in [0.0, 0.01, 0.05, 0.2]:
    t = solve(fa,fb,pa+rng.normal(0,sigma,pa.shape),pb+rng.normal(0,sigma,pb.shape),
              np.ones(len(pa)),F)
    print(f"    sigma={sigma:<5} -> max|t_hat - t_true| = {np.abs(t-t_true).max():.5f}")

def irls(fa,fb,pa,pb,F,iters=15,delta=0.02):
    w=np.ones(len(pa))
    for _ in range(iters):
        t=solve(fa,fb,pa,pb,w,F)
        r=np.linalg.norm((pa+t[fa])-(pb+t[fb]),axis=1)
        w=1.0/np.maximum(r,delta)
    return t

print("\n[D] outlier robustness: plain least squares vs IRLS")
for frac in [0.0,0.1,0.2,0.4]:
    pb_o = pb.copy()
    n = int(frac*len(pb_o))
    if n:
        idx = rng.choice(len(pb_o), n, replace=False)
        pb_o[idx] += rng.normal(0,3.0,(n,3))
    e_ls = np.abs(solve(fa,fb,pa,pb_o,np.ones(len(pa)),F)-t_true).max()
    e_ir = np.abs(irls(fa,fb,pa,pb_o,F)-t_true).max()
    print(f"    {int(frac*100):3d}% outliers -> LS err={e_ls:7.4f}   IRLS err={e_ir:7.4f}"
          f"   ({'IRLS better' if e_ir<e_ls else 'no gain'})")

print("\n[E] normal-opposition filter: can it remove outliers BEFORE the solve?")
# true matches have opposite normals; outliers have random ones
n_true = rng.normal(size=(len(pa),3)); n_true/=np.linalg.norm(n_true,axis=1,keepdims=True)
nA, nB = n_true, -n_true + rng.normal(0,0.1,n_true.shape)
nB /= np.linalg.norm(nB,axis=1,keepdims=True)
idx = rng.choice(len(pa), int(0.3*len(pa)), replace=False)
nB[idx] = rng.normal(size=(len(idx),3)); nB[idx]/=np.linalg.norm(nB[idx],axis=1,keepdims=True)
pb_o = pb.copy(); pb_o[idx] += rng.normal(0,3.0,(len(idx),3))
keep = (nA*nB).sum(1) < -0.5           # normals roughly opposite
tp = keep[idx].sum(); print(f"    filter keeps {keep.sum()}/{len(keep)} matches; "
      f"of the {len(idx)} true outliers it wrongly keeps {tp}")
e_raw = np.abs(solve(fa,fb,pa,pb_o,np.ones(len(pa)),F)-t_true).max()
e_flt = np.abs(solve(fa[keep],fb[keep],pa[keep],pb_o[keep],np.ones(keep.sum()),F)-t_true).max()
e_both= np.abs(irls(fa[keep],fb[keep],pa[keep],pb_o[keep],F)-t_true).max()
print(f"    no filter LS err={e_raw:.4f} | normal-filtered LS err={e_flt:.4f} | "
      f"filtered + IRLS err={e_both:.4f}")
