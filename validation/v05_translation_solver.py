"""
Design doc S1.8 asks for  min_t ( E_pos + a*E_normal + b*E_collision ).

Two things to check:
 (A) E_normal = sum ||n_A + n_B||^2 does NOT depend on t at all -- normals are
     translation invariant. So it cannot be a term in a translation objective;
     it is a *matching quality* criterion. Confirm numerically.
 (B) E_pos alone is a quadratic in t with a closed-form solution: a weighted
     graph Laplacian system, solvable exactly per coordinate, unique up to a
     global translation (gauge freedom). Confirm it recovers known translations.
"""
import numpy as np
rng = np.random.default_rng(5)

# ---------- (A) E_normal is translation-invariant ----------------------------
nA = rng.normal(size=(50,3)); nB = -nA + rng.normal(0,.05,(50,3))
E_normal = lambda tA, tB: ((nA + nB)**2).sum()      # normals ignore t entirely
print(f"[A] E_normal at t=0        : {E_normal(0,0):.6f}")
print(f"    E_normal at t=(5,-3,2) : {E_normal(np.array([5,-3,2]), np.array([-1,7,0])):.6f}")
print("    -> identical: E_normal contributes NO gradient w.r.t. translation.\n")

# ---------- (B) closed-form E_pos solve --------------------------------------
def solve_translations(frag_a, frag_b, p_a, p_b, weights, num_fragments, anchor=0):
    """min_t sum_p w_p || (p_a + t[A_p]) - (p_b + t[B_p]) ||^2

    Gradient zero  ->  L t = c, with L the weighted fragment Laplacian.
    L is singular (adding a constant to every t changes nothing), so we pin one
    fragment and solve the reduced system; lstsq would also work via pinv.
    """
    L = np.zeros((num_fragments, num_fragments))
    c = np.zeros((num_fragments, 3))
    d = p_a - p_b
    for A, B, w, dp in zip(frag_a, frag_b, weights, d):
        L[A, A] += w; L[B, B] += w; L[A, B] -= w; L[B, A] -= w
        c[A] -= w * dp; c[B] += w * dp
    free = [i for i in range(num_fragments) if i != anchor]
    t = np.zeros((num_fragments, 3))
    if free:
        t[free] = np.linalg.lstsq(L[np.ix_(free, free)], c[free], rcond=None)[0]
    return t

# --- build a synthetic scene: F fragments, known ground-truth translations ---
F = 5
t_true = rng.normal(0, 2.0, size=(F,3)); t_true -= t_true[0]      # anchor frag 0
frag_a, frag_b, p_a, p_b, w = [], [], [], [], []
for A in range(F):
    for B in range(A+1, F):
        if rng.random() < 0.6:                       # not every pair touches
            continue
        m = rng.integers(8, 30)
        shared = rng.normal(size=(m,3))              # true interface location
        # fragment-local coords: shared point minus that fragment's translation
        p_a.append(shared - t_true[A]); p_b.append(shared - t_true[B])
        frag_a += [A]*m; frag_b += [B]*m; w += [1.0]*m
p_a = np.concatenate(p_a); p_b = np.concatenate(p_b)
t_hat = solve_translations(np.array(frag_a), np.array(frag_b), p_a, p_b, np.array(w), F)
print(f"[B] exact correspondences  : max|t_hat - t_true| = {np.abs(t_hat-t_true).max():.2e}")

# --- with noise on the correspondences ---------------------------------------
for sigma in [0.01, 0.05, 0.2]:
    pa_n = p_a + rng.normal(0, sigma, p_a.shape)
    pb_n = p_b + rng.normal(0, sigma, p_b.shape)
    th = solve_translations(np.array(frag_a), np.array(frag_b), pa_n, pb_n, np.array(w), F)
    print(f"    noise sigma={sigma:<5}      : max|t_hat - t_true| = {np.abs(th-t_true).max():.4f}")

# --- with OUTLIER correspondences (what mutual-NN matching actually produces) -
print("\n[C] robustness to outlier matches (plain least squares vs IRLS reweighting)")
def irls(frag_a, frag_b, p_a, p_b, F, iters=12, delta=0.05):
    """Iteratively reweighted least squares with a Huber-like weight -- the
    standard fix for a quadratic objective fed noisy correspondences."""
    w = np.ones(len(p_a))
    for _ in range(iters):
        t = solve_translations(frag_a, frag_b, p_a, p_b, w, F)
        r = np.linalg.norm((p_a + t[frag_a]) - (p_b + t[frag_b]), axis=1)
        w = 1.0 / np.maximum(r, delta)          # Huber/L1-ish
    return t

fa, fb = np.array(frag_a), np.array(frag_b)
for frac in [0.0, 0.1, 0.3]:
    pa_o, pb_o = p_a.copy(), p_b.copy()
    n_out = int(frac * len(pa_o))
    if n_out:
        idx = rng.choice(len(pa_o), n_out, replace=False)
        pb_o[idx] += rng.normal(0, 3.0, (n_out,3))       # wrong matches
    t_ls = solve_translations(fa, fb, pa_o, pb_o, np.ones(len(pa_o)), F)
    t_ir = irls(fa, fb, pa_o, pb_o, F)
    print(f"    {int(frac*100):3d}% outliers : plain LS err={np.abs(t_ls-t_true).max():7.4f}   "
          f"IRLS err={np.abs(t_ir-t_true).max():7.4f}")
