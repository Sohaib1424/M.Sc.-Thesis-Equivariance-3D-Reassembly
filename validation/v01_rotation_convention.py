"""
Does the model's equivariance law match the law its TARGET obeys?

Model (current code): R_pred = [b1|b2|b3] from Gram-Schmidt on equivariant
vectors a1,a2. Under an input rotation Q:  a_i -> Q a_i, so R_pred -> Q R_pred.

Target: R_gt = R_diffuse^T. If the diffused input is further rotated by Q, the
effective diffusion becomes Q R_diffuse, so R_gt -> (Q R_diffuse)^T = R_gt Q^T.

Q R_pred  vs  R_pred Q^T  --  these are NOT the same map.
"""
import numpy as np
rng = np.random.default_rng(0)

def rand_rot(rng):
    A = rng.normal(size=(3, 3))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def gram_schmidt(a1, a2):
    b1 = a1 / np.linalg.norm(a1)
    a2o = a2 - (b1 @ a2) * b1
    b2 = a2o / np.linalg.norm(a2o)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)      # columns b1,b2,b3

# ---- 1. Gram-Schmidt is LEFT-equivariant: GS(Qa1,Qa2) == Q GS(a1,a2) --------
a1, a2 = rng.normal(size=3), rng.normal(size=3)
Q = rand_rot(rng)
lhs, rhs = gram_schmidt(Q @ a1, Q @ a2), Q @ gram_schmidt(a1, a2)
print(f"[1] GS(Qa) == Q GS(a)                      max|diff| = {np.abs(lhs-rhs).max():.2e}")

# ---- 2. The TARGET's transformation law ------------------------------------
R_d = rand_rot(rng)                 # diffusion applied to the clean fragment
X_clean = rng.normal(size=(40, 3)); X_clean -= X_clean.mean(0)
X_diff  = X_clean @ R_d.T           # rows are points: x' = R_d x

R_gt  = R_d.T
print(f"[2] R_gt recovers clean geometry           max|diff| = "
      f"{np.abs(X_diff @ R_gt.T - X_clean).max():.2e}")

R_d2   = Q @ R_d                    # same scene, rotated further by Q
R_gt2  = R_d2.T
print(f"    R_gt(rotated) == R_gt @ Q^T            max|diff| = "
      f"{np.abs(R_gt2 - R_gt @ Q.T).max():.2e}")
print(f"    R_gt(rotated) == Q @ R_gt   (model's)  max|diff| = "
      f"{np.abs(R_gt2 - Q @ R_gt).max():.2e}   <-- MISMATCH")

# ---- 3. Fix: output the TRANSPOSE of the Gram-Schmidt frame -----------------
def head_current(a1, a2):  return gram_schmidt(a1, a2)
def head_fixed(a1, a2):    return gram_schmidt(a1, a2).T

for name, head in [("current  R_pred = F  ", head_current),
                   ("fixed    R_pred = F^T", head_fixed)]:
    Rp  = head(a1, a2)
    Rp2 = head(Q @ a1, Q @ a2)                 # network sees the rotated scene
    err = np.abs(Rp2 - Rp @ Q.T).max()         # required law: R_pred @ Q^T
    print(f"[3] {name}: obeys target law? max|diff| = {err:.2e}  "
          f"{'OK' if err < 1e-12 else 'VIOLATED'}")

# ---- 4. End-to-end: can a *perfectly canonicalizing* frame solve the task? --
# Model learns frame F with F(clean) = I. Then F(diffused) = R_d.
F_diff = R_d
print(f"[4] R_pred = F^T reproduces R_gt exactly   max|diff| = "
      f"{np.abs(F_diff.T - R_gt).max():.2e}")
print(f"    R_pred = F   reproduces R_gt exactly   max|diff| = "
      f"{np.abs(F_diff   - R_gt).max():.2e}   <-- only if R_d symmetric")
