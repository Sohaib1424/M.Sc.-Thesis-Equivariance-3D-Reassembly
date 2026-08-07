"""
Which formulation of the SO(3) geodesic angle should be used to REPORT
RMSE(R), and what it does to the gradient if used as a loss.

The original code used arccos((tr(R1^T R2) - 1) / 2) with the argument clamped
to 1 - 1e-7 to keep it finite. Two problems, both measured below.
"""
import numpy as np

def R_axis(deg, ax=(0, 0, 1.)):
    ax = np.asarray(ax, float); ax /= np.linalg.norm(ax); th = np.deg2rad(deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def acos_form(R1, R2, eps=1e-7, dt=np.float32):
    R1, R2 = R1.astype(dt), R2.astype(dt)
    D = R1.T @ R2
    return np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1 + eps, 1 - eps)))

def asin_form(R1, R2, dt=np.float32):
    R1, R2 = R1.astype(dt), R2.astype(dt)
    return np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(R1 - R2) / (2 * np.sqrt(2)), 0, 1)))

def atan2_form(R1, R2, dt=np.float32):
    """cos from the trace, sin from the skew part, combined by atan2."""
    R1, R2 = R1.astype(dt), R2.astype(dt)
    D = R1.T @ R2
    return np.degrees(np.arctan2(np.linalg.norm(D - D.T) / (2 * np.sqrt(2)),
                                 (np.trace(D) - 1) / 2))

print("[A] accuracy against known angles, float32")
print(f"{'true deg':>10} {'acos+eps':>12} {'asin':>12} {'atan2':>12}")
print("-" * 50)
for true in [0.0, 0.001, 0.01, 0.1, 1.0, 45.0, 90.0, 179.0, 179.99]:
    R, I = R_axis(true), np.eye(3)
    print(f"{true:>10.4f} {acos_form(I,R):>12.5f} {asin_form(I,R):>12.5f} {atan2_form(I,R):>12.5f}")

rng = np.random.default_rng(0)
print("\n[B] worst-case error over 0..180 degrees, random axes, float32")
for name, fn in [("acos+eps", acos_form), ("asin", asin_form), ("atan2", atan2_form)]:
    errs = [abs(fn(np.eye(3), R_axis(t, rng.normal(size=3))) - t)
            for t in np.linspace(0, 180, 400)]
    print(f"  {name:9s} max={max(errs):.6f} deg   mean={np.mean(errs):.7f}")

print("\n[C] two IDENTICAL rotations -- the metric must be able to read zero")
Rr = np.linalg.qr(rng.normal(size=(3, 3)))[0]
if np.linalg.det(Rr) < 0: Rr[:, 0] *= -1
print(f"  acos+eps : {acos_form(Rr,Rr):.6f} deg   <-- the eps clamp is a hard FLOOR")
print(f"  asin     : {asin_form(Rr,Rr):.6f} deg")
print(f"  atan2    : {atan2_form(Rr,Rr):.6f} deg")
print("  -> the floor is why test_composite_loss_is_zero_for_a_perfect_prediction failed")

print("\n[D] gradient magnitude as the prediction converges (finite differences)")
def theta_acos(R1, R2, eps=1e-7):
    D = R1.T @ R2; return np.arccos(np.clip((np.trace(D)-1)/2, -1+eps, 1-eps))
def theta_atan2(R1, R2):
    D = R1.T @ R2
    return np.arctan2(np.linalg.norm(D-D.T)/(2*np.sqrt(2)), (np.trace(D)-1)/2)
def chordal(R1, R2): return ((R1 - R2) ** 2).sum()

def gnorm(fn, R, Rgt, h=1e-6):
    g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Rp = R.copy(); Rp[i, j] += h
            Rm = R.copy(); Rm[i, j] -= h
            g[i, j] = (fn(Rp, Rgt) - fn(Rm, Rgt)) / (2 * h)
    return np.linalg.norm(g)

print(f"{'offset deg':>11} {'geodesic(acos)':>16} {'geodesic(atan2)':>17} {'chordal':>12}")
print("-" * 60)
for d in [30.0, 5.0, 1.0, 0.1, 0.01, 0.001]:
    R = R_axis(d)
    print(f"{d:>11.3f} {gnorm(theta_acos,R,np.eye(3)):>16.3f} "
          f"{gnorm(theta_atan2,R,np.eye(3)):>17.4f} {gnorm(chordal,R,np.eye(3)):>12.6f}")

print("""
CONCLUSIONS
  1. atan2 is the right way to REPORT the angle: exact to 1e-5 deg across the
     whole range, and exactly zero for identical rotations.
  2. The gradient explosion attributed to "the geodesic loss" was an artefact
     of the arccos parameterisation, not of geodesic distance. With atan2 the
     gradient is a well-behaved constant 1/sqrt(2) ~ 0.707.
  3. Chordal is still the better default LOSS, for a different reason: its
     gradient vanishes smoothly at the optimum (L2-like), while the geodesic
     angle's stays constant with a kink at zero (L1-like).
""")
