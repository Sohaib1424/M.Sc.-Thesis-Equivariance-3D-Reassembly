r"""
Cross-section of a fragment for the fracture-labelling figure: a closed polygon
whose upper part is the smooth intact surface and whose lower part is a jagged
fracture surface. Each segment plays the role of a face, consecutive segments
are adjacent faces. Applies the dihedral rule exactly as the pipeline does
(|m_f . m_g| < 0.9 marks a sharp adjacency; a face is labelled when floor(k/2)
<= sigma, i.e. with k = 2 neighbours here, when it has a sharp neighbour) and
writes figures/src/dihedral.tex.
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"   # figures/src

rng = np.random.default_rng(3)
pts = []
# smooth intact surface: an arc from left to right over the top
for t in np.linspace(np.pi * 0.98, np.pi * 0.02, 17):
    pts.append([2.3 * np.cos(t), 1.25 * np.sin(t) + 0.05])
# jagged fracture surface back from right to left along the bottom
x_right, x_left = pts[-1][0], pts[0][0]
n_j = 11
for k in range(1, n_j):
    x = x_right + (x_left - x_right) * k / n_j
    y = -0.05 + rng.uniform(-0.33, 0.33) - 0.15 * np.sin(np.pi * k / n_j)
    pts.append([x, y])
P = np.array(pts)
n = len(P)
seg = [(P[i], P[(i + 1) % n]) for i in range(n)]


def normal(a, b):
    d = b - a
    m = np.array([d[1], -d[0]])          # outward for a clockwise polygon
    return m / np.linalg.norm(m)


# orientation check: make the normals point outward
area = 0.5 * sum(P[i, 0] * P[(i + 1) % n, 1] - P[(i + 1) % n, 0] * P[i, 1] for i in range(n))
sign = -1.0 if area < 0 else 1.0
M = np.array([sign * normal(a, b) for a, b in seg])
cos_next = np.array([abs(M[i] @ M[(i + 1) % n]) for i in range(n)])   # joint between seg i and i+1
sharp = cos_next < 0.9
labelled = np.array([sharp[i] or sharp[(i - 1) % n] for i in range(n)])   # k = 2: one sharp neighbour
n_smooth = 16                       # segments 0..15 are the intact arc
truth = np.array([i >= n_smooth for i in range(n)])

out = []
thin = [f"({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f})" for (a, b), l in zip(seg, labelled) if not l]
thick = [f"({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f})" for (a, b), l in zip(seg, labelled) if l]
out.append("\\def\\SecFill{" + " -- ".join(f"({p[0]:.3f},{p[1]:.3f})" for p in P) + " -- cycle}")
out.append("\\def\\SecThin{" + " ".join(thin) + "}")
out.append("\\def\\SecThick{" + " ".join(thick) + "}")
joints = [f"({P[(i + 1) % n][0]:.3f},{P[(i + 1) % n][1]:.3f})" for i in range(n) if sharp[i]]
out.append("\\def\\SecSharp{" + ", ".join(joints) + "}")
arrows = []
for i, (a, b) in enumerate(seg):
    mid = (a + b) / 2
    tip = mid + 0.42 * M[i]
    arrows.append(f"\\draw[vecarr, line width=0.7pt, -{{Stealth[length=1.5mm,width=1.1mm]}}] ({mid[0]:.3f},{mid[1]:.3f}) -- ({tip[0]:.3f},{tip[1]:.3f});")
out.append("\\def\\SecNormals{" + " ".join(arrows) + "}")
open(SRC / "dihedral.tex", "w").write("\n".join(out) + "\n")
fp = int((labelled & ~truth).sum()); fn = int((~labelled & truth).sum())
print(f"{n} faces, {int(sharp.sum())} sharp joints, {int(labelled.sum())} labelled, "
      f"{int(truth.sum())} truly fracture; false pos {fp}, false neg {fn}; cos range {cos_next.min():.2f}..{cos_next.max():.2f}")
