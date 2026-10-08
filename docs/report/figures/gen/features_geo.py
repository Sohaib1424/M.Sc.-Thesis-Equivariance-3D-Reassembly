r"""
Projected 3-D geometry for the input-feature figure. Writes
figures/src/features_geo.tex with macros:
  \FanDraw      a vertex fan (panel a), faces shaded by a fixed light
  \FanCoords    named coordinates: (fv) vertex, (fn) tip of its normal, (fo) centroid
  \RoofDraw     two faces sharing one edge (panel b)
  \RoofCoords   (ru) source, (rv) destination, (ra)/(rb) face centroids,
                (ran)/(rbn) normal tips, (n1)/(n2) tips in canonical order
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"   # figures/src

AZ, EL = np.deg2rad(-28), np.deg2rad(36)
Rz = np.array([[np.cos(AZ), -np.sin(AZ), 0], [np.sin(AZ), np.cos(AZ), 0], [0, 0, 1]])
Rx = np.array([[1, 0, 0], [0, np.cos(EL - np.pi / 2), -np.sin(EL - np.pi / 2)],
               [0, np.sin(EL - np.pi / 2), np.cos(EL - np.pi / 2)]])
VIEW = Rx @ Rz
LIGHT = np.array([0.3, -0.5, 0.8]); LIGHT /= np.linalg.norm(LIGHT)


def P(p):
    q = VIEW @ np.asarray(p, float)
    return q[0], q[1]


def depth(p):
    return (VIEW @ np.asarray(p, float))[2]


def c(p):
    x, y = P(p)
    return f"({x:.3f},{y:.3f})"


def unit(v):
    return v / np.linalg.norm(v)


def shade(n):
    k = abs(float(n @ LIGHT))
    return int(round(26 - 18 * k))          # percent of ink: 8 .. 26


out = []

# ---------------- panel (a): a vertex fan ----------------
v = np.array([0.0, 0.0, 0.18])
ring = []
for k in range(6):
    t = np.deg2rad(15 + 60 * k + (8 if k % 2 else -6))
    r = 1.0 + (0.12 if k % 2 else -0.05)
    ring.append(np.array([r * np.cos(t), r * np.sin(t), 0.22 * np.sin(2 * t) - 0.05]))
faces = [(v, ring[k], ring[(k + 1) % 6]) for k in range(6)]
weighted = sum(np.cross(b - a, cc - a) for a, b, cc in faces)
nv = unit(weighted)
draws = []
for a, b, cc in sorted(faces, key=lambda f: -np.mean([depth(p) for p in f])):
    n = unit(np.cross(b - a, cc - a))
    draws.append(f"\\filldraw[fill=ink!{shade(n)}, draw=ink2, line width=0.5pt] {c(a)} -- {c(b)} -- {c(cc)} -- cycle;")
out.append("\\def\\FanDraw{" + " ".join(draws) + "}")
o = np.array([-0.55, -1.55, -1.05])                 # the fragment centroid, below the patch
coords = [f"\\coordinate (fv) at {c(v)};", f"\\coordinate (fn) at {c(v + 1.05 * nv)};",
          f"\\coordinate (fo) at {c(o)};"]
for k, p in enumerate(ring):
    coords.append(f"\\coordinate (fr{k}) at {c(p)};")
out.append("\\def\\FanCoords{" + " ".join(coords) + "}")

# ---------------- panel (b): two faces on one edge ----------------
# Designed in screen-aligned axes: x right, y up, z towards the viewer. The edge
# runs up the page, the faces open like a book, folded away from the viewer
# (a ridge), and the scene is turned slightly so depth reads.
tilt = np.deg2rad(-30)
turn = np.deg2rad(6)
Ry = np.array([[np.cos(turn), 0, np.sin(turn)], [0, 1, 0], [-np.sin(turn), 0, np.cos(turn)]])
Rx2 = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
V2 = Rx2 @ Ry


def c2(p):
    q = V2 @ np.asarray(p, float)
    return f"({q[0]:.3f},{q[1]:.3f})"


def depth2(p):
    return (V2 @ np.asarray(p, float))[2]


u = np.array([0.0, -1.0, 0.0])        # source
w = np.array([0.0, 1.0, 0.0])         # destination
wa = np.array([-1.25, 0.15, -0.55])   # apex of face a (left slope)
wb = np.array([1.2, -0.2, -0.5])      # apex of face b (right slope)
fa = (u, w, wa)
fb = (w, u, wb)
ma = unit(np.cross(fa[1] - fa[0], fa[2] - fa[0]))
mb = unit(np.cross(fb[1] - fb[0], fb[2] - fb[0]))
s = float(np.cross(ma, mb) @ (w - u))            # (m_a x m_b) . (x_dst - x_src)
n1, n2 = (ma, mb) if s >= 0 else (mb, ma)
ca, cb = sum(fa) / 3, sum(fb) / 3
cos_ab = float(ma @ mb)
draws = []
for f in sorted([fa, fb], key=lambda f: np.mean([depth2(p) for p in f])):
    n = unit(np.cross(f[1] - f[0], f[2] - f[0]))
    draws.append(f"\\filldraw[fill=ink!{shade(n)}, draw=ink2, line width=0.5pt] {c2(f[0])} -- {c2(f[1])} -- {c2(f[2])} -- cycle;")
out.append("\\def\\RoofDraw{" + " ".join(draws) + "}")
L = 1.0
first = ca if s >= 0 else cb
second = cb if s >= 0 else ca
coords = [f"\\coordinate (ru) at {c2(u)};", f"\\coordinate (rv) at {c2(w)};",
          f"\\coordinate (r1) at {c2(first)};", f"\\coordinate (r2) at {c2(second)};",
          f"\\coordinate (r1n) at {c2(first + L * n1)};", f"\\coordinate (r2n) at {c2(second + L * n2)};",
          f"\\coordinate (ruo) at {c2(u + np.array([0.18, 0, 0.1]))};",
          f"\\coordinate (rvo) at {c2(w + np.array([0.18, 0, 0.1]))};"]
out.append("\\def\\RoofCoords{" + " ".join(coords) + "}")
out.append(f"\\def\\RoofOrder{{{'ab' if s >= 0 else 'ba'}}}")
out.append(f"\\def\\RoofCos{{{cos_ab:.2f}}}")
open(SRC / "features_geo.tex", "w").write("\n".join(out) + "\n")
print("vertex normal", nv.round(3), "| roof: m_a", ma.round(3), "m_b", mb.round(3),
      "triple", round(s, 3), "order", "ab" if s >= 0 else "ba", "cos", round(cos_ab, 3))
