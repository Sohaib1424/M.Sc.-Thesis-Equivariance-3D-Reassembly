r"""
Geometry of the 2-D cartoon used by several figures: one flat object broken
into three pieces along a Y-shaped crack.

Writes figures/src/cartoon.tex, which defines, for each piece P in {A, B, C}:

  \PieceP        closed outline, in the piece's own centred frame (cm)
  \CrackP        the fracture part of its outline (open polylines)
  \IntactP       the intact part of its outline (open polyline)
  \NameCracksP   \coordinate (P-c-k) for every crack vertex c-k (shared ids)
  \TokensP       comma list of token positions for \foreach
  \cPx, \cPy     its centroid in the object's frame
  \rP            its radius (max distance from the centroid)
and \dScene (the largest radius) plus the unit circle radius used in figures.

Everything is computed here so that the numbers a figure prints (centroids,
radii, the divisor) are the numbers of the drawing.
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"   # figures/src

rng = np.random.default_rng(7)

A_AX, B_AX, N_EXP = 2.1, 1.35, 2.6          # superellipse half-axes, exponent
J = np.array([0.05, -0.12])                  # crack junction


def boundary(t):
    c, s = np.cos(t), np.sin(t)
    x = A_AX * np.sign(c) * np.abs(c) ** (2 / N_EXP)
    y = B_AX * np.sign(s) * np.abs(s) ** (2 / N_EXP)
    return np.array([x, y])


def arc(t0, t1, n):
    if t1 < t0:
        t1 += 2 * np.pi
    return [boundary(t) for t in np.linspace(t0, t1, n)]


def crack(p, q, n=6, amp=0.11):
    """Jagged polyline p -> q (both endpoints included)."""
    pts = [p]
    d = q - p
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    for k in range(1, n):
        f = k / n
        off = amp * (rng.uniform(-1, 1)) * (1 - abs(2 * f - 1) ** 3)
        pts.append(p + f * d + off * nrm)
    pts.append(q)
    return pts


T1, T2, T3 = np.deg2rad(100), np.deg2rad(232), np.deg2rad(318)
P1, P2, P3 = boundary(T1), boundary(T2), boundary(T3)
cracks = {0: crack(P1, J), 1: crack(P2, J), 2: crack(P3, J)}   # each runs P -> J

# piece outlines, counter-clockwise: arc, then crack back to J, then crack out
pieces = {
    "A": dict(arc=arc(T1, T2, 16), out=(1, +1), back=(0, -1)),   # arc P1->P2, P2->J, J->P1
    "B": dict(arc=arc(T2, T3, 12), out=(2, +1), back=(1, -1)),
    "C": dict(arc=arc(T3, T1, 16), out=(0, +1), back=(2, -1)),
}


def outline(name):
    p = pieces[name]
    arc_pts = p["arc"]
    c_out, _ = p["out"]           # crack from the arc's end to J
    c_back, _ = p["back"]         # crack from J back to the arc's start
    to_j = cracks[c_out]          # P_end -> J
    from_j = cracks[c_back][::-1]  # J -> P_start
    pts = arc_pts + to_j[1:] + from_j[1:-1]
    ids = [None] * len(arc_pts) + [(c_out, k) for k in range(1, len(to_j))] + \
          [(c_back, len(cracks[c_back]) - 1 - k) for k in range(1, len(from_j) - 1)]
    # the arc's end point is crack c_out's vertex 0, its start is crack c_back's vertex 0
    ids[len(arc_pts) - 1] = (c_out, 0)
    ids[0] = (c_back, 0)
    return np.array(pts), ids


def area_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    area = cross.sum() / 2
    cx = ((x + xs) * cross).sum() / (6 * area)
    cy = ((y + ys) * cross).sum() / (6 * area)
    return area, np.array([cx, cy])


def path(points, closed=False):
    s = " -- ".join(f"({p[0]:.3f},{p[1]:.3f})" for p in points)
    return s + (" -- cycle" if closed else "")


out = []
info = {}
for name in "ABC":
    poly, ids = outline(name)
    area, c = area_centroid(poly)
    r = np.max(np.linalg.norm(poly - c, axis=1))
    info[name] = (area, c, r, poly, ids)

d_scene = max(v[2] for v in info.values())

# tokens: 14 in total, split by crack length, farthest-point sampled along the
# crack polyline of each piece (geodesic = arc length along the crack)
def crack_points(name):
    """Crack vertices of a piece in path order: arc end -> J -> arc start."""
    poly, ids = info[name][3], info[name][4]
    n_arc = len(pieces[name]["arc"])
    order = list(range(n_arc - 1, len(poly))) + [0]
    return [(i, ids[i]) for i in order]


lengths = {}
for name in "ABC":
    pts = [info[name][3][i] for i, _ in crack_points(name)]
    lengths[name] = sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))
T_TOTAL = 15
share = {n: T_TOTAL * lengths[n] / sum(lengths.values()) for n in "ABC"}
alloc = {n: int(np.floor(share[n])) for n in "ABC"}
rest = T_TOTAL - sum(alloc.values())
for n in sorted("ABC", key=lambda n: -(share[n] - alloc[n]))[:rest]:
    alloc[n] += 1


def fps_along(points, k):
    pts = np.array(points)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    centre = pts.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(pts - centre, axis=1)))]
    while len(chosen) < min(k, len(pts)):
        dist = np.min(np.abs(seg[:, None] - seg[chosen][None, :]), axis=1)
        chosen.append(int(np.argmax(dist)))
    return sorted(chosen)


for name in "ABC":
    area, c, r, poly, ids = info[name]
    local = poly - c
    L = name
    out.append(f"% piece {L}: area {area:.3f}, radius {r:.3f}")
    out.append(f"\\def\\c{L}x{{{c[0]:.3f}}}\\def\\c{L}y{{{c[1]:.3f}}}\\def\\r{L}{{{r:.3f}}}")
    far = local[int(np.argmax(np.linalg.norm(local, axis=1)))]
    out.append(f"\\def\\far{L}x{{{far[0]:.3f}}}\\def\\far{L}y{{{far[1]:.3f}}}")
    out.append(f"\\def\\Piece{L}{{{path(local, closed=True)}}}")
    # intact part: the arc
    n_arc = len(pieces[name]["arc"])
    out.append(f"\\def\\Intact{L}{{{path(local[:n_arc])}}}")
    # fracture part: from the arc's end round to its start, through J
    crack_idx = list(range(n_arc - 1, len(local))) + [0]
    out.append(f"\\def\\Crack{L}{{{path(local[crack_idx])}}}")
    names = [f"\\coordinate ({L}-{cc}-{k}) at ({local[i][0]:.3f},{local[i][1]:.3f});"
             for i, idv in enumerate(ids) if idv is not None for cc, k in [idv]]
    out.append(f"\\def\\NameCracks{L}{{{' '.join(names)}}}")
    cp = crack_points(name)
    pick = fps_along([local[i] for i, _ in cp], alloc[name])
    toks = [local[cp[j][0]] for j in pick]
    out.append(f"\\def\\Tokens{L}{{{', '.join(f'{t[0]:.3f}/{t[1]:.3f}' for t in toks)}}}")
    out.append(f"\\def\\TokenIds{L}{{{', '.join(f'{cp[j][1][0]}-{cp[j][1][1]}' for j in pick)}}}")

out.append(f"\\def\\dScene{{{d_scene:.3f}}}")
out.append(f"\\def\\invd{{{1 / d_scene:.4f}}}")
open(SRC / "cartoon.tex", "w").write("\n".join(out) + "\n")
for name in "ABC":
    area, c, r, _, _ = info[name]
    print(name, f"area {area:.3f} centroid ({c[0]:.3f},{c[1]:.3f}) radius {r:.3f} tokens {alloc[name]} crack length {lengths[name]:.2f}")
print("d =", round(d_scene, 3))
