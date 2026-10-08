r"""
A Voronoi cell decomposition of a rectangle, for the data-format figure.
Breaking Bad stores an object pre-cut into cells; a break pattern assigns each
cell to a piece. Writes figures/src/cells.tex with:
  \CellFills{<style prefix>}  fills every cell with colour piece<A|B|C>
  \CellWalls                   every interior wall (thin)
  \PieceWalls                  walls between cells of different pieces (thick)
  \CellOutline                 the outer boundary
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"   # figures/src
from scipy.spatial import Voronoi

rng = np.random.default_rng(11)
W, H = 4.0, 2.6
n = 46
seeds = np.c_[rng.uniform(0, W, n), rng.uniform(0, H, n)]
# relax a little (Lloyd-like jitter) so cells are not slivers
for _ in range(3):
    mirrored = np.vstack([seeds,
                          np.c_[-seeds[:, 0], seeds[:, 1]], np.c_[2 * W - seeds[:, 0], seeds[:, 1]],
                          np.c_[seeds[:, 0], -seeds[:, 1]], np.c_[seeds[:, 0], 2 * H - seeds[:, 1]]])
    vor = Voronoi(mirrored)
    new = []
    for i in range(n):
        reg = vor.regions[vor.point_region[i]]
        poly = vor.vertices[reg]
        new.append(poly.mean(axis=0))
    seeds = np.array(new)

mirrored = np.vstack([seeds,
                      np.c_[-seeds[:, 0], seeds[:, 1]], np.c_[2 * W - seeds[:, 0], seeds[:, 1]],
                      np.c_[seeds[:, 0], -seeds[:, 1]], np.c_[seeds[:, 0], 2 * H - seeds[:, 1]]])
vor = Voronoi(mirrored)

J = np.array([2.05, 1.15])


def piece_of(p):
    a = np.degrees(np.arctan2(p[1] - J[1], p[0] - J[0])) % 360
    if 100 <= a < 232:
        return "A"
    if 232 <= a < 318:
        return "B"
    return "C"


labels = [piece_of(s) for s in seeds]


def order(poly):
    c = poly.mean(axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


fills, walls, piece_walls = [], [], []
for i in range(n):
    reg = vor.regions[vor.point_region[i]]
    poly = order(np.clip(vor.vertices[reg], [0, 0], [W, H]))
    path = " -- ".join(f"({x:.3f},{y:.3f})" for x, y in poly) + " -- cycle"
    fills.append(f"\\filldraw[fill=piece{labels[i]}, draw=piece{labels[i]}, line width=0.4pt] {path};")
for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
    if p >= n or q >= n or -1 in rv:
        continue
    a, b = vor.vertices[rv[0]], vor.vertices[rv[1]]
    seg = f"({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f})"
    walls.append(seg)
    if labels[p] != labels[q]:
        piece_walls.append(seg)

out = [
    "\\def\\CellFills{" + " ".join(fills) + "}",
    "\\def\\CellWalls{" + " ".join(walls) + "}",
    "\\def\\PieceWalls{" + " ".join(piece_walls) + "}",
    f"\\def\\CellOutline{{(0,0) rectangle ({W},{H})}}",
    f"\\def\\NumCells{{{n}}}",
]
open(SRC / "cells.tex", "w").write("\n".join(out) + "\n")
print(n, "cells;", {k: labels.count(k) for k in "ABC"}, len(walls), "walls,", len(piece_walls), "between pieces")
