# Numerical validations

Each script here checks one claim made in `docs/CHANGES.md`, in pure numpy, and
prints its evidence. They were written *before* the corresponding torch code, so
the mathematics was confirmed independently of the implementation.

All of them run without torch:

```bash
for f in validation/v*.py; do echo "=== $f"; python "$f"; done
```

| script | claim it checks | CHANGES.md |
|---|---|---|
| `v01_rotation_convention.py` | the Gram-Schmidt frame is left-equivariant while the target transforms right; the head must return `Fᵀ` | §1 |
| `v02_segment_attention.py` | segment attention is bit-identical to dense masked attention, at 20× less memory | §2 |
| `v03_resolve_faces.py` | the vectorised duplicate-face resolution matches the reference loop on 400/400 cases | §11 |
| `v04_decimation.py`, `v04b_interface.py` | a shared voxel grid preserves cross-fragment interfaces better than per-fragment grids | §3 |
| `v05_translation_solver.py`, `v05b_translation.py` | `E_normal` has zero translation gradient; the Laplacian solve is exact; IRLS survives 40% outliers | §4, §5 |
| `v06_assembly_e2e.py` | matching → solving → scoring end to end; disconnection detection; Euler vs geodesic | §5, §6, §7 |
| `v07_geodesic_angle.py` | the `arccos` angle has a 0.028° floor and an exploding gradient; `atan2` has neither | §10 |
