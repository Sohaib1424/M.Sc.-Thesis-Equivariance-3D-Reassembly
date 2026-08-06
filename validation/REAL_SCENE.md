# Running a real scene

`run_real_scene.py` drives one real Breaking Bad scene directory through the
project's own numpy code paths — `load_scene`'s logic, `decimate_scene`,
`find_shared_points` — and reports what happened at each stage.

```bash
python validation/run_real_scene.py /path/to/<scene_hash>
```

The argument is a directory laid out the way the dataset ships:

```
<scene_hash>/
  compressed_mesh.obj
  compressed_data.npz
  fractured_0/compressed_fracture.npy
  fractured_1/compressed_fracture.npy
  ...
  mode_0/compressed_fracture.npy
  ...
```

## What it reports

- how many fracture subdirectories exist, split by prefix, and their label dtypes
- fractures that yield fewer than two pieces (nothing to reassemble)
- fragments built per fracture, and the fragment size distribution
- **how many vertices are shared across fragments** — the signal the two
  interface-embedding losses are supervised by, and the one thing that fails
  silently if it is absent
- decimation behaviour across vertex budgets, including whether the budget was
  actually reachable

## Why it exists

It found the bug in `docs/CHANGES.md` §13: decimation could report success
while reducing nothing, and the meaningless voxel size it returned then widened
the correspondence tolerance to half the object — quietly turning two loss
terms into noise. No synthetic test had caught it, because synthetic scenes do
not have fifteen fragments already sitting below the per-fragment vertex floor.

Worth pointing at a few scenes from different categories before committing to a
long training run. What to look for:

- `scenes with ZERO interface correspondences` should be 0
- the correspondence tolerance should stay small relative to the object extent
- fragment counts must be identical before and after decimation

## Relationship to `scripts/smoke_test.py`

This harness replicates `load_scene`'s three `igl` calls in numpy and supplies a
minimal mesh stand-in, so it runs **without trimesh or igl**. That is its only
advantage.

In an environment that has the full stack, `scripts/smoke_test.py` is the
better tool: it covers the same ground and then continues through feature
construction, collation, the model, the equivariance law, the loss, and a
backward pass. Use this one when the geometry libraries are unavailable, or
when you specifically want to inspect the data layer in isolation.
