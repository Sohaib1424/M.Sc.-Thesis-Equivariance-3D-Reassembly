# Visualisation

All of these take `--root-dir` and `--save`. On Kaggle or any headless machine
pass `--save out.png`; without it they try to open an interactive window and
fail.

| script | shows |
|---|---|
| `vis_default.py` | fragments in their assembled positions |
| `vis_diffused.py` | after the scattering transform — the network's input |
| `vis_fractures.py` | extracted fracture surfaces of an assembled scene |
| `vis_diffused_fractures.py` | fracture surfaces of a scattered scene (`input_source: frac`) |
| `vis_diffused_and_fractures.py` | scattered fragments and their fracture surfaces, side by side |

```bash
python scripts/visualize/vis_default.py --root-dir data --save assembled.png
python scripts/visualize/vis_fractures.py --root-dir data --seed 7
```

Common options: `--scene <dir>` and `--fracture fractured_3` pin an exact
scene, `--seed` reproduces a random choice, `--split test` draws only from one
split, `--fracture-pattern 'fractured_*'` excludes the `mode_*` fractures.

For a **trained model's output against ground truth**, use the one a level up:

```bash
python scripts/vis_prediction.py --checkpoint checkpoints/best.pt \
    --root-dir data --save prediction.png
```

These are ports of the original `vis_*.py` scripts. Three changes worth
knowing: they take `--root-dir` instead of assuming a fixed location, they can
save a PNG instead of requiring a display, and fragment colours are spaced
around the hue circle rather than drawn uniformly at random — on a 50-fragment
scene, uniform random colours produce several near-identical shades, which
defeats the purpose of colouring them.
