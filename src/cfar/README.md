# CFAR Detector

`src/cfar` is the classical CFAR baseline for this repo. It operates on the released grayscale PPI PNGs and writes COCO-format vessel detections.

## What It Does

The implemented variants are:

- `CA-CFAR`: cell-averaging background estimate
- `GO-CFAR`: greatest-of split-window estimate
- `SO-CFAR`: smallest-of split-window estimate
- `OS-CFAR`: order-statistic estimate

All four are applied in image space, not on native radar range bins. The detector:

1. Masks the image to the measured radar disc.
2. Normalizes intensities to `float32` in `[0, 1]`.
3. Builds a per-pixel CFAR threshold map from a guard window and a reference window.
4. Thresholds the image into a binary detection mask.
5. Extracts connected components and converts them to bounding boxes plus scores.
6. Filters detections back to the radar disc and writes COCO predictions.

Core files:

- [detector.py](detector.py) implements the CFAR threshold map and box extraction
- [config.yaml](config.yaml) holds the default CA / GO / SO / OS parameters
- [run.py](run.py) runs one variant on a COCO annotation file and prints COCO AP

## How The Variants Work

`CA-CFAR`:

- Estimates local clutter as the mean of the reference cells around the cell under test
- Works well when clutter is fairly homogeneous

`GO-CFAR`:

- Splits the reference support into two halves and uses the larger estimate
- More conservative near clutter edges or asymmetric backgrounds

`SO-CFAR`:

- Splits the support and uses the smaller estimate
- More permissive, but can overfire in heterogeneous clutter

`OS-CFAR`:

- Sorts the reference cells and thresholds against a chosen rank
- More robust to outliers and interfering targets, but much slower in the exact CPU formulation

## What We Intentionally Skipped

This repo only ships rendered PPI images, not raw radar cubes or native range–azimuth sweeps. That matters.

Because of that, this CFAR baseline does **not** implement radar-domain variants that depend on:

- native range–azimuth cells or beam ordering
- per-azimuth scanline processing in the original polar grid
- range–Doppler or Doppler-aware CFAR
- I/Q or slow-time access
- clutter-map / knowledge-aided variants that assume calibrated radar-native measurements

In other words, the implemented CFAR family is the subset that can be defended on rasterized PPI imagery. Once the radar data has already been rendered into Cartesian PNGs, the original range-cell structure is gone, so some classical CFAR formulations are no longer faithful to the sensor domain.

## CLI

Run one variant from the repo root:

```bash
python -m src.cfar.run \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --variant ca_cfar
```

Default output behavior:

- If `--out` is omitted, predictions go to `results/cfar/<variant>_predictions.json`
- `--out-dir` is still accepted for compatibility and expands to the same filename pattern

Examples:

```bash
python -m src.cfar.run \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --variant os_cfar

python -m src.cfar.run \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --variant ca_cfar \
    --out results/cfar/ca_cfar_predictions.json
```

## Experiment Harness

This workspace also contains an expanded CFAR experiment matrix:

- [experiments/matrix.py](experiments/matrix.py) defines the 40 experiments
- [experiments/run_matrix.py](experiments/run_matrix.py) executes the matrix and writes CSV / per-experiment JSON
- [experiments/grid.py](experiments/grid.py) performs the grid search over window size, `pfa`, `min_area`, and `OS-k`

The matrix is:

- `P`: pooled random split, 5 seeds, 2 methods = 10 experiments
- `S`: 6 site permutations, 2 methods = 12 experiments
- `T`: 3x3 transfer grid, 2 methods = 18 experiments

Only `CA` and `OS` are used in that matrix harness. `GO` and `SO` are available in the single-run detector, but the larger experiment sweep in this workspace focuses on the two methods that were carried forward into the cached GPU evaluation pipeline.

## Results Present In This Repo

Current artifacts already under [../../results/cfar](../../results/cfar):

- [../../results/cfar/matrix.csv](../../results/cfar/matrix.csv)
- [../../results/cfar/ca_matrix.log](../../results/cfar/ca_matrix.log)
- [../../results/cfar/per_experiment](../../results/cfar/per_experiment)

The current matrix file contains:

- `40` experiments total
- `10` pooled (`P`)
- `12` site-permutation (`S`)
- `18` transfer (`T`)
- `20` `CA` runs and `20` `OS` runs

Best `AP50` in each block from the current `matrix.csv`:

| Block | Best Experiment | AP50 | AP |
|---|---|---:|---:|
| `P-CA` | `P-CA-3` | `0.6026` | `0.5512` |
| `P-OS` | `P-OS-4` | `0.5964` | `0.5472` |
| `S-CA` | `S-CA-6` | `0.8222` | `0.8033` |
| `S-OS` | `S-OS-6` | `0.8078` | `0.7908` |
| `T-CA` | `T-CA-MvMv` | `0.8698` | `0.8468` |
| `T-OS` | `T-OS-MvMv` | `0.8228` | `0.7993` |

Overall best row in the current matrix:

- `T-CA-MvMv`
- `AP50 = 0.8698`
- `AP = 0.8468`

## Notes

- The released COCO file is single-class: `category_id = 1` / `vessel`
- CFAR scores here are postprocessed connected-component scores, not calibrated probabilities
- On this dataset, CFAR is best treated as a strong classical baseline on rendered radar imagery, not as a raw-radar detector
