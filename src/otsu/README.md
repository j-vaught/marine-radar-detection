# Otsu / SSOTSU

`src/otsu` is a native Rust/CUDA detector crate for two classical thresholding methods:

- `otsu`: global masked Otsu on the circular PPI ROI, then connected components
- `ssotsu`: blockwise local Otsu seed generation plus threshold-propagation region growth

The folder is intentionally native-only:

- Rust for CLI, evaluation, matrix search, and the fastest current CPU path
- CUDA C++ for optional thresholding and postprocess kernels when GPU execution is explicitly requested

## Why this rewrite exists

The old `src/otsu` Python package was a single-run script. It could score one config, but it was not shaped for rigorous experiments or fast repeated search. The native crate changes that in two ways:

1. It removes Python from the hot path entirely.
2. It splits each detector into a heavy preprocessing stage and a lighter bbox/scoring stage so the matrix runner can reuse expensive work across parameter sweeps.

## Commands

```bash
cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images --backend cuda --method otsu

cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json --method otsu

cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    bench --ann data/processed/annotations/all.json --images data/processed/images --backend cuda --method ssotsu --preload

cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    matrix --ann data/processed/annotations/all.json --images data/processed/images --backend cuda --grid small --preload
```

The CLI defaults to `--backend cpu` because that is currently the fastest end-to-end backend on this workstation in release mode.

Default outputs:

- `run` writes to `results/otsu/<method>_predictions.json` unless `--out` is supplied
- `matrix` writes to `results/otsu/matrix.csv` and `results/otsu/per_experiment/*.json`

## How it works

### Otsu

1. Crop work to the circular ROI bounding box.
2. Compute a masked histogram over non-zero pixels.
3. Solve the global Otsu threshold.
4. Emit a binary ROI mask.
5. Run connected components in Rust.
6. Score each component as `peak_intensity / threshold_at_centroid`, normalized per image.

### SSOTSU

1. Split the ROI into `block_size x block_size` tiles.
2. For each tile, compute local masked Otsu if the tile has enough non-zero support and variance.
3. Form seeds where `pixel > local_otsu * seed_threshold_factor`.
4. Convert each seed to a region-growth requirement `ceil(seed_intensity * region_grow_threshold)`.
5. Propagate the lowest surviving requirement through 4-connected neighbors.
6. Run connected components and score boxes from the per-tile seed threshold map.

The CUDA path keeps thresholding and postprocess kernels on device and can reuse a device-resident ROI cache across repeated detections. The matrix runner also caches raw candidates per `(dilation, shape-needed)` so configs that only vary area limits or box padding do not rerun connected components.

## Experimental matrix

The crate mirrors the native wavelet experiment structure:

- pooled random split: 5 seeds
- site permutation: 6 runs
- transfer: 3x3 site grid
- methods: `OTSU`, `SSOTSU`

So the full matrix is `40` experiments total.

The matrix runner does not treat every config equally expensive:

- `OTSU` heavy stage is shared across all bbox/postprocess settings
- `SSOTSU` heavy stage is shared across configs with the same `block_size`, `seed_threshold_factor`, and `region_grow_threshold`

That reuse is the main runtime optimization for search.

## Tests run

Release verification used:

```bash
CARGO_TARGET_DIR=/home/j-vaught/marine-radar-detection/target/otsu \
cargo test --release --manifest-path src/otsu/Cargo.toml --features cuda
```

Current test coverage:

- unit tests for Otsu thresholding and kernel geometry
- synthetic CPU golden tests for `otsu` and `ssotsu`
- CUDA parity test against CPU on synthetic inputs

## Current checked results

On April 20, 2026, release-mode checks with the current code produced:

- `Otsu CPU preload`: `3.427 ms/image` over 200 images
- `Otsu CUDA preload`: `3.880 ms/image` over 200 images
- single pooled matrix slice `P-OTSU-0`, `grid=small`, `--preload`:
  - CPU: `22.22 s` wall time, internal row runtime `18.5 s`
  - CUDA: `50.29 s` wall time, internal row runtime `46.5 s`

Small CUDA smoke runs on 64 real images produced:

- `results/otsu/otsu_smoke_predictions.json`
- `results/otsu/ssotsu_smoke_predictions.json`

Those are correctness and runtime checks, not tuned final experiments. In particular, untuned `ssotsu` is still very noisy on this corpus; use the matrix runner to tune it before treating it as a serious baseline. At the moment, CUDA is functionally correct but not the fastest full-search backend for Otsu on this machine.
