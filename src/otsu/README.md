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

## Completed small-grid matrix

The full `40`-experiment small-grid matrix finished on `April 20, 2026` with the CPU backend:

```bash
src/otsu/target/release/otsu matrix \
  --ann data/processed/annotations/all.json \
  --images data/processed/images \
  --backend cpu \
  --grid small \
  --preload
```

Artifacts live in:

- `results/otsu/matrix.csv`
- `results/otsu/per_experiment/*.json`
- `results/otsu/matrix_small_cpu.log`

High-level outcome:

- `OTSU` beat `SSOTSU` on average on this small search grid.
- `OTSU` mean over all 20 experiments:
  - `AP50 = 0.5759`
  - `AP = 0.4121`
  - `FA/img = 3.18`
  - mean runtime `10.7 s`
- `SSOTSU` mean over all 20 experiments:
  - `AP50 = 0.4656`
  - `AP = 0.3041`
  - `FA/img = 5.72`
  - mean runtime `346.1 s`

Condition-level means:

- `P / OTSU`: `AP50 0.6006`, `AP 0.4105`, `FA/img 2.27`
- `P / SSOTSU`: `AP50 0.5429`, `AP 0.3483`, `FA/img 2.22`
- `S / OTSU`: `AP50 0.5584`, `AP 0.4007`, `FA/img 3.46`
- `S / SSOTSU`: `AP50 0.4473`, `AP 0.2903`, `FA/img 3.79`
- `T / OTSU`: `AP50 0.5738`, `AP 0.4206`, `FA/img 3.51`
- `T / SSOTSU`: `AP50 0.4349`, `AP 0.2887`, `FA/img 8.95`

Best observed rows by `AP50`:

- `T-OTSU-MvMv`: `AP50 0.7922`, `AP 0.7574`, `FA/img 1.49`
- `S-OTSU-6`: `AP50 0.7711`, `AP 0.4910`, `FA/img 1.47`
- `T-OTSU-DaMv`: `AP50 0.7711`, `AP 0.4910`, `FA/img 1.47`
- `T-SSOTSU-MvMv`: `AP50 0.7281`, `AP 0.6622`, `FA/img 1.49`
- `S-SSOTSU-6`: `AP50 0.7272`, `AP 0.6507`, `FA/img 1.46`
- `T-SSOTSU-DaMv`: `AP50 0.7272`, `AP 0.6507`, `FA/img 1.46`

The main practical conclusion is that the native CPU `OTSU` baseline is currently the strongest cost/performance point in this folder. `SSOTSU` can be competitive on some MANV-centered splits, but on this grid it is slower and less stable overall.
