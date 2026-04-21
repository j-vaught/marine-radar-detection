# Morphological Detector

`src/morphological` is the native Rust/CUDA implementation of the classical morphology baseline for this repo. It replaces the old Python morphology runner with a crate that keeps the detector logic in Rust and uses CUDA for the preprocessing hot path when the `cuda` feature is enabled.

## What It Does

The detector is a single-class vessel detector over grayscale radar PPI images. It takes `data/processed/images/*.png` plus COCO image metadata from `data/processed/annotations/*.json`, then emits COCO-format predictions.

The baseline is intentionally simple:

1. Apply the circular PPI mask and crop work to the measured radar-disc ROI.
2. Compute a masked Otsu threshold over nonzero pixels.
3. Threshold the image into a binary target mask.
4. Apply elliptical opening and closing.
5. Optionally dilate the binary mask before box extraction.
6. Run connected-components on the ROI mask.
7. Filter components by area and optional eccentricity.
8. Convert surviving components to bounding boxes and normalized scores.

The implementation is in:

- [src/detector.rs](src/detector.rs) for thresholding, morphology, ROI handling, and scoring
- [src/ccl.rs](src/ccl.rs) for connected-components and shape statistics
- [src/cuda.rs](src/cuda.rs) and [cuda/kernels.cu](cuda/kernels.cu) for the CUDA preprocessing path
- [src/main.rs](src/main.rs) for the `run`, `eval`, and `bench` CLI
- [tests/golden.rs](tests/golden.rs) for CPU golden tests and CUDA parity smoke tests

## CPU And CUDA Backends

`cpu`:

- Full detector path in Rust
- ROI-scoped masked Otsu
- Row-parallel morphology with `rayon`
- Union-find connected-components and component-stat reduction in Rust

`cuda`:

- CUDA accelerates masked Otsu, thresholding, and morphology preprocessing
- The binary mask then comes back to Rust for connected-components, filtering, and scoring

So the CUDA backend is currently a hybrid path, not a fully GPU-resident detector.

## CLI

Run from the repo root:

```bash
cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images --backend cpu
```

Default paths:

- `run` writes predictions to `results/morphological/predictions.json` if `--out` is omitted
- `eval` reads `results/morphological/predictions.json` if `--pred` is omitted
- `bench` and `eval` default to `<repo>/.venv/bin/python` for the Python-side comparison/eval helper

Useful commands:

```bash
cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json

cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    bench --ann data/processed/annotations/all.json --images data/processed/images \
    --count 200 --backend cpu --compare-python

cargo run --release --features cuda --manifest-path src/morphological/Cargo.toml -- \
    bench --ann data/processed/annotations/all.json --images data/processed/images \
    --count 200 --backend cuda --compare-python
```

## Validation

The detector has two built-in correctness checks:

- CPU golden tests on real images from the processed corpus
- CUDA parity check against CPU on a smoke image when CUDA is available

Run them with:

```bash
CARGO_TARGET_DIR=target/morphological cargo test --release \
    --manifest-path src/morphological/Cargo.toml --features cuda
```

## Experiments Run

The following native runs were executed on **April 20, 2026**.

### 1. Release CPU Benchmark

Command shape:

```bash
cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    bench --ann data/processed/annotations/all.json --images data/processed/images \
    --count 200 --backend cpu --compare-python
```

Observed result:

- Native CPU: `10.150 ms/image` over 200 images
- Legacy Python baseline: `29.344 ms/image`
- Speedup: `2.89x`

This is an end-to-end comparison on a 200-image sample, not a detector-core-only microbenchmark.

### 2. Full-Corpus COCO Evaluation

The full processed corpus was evaluated on **2239 images** / **7647 annotations**. This is a convenience full-corpus run, not a paper-style held-out train/val/test split.

Overall metrics:

- `AP@[0.50:0.95]`: `0.316`
- `AP50`: `0.468`
- `AP75`: `0.332`
- `AR@100`: `0.804`

Per-site summary:

| Site | AP | AP50 | AP75 | AR@100 |
|---|---:|---:|---:|---:|
| `DAAN` | `0.311` | `0.537` | `0.314` | `0.660` |
| `DARC` | `0.241` | `0.425` | `0.227` | `0.642` |
| `MANV` | `0.407` | `0.445` | `0.437` | `0.949` |

Supplementary greedy IoU summary at prediction score `>= 0.5`:

- Mean matched IoU: `0.886`
- Median matched IoU: `0.905`
- P90 matched IoU: `1.000`

Artifacts from that run now live in:

- [../../results/morphological/eval_20260420](../../results/morphological/eval_20260420)

Important interpretation note:

- The current released COCO file is single-class only: `category_id = 1` / `vessel`
- Extra annotation fields like `ais_obj_id`, `ais_cluster`, `heading`, and `blob_peak` are metadata, not separate detection classes

## Limits

- The CUDA path does not yet keep connected-components and postprocessing on device.
- The full-corpus evaluation above is useful for smoke-testing and qualitative inspection, but not for rigorous model selection.
- Scores are normalized per image, so this baseline should be treated as a classical reference detector, not a calibrated probabilistic model.
