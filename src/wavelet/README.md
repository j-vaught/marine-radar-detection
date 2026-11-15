# Wavelet

`src/wavelet` is a native Rust/CUDA detector crate for the stationary wavelet baseline.

The folder is intentionally native-only:

- Rust for CLI, preprocessing orchestration, connected components, evaluation, and matrix search
- CUDA C++ for optional batched wavelet preprocessing

## Method

The detector applies a `db4` stationary wavelet transform, thresholds the response map, extracts connected components, and turns them into COCO-format detections.

The main tunable controls are:

- `level`
- `threshold_sigma`
- `min_area`
- `max_area`
- `dilation`
- `box_padding`

## Commands

```bash
cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images \
    --out results/wavelet/predictions.json --backend cuda

cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json --pred results/wavelet/predictions.json

cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    bench --ann data/processed/annotations/all.json --images data/processed/images \
    --backend cpu --preload

cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    matrix --ann data/processed/annotations/all.json --images data/processed/images \
    --backend cuda --grid small --preload
```

Default matrix outputs go under `results/wavelet/`.

## Structure

- [Cargo.toml](Cargo.toml): crate manifest
- [build.rs](build.rs): CUDA build hook
- [src/main.rs](src/main.rs): CLI entrypoints
- [src/detector.rs](src/detector.rs): CPU preprocessing and postprocess
- [src/cuda.rs](src/cuda.rs): Rust/CUDA bridge
- [cuda/kernels.cu](cuda/kernels.cu): CUDA kernels
- [src/matrix.rs](src/matrix.rs): experiment matrix definitions
- [src/eval.rs](src/eval.rs): COCO-style evaluation helpers

## Cleanup note

The older Python implementation was removed once the native crate existed, so this folder now only contains Rust/CUDA source plus crate metadata.

The response mean and population standard deviation at each level use pixels
inside the circle whose original input value is nonzero. Both backends apply a
minimum standard deviation of `1e-6` before forming the threshold. Pixels outside
that region receive zero in both output maps.
