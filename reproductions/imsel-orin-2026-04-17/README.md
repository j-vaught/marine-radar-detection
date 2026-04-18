# imsel Orin Classical Reproduction (2026-04-17)

This folder contains a clean, branch-local reproduction bundle for the classical DAAN test-set benchmark from `code-v1.0`, generated on an NVIDIA Orin machine and kept separate from the repo's ignored `data/` and `results/` trees.

## What is included

- `RUNBOOK.md`: end-to-end setup and execution notes for this machine
- `classical_reproduction_comparison.csv`: observed vs. reference metrics and prediction counts
- `logs/`: completed method logs with the pycocotools summaries
- `predictions/`: completed method prediction JSONs in COCO result format

## What is not included

- Extracted dataset files under `data/processed/`
- Release tarballs under `dist/`
- The ignored `results/` tree itself
- `os_cfar`: this run was intentionally not completed here because it was taking multiple hours on CPU and is not part of the repo's published comparison table

## Dataset and split used

The working dataset asset was downloaded with GitHub CLI from release `code-v1.1`:

```bash
gh release download code-v1.1 --repo j-vaught/marine-radar-detection --pattern "processed-v1.tar.zst*"
```

The extracted archive contained:

- `data/processed/images/`
- `data/processed/annotations/all.json`

Paper-compatible split convention used for this reproduction:

- `MANV_*` -> `train`
- `DARC_*` -> `val`
- `DAAN_*` -> `test`

Observed split counts:

- `train.json`: 952 images, 3925 annotations
- `val.json`: 527 images, 1784 annotations
- `test.json`: 760 images, 1938 annotations

## Completed methods

- `morphological`
- `otsu`
- `ssotsu`
- `ca_cfar`
- `go_cfar`
- `so_cfar`
- `wavelet`

## High-level outcome

The completed DAAN classical runs reproduced the repo's published reference table very closely:

- prediction counts matched exactly for every completed published method
- metric deltas were very small and consistent with normal environment/package drift
- `morphological` and `otsu` remained the strongest classical baselines on DAAN
- `wavelet` remained effectively zero, matching the repo's reported failure mode

See `classical_reproduction_comparison.csv` for the exact observed/reference values and deltas.
