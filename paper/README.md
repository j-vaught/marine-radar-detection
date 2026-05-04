# Replication And Journal-Paper Materials

This repository is the codebase for the study, but the complete publication package is split across Git-tracked source and release-sized result artifacts.

## What should be public

- Git repo: `src/`, `tools/`, `paper/`, the top-level `README.md`, and the tracked figure/discussion files under `results/figures/paper/`.
- Release asset 1: `processed-v1.tar.zst` for `data/processed/`.
- Release asset 2: `paper-artifacts-v1.tar.zst` for the ignored result files needed to reproduce the figures and support a journal manuscript.

## What the paper-artifacts bundle contains

- `results/cfar/matrix.csv`, `results/otsu/matrix.csv`, `results/wavelet/matrix.csv`, `results/unet_matrix/matrix.csv`, `results/faster_rcnn_matrix/matrix.csv`, `results/rtdetr_matrix/matrix.csv`, and `results/yolo26_matrix/matrix.csv`
- `results/*/per_experiment/*.json` winner/summary rows used by the paper plots
- `results/figures/paper/` including the failure-analysis zoom pack and classical cache
- `results/tide/` pooled TIDE summaries
- The specific learned-model `test_gt.json` and `test_predictions.json` files used by the hard-transfer and RT-DETR-collapse panels

That bundle is the lightweight supplement to the repo. It avoids publishing the multi-gigabyte training directories while still giving a collaborator enough material to regenerate the figure pack, audit the reported numbers, and draft a journal paper from the outputs.

## Build It Locally

```bash
./.venv/bin/python tools/package_paper_artifacts.py
```

This writes:

- `dist/paper-artifacts-v1/`
- `dist/paper-artifacts-v1.tar.zst`
- `dist/paper-artifacts-v1.tar.zst.sha256`

The bundle includes a `manifest.json` with file hashes plus Git metadata from the source checkout.

## Use The Bundle

After downloading or building the bundle:

```bash
curl -LO https://github.com/j-vaught/marine-radar-detection/releases/download/code-v1.2/paper-artifacts-v1.tar.zst
curl -LO https://github.com/j-vaught/marine-radar-detection/releases/download/code-v1.2/paper-artifacts-v1.tar.zst.sha256
sha256sum -c paper-artifacts-v1.tar.zst.sha256
tar --zstd -xf paper-artifacts-v1.tar.zst
rsync -a paper-artifacts-v1/repo_overlay/ /path/to/marine-radar-detection/
```

Then the main paper figure pack can be regenerated with:

```bash
./.venv/bin/python tools/plot_paper_figures.py
```

## Manuscript Inputs

- Figure inventory: [results/figures/paper/README.md](../results/figures/paper/README.md)
- Paper-ready narrative: [results/figures/paper/DISCUSSION.md](../results/figures/paper/DISCUSSION.md)
- Publication checklist: [paper/PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)

## Intentionally Excluded

- Raw training runs under `runs/`
- Full model checkpoints for every learned experiment
- Full matrix directories for every learned run outside the specific payloads copied into the paper bundle

Those files are useful for internal work, but they are not required for a collaborator to reproduce the reported tables/figures or to write a journal paper from the released evidence.
