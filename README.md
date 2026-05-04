# Classical vs. Deep Learning Object Detection in X-Band Marine Radar Imagery

A cross-paradigm benchmark of 8 object detection methods — 4 classical and 4 deep learning — on the DLR DAAN / DARC / MANV corpus.

**Authors:** J.C. Vaught and Ty Dangerfield — University of South Carolina CSCE 763.
**Paper materials:** [paper/README.md](paper/README.md)
**Release:** [release `code-v1.3`](https://github.com/j-vaught/marine-radar-detection/releases/tag/code-v1.3)

## Methods

Each method is a self-contained folder under `src/`. Owners are free to reshape inputs however they want — the only shared inputs are the COCO-format annotations and PPI images shipped in the data release.

| Folder | Methods | Entry point(s) |
|---|---|---|
| `src/cfar/` | CA / GO / SO / OS-CFAR | `run.py` |
| `src/morphological/` | Otsu + opening/closing + CC | `Cargo.toml` (`run`, `eval`, `bench`) |
| `src/otsu/` | Otsu, SSOTSU | `Cargo.toml` (`run`, `eval`, `bench`, `matrix`) |
| `src/wavelet/` | Stationary wavelet transform (db4, 3 levels) | `Cargo.toml` (`run`, `eval`, `bench`, `sweep-bench`, `matrix`) |
| `src/yolo/` | YOLOv11 (Ultralytics) | `train.py`, `infer.py`, `coco_to_yolo.py` |
| `src/rtdetr/` | RT-DETR (Ultralytics) | `train.py`, `infer.py`, `coco_to_yolo.py` |
| `src/faster_rcnn/` | Faster R-CNN (torchvision) | `train.py`, `infer.py`, `dataset.py` |
| `src/unet/` | U-Net segmentation | `train.py`, `infer.py`, `dataset.py` |

Each Python method folder also contains a `config.yaml` (hyperparameters) and a `detector.py` or `model.py` (core logic). The method folders with dedicated READMEs are: [CFAR](src/cfar/README.md), [Morphological](src/morphological/README.md), [Otsu](src/otsu/README.md), [Wavelet](src/wavelet/README.md), [Faster R-CNN](src/faster_rcnn/README.md), [U-Net](src/unet/README.md).

## Project structure

```
marine-radar-detection/
├── data/
│   ├── raw/HAXR/              # 5.1 GB — kept for future work
│   └── processed/
│       ├── images/            # 2,239 PPI PNGs (from data-v1.0)
│       └── annotations/       # COCO JSONs (full + tiled)
├── dist/                      # gitignored — release tarballs
├── paper/                     # replication / publication docs
├── src/
│   ├── cfar/ morphological/ otsu/ wavelet/          # classical
│   └── yolo/ rtdetr/ faster_rcnn/ unet/             # deep learning
├── pyproject.toml
└── README.md
```

No shared `common/`, `eval/`, `scripts/`, `configs/`, or `Makefile`. Each method folder is self-contained so a teammate can take ownership of one detector without touching anyone else's code. Helper functions (circular PPI mask, COCO → YOLO conversion, pycocotools AP) are inlined per folder; ~200 lines of intentional duplication buys team sovereignty.

## Setup

Requires Python 3.10+ and (for DL) CUDA 12.4.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

## Fetch the dataset

This repository is currently private. The commands below assume the user already has GitHub access and has authenticated the GitHub CLI with `gh auth login`.

```bash
gh release download code-v1.3 -p 'processed-v1.tar.zst' -p 'processed-v1.tar.zst.sha256'
sha256sum -c processed-v1.tar.zst.sha256
zstd -dc processed-v1.tar.zst | tar -xf - -C data/
```

This populates `data/processed/images/` (2,239 PPI PNGs) and `data/processed/annotations/all.json` — the full AIS-anchored ground truth in one COCO file, unsplit. You decide train/val/test however you want (the paper uses filename prefix: MANV → train, DARC → val, DAAN → test). Tiling, YOLO-format conversion, mask generation, and any format munging are also your call.

## Running a method

Most method folders run as Python modules. `src/morphological`, `src/otsu`, and `src/wavelet` are native Rust/CUDA crates.

**Classical detectors** — the Python methods below print COCO AP at the end. The native morphology crate writes COCO-format predictions JSON. The example below evaluates on the full corpus; substitute your own test split if you split the data:

```bash
python -m src.cfar.run          --ann data/processed/annotations/all.json --images data/processed/images --variant ca_cfar
cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images --backend cpu
cargo run --release --manifest-path src/morphological/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json
cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images --backend cuda --method otsu
cargo run --release --features cuda --manifest-path src/otsu/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json --method otsu
cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images \
    --out results/wavelet/predictions.json --backend cuda
cargo run --release --features cuda --manifest-path src/wavelet/Cargo.toml -- \
    eval --ann data/processed/annotations/all.json --pred results/wavelet/predictions.json
```

By default the native morphology crate writes to `results/morphological/predictions.json` and `eval` reads the same path back unless `--out` / `--pred` are supplied explicitly.

The CFAR runner writes to `results/cfar/<variant>_predictions.json` by default unless `--out` is supplied explicitly.

The native Otsu crate writes to `results/otsu/<method>_predictions.json` by default unless `--out` is supplied explicitly. Its `matrix` command writes `results/otsu/matrix.csv` plus per-experiment JSON rows under `results/otsu/per_experiment/`.

To use the CUDA preprocessing backend instead of the CPU path:

```bash
cargo run --release --features cuda --manifest-path src/morphological/Cargo.toml -- \
    run --ann data/processed/annotations/all.json --images data/processed/images --backend cuda
```

**Deep learning detectors** — train then infer. Each folder takes explicit train/val paths so you can use whatever split + tiling layout you produced:

```bash
# YOLO / RT-DETR need YOLO-format labels; supply your own split files
python -m src.yolo.coco_to_yolo \
    --train-ann <your_train.json> --val-ann <your_val.json> --test-ann <your_test.json> \
    --train-images <your_train_dir> --val-images <your_val_dir> --test-images <your_test_dir> \
    --out data/processed/yolo

python -m src.yolo.train  --data data/processed/yolo/data.yaml
python -m src.yolo.infer  --weights results/yolo/best_seed42.pt \
                          --ann <your_test.json> --images <your_test_dir>

# Faster R-CNN
python -m src.faster_rcnn.train \
    --train-ann <your_train.json> --val-ann <your_val.json> \
    --train-images <your_train_dir> --val-images <your_val_dir>

# U-Net (segmentation — needs paired image + binary-mask directories you build yourself)
python -m src.unet.train \
    --train-images <your_train_images> --train-masks <your_train_masks> \
    --val-images   <your_val_images>   --val-masks   <your_val_masks>
```

Each `infer.py` / `run.py` writes a `results/<method>/*_predictions.json` and prints a COCO AP summary via pycocotools. No central evaluation step.

## Reproducing the paper

Start here:

- [paper/README.md](paper/README.md) for the publication package layout
- [paper/PUBLICATION_CHECKLIST.md](paper/PUBLICATION_CHECKLIST.md) for what still has to be committed or uploaded before everything is truly public
- [results/figures/paper/README.md](results/figures/paper/README.md) for the figure inventory
- [results/figures/paper/DISCUSSION.md](results/figures/paper/DISCUSSION.md) for the paper-ready results narrative

The ignored result inputs used by the paper figures are bundled with:

```bash
./.venv/bin/python tools/package_paper_artifacts.py
```

This writes `dist/paper-artifacts-v1.tar.zst` and `dist/paper-artifacts-v1.tar.zst.sha256`. The bundle carries the matrix CSVs, per-experiment summaries, TIDE outputs, the full paper figure pack, and the specific prediction payloads needed for the failure-analysis panels.

To fetch the published paper bundle from GitHub instead of building it locally:

```bash
gh release download code-v1.3 -p 'paper-artifacts-v1.tar.zst' -p 'paper-artifacts-v1.tar.zst.sha256'
sha256sum -c paper-artifacts-v1.tar.zst.sha256
tar --zstd -xf paper-artifacts-v1.tar.zst
rsync -a paper-artifacts-v1/repo_overlay/ .
```
