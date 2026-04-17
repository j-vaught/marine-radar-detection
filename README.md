# Classical vs. Deep Learning Object Detection in X-Band Marine Radar Imagery

A cross-paradigm benchmark of 8 object detection methods — 4 classical and 4 deep learning — on the DLR DAAN / DARC / MANV corpus.

**Authors:** J.C. Vaught and Ty Dangerfield — University of South Carolina CSCE 763.
**Paper:** [`report/radar_detection_report.pdf`](report/radar_detection_report.pdf)
**Data:** [release `data-v1.0`](https://github.com/j-vaught/marine-radar-detection/releases/tag/data-v1.0)

## Methods

Each method is a self-contained folder under `src/`. Owners are free to reshape inputs however they want — the only shared inputs are the COCO-format annotations and PPI images shipped in the data release.

| Folder | Methods | Entry point(s) |
|---|---|---|
| `src/cfar/` | CA / GO / SO / OS-CFAR | `run.py` |
| `src/morphological/` | Otsu + opening/closing + CC | `run.py` |
| `src/otsu/` | Otsu, SSOTSU | `run.py` |
| `src/wavelet/` | Stationary wavelet transform (db4, 3 levels) | `run.py` |
| `src/yolo/` | YOLOv11 (Ultralytics) | `train.py`, `infer.py`, `coco_to_yolo.py` |
| `src/rtdetr/` | RT-DETR (Ultralytics) | `train.py`, `infer.py`, `coco_to_yolo.py` |
| `src/faster_rcnn/` | Faster R-CNN (torchvision) | `train.py`, `infer.py`, `dataset.py` |
| `src/unet/` | U-Net segmentation | `train.py`, `infer.py`, `dataset.py` |

Each folder also contains a `config.yaml` (hyperparameters) and a `detector.py` or `model.py` (core logic).

## Project structure

```
marine-radar-detection/
├── data/
│   ├── raw/HAXR/              # 5.1 GB — kept for future work
│   └── processed/
│       ├── images/            # 2,239 PPI PNGs (from data-v1.0)
│       └── annotations/       # COCO JSONs (full + tiled)
├── dist/                      # gitignored — release tarballs
├── report/                    # Typst source + compiled PDFs
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

```bash
curl -LO https://github.com/j-vaught/marine-radar-detection/releases/download/data-v1.0/processed-v1.tar.zst
curl -LO https://github.com/j-vaught/marine-radar-detection/releases/download/data-v1.0/processed-v1.tar.zst.sha256
sha256sum -c processed-v1.tar.zst.sha256
zstd -dc processed-v1.tar.zst | tar -xf - -C data/
```

This populates `data/processed/images/` (2,239 PPI PNGs) and `data/processed/annotations/` (COCO JSONs including AIS-anchored ground truth). Everything else — tiles, YOLO-format labels, masks — is regenerable per-owner inside the relevant `src/<method>/` folder.

## Running a method

Every method folder runs as a Python module. Pass the COCO annotation JSON and the image directory.

**Classical detectors** — single command, prints COCO AP at the end:

```bash
python -m src.cfar.run          --ann data/processed/annotations/test.json --images data/processed/images --variant ca_cfar
python -m src.morphological.run --ann data/processed/annotations/test.json --images data/processed/images
python -m src.otsu.run          --ann data/processed/annotations/test.json --images data/processed/images --variant ssotsu
python -m src.wavelet.run       --ann data/processed/annotations/test.json --images data/processed/images
```

**Deep learning detectors** — train then infer:

```bash
# YOLO / RT-DETR need YOLO-format labels (once, per owner)
python -m src.yolo.coco_to_yolo \
    --train-ann data/processed/annotations/train_tiled.json \
    --val-ann   data/processed/annotations/val_tiled.json \
    --test-ann  data/processed/annotations/test_tiled.json \
    --train-images data/processed/tiles/train \
    --val-images   data/processed/tiles/val \
    --test-images  data/processed/tiles/test \
    --out data/processed/yolo

python -m src.yolo.train  --data data/processed/yolo/data.yaml
python -m src.yolo.infer  --weights results/yolo/best_seed42.pt \
                          --ann data/processed/annotations/test_tiled.json \
                          --images data/processed/tiles/test

# Faster R-CNN / U-Net consume COCO directly via their local dataset.py
python -m src.faster_rcnn.train --data data/processed
python -m src.faster_rcnn.infer --weights results/faster_rcnn/best_seed42.pth \
                                --ann data/processed/annotations/test_tiled.json \
                                --images data/processed/tiles/test
```

Each `infer.py` / `run.py` writes a `results/<method>/*_predictions.json` and prints a COCO AP summary via pycocotools. No central evaluation step.

## Reproducing the paper

The PDF at [`report/radar_detection_report.pdf`](report/radar_detection_report.pdf) corresponds to release tag `data-v1.0` (matching AIS-anchored GT, calibration, and split convention). All detector hyperparameters live in each method's `config.yaml`.

Typst source:

```bash
typst compile report/report.typ report/radar_detection_report.pdf
```
