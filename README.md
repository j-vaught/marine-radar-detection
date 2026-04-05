# Classical vs. Deep Learning Object Detection in X-Band Marine Radar Imagery

A cross-paradigm benchmark comparing 11 object detection methods -- 7 classical and 4 deep learning -- on X-band marine radar PPI imagery using the DLR DAAN and HAXR datasets.

**Authors:** J.C. Vaught and Ty Dangerfield

## Methods

### Classical (7)

| Method | Description |
|--------|-------------|
| CA-CFAR | Cell-Averaging Constant False Alarm Rate detector |
| GO-CFAR | Greatest-Of CFAR (clutter edge robust) |
| SO-CFAR | Smallest-Of CFAR (multi-target robust) |
| OS-CFAR | Order-Statistic CFAR |
| Morphological | Otsu binarization + opening/closing + connected components |
| Wavelet-SWT | 2D stationary wavelet transform with HF subband thresholding |
| SSOTSU | Speed-up seed point Otsu for small-target detection |

### Deep Learning (4)

| Method | Description |
|--------|-------------|
| YOLOv11-OBB | Oriented bounding box detection with SAHI tiled inference |
| Faster R-CNN | Radar-optimized anchor generation and scale normalization |
| U-Net | Per-pixel segmentation with bounding box extraction |
| RT-DETR | Real-Time DEtection TRansformer (NMS-free, end-to-end) |

## Project Structure

```
marine-radar-detection/
├── configs/
│   ├── classical/              # CFAR, morphological, wavelet, Otsu configs
│   └── deep_learning/          # YOLOv11, Faster R-CNN, U-Net, RT-DETR configs
├── data/
│   ├── raw/
│   │   ├── DAAN/               # DLR DAAN dataset (760 PPI frames)
│   │   └── HAXR/               # DLR HAXR dataset
│   └── processed/              # Tiles, annotations, and splits (generated)
├── src/
│   ├── data/                   # Preprocessing, tiling, datasets, temporal splitting
│   ├── classical/              # All 7 classical detection implementations
│   ├── deep_learning/          # All 4 DL model wrappers
│   ├── evaluation/             # COCO metrics, TIDE analysis, benchmarking, stats
│   └── visualization/          # Pareto plots, AP comparisons, detection overlays
├── scripts/                    # Pipeline entry points
├── results/                    # Outputs (predictions, figures, tables)
├── Makefile                    # Convenience targets
└── pyproject.toml              # Dependencies
```

## Setup

Requires Python 3.10+ and CUDA 12.4.

```bash
# Create virtual environment and install everything
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Or use the Makefile:

```bash
make setup
```

## Usage

### 1. Prepare Data

Place the DLR DAAN and HAXR datasets in `data/raw/`, then preprocess:

```bash
make preprocess-daan
make preprocess-haxr
```

This tiles all PPI frames to 640x640, clips annotations, and creates temporal train/val/test splits.

### 2. Run Classical Methods

```bash
make classical
```

Runs all 7 classical detectors on the test split and saves COCO-format predictions to `results/classical/`.

### 3. Train Deep Learning Models

Each model is trained with 3 random seeds for statistical comparison:

```bash
make train-yolo      # YOLOv11-OBB + SAHI
make train-frcnn     # Faster R-CNN
make train-unet      # U-Net segmentation
make train-rtdetr    # RT-DETR
```

### 4. Evaluate

```bash
make evaluate
```

Computes COCO metrics (AP@0.5, AP@0.75, AP@[0.50:0.95], AR) stratified by object size, plus TIDE error decomposition for all 11 methods.

### 5. Generate Report Figures

```bash
make figures
```

Produces AP comparison bar charts, accuracy vs. latency Pareto plots, and TIDE error breakdowns.

## Evaluation

- **Metrics:** COCO AP/AR stratified by object size (small, medium, large)
- **Error analysis:** TIDE framework decomposing errors into classification, localization, duplicate, background, and missed components
- **Benchmarking:** Inference latency, FLOPs, and parameter count
- **Statistical testing:** Pairwise bootstrap tests (10,000 resamples) with Bonferroni correction across the 3-seed DL runs

## Dataset

**DLR DAAN:** 760 X-band PPI frames tiled to 3,040 patches at 640x640 with 11,893 annotated instances. Bounding box, oriented bounding box, and segmentation annotations. Temporal splitting by recording session prevents data leakage.

## References

1. Finn & Johnson, "Adaptive detection mode with threshold control," *RCA Review*, 1968.
2. Rohling, "Radar CFAR thresholding in clutter and multiple target situations," *IEEE Trans. AES*, 1983.
3. Duk et al., "Target detection in sea-clutter using stationary wavelet transforms," *IEEE Trans. AES*, 2017.
4. Li et al., "A speed-up seed point Otsu method for ship detection," *IEEE IGARSS*, 2016.
5. Xu et al., "Adaptive clustering-based marine radar sea clutter normalization," *J. Sensors*, 2021.
6. Chen et al., "Automatic detection of scanning radar images based on Radar PPInet," *Remote Sensing*, 2021.
7. Chen et al., "Marine-Faster R-CNN for navigation radar PPI images," *Front. Inf. Technol. Electron. Eng.*, 2022.
8. Heuer et al., "Marine radar object detection using SAHI enhanced YOLOv11-OBB," *Sensors*, 2025.
9. Yee, "Onet: Twin U-Net for semantic segmentation in radar images," *IEEE Trans. Image Process.*, 2025.
10. Zhang et al., "CCDN-DETR for multi-class SAR object detection," *Sensors*, 2024.
11. Bolya et al., "TIDE: A general toolbox for identifying object detection errors," *ECCV*, 2020.
12. Zhang & Li, "Survey on deep learning-based marine object detection," *J. Adv. Transp.*, 2021.
