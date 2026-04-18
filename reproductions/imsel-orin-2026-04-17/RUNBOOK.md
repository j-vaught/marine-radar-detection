# Marine Radar Detection Reproduction Runbook

This runbook is for the SSD-based checkout at:

`/media/imsel/imsel_ssd/projects/marine-radar-detection`

It follows the `code-v1.0` repo first, starts with the classical methods, and records the current blocker: the dataset archive URLs referenced by the repo are not publicly resolving from this machine.

## 1. Repo and environment

Repo checkout:

```bash
cd /media/imsel/imsel_ssd/projects
git clone --branch code-v1.0 https://github.com/j-vaught/marine-radar-detection.git
cd marine-radar-detection
git describe --tags --exact-match
git rev-parse HEAD
```

Expected:

```text
code-v1.0
907ef94ea0006174c2270bb5e32bf40615d03a66
```

Classical-only Python environment created on the SSD:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection
mkdir -p /media/imsel/imsel_ssd/.pip-cache /media/imsel/imsel_ssd/tmp
python3 -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR=/media/imsel/imsel_ssd/.pip-cache
export TMPDIR=/media/imsel/imsel_ssd/tmp
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy pandas tqdm pyyaml opencv-python scikit-image PyWavelets pycocotools matplotlib seaborn
```

Notes:

- Root storage on this machine is effectively full, so all work should stay on the SSD.
- The full repo dependency install was intentionally deferred because the DL stack needs `torch` and this Jetson/Orin environment should be handled separately after the classical/data path is unblocked.

## 2. Dataset acquisition

Anonymous `curl` against the README release URLs returned `Not Found` from this machine, but `gh release download` works correctly.

Working fetch path:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection/dist
gh release download code-v1.1 --repo j-vaught/marine-radar-detection --pattern "processed-v1.tar.zst*"
cd ..
sha256sum -c dist/processed-v1.tar.zst.sha256
mkdir -p data
zstd -dc dist/processed-v1.tar.zst | tar -xf - -C data
```

Notes:

- The checksum file references `dist/processed-v1.tar.zst`, so run `sha256sum -c` from the repo root, not from inside `dist/`.
- The downloaded asset is the `code-v1.1` data release, which contains:
  - `data/processed/images/`
  - `data/processed/annotations/all.json`
- The git tags `code-v1.0`, `code-v1.1`, and `data-v1.0` all exist.
- The `data-v1.0` git tag contains code/results artifacts, not the image archive itself.

## 3. Build the paper split from `all.json`

The extracted archive is unsplit. For the paper-compatible split:

- `MANV_*` -> train
- `DARC_*` -> val
- `DAAN_*` -> test

Generate split JSONs:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection
source .venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
base = Path("data/processed/annotations")
coco = json.load(open(base / "all.json"))
groups = {"MANV": [], "DARC": [], "DAAN": []}
for img in coco["images"]:
    groups[img["file_name"][:4]].append(img)
img_to_group = {img["id"]: key for key, imgs in groups.items() for img in imgs}
ann_groups = {"MANV": [], "DARC": [], "DAAN": []}
for ann in coco["annotations"]:
    ann_groups[img_to_group[ann["image_id"]]].append(ann)
for key, out_name in {"MANV": "train.json", "DARC": "val.json", "DAAN": "test.json"}.items():
    json.dump(
        {"images": groups[key], "annotations": ann_groups[key], "categories": coco["categories"]},
        open(base / out_name, "w"),
    )
    print(out_name, len(groups[key]), len(ann_groups[key]))
PY
```

Observed counts on this machine:

```bash
train.json 952 3925
val.json 527 1784
test.json 760 1938
```

Verify expected files:

```bash
find data/processed -maxdepth 3 | sed -n '1,120p'
```

## 4. Classical execution order

Run from the repo root:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection
source .venv/bin/activate
```

### 4.1 Morphological

```bash
python -m src.morphological.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images
```

Output:

- `results/morphological/morphological_predictions.json`
- printed COCO AP summary

### 4.2 Otsu

```bash
python -m src.otsu.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant otsu
```

Output:

- `results/otsu/otsu_predictions.json`
- printed COCO AP summary

### 4.3 SSOTSU

```bash
python -m src.otsu.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant ssotsu
```

Output:

- `results/otsu/ssotsu_predictions.json`
- printed COCO AP summary

### 4.4 CA-CFAR

```bash
python -m src.cfar.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant ca_cfar
```

Output:

- `results/cfar/ca_cfar_predictions.json`
- printed COCO AP summary

### 4.5 GO-CFAR

```bash
python -m src.cfar.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant go_cfar
```

### 4.6 SO-CFAR

```bash
python -m src.cfar.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant so_cfar
```

### 4.7 OS-CFAR

```bash
python -m src.cfar.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images \
  --variant os_cfar
```

### 4.8 Wavelet

```bash
python -m src.wavelet.run \
  --ann data/processed/annotations/test.json \
  --images data/processed/images
```

Output:

- `results/wavelet/wavelet_predictions.json`
- printed COCO AP summary

## 5. Reference values to compare against

These are the repo’s published DAAN test-set reference values from `data-v1.0/results/evaluation_summary_v3.csv` and `report/report.typ`.

| Method | AP@0.50 | AP@0.75 | AP@0.50:0.95 | AR@100 | Preds |
|---|---:|---:|---:|---:|---:|
| Morphological | 0.537408 | 0.314655 | 0.311198 | 0.661197 | 4289 |
| Otsu | 0.533529 | 0.337542 | 0.318168 | 0.676058 | 4663 |
| CA-CFAR | 0.482965 | 0.407970 | 0.387672 | 0.832043 | 4078 |
| SO-CFAR | 0.341926 | 0.315177 | 0.255130 | 0.672136 | 4268 |
| GO-CFAR | 0.314603 | 0.224139 | 0.203789 | 0.534520 | 3501 |
| SSOTSU | 0.030232 | 0.008771 | 0.013457 | 0.449432 | 49163 |
| Wavelet | 0.000005 | 0.000000 | 0.000001 | 0.000052 | 5916 |

Interpretation:

- Morphological and Otsu are the best classical baselines on DAAN.
- CA-CFAR has lower AP than Morphological/Otsu on DAAN but very high recall.
- Wavelet is effectively a failed baseline on this split.

## 6. Per-dataset classical references

From `data-v1.0/results/classical_by_dataset.csv`:

- `MANV` strongly favors CFAR, with `CA-CFAR AP@0.50 = 0.785235`.
- `DAAN` and `DARC` favor Morphological and Otsu.
- `Wavelet` is effectively zero on all three sub-datasets.

## 7. Deep-learning next steps after classical

Do not start this until dataset access is fixed and the Jetson `torch` path is confirmed.

Main points:

- `code-v1.0` expects tiled datasets and YOLO-format conversion for YOLO / RT-DETR.
- Faster R-CNN and U-Net still require the tiled inputs/annotations or mask datasets that are derived from the processed corpus.
- On this machine, install `torch` carefully for Jetson/Orin instead of assuming the generic CUDA 12.4 wheel is the correct final path.

## 8. Immediate unblock options

Any one of these will let the reproduction continue:

1. Provide a working `processed-v1.tar.zst` URL.
2. Place the extracted dataset under `data/processed/` manually on the SSD checkout.
3. Provide access credentials or an alternate distribution source if the release assets were made private or removed.
