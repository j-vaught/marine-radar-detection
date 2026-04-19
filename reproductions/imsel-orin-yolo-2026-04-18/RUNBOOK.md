# YOLOv11 Seed 42 Runbook

This runbook records the completed Jetson Orin YOLO baseline from the SSD-based checkout at:

`/media/imsel/imsel_ssd/projects/marine-radar-detection`

## 1. Environment

Machine snapshot:

- host: `ubuntu`
- kernel: `Linux ubuntu 5.15.148-tegra`
- architecture: `aarch64`
- date captured: `2026-04-18T22:44:09-04:00`

Python environment:

- local venv at `.venv`
- Jetson-compatible `torch 2.8.0`
- `ultralytics 8.4.38`
- `sahi 0.11.36`

Important Jetson runtime notes from this run:

- `MAXN` triggered Jetson over-current throttling under YOLO training
- the successful rerun used `nvpmodel MODE_50W`
- YOLO config was reduced to `batch_size=8` and `workers=4`

## 2. Data used

Prepared local deep-learning artifacts:

- `data/processed/annotations/train_tiled.json`
- `data/processed/annotations/val_tiled.json`
- `data/processed/annotations/test_tiled.json`
- `data/processed/tiles/{train,val,test}`
- `data/processed/yolo/data.yaml`

Observed tiled counts:

- train: `5712` images / `11669` annotations
- val: `3162` images / `4256` annotations
- test: `4560` images / `6980` annotations

## 3. Commands run

Training:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection
source .venv/bin/activate
sudo nvpmodel -m 3
python -m src.yolo.train --data data/processed/yolo/data.yaml --seeds 42
```

Held-out test inference:

```bash
cd /media/imsel/imsel_ssd/projects/marine-radar-detection
source .venv/bin/activate
python -m src.yolo.infer \
  --weights results/yolo/best_seed42.pt \
  --ann data/processed/annotations/test_tiled.json \
  --images data/processed/tiles/test
```

## 4. Outputs collected into this bundle

- training log: `logs/yolo_seed42_train.log`
- held-out test eval: `logs/yolo_seed42_test_eval.log`
- checkpoint: `weights/best_seed42.pt`
- predictions: `predictions/best_seed42_predictions.json`
- Ultralytics artifacts:
  - `artifacts/args.yaml`
  - `artifacts/results.csv`
  - `artifacts/results.png`
  - `artifacts/confusion_matrix.png`
  - `artifacts/BoxPR_curve.png`

## 5. Metrics

Validation metrics recorded by Ultralytics:

- best recorded validation epoch by `mAP50-95(B)`: epoch `1`
- best validation `precision(B)`: `0.48874`
- best validation `recall(B)`: `0.76739`
- best validation `mAP50(B)`: `0.65833`
- best validation `mAP50-95(B)`: `0.48368`

Held-out test COCO metrics from `predictions/best_seed42_predictions.json`:

- `AP@[0.50:0.95] = 0.5482294695`
- `AP50 = 0.7383038421`
- `AP75 = 0.6812622075`
- `AP_small = 0.5482178302`
- `AP_medium = 0.6432586063`
- `AR@1 = 0.3411318052`
- `AR@10 = 0.7709169054`
- `AR@100 = 0.7719054441`

## 6. Artifact hashes

- `weights/best_seed42.pt`
  - `sha256: 48dfa2ccac782abd8c4a4e752908e6521abf9117e736e370db3b37c6c4327241`
- `predictions/best_seed42_predictions.json`
  - `sha256: 57d1b3a83192d2d53513161c2f0c2433b51255c4686395645b87698e06486c13`
- `logs/yolo_seed42_train.log`
  - `sha256: f6c0bc614a8e70f8b4625443fdffadf5c7a951410e4a768c23e1d6cba68116e9`
