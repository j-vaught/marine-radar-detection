#!/usr/bin/env bash
# Train all 4 DL models in parallel, one per GPU.
# Each model runs 3 seeds sequentially on its assigned GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python"
DATA_DIR="data/processed"
YOLO_DATA="data/processed/yolo/data.yaml"
OUTPUT_DIR="results/deep_learning"

echo "Starting parallel training on 4 GPUs..."
echo "Logs: ${OUTPUT_DIR}/{model}/train.log"

mkdir -p "${OUTPUT_DIR}"

# GPU 0: YOLOv11-OBB
$PYTHON scripts/train_dl.py \
    --model yolov11 \
    --config configs/deep_learning/yolov11.yaml \
    --data "$YOLO_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --device 0 \
    2>&1 | tee "${OUTPUT_DIR}/yolov11_train.log" &
PID_YOLO=$!

# GPU 1: RT-DETR
$PYTHON scripts/train_dl.py \
    --model rt_detr \
    --config configs/deep_learning/rt_detr.yaml \
    --data "$YOLO_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --device 1 \
    2>&1 | tee "${OUTPUT_DIR}/rt_detr_train.log" &
PID_RTDETR=$!

# GPU 2: Faster R-CNN
$PYTHON scripts/train_dl.py \
    --model faster_rcnn \
    --config configs/deep_learning/faster_rcnn.yaml \
    --data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device 2 \
    2>&1 | tee "${OUTPUT_DIR}/faster_rcnn_train.log" &
PID_FRCNN=$!

# GPU 3: U-Net
$PYTHON scripts/train_dl.py \
    --model unet \
    --config configs/deep_learning/unet.yaml \
    --data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device 3 \
    2>&1 | tee "${OUTPUT_DIR}/unet_train.log" &
PID_UNET=$!

echo ""
echo "PIDs: YOLOv11=$PID_YOLO  RT-DETR=$PID_RTDETR  FRCNN=$PID_FRCNN  U-Net=$PID_UNET"
echo "Waiting for all training jobs to complete..."

wait $PID_YOLO && echo "YOLOv11 done" || echo "YOLOv11 FAILED"
wait $PID_RTDETR && echo "RT-DETR done" || echo "RT-DETR FAILED"
wait $PID_FRCNN && echo "Faster R-CNN done" || echo "Faster R-CNN FAILED"
wait $PID_UNET && echo "U-Net done" || echo "U-Net FAILED"

echo ""
echo "All training complete."
