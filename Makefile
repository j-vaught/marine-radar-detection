SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup preprocess-daan preprocess-haxr classical train-yolo train-frcnn train-unet train-rtdetr evaluate figures clean

help:
	@echo "Available targets:"
	@echo "  setup           - Create venv and install dependencies"
	@echo "  preprocess-daan - Preprocess DLR DAAN dataset"
	@echo "  preprocess-haxr - Preprocess DLR HAXR dataset"
	@echo "  classical       - Run all classical detection methods"
	@echo "  train-yolo      - Train YOLOv11-OBB"
	@echo "  train-frcnn     - Train Faster R-CNN"
	@echo "  train-unet      - Train U-Net"
	@echo "  train-rtdetr    - Train RT-DETR"
	@echo "  evaluate        - Run evaluation on all predictions"
	@echo "  figures         - Generate report figures"
	@echo "  clean           - Remove results and processed data"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -e ".[dev]"

preprocess-daan:
	$(PYTHON) scripts/preprocess.py --dataset DAAN --raw-dir data/raw/DAAN --apply-clahe

preprocess-haxr:
	$(PYTHON) scripts/preprocess.py --dataset HAXR --raw-dir data/raw/HAXR --apply-clahe

classical:
	$(PYTHON) scripts/run_classical.py \
		--ann-path data/processed/annotations/DAAN_test.json \
		--image-dir data/processed/tiles

train-yolo:
	$(PYTHON) scripts/train_dl.py --model yolov11 \
		--config configs/deep_learning/yolov11.yaml \
		--data data/processed

train-frcnn:
	$(PYTHON) scripts/train_dl.py --model faster_rcnn \
		--config configs/deep_learning/faster_rcnn.yaml \
		--data data/processed

train-unet:
	$(PYTHON) scripts/train_dl.py --model unet \
		--config configs/deep_learning/unet.yaml \
		--data data/processed

train-rtdetr:
	$(PYTHON) scripts/train_dl.py --model rt_detr \
		--config configs/deep_learning/rt_detr.yaml \
		--data data/processed

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--gt-path data/processed/annotations/DAAN_test.json \
		--pred-dir results/ \
		--output-dir results \
		--run-tide

figures:
	$(PYTHON) scripts/generate_report_figures.py \
		--results-csv results/evaluation_summary.csv \
		--output-dir results/figures

clean:
	rm -rf results/classical results/deep_learning results/figures results/tables
	rm -rf results/evaluation_summary.csv
