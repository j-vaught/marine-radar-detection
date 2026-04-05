SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

DATA_DIR := data/processed
ANN_DIR := $(DATA_DIR)/annotations
YOLO_DATA := $(DATA_DIR)/yolo/data.yaml
RESULTS := results

.PHONY: help setup preprocess preprocess-dlr preprocess-haxr \
        classical train-all train-yolo train-frcnn train-unet train-rtdetr \
        infer-dl evaluate stats figures clean

help:
	@echo "=== Marine Radar Detection Benchmark ==="
	@echo ""
	@echo "Data preparation:"
	@echo "  preprocess      - Full pipeline (DLR + HAXR -> tiles + YOLO + masks)"
	@echo "  preprocess-dlr  - DLR datasets only (DAAN/DARC/MANV)"
	@echo "  preprocess-haxr - HAXR dataset only"
	@echo ""
	@echo "Classical methods:"
	@echo "  classical       - Run all 8 classical detectors on test set"
	@echo ""
	@echo "Deep learning:"
	@echo "  train-all       - Train all 4 DL models in parallel (4 GPUs)"
	@echo "  train-yolo      - Train YOLOv11-OBB (GPU 0)"
	@echo "  train-frcnn     - Train Faster R-CNN (GPU 2)"
	@echo "  train-unet      - Train U-Net (GPU 3)"
	@echo "  train-rtdetr    - Train RT-DETR (GPU 1)"
	@echo "  infer-dl        - Run all trained DL models on test set"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate        - COCO metrics + TIDE for all methods"
	@echo "  stats           - Bootstrap significance tests"
	@echo "  figures         - Generate all report figures"
	@echo ""
	@echo "  clean           - Remove results (keeps processed data)"
	@echo "  clean-all       - Remove results AND processed data"

# ── Setup ──

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -e ".[dev]"
	$(PIP) install h5py scikit-learn

# ── Preprocessing ──

preprocess:
	$(PYTHON) scripts/preprocess.py \
		--config configs/data_split.yaml \
		--raw-dir data/raw \
		--output-dir $(DATA_DIR) \
		--apply-clahe

preprocess-dlr:
	$(PYTHON) scripts/preprocess.py \
		--config configs/data_split.yaml \
		--raw-dir data/raw \
		--output-dir $(DATA_DIR) \
		--skip-haxr \
		--apply-clahe

preprocess-haxr:
	$(PYTHON) scripts/preprocess.py \
		--config configs/data_split.yaml \
		--raw-dir data/raw \
		--output-dir $(DATA_DIR) \
		--skip-dlr

# ── Classical Methods ──

classical:
	$(PYTHON) scripts/run_classical.py \
		--ann-path $(ANN_DIR)/test.json \
		--image-dir $(DATA_DIR)/images \
		--output-dir $(RESULTS)/classical

# ── DL Training ──

train-all:
	chmod +x scripts/train_all.sh
	bash scripts/train_all.sh

train-yolo:
	$(PYTHON) scripts/train_dl.py --model yolov11 \
		--config configs/deep_learning/yolov11.yaml \
		--data $(YOLO_DATA) \
		--output-dir $(RESULTS)/deep_learning \
		--device 0

train-frcnn:
	$(PYTHON) scripts/train_dl.py --model faster_rcnn \
		--config configs/deep_learning/faster_rcnn.yaml \
		--data $(DATA_DIR) \
		--output-dir $(RESULTS)/deep_learning \
		--device 2

train-unet:
	$(PYTHON) scripts/train_dl.py --model unet \
		--config configs/deep_learning/unet.yaml \
		--data $(DATA_DIR) \
		--output-dir $(RESULTS)/deep_learning \
		--device 3

train-rtdetr:
	$(PYTHON) scripts/train_dl.py --model rt_detr \
		--config configs/deep_learning/rt_detr.yaml \
		--data $(YOLO_DATA) \
		--output-dir $(RESULTS)/deep_learning \
		--device 1

# ── DL Inference ──

infer-dl:
	@for model in yolov11 faster_rcnn unet rt_detr; do \
		echo "\n=== Inference: $$model ==="; \
		$(PYTHON) scripts/run_dl_inference.py \
			--model $$model \
			--weights-dir $(RESULTS)/deep_learning/$$model \
			--config configs/deep_learning/$$model.yaml \
			--test-ann $(ANN_DIR)/test_tiled.json \
			--image-dir $(DATA_DIR)/tiles/test \
			--output-dir $(RESULTS)/deep_learning; \
	done

# ── Evaluation ──

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--gt-path $(ANN_DIR)/test.json \
		--classical-dir $(RESULTS)/classical \
		--dl-dir $(RESULTS)/deep_learning \
		--output-dir $(RESULTS) \
		--run-tide

stats:
	$(PYTHON) scripts/run_stats.py \
		--gt-path $(ANN_DIR)/test.json \
		--pred-dir $(RESULTS) \
		--output-dir $(RESULTS)/tables \
		--n-resamples 10000

figures:
	$(PYTHON) scripts/generate_report_figures.py \
		--results-csv $(RESULTS)/evaluation_summary.csv \
		--output-dir $(RESULTS)/figures

# ── Cleanup ──

clean:
	rm -rf $(RESULTS)/classical $(RESULTS)/deep_learning
	rm -rf $(RESULTS)/figures $(RESULTS)/tables $(RESULTS)/tide
	rm -f $(RESULTS)/evaluation_summary.csv

clean-all: clean
	rm -rf $(DATA_DIR)
