# imsel Orin YOLO Reproduction (2026-04-18)

This folder contains a clean, branch-local reproduction bundle for the first completed YOLOv11 baseline run from the SSD-based Jetson Orin checkout, kept separate from the repo's ignored `data/` and `results/` trees.

## What is included

- `RUNBOOK.md`: setup, training, and evaluation notes for this run
- `yolo_seed42_summary.csv`: compact summary of the recorded validation and held-out test metrics
- `logs/`: the completed training log and the held-out test COCO evaluation log
- `predictions/`: the test-set COCO result JSON written by `src.yolo.infer`
- `weights/`: the stable `best_seed42.pt` checkpoint
- `artifacts/`: selected Ultralytics output files (`args.yaml`, `results.csv`, `results.png`, confusion matrix, PR curve)

## What is not included

- Extracted dataset files under `data/processed/`
- The ignored root-level `results/` tree itself
- Intermediate failed MAXN run directories
- Container-only bootstrap artifacts and venv directories

## Dataset and split used

This run used the tiled deep-learning artifacts already prepared in the local ignored data tree:

- `train_tiled.json`: `5712` images, `11669` annotations
- `val_tiled.json`: `3162` images, `4256` annotations
- `test_tiled.json`: `4560` images, `6980` annotations

Paper-compatible split convention:

- `MANV_*` -> train
- `DARC_*` -> val
- `DAAN_*` -> test

## High-level outcome

Completed checkpoint:

- `weights/best_seed42.pt`

Held-out DAAN test-tile evaluation from `predictions/best_seed42_predictions.json`:

- `AP@[0.50:0.95] = 0.548229`
- `AP50 = 0.738304`
- `AP75 = 0.681262`
- `AR@100 = 0.771905`

Recorded validation metrics in Ultralytics `results.csv` peaked at epoch `1`, which is consistent with the run stopping after `21` epochs under `patience: 20`.

## Notes

- The final successful run used Jetson `MODE_50W`, `batch_size=8`, and `workers=4` to avoid the over-current throttling seen in `MAXN`.
- Test inference was re-run after fixing a bug that previously reloaded the YOLO checkpoint once per image.
