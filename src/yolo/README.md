# YOLO26

`src/yolo` is the completed Ultralytics YOLO26 baseline for the deep-learning side of this repo.

It includes:

- standard train / infer entrypoints
- a full pooled / site / transfer experiment matrix
- validation-time score-threshold tuning before final test evaluation

## Entry points

- [train.py](train.py): train one or more seeds on explicit YOLO-format splits
- [infer.py](infer.py): run a trained checkpoint on a COCO test split
- [experiments.py](experiments.py): run the full P/S/T matrix
- [matrix.py](matrix.py): enumerate the experiment definitions
- [coco_to_yolo.py](coco_to_yolo.py): convert COCO boxes into YOLO label files
- [model.py](model.py): model construction, training, inference, and SAHI helpers
- [config.yaml](config.yaml): model, training, and postprocessing defaults

## Commands

```bash
python -m src.yolo.train \
    --data data/processed/yolo/data.yaml \
    --config src/yolo/config.yaml

python -m src.yolo.infer \
    --weights runs/detect/yolo_seed0/weights/best.pt \
    --ann data/processed/annotations/all.json \
    --images data/processed/images

python -m src.yolo.experiments \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --output-dir results/yolo26_matrix \
    --resume
```

## Matrix status

This method is finished at the experiment level.

The completed matrix lives in:

- `results/yolo26_matrix/`
- `results/yolo26_matrix/matrix.csv`
- `results/yolo26_matrix/per_experiment/*.json`

The result tree contains all `20` expected experiments:

- `5` pooled
- `6` site-permutation
- `9` transfer

## Result summary

Across the completed matrix:

- mean `test_ap50 = 0.7939`
- mean `test_ap = 0.6916`
- mean `test_ap75 = 0.7851`
- best row: `T-Y26-MvMv` with `AP50 = 0.9642`
- worst row: `T-Y26-MvDc` with `AP50 = 0.3470`

YOLO26 is a strong completed baseline. It matches the repo-wide pattern of excellent pooled performance with asymmetric transfer behavior across sites.
