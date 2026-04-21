# RT-DETR

`src/rtdetr` is the completed Ultralytics RT-DETR baseline for the deep-learning side of this repo.

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
- [model.py](model.py): model construction, training, and inference helpers
- [config.yaml](config.yaml): model, training, and postprocessing defaults

## Commands

```bash
python -m src.rtdetr.train \
    --data data/processed/yolo/data.yaml \
    --config src/rtdetr/config.yaml

python -m src.rtdetr.infer \
    --weights runs/detect/rtdetr_seed0/weights/best.pt \
    --ann data/processed/annotations/all.json \
    --images data/processed/images

python -m src.rtdetr.experiments \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --output-dir results/rtdetr_matrix \
    --resume
```

## Matrix status

This method is finished at the experiment level.

The completed matrix lives in:

- `results/rtdetr_matrix/`
- `results/rtdetr_matrix/matrix.csv`
- `results/rtdetr_matrix/per_experiment/*.json`

The result tree contains all `20` expected experiments:

- `5` pooled
- `6` site-permutation
- `9` transfer

## Result summary

Across the completed matrix:

- mean `test_ap50 = 0.1356`
- mean `test_ap = 0.0924`
- mean `test_ap75 = 0.1081`
- best row: `P-RTDETR-1` with `AP50 = 0.8040`
- worst row: `P-RTDETR-4` with `AP50 = 0.0000`

RT-DETR is a completed baseline, but it is materially less robust than Faster R-CNN or YOLO26 on this corpus.
