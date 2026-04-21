# Faster R-CNN

`src/faster_rcnn` is the completed torchvision Faster R-CNN baseline for the deep-learning side of this repo.

It includes:

- standard train / infer entrypoints
- a full pooled / site / transfer experiment matrix
- threshold tuning on validation predictions before final test evaluation

## Entry points

- [train.py](train.py): train one or more seeds on explicit train/val paths
- [infer.py](infer.py): run a trained checkpoint on a COCO test split
- [experiments.py](experiments.py): run the full P/S/T matrix
- [matrix.py](matrix.py): enumerate the experiment definitions
- [dataset.py](dataset.py): COCO dataset adapter
- [model.py](model.py): model construction, training, inference helpers
- [config.yaml](config.yaml): model, training, and postprocessing defaults

## Commands

```bash
python -m src.faster_rcnn.train \
    --train-ann data/processed/annotations/all.json \
    --val-ann data/processed/annotations/all.json \
    --train-images data/processed/images \
    --val-images data/processed/images

python -m src.faster_rcnn.infer \
    --weights results/faster_rcnn/best_seed0.pth \
    --ann data/processed/annotations/all.json \
    --images data/processed/images

python -m src.faster_rcnn.experiments \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --output-dir results/faster_rcnn_matrix \
    --resume
```

## Matrix status

This method is finished at the experiment level.

The completed matrix lives in:

- `results/faster_rcnn_matrix/`
- `results/faster_rcnn_matrix/matrix.csv`
- `results/faster_rcnn_matrix/per_experiment/*.json`

The result tree contains all `20` expected experiments:

- `5` pooled
- `6` site-permutation
- `9` transfer

## Result summary

Across the completed matrix:

- mean `test_ap50 = 0.8058`
- mean `test_ap = 0.6677`
- mean `test_ap75 = 0.7936`
- mean `train_seconds = 277.0`

This makes Faster R-CNN the strongest completed deep-learning baseline currently present in the repo.
