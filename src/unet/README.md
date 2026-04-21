# U-Net

`src/unet` is the completed segmentation-to-box baseline for the deep-learning side of this repo.

It includes:

- standard train / infer entrypoints
- a full pooled / site / transfer experiment matrix
- validation-time postprocessing tuning before final test evaluation

## Entry points

- [train.py](train.py): train one or more seeds on explicit image/mask paths
- [infer.py](infer.py): run a trained checkpoint and emit COCO detections
- [experiments.py](experiments.py): run the full P/S/T matrix
- [matrix.py](matrix.py): enumerate the experiment definitions
- [dataset.py](dataset.py): segmentation dataset and split helpers
- [model.py](model.py): U-Net model, training, inference helpers
- [utils.py](utils.py): mask-to-box conversion and postprocess helpers
- [config.yaml](config.yaml): model, training, and postprocessing defaults

## Commands

```bash
python -m src.unet.train \
    --train-images data/processed/images \
    --train-masks data/processed/masks \
    --val-images data/processed/images \
    --val-masks data/processed/masks

python -m src.unet.infer \
    --weights results/unet/best_seed0.pth \
    --ann data/processed/annotations/all.json \
    --images data/processed/images

python -m src.unet.experiments \
    --ann data/processed/annotations/all.json \
    --images data/processed/images \
    --output-dir results/unet_matrix \
    --resume
```

## Matrix status

This method is finished at the experiment level.

The completed matrix lives in:

- `results/unet_matrix/`
- `results/unet_matrix/matrix.csv`
- `results/unet_matrix/per_experiment/*.json`

There is also an older non-matrix result tree in `results/unet/`, but `results/unet_matrix/` is the complete comparable experiment set.

## Result summary

Across the completed matrix:

- mean `test_ap50 = 0.6747`
- mean `test_ap = 0.4106`
- mean `test_ap75 = 0.4594`
- mean `train_seconds = 136.8`

U-Net is a completed and useful baseline, but it is materially weaker than Faster R-CNN on this corpus.
