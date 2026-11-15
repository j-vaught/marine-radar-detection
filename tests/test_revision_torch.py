"""Focused checks of revision data handling without running training."""

import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# The common protocol and evaluator are supplied by the revision framework.
for name in ("protocol", "evaluation"):
    try:
        importlib.import_module(f"src.revision.{name}")
    except ModuleNotFoundError:
        sys.modules[f"src.revision.{name}"] = types.ModuleType(name)

from src.revision.torch_models import RadarDataset, _merge


def coco():
    return {
        "images": [{"id": 1, "file_name": "tile.png", "height": 1030, "width": 1062}],
        "annotations": [{"image_id": 1, "bbox": [500, 500, 30, 20]}],
        "categories": [{"id": 1}],
    }


def test_roi_and_native_detection_boxes(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "tile.png"), np.full((1030, 1062), 255, np.uint8))
    image, target = RadarDataset(coco(), tmp_path, "faster_rcnn", (640, 640))[0]
    assert image.shape == (3, 1030, 1062)
    assert image[0, 0, 0] == 0
    assert image[0, 515, 531] == 1
    assert target["boxes"].tolist() == [[500, 500, 530, 520]]


def test_unet_resize_and_missing_image(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "tile.png"), np.full((1030, 1062), 255, np.uint8))
    dataset = RadarDataset(coco(), tmp_path, "unet", (640, 640))
    image, mask = dataset[0]
    assert image.shape == mask.shape == (1, 640, 640)
    assert image[0, 0, 0] == mask[0, 0, 0] == 0
    assert mask.sum() > 0
    (tmp_path / "tile.png").unlink()
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_nested_overrides_preserve_unspecified_values():
    assert _merge({"training": {"epochs": 2, "workers": 0}}, {"training": {"epochs": 3}}) == {
        "training": {"epochs": 3, "workers": 0}
    }


def test_unet_pixel_threshold_grid_uses_one_inference_per_image(tmp_path: Path, monkeypatch):
    from src.revision.torch_models import _unet_predictions_grid
    from src.unet import model as unet_model

    cv2.imwrite(str(tmp_path / "tile.png"), np.full((1030, 1062), 255, np.uint8))
    calls = []

    def predict(*args, **kwargs):
        calls.append(1)
        prob = np.zeros((1030, 1062), dtype=np.float32)
        prob[500:520, 500:520] = 0.45
        return prob

    monkeypatch.setattr(unet_model, "predict_prob_map", predict)
    post = {"min_area": 1, "max_area": 1000, "dilation": 0, "box_padding": 0}
    predictions = _unet_predictions_grid(
        types.SimpleNamespace(eval=lambda: None),
        coco(),
        tmp_path,
        (640, 640),
        "cpu",
        post,
        [0.3, 0.5],
    )
    assert len(calls) == 1
    assert len(predictions[0.3]) == 1
    assert predictions[0.5] == []
    assert predictions[0.3][0]["bbox"] == [500.0, 500.0, 20.0, 20.0]
