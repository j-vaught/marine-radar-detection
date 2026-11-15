"""Boundary and deterministic replay checks for the frozen DARC input policy."""

import hashlib

import cv2
import numpy as np
import pytest
import torch

from src.revision import darc_data as data
from src.revision.torch_models import RadarDataset


def image_with_support(value, count=256):
    image = np.full(data.SHAPE, 3, dtype=np.uint8)
    pixels = np.flatnonzero(data.disc_mask())[:count]
    image.flat[pixels] = value
    return image


def test_intensity_boundaries_mask_pedestal_and_no_uint8_underflow():
    image = image_with_support(19)
    image[515, 531] = 0
    actual, stats = data.normalize_intensity(image, "I1")
    assert not stats["intensity_fallback"]  # Exactly 256 pixels and span 16 are valid.
    assert stats["high"] == 19
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    assert actual.min() == 0 and actual.max() == 1
    assert actual[515, 531] == 0 and actual[515, 532] == 0
    assert np.all(actual[~data.disc_mask()] == 0)
    for value, count, reason in ((19, 255, "low_support"), (18, 256, "small_span")):
        image = image_with_support(value, count)
        normalized, stats = data.normalize_intensity(image, "I1")
        baseline, _ = data.normalize_intensity(image, "I0")
        assert stats["intensity_fallback_reason"] == reason
        np.testing.assert_array_equal(normalized, baseline)
    empty, stats = data.normalize_intensity(np.zeros(data.SHAPE, np.uint8), "I1")
    assert stats["high"] is None and not empty.any()


def test_linear_quantile_clipping_and_type_rejection():
    image = image_with_support(100, 400)
    support = np.flatnonzero(data.disc_mask())[:400]
    image.flat[support[-2:]] = 255
    actual, stats = data.normalize_intensity(image, "I1")
    expected_high = np.percentile(np.r_[np.full(398, 100), 255, 255], 99.5, method="linear")
    assert stats["high"] == expected_high
    assert actual.flat[support[-1]] == 1
    with pytest.raises(ValueError):
        data.normalize_intensity(image.astype(np.float32), "I1")


def test_hash_uniform_independent_known_vector():
    prefix = "darc-processed-v1:zoom:DARC:0:0:913"
    assert data.hash_uniform(f"{prefix}:gate") == 0.22368688660433178
    assert data.hash_uniform(f"{prefix}:scale0") == 0.7365349994985514
    assert data.hash_uniform(f"{prefix}:scale1") == 0.25080395974528924


def test_stream_odd_cycle_no_dropped_examples_and_fixed_order():
    images = [{"id": image_id} for image_id in (3, 1, 2)]
    stream = data.train_stream(images, "DARC", 0, 8)
    expected = []
    for cycle in range(3):
        expected += sorted(
            (1, 2, 3),
            key=lambda image_id: (
                hashlib.sha256(
                    f"darc-processed-v1:train-order:DARC:0:{cycle}:{image_id}".encode()
                ).digest(),
                image_id,
            ),
        )
    assert stream == list(enumerate(expected[:8]))
    assert stream == data.train_stream(list(reversed(images)), "DARC", 0, 8)
    assert {image_id for _, image_id in stream[:3]} == {1, 2, 3}
    with pytest.raises(ValueError):
        data.train_stream([{"id": 1}, {"id": 1}], "DARC", 0)


@pytest.mark.parametrize(
    "source,n,quotas",
    [
        ("DAAN", 456, [27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 26]),
        ("MANV", 571, [26, 27, 26, 27, 26, 27, 27, 26, 26, 26, 26, 26]),
    ],
)
def test_count_subset_integer_quotas_and_temporal_order(source, n, quotas):
    images = [{"id": k + 1, "frame": 2 * k} for k in range(n)]
    selected = data.select_count_subset(images, source, 1)
    assert len(selected) == len(set(selected)) == 316 and selected == sorted(selected)
    assert selected == data.select_count_subset(list(reversed(images)), source, 1)
    for k, quota in enumerate(quotas):
        assert sum(k * n // 12 < image_id <= (k + 1) * n // 12 for image_id in selected) == quota
    with pytest.raises(ValueError):
        data.select_count_subset(images, source, 1, n + 1)


def test_affine_inverse_continuous_geometry_and_all_corners():
    boxes = np.asarray([[510.2, 500.3, 540.7, 530.9]])
    for scale in (2**-0.5, 2**0.5):
        actual = data.affine_boxes(boxes, scale)
        np.testing.assert_allclose(data.affine_boxes(actual, 1 / scale), boxes, atol=1e-12)
        assert data.valid_boxes(actual)
    assert not data.valid_boxes([[900, 850, 910, 860]])  # Inside canvas, outside disc.
    assert data.valid_boxes([[900, 850, 910, 860]], require_disc=False)
    assert not data.valid_boxes([[10, 10, 10, 20]])
    assert not data.valid_boxes([[np.nan, 500, 520, 520]])


def test_gate_identity_redraw_and_exhausted_rejection(monkeypatch):
    image = np.zeros(data.SHAPE, np.float32)
    image[500:530, 510:540] = 1
    boxes = [[510, 500, 540, 530]]
    image0, boxes0, stats = data.transform_training(image, boxes, "DARC", 0, 0, 913, "S1")
    np.testing.assert_array_equal(image0, image)
    np.testing.assert_array_equal(boxes0, boxes)
    assert stats["scale_identity_reason"] == "gate"
    peripheral = [[980, 505, 990, 525]]
    draws = iter((0.9, 0.99, 0.0))
    monkeypatch.setattr(data, "hash_uniform", lambda _: next(draws))
    _, warped, stats = data.transform_training(image, peripheral, "DARC", 0, 1, 2, "S1")
    assert stats["scale_rejections"] == 1 and stats["scale"] < 1
    assert data.valid_boxes(warped) and len(warped) == 1
    draws = iter((0.9, 0.99, 0.99))
    image1, boxes1, stats = data.transform_training(image, peripheral, "DARC", 0, 1, 2, "S1")
    assert stats["scale_rejections"] == 2 and stats["scale_identity_reason"] == "rejected"
    np.testing.assert_array_equal(image1, image)
    np.testing.assert_array_equal(boxes1, peripheral)


def test_zoom_matches_impulse_and_geometry_shared_between_intensities(monkeypatch):
    image = np.zeros(data.SHAPE, np.float32)
    image[515, 551] = 1
    boxes = [[549, 513, 553, 517]]
    monkeypatch.setattr(data, "hash_uniform", lambda text: 0.9)
    actual, transformed, stats = data.transform_training(image, boxes, "DARC", 2, 4, 3, "S1")
    _, transformed2, stats2 = data.transform_training(image * 0.5, boxes, "DARC", 2, 4, 3, "S1")
    assert stats == stats2
    np.testing.assert_array_equal(transformed, transformed2)
    peak_y, peak_x = np.unravel_index(actual.argmax(), actual.shape)
    assert peak_y == 515 and abs(peak_x - (531 + 20 * stats["scale"])) <= 1
    assert np.all(actual[~data.disc_mask()] == 0)


def test_dataset_bridge_input_hash_cache_and_canonical_outside_disc_labels(tmp_path):
    image = np.arange(np.prod(data.SHAPE), dtype=np.uint8).reshape(data.SHAPE)
    path = tmp_path / "one.png"
    cv2.imwrite(str(path), image)
    coco = {
        "images": [{"id": 1, "file_name": "one.png", "height": 1024, "width": 1050}],
        "annotations": [{"id": 2, "image_id": 1, "bbox": [900, 850, 10, 10]}],
    }
    expected = {1: hashlib.sha256(path.read_bytes()).hexdigest()}
    dataset = data.DARCData(coco, tmp_path, expected_hashes=expected, cache_size=1)
    actual, target, stats = dataset[0]
    legacy, legacy_target = RadarDataset(coco, tmp_path, "faster_rcnn", (640, 640))[0]
    assert torch.equal(actual, legacy)
    assert all(torch.equal(target[key], legacy_target[key]) for key in target)
    assert torch.equal(actual, dataset[0][0]) and len(dataset.cache) == 1
    assert stats["scale"] == 1 and stats["presentation_index"] is None
    with pytest.raises(ValueError, match="no expected digest"):
        data.DARCData(coco, tmp_path, expected_hashes={})[0]
    with pytest.raises(ValueError, match="digest mismatch"):
        data.DARCData(coco, tmp_path, expected_hashes={1: "wrong"})[0]
    with pytest.raises(ValueError, match="explicit training"):
        data.DARCData(coco, tmp_path, scale="S1")
