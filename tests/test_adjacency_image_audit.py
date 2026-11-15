import hashlib

import cv2
import numpy as np
import pytest

from src.revision.adjacency_image_audit import audit, compare_partition, perceptual_hash
from src.revision.protocol import save_json


def test_phash_mask_dc_and_constant():
    # All changes outside the fixed circle must leave the result unchanged.
    base = np.zeros((1024, 1050), dtype=np.uint8)
    base[515, 531] = 255
    modified = base.copy()
    modified[0, 0] = 255
    assert perceptual_hash(base) == perceptual_hash(modified)
    assert perceptual_hash(base) >> 63 == 0
    assert perceptual_hash(np.zeros_like(base)) == 0
    assert perceptual_hash(base) == perceptual_hash(base.copy())
    with pytest.raises(ValueError):
        perceptual_hash(np.zeros((2, 2, 3), dtype=np.uint8))


def test_nearest_ties_and_index_distance():
    records = {
        1: {"frame": 10, "sha256": "a"},
        2: {"frame": 30, "sha256": "b"},
        3: {"frame": 29, "sha256": "b"},
    }
    result = compare_partition(records, {1: 0, 2: 0, 3: 1}, [2, 1], [3])
    row = result["frames"][0]
    assert row["nearest_train_image_id"] == 1
    assert row["hamming_distance"] == 1
    assert row["nearest_hash_frame_index_distance"] == 19
    assert row["minimum_frame_index_distance"] == 1
    assert row["exact_sha256_train_image_ids"] == [2]
    assert result["summary"]["hamming_threshold_candidate_fractions"] == {"0": 0, "4": 1, "8": 1}
    with pytest.raises(ValueError):
        compare_partition(records, {1: 0}, [1], [1])


def test_manifest_hash_verification_and_arms(tmp_path):
    images = []
    for i in range(1, 5):
        image = np.zeros((1024, 1050), np.uint8)
        image[500, 500 + i] = i * 40
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        raw = encoded.tobytes()
        path = tmp_path / f"{i}.png"
        path.write_bytes(raw)
        images.append(
            {
                "id": i,
                "site": "A",
                "frame": i,
                "file_name": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    m = {
        "images": images,
        "image_count": 4,
        "sites": {
            "A": {
                "partitions": {"train": [1, 2], "test": [4]},
                "adjacency": {str(i): [2, 3] for i in range(3)},
            }
        },
    }
    p = tmp_path / "manifest.json"
    save_json(p, m)
    result = audit(p, tmp_path)
    assert result["verified_image_count"] == 4
    assert set(result["sites"]["A"]) == {"buffered", "adjacency_0", "adjacency_1", "adjacency_2"}
    (tmp_path / "1.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit(p, tmp_path)
