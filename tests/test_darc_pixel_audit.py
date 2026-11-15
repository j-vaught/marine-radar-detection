import hashlib
import json

import cv2
import numpy as np
import pytest

from src.revision.darc_pixel_audit import (
    QUANTILES,
    audit,
    box_samples,
    histogram_quantiles,
    write_csv,
)


def test_histogram_quantiles_match_array_quantiles():
    samples = np.array([0, 0, 1, 2, 17, 17, 201, 255], dtype=np.uint8)
    result = histogram_quantiles(np.bincount(samples, minlength=256))
    np.testing.assert_allclose(list(result.values()), np.quantile(samples, QUANTILES))
    assert histogram_quantiles(np.zeros(256, dtype=np.int64)) == {}


def test_box_analysis_measures_pixels_and_excludes_other_boxes_from_annulus():
    signal = np.full((64, 64), 12, dtype=np.uint8)
    inside = np.ones(signal.shape, dtype=bool)
    inside[30, 30] = False
    signal[30:34, 30:34] = 71
    signal[30, 30] = 255
    signal[30:34, 39:45] = 255
    annotations = [
        {"bbox": [30, 30, 4, 4], "blob_peak": 75},
        {"bbox": [39, 30, 6, 4], "blob_peak": 255},
    ]
    first, second = box_samples(signal, inside, annotations)
    assert first["in_disc_box_pixels"] == 15
    assert first["box_max"] == 71
    assert first["box_max_minus_stored_peak"] == -4
    assert first["annulus_pixels"] == 616
    assert first["annulus_p50"] == 12
    assert first["box_p90_minus_annulus_p90"] == 59
    assert second["box_255_fraction"] == 1


def test_audit_verifies_selected_image_hash_and_outputs_summary(tmp_path):
    signal = np.zeros((530, 540), dtype=np.uint8)
    signal[513:517, 529:533] = 240
    annotations = [{"id": 1, "image_id": 1, "bbox": [529, 513, 4, 4], "blob_peak": 241}]
    images = []
    records = []
    for image_id in (1, 2):
        name = f"DARC_{image_id}.png"
        path = tmp_path / name
        assert cv2.imwrite(str(path), signal)
        images.append(
            {
                "id": image_id,
                "dataset": "DARC",
                "frame": image_id,
                "file_name": name,
                "height": 530,
                "width": 540,
            }
        )
        records.append(
            {
                "id": image_id,
                "site": "DARC",
                "frame": image_id,
                "file_name": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations}))
    manifest = {
        "annotation_sha256": hashlib.sha256(ann_path.read_bytes()).hexdigest(),
        "image_count": 2,
        "annotation_count": 1,
        "images": records,
        "sites": {"DARC": {"partitions": {"train": [1], "test": [2]}}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "DARC_2.png").write_bytes(b"corrupt holdout")

    result = audit(manifest_path, ann_path, tmp_path, ["DARC"], ["train"])
    assert result["image_hashes"]["verified"] == 1
    [group] = result["groups"]
    assert group["pooled_in_disc"]["disc_positive_quantiles"]["p50"] == 240
    assert group["counts"]["box_max_below_stored_peak"] == 1
    assert group["per_box"]["box_max_minus_stored_peak"]["mean"] == -1
    csv_path = tmp_path / "audit.csv"
    write_csv(result, csv_path)
    assert "DARC,train,per_box,box_max,mean,1,240.0" in csv_path.read_text()

    with pytest.raises(ValueError, match="Image digest"):
        audit(manifest_path, ann_path, tmp_path)
    ann_path.write_text("{}")
    with pytest.raises(ValueError, match="Annotation digest"):
        audit(manifest_path, ann_path, tmp_path)
