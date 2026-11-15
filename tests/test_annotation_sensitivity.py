from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.revision.annotation_sensitivity import (
    _load_references,
    build_reference_views,
    center_agreement,
    score_reference_views,
)
from src.revision.protocol import save_json, sha256_file


def _image(image_id, site):
    return {"id": image_id, "width": 40, "height": 30, "dataset": site}


def _annotation(ann_id, image_id, bbox):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "segmentation": [],
        "blob_peak": 255,
    }


def _prediction(bbox, score):
    return {"image_id": 1, "category_id": 1, "bbox": bbox, "score": score}


def _release():
    return {
        "images": [_image(1, "DARC"), _image(2, "DAAN"), _image(3, "DAAN")],
        "annotations": [
            _annotation(1, 1, [0, 1, 2, 4]),
            _annotation(2, 1, [39, 29, 1, 1]),
            _annotation(3, 1, [10, 12, 10, 5]),
            _annotation(4, 2, [5, 5, 12, 8]),
            _annotation(5, 2, [10, 12, 16, 16]),
            _annotation(6, 3, [1, 1, 1, 1]),
        ],
        "categories": [{"id": 1, "name": "vessel"}],
    }


class AnnotationSensitivityTest(unittest.TestCase):
    def test_clip_transform_and_retain_inventory(self):
        released = _release()
        original = json.dumps(released)
        views = build_reference_views(released, (8, 6))
        self.assertEqual(len(views), 12)
        ids = [(a["id"], a["image_id"], a["category_id"]) for a in released["annotations"]]
        for reference, _ in views.values():
            self.assertEqual(
                ids, [(a["id"], a["image_id"], a["category_id"]) for a in reference["annotations"]]
            )
            for annotation in reference["annotations"]:
                x, y, w, h = annotation["bbox"]
                self.assertGreaterEqual(min(x, y), 0)
                self.assertGreater(min(w, h), 0)
                self.assertLessEqual(x + w, 40)
                self.assertLessEqual(y + h, 30)
                self.assertAlmostEqual(annotation["area"], w * h)
        self.assertEqual(json.dumps(views["baseline"][0]), original)
        for variant, expected in {
            "scale_0.8": [11, 12.5, 8, 4],
            "scale_1.2": [9, 11.5, 12, 6],
            "shift_x_-4": [6, 12, 10, 5],
            "shift_x_2": [12, 12, 10, 5],
            "shift_y_-2": [10, 10, 10, 5],
            "shift_y_4": [10, 16, 10, 5],
            "fixed_source_envelope": [11, 11.5, 8, 6],
        }.items():
            self.assertEqual(views[variant][0]["annotations"][2]["bbox"], expected)
        shifted_left, ledger = views["shift_x_-4"]
        self.assertEqual(shifted_left["annotations"][0]["bbox"], [0, 1, 1, 4])
        self.assertIn(1, ledger["edge_retained_annotation_ids"])
        self.assertIn(1, ledger["clipped_annotation_ids"])
        shifted_right, ledger = views["shift_x_2"]
        self.assertEqual(shifted_right["annotations"][1]["bbox"], [39, 29, 1, 1])
        self.assertIn(2, ledger["edge_retained_annotation_ids"])
        shifted_up, ledger = views["shift_y_-2"]
        self.assertEqual(shifted_up["annotations"][0]["bbox"], [0, 0, 2, 3])
        self.assertIn(1, ledger["clipped_annotation_ids"])
        self.assertNotIn(1, ledger["edge_retained_annotation_ids"])
        views["baseline"][0]["images"][0]["width"] = 0
        shifted_up["annotations"][0]["segmentation"].append(123)
        self.assertEqual(json.dumps(released), original)

    def test_envelope_uses_only_frozen_source_training_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            ann = Path(directory) / "release.json"
            save_json(ann, _release())
            manifest = {
                "annotation_sha256": sha256_file(ann),
                "sites": {
                    "DAAN": {"partitions": {"train": [2], "validation": [3]}},
                    "DARC": {"partitions": {"test": [1]}},
                },
            }
            target_test, target_site, envelope = _load_references(ann, manifest, "DAAN", "DARC")
        self.assertEqual(envelope["width_height_px"], (14.0, 12.0))
        self.assertEqual(envelope["train_image_ids"], [2])
        self.assertEqual(len(target_test["annotations"]), 3)
        self.assertEqual(target_site, {1})

    def test_one_to_one_center_matching_and_ap_scoring(self):
        release = _release()
        release["images"] = [_image(1, "DARC")]
        release["annotations"] = [
            _annotation(1, 1, [10, 5, 2, 2]),
            _annotation(2, 1, [13, 5, 2, 2]),
        ]
        mutually_ambiguous = [
            _prediction([11.5, 5, 2, 2], 0.95),
            _prediction([8.5, 5, 2, 2], 0.8),
        ]
        rows = center_agreement(release, mutually_ambiguous)["rows"]
        self.assertEqual(rows[0]["radius_px"], 2)
        self.assertEqual(rows[0]["matched_centers"], 2)
        self.assertEqual(
            center_agreement(release, mutually_ambiguous[:1])["rows"][0]["matched_centers"], 1
        )

        perfect = [_prediction(a["bbox"], 0.9) for a in release["annotations"]]
        results = score_reference_views(
            build_reference_views(release, (8, 6)), {"otsu": perfect, "other": []}
        )
        self.assertAlmostEqual(results["baseline"]["methods"]["otsu"]["coco"]["ap"], 1.0)
        self.assertEqual(results["baseline"]["methods"]["other"]["coco"]["ap"], 0.0)
        self.assertEqual(results["baseline"]["ranks"]["ap"], {"otsu": 1, "other": 2})
        self.assertLess(
            results["fixed_source_envelope"]["otsu_gap_change_from_baseline"]["other"]["ap"], 0
        )


if __name__ == "__main__":
    unittest.main()
