from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.revision.protocol import save_json, sha256_file
from src.revision.provenance_audit import _compare_components, audit


class ProvenanceAuditTest(unittest.TestCase):
    def test_replay_fixed_components_and_strict_otsu_eligibility(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image_path = root / "DAAN_000.png"
            annotation_path = root / "all.json"
            image = np.zeros((1024, 1050), dtype=np.uint8)
            image[400:407, 500:507] = 30
            image[600:610, 500:511] = 200
            image[0:10, 0:10] = 255  # Outside the radar disc.
            self.assertTrue(cv2.imwrite(str(image_path), image))
            release = {
                "images": [
                    {
                        "id": 1,
                        "width": 1050,
                        "height": 1024,
                        "dataset": "DAAN",
                        "file_name": image_path.name,
                    }
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [500, 400, 7, 7], "blob_peak": 30},
                    {"id": 2, "image_id": 1, "bbox": [500, 600, 11, 10], "blob_peak": 200},
                ],
            }
            save_json(annotation_path, release)
            report = audit(annotation_path, root)
            self.assertEqual(report["annotations_sha256"], sha256_file(annotation_path))
            self.assertEqual(report["frames"][0]["sha256"], sha256_file(image_path))

        frame = report["frames"][0]
        self.assertEqual(frame["masked_otsu_threshold"], 30)
        self.assertEqual(frame["released_annotations_at_or_below_otsu_ids"], [1])
        by_variant = report["total"]["comparators"]
        self.assertEqual(by_variant["fixed_components"]["bbox_and_peak_exact"], 2)
        for key in ("otsu_components", "fixed_components_otsu_peak_gate"):
            self.assertEqual(by_variant[key]["candidate_components"], 1)
            self.assertEqual(by_variant[key]["bbox_and_peak_exact"], 1)
        self.assertEqual(report["sites"]["DAAN"], report["total"])

    def test_bbox_matches_allocate_exact_pairs_before_peak_mismatches(self):
        box = (10, 10, 10, 10)
        result = _compare_components(
            [(box, 50)],
            [
                {"id": 1, "bbox": box, "blob_peak": 20},
                {"id": 2, "bbox": box, "blob_peak": 50},
                {"id": 3, "bbox": box, "blob_peak": 50},
            ],
        )
        self.assertEqual(result["bbox_and_peak_exact_annotation_ids"], [2])
        self.assertEqual(result["unreproduced_bbox_annotation_ids"], [1, 3])


if __name__ == "__main__":
    unittest.main()
