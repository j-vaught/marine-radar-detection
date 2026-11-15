"""Synthetic contract checks for the historical candidate replay, without real raw data."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.revision.raw_rebuild import compare_release, rebuild
from src.revision.vendor.ec6d088.ais_anchored_gt import (
    AISSeed,
    Blob,
    extract_blobs,
    greedy_match,
    polar_to_pixel,
)
from src.revision.vendor.ec6d088.radar_mask import extract_radar_signal


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_fixture(root: Path, *, exclusions: bool = False) -> Path:
    """Write a raw-format DAAN capture whose only seed matches a 5 by 5 red blob."""
    raw = root / "raw"
    images = raw / "DAAN" / "radarImages"
    images.mkdir(parents=True)
    bgr = np.zeros((1024, 1050, 3), dtype=np.uint8)
    bgr[414:419, 510:515, 2] = 200
    # Raw polar point projects to (512, 416) under the archived DAAN calibration.
    entry = [7, 0.0, [100.0 * 5750.0 / 466.0, 0.0]]
    polar = {"0": entry}
    if not cv2.imwrite(str(images / "000.png"), bgr):
        raise AssertionError("Could not create synthetic PNG")
    if exclusions:
        polar.update({"1": entry, "2": None, "3": entry, "4": []})
        for name in ("002.png", "004.png"):
            if not cv2.imwrite(str(images / name), bgr):
                raise AssertionError("Could not create synthetic PNG")
        (images / "003.png").write_bytes(b"deliberately unreadable PNG")
    (raw / "DAAN" / "polarMeas.json").write_text(json.dumps(polar))
    return raw


class HistoricalRawContractTest(unittest.TestCase):
    def test_polar_projection_uses_north_zero_clockwise_and_site_origin(self):
        expected = ((512, 50), (978, 516), (512, 982), (46, 516))
        for theta, point in zip((0, math.pi / 2, math.pi, 3 * math.pi / 2), expected):
            actual = polar_to_pixel(5750.0, theta, cx=512, cy=516, r_px_max=466, max_range_m=5750.0)
            self.assertAlmostEqual(actual[0], point[0])
            self.assertAlmostEqual(actual[1], point[1])
        self.assertEqual(
            polar_to_pixel(0.0, 1.0, cx=514, cy=521, r_px_max=466, max_range_m=5600),
            (514.0, 521.0),
        )

    def test_site_palette_keeps_white_for_darc_and_rejects_white_for_manv(self):
        bgr = np.array([[[140, 0, 0], [255, 255, 255], [0, 255, 255], [150, 200, 200]]], np.uint8)
        for site in ("DAAN", "DARC"):
            np.testing.assert_array_equal(extract_radar_signal(bgr, site), [[0, 255, 255, 200]])
        np.testing.assert_array_equal(extract_radar_signal(bgr, "MANV"), [[0, 0, 255, 0]])

    def test_fixed_threshold_is_strict_and_uses_unweighted_component_centroid(self):
        bgr = np.zeros((40, 40, 3), dtype=np.uint8)
        bgr[5:10, 5:10, 2] = 25
        bgr[25:30, 25:30, 2] = 26
        bgr[25, 25, 2] = 255
        blobs = extract_blobs(bgr, radar_mask=np.full((40, 40), 255, np.uint8), dataset="DARC")
        self.assertEqual(len(blobs), 1)
        blob = blobs[0]
        self.assertEqual((blob.x1, blob.y1, blob.x2, blob.y2), (25, 25, 30, 30))
        self.assertEqual((blob.cx, blob.cy), (27.0, 27.0))
        self.assertEqual((blob.area, blob.peak), (25, 255))

    def test_greedy_match_rejects_exact_radius_boundary(self):
        seed = AISSeed(obj_id=1, heading=0.0, cluster_idx=0, centroid_px=(0, 0), n_points=1)
        boundary = Blob(cx=20.0, cy=0.0, x1=18, y1=-2, x2=22, y2=2, area=16, peak=200)
        result = greedy_match([seed], [boundary], match_radius=20)
        self.assertEqual(result.matched, [])
        self.assertEqual(result.unmatched_ais, [seed])
        boundary.cx = 19.999
        self.assertEqual(len(greedy_match([seed], [boundary], match_radius=20).matched), 1)


class RawRebuildTest(unittest.TestCase):
    def test_fresh_output_guard_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve existing outputs")
            with self.assertRaises((FileExistsError, ValueError)):
                rebuild(raw, output, datasets=("DAAN",))
            self.assertEqual(sentinel.read_text(), "preserve existing outputs")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_failed_conversion_records_failure_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            output = root / "failed"
            with (
                patch(
                    "src.revision.raw_rebuild.historical.convert_all_datasets",
                    side_effect=RuntimeError("synthetic conversion failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic conversion failure"),
            ):
                rebuild(raw, output, datasets=("DAAN",))
            failure = _read_json(output / "manifest.json")
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error"]["type"], "RuntimeError")
            self.assertEqual(failure["error"]["message"], "synthetic conversion failure")
            before_retry = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                rebuild(raw, output, datasets=("DAAN",))
            self.assertEqual((output / "manifest.json").read_bytes(), before_retry)

    def test_missing_requested_site_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            output = root / "new"
            with self.assertRaises((FileNotFoundError, ValueError)):
                rebuild(raw, output, datasets=("DAAN", "DARC"))
            self.assertFalse(output.exists())

    def test_variants_have_identical_labels_and_repeatable_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            outputs = [root / "first", root / "second"]
            for output in outputs:
                rebuild(raw, output, datasets=("DAAN",))
                self.assertTrue((output / "manifest.json").is_file())
                self.assertTrue((output / "raw_inventory.json").is_file())
                self.assertTrue((output / "frame_ledger.json").is_file())
            off = outputs[0] / "variants" / "clahe_off"
            on = outputs[0] / "variants" / "clahe_on"
            labels = _read_json(off / "annotations" / "all.json")
            self.assertEqual(labels, _read_json(on / "annotations" / "all.json"))
            self.assertEqual(len(labels["images"]), 1)
            self.assertEqual(len(labels["annotations"]), 1)
            self.assertEqual(labels["annotations"][0]["bbox"], [510, 414, 5, 5])
            self.assertEqual(labels["annotations"][0]["blob_peak"], 200)
            raw_pixels = cv2.imread(str(off / "images" / "DAAN_000.png"), cv2.IMREAD_UNCHANGED)
            contrast_pixels = cv2.imread(str(on / "images" / "DAAN_000.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(raw_pixels.shape, (1024, 1050))
            self.assertEqual(raw_pixels.dtype, np.uint8)
            self.assertFalse(np.array_equal(raw_pixels, contrast_pixels))
            for variant in ("clahe_off", "clahe_on"):
                relative = Path("variants") / variant
                for filename in (Path("annotations/all.json"), Path("images/DAAN_000.png")):
                    self.assertEqual(
                        (outputs[0] / relative / filename).read_bytes(),
                        (outputs[1] / relative / filename).read_bytes(),
                    )

    def test_exclusion_ledger_and_hashed_inventory_preserve_frame_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root, exclusions=True)
            # An image omitted from polarMeas must remain visible in the audit ledger.
            unindexed = raw / "DAAN" / "radarImages" / "005.png"
            unindexed.write_bytes((unindexed.parent / "000.png").read_bytes())
            output = root / "rebuilt"
            result = rebuild(raw, output, datasets=("DAAN",))
            ledger = _read_json(output / "frame_ledger.json")
            by_frame = {row["frame"]: row for row in ledger["frames"]}
            self.assertEqual(
                {frame: row["status"] for frame, row in by_frame.items()},
                {
                    0: "included",
                    1: "missing_png",
                    2: "null_polar",
                    3: "unreadable_png",
                    4: "included",
                    5: "no_polar_entry",
                },
            )
            self.assertEqual(by_frame[2]["polar_entry"], "null")
            self.assertEqual(by_frame[4]["polar_entry"], "empty")
            labels = _read_json(output / "variants/clahe_off/annotations/all.json")
            self.assertEqual([image["frame"] for image in labels["images"]], [0, 4])
            self.assertEqual(len(labels["annotations"]), 1)
            for filename, manifest_key in (
                ("raw_inventory.json", "raw_inventory_sha256"),
                ("frame_ledger.json", "frame_ledger_sha256"),
            ):
                self.assertEqual(
                    result[manifest_key],
                    hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["interpretation"], "historical_candidate_reconstruction")

    def test_release_comparison_uses_frame_keys_and_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            first = root / "reference_candidate"
            rebuild(raw, first, datasets=("DAAN",))
            off = first / "variants/clahe_off"
            release = copy.deepcopy(_read_json(off / "annotations/all.json"))
            release["images"][0]["id"] = 101
            release["annotations"][0]["id"] = 701
            release["annotations"][0]["image_id"] = 101
            release_annotations = root / "release.json"
            release_annotations.write_text(json.dumps(release))
            release_images = root / "release_images"
            release_images.mkdir()
            pixels = cv2.imread(str(off / "images/DAAN_000.png"), cv2.IMREAD_UNCHANGED)
            release_image = release_images / "DAAN_000.png"
            self.assertTrue(
                cv2.imwrite(str(release_image), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            )
            self.assertNotEqual(
                release_image.read_bytes(), (off / "images/DAAN_000.png").read_bytes()
            )
            output = root / "compared"
            rebuild(
                raw,
                output,
                datasets=("DAAN",),
                release_annotations=release_annotations,
                release_images=release_images,
            )
            for variant in ("clahe_off", "clahe_on"):
                comparison = _read_json(output / "variants" / variant / "comparison.json")
                self.assertEqual(len(comparison["frames"]), 1)
                row = comparison["frames"][0]
                self.assertEqual((row["dataset"], row["frame"]), ("DAAN", 0))
                self.assertEqual(row["status"], "compared")
                self.assertTrue(row["labels"]["exact"])
                self.assertEqual(row["pixels"]["equal"], variant == "clahe_off")

    def test_release_comparison_detects_peak_mismatch_with_identical_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _write_fixture(root)
            output = root / "candidate"
            rebuild(raw, output, datasets=("DAAN",))
            variant = output / "variants/clahe_off"
            rebuilt = _read_json(variant / "annotations/all.json")
            release = copy.deepcopy(rebuilt)
            release["annotations"][0]["blob_peak"] = 201
            self.assertEqual(rebuilt["annotations"][0]["bbox"], release["annotations"][0]["bbox"])
            release_annotations = root / "release.json"
            release_annotations.write_text(json.dumps(release))
            comparison = compare_release(
                rebuilt,
                variant / "images",
                release_annotations,
                variant / "images",
            )
            row = comparison["frames"][0]
            self.assertTrue(row["pixels"]["equal"])
            self.assertFalse(row["labels"]["exact"])
            self.assertEqual(row["labels"]["matched_count"], 0)
            self.assertEqual(len(row["labels"]["rebuild_only"]), 1)
            self.assertEqual(len(row["labels"]["release_only"]), 1)


if __name__ == "__main__":
    unittest.main()
