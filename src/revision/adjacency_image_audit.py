"""Audit image-similarity candidates across fixed frame-index partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from .protocol import _safe_image_path, save_json, sha256_file

METHOD = {
    "name": "masked_dct_phash_63_ac_bits",
    "mask": {"center_xy": [531, 515], "radius": 466, "outside_value": 0},
    "input": "OpenCV grayscale decoding; byte SHA-256 verified against manifest",
    "resize": {"width": 32, "height": 32, "interpolation": "cv2.INTER_AREA"},
    "transform": "float64 cv2.dct; top-left 8x8, row-major",
    "bits": "DC bit zero; each of 63 AC bits is 1 iff coefficient > median of 63 AC coefficients; row-major packed most-significant bit first",
    "matching": "minimum Hamming distance; ties use lowest numeric training image ID",
    "thresholds": [0, 4, 8],
    "scope": "Image-similarity candidates do not certify duplicate frames, vessel identities, tracks, or temporal independence.",
}


class FrameMatch(TypedDict):
    test_image_id: int
    test_frame: int
    nearest_train_image_id: int
    nearest_train_frame: int
    hamming_distance: int
    nearest_hash_frame_index_distance: int
    minimum_frame_index_distance: int
    exact_sha256_train_image_ids: list[int]


def perceptual_hash(gray: np.ndarray) -> int:
    """Return a 64-bit integer with the most-significant (DC) bit fixed at zero."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("Expected a nonempty uint8 grayscale image")
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    masked = np.where((xx - 531) ** 2 + (yy - 515) ** 2 <= 466**2, gray, 0).astype(np.uint8)
    resized = cv2.resize(masked, (32, 32), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(resized.astype(np.float64))[:8, :8].ravel()
    median = np.median(coeff[1:])
    value = 0
    for bit in [False, *(coeff[1:] > median)]:
        value = (value << 1) | int(bit)
    return value


def compare_partition(records: dict, hashes: dict, train_ids: list, test_ids: list) -> dict:
    """Find the closest training image to every test image, and index minima."""
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        raise ValueError("Training and test IDs must be nonempty and disjoint")
    train_ids = sorted(train_ids)
    rows: list[FrameMatch] = []
    for test_id in test_ids:
        test = records[test_id]
        closest = min(train_ids, key=lambda i: ((hashes[i] ^ hashes[test_id]).bit_count(), i))
        distance = (hashes[closest] ^ hashes[test_id]).bit_count()
        index_distance = abs(records[closest]["frame"] - test["frame"])
        minimum_index_distance = min(abs(records[i]["frame"] - test["frame"]) for i in train_ids)
        exact_ids = [i for i in train_ids if records[i]["sha256"] == test["sha256"]]
        rows.append(
            {
                "test_image_id": test_id,
                "test_frame": test["frame"],
                "nearest_train_image_id": closest,
                "nearest_train_frame": records[closest]["frame"],
                "hamming_distance": distance,
                "nearest_hash_frame_index_distance": index_distance,
                "minimum_frame_index_distance": minimum_index_distance,
                "exact_sha256_train_image_ids": exact_ids,
            }
        )
    n = len(rows)
    return {
        "train_count": len(train_ids),
        "test_count": n,
        "frames": rows,
        "summary": {
            "hamming_threshold_candidate_counts": {
                str(k): sum(r["hamming_distance"] <= k for r in rows) for k in (0, 4, 8)
            },
            "hamming_threshold_candidate_fractions": {
                str(k): sum(r["hamming_distance"] <= k for r in rows) / n for k in (0, 4, 8)
            },
            "test_images_with_exact_sha256_train_match": sum(
                bool(r["exact_sha256_train_image_ids"]) for r in rows
            ),
            "minimum_frame_index_distance": min(r["minimum_frame_index_distance"] for r in rows),
        },
    }


def audit(manifest_path: Path, image_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    records = {r["id"]: r for r in manifest["images"]}
    if len(records) != len(manifest["images"]) or len(records) != manifest["image_count"]:
        raise ValueError("Manifest image IDs are not unique or image count is inconsistent")
    hashes = {}
    for image_id, record in records.items():
        raw = _safe_image_path(image_dir, record["file_name"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {image_id}")
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Image decode failed: {image_id}")
        hashes[image_id] = perceptual_hash(gray)
    sites = {}
    for name, site in sorted(manifest["sites"].items()):
        parts = site["partitions"]
        test = parts["test"]
        arms = {
            "buffered": parts["train"],
            **{f"adjacency_{seed}": site["adjacency"][str(seed)] for seed in range(3)},
        }
        for ids in [test, *arms.values()]:
            if len(ids) != len(set(ids)) or any(
                i not in records or records[i]["site"] != name for i in ids
            ):
                raise ValueError(f"Invalid image IDs for site {name}")
        sites[name] = {
            arm: compare_partition(records, hashes, ids, test) for arm, ids in arms.items()
        }
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "method": METHOD,
        "verified_image_count": len(hashes),
        "image_hashes_hex": {str(i): f"{value:016x}" for i, value in sorted(hashes.items())},
        "sites": sites,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    save_json(args.out, audit(args.manifest, args.images))
    print(f"Wrote {args.out}; sha256={sha256_file(args.out)}")


if __name__ == "__main__":
    main()
