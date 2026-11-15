"""Replay a frozen historical preprocessing candidate into a new, audited dataset.

The historical modules retain their original naming and comments. This wrapper
makes no independent AIS or submitted-pipeline equivalence claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2 as _cv2
import numpy as np

from .vendor.ec6d088 import ais_anchored_gt as historical
from .vendor.ec6d088.preprocessing import apply_clahe
from .vendor.ec6d088.radar_mask import create_radar_mask, extract_radar_signal

# The compiled OpenCV API is broader than this archived wheel's type stubs.
cv2: Any = _cv2

SITES = ("DAAN", "DARC", "MANV")
VARIANTS = {"clahe_off": False, "clahe_on": True}
RECIPE_VERSION = "historical-ec6d088-replay-v1"
VENDOR_ROOT = Path(__file__).parent / "vendor" / "ec6d088"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _frozen_source() -> dict:
    source = json.loads((VENDOR_ROOT / "SOURCE.json").read_text())
    for name, expected in source["files"].items():
        data = (VENDOR_ROOT / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if hashlib.sha256(data).hexdigest() != expected["sha256"] or blob != expected["git_blob"]:
            raise ValueError(f"Frozen historical source does not match its manifest: {name}")
    return source


def _inventory(raw_root: Path, datasets: tuple[str, ...]) -> dict:
    files = []
    for dataset in datasets:
        site = raw_root / dataset
        for path in sorted(site.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Raw inventory requires ordinary files/directories: {path}")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(raw_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return {"schema_version": 1, "datasets": list(datasets), "files": files}


def _frame_ledger(raw_root: Path, datasets: tuple[str, ...]) -> dict:
    frames = []
    counts: dict[str, Counter] = {}
    for dataset in datasets:
        site = raw_root / dataset
        data = json.loads((site / "polarMeas.json").read_text())
        if not isinstance(data, dict):
            raise TypeError(f"Expected a frame-keyed JSON object for {dataset}")
        keys = sorted(data, key=int)
        if len({int(key) for key in keys}) != len(keys):
            raise ValueError(f"Ambiguous numeric frame keys for {dataset}")
        expected_images = set()
        counts[dataset] = Counter()
        for key in keys:
            frame = int(key)
            image_path = site / "radarImages" / f"{frame:03d}.png"
            expected_images.add(image_path.name)
            entry = data[key]
            row = {
                "dataset": dataset,
                "polar_key": key,
                "frame": frame,
                "raw_image": image_path.relative_to(raw_root).as_posix(),
                "polar_entry": "null" if entry is None else "empty" if entry == [] else "present",
            }
            if not image_path.is_file():
                status = "missing_png"
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    status = "unreadable_png"
                elif entry is None:
                    status = "null_polar"
                else:
                    if image.shape != (1024, 1050, 3) or image.dtype != np.uint8:
                        raise ValueError(
                            f"Historical replay expects 1050x1024 uint8 BGR: {image_path}"
                        )
                    status = "included"
                    row["decoded_shape"] = list(image.shape)
            row["status"] = status
            counts[dataset][status] += 1
            frames.append(row)
        for path in sorted((site / "radarImages").glob("*.png")):
            if path.name not in expected_images:
                frames.append(
                    {
                        "dataset": dataset,
                        "polar_key": None,
                        "frame": int(path.stem) if path.stem.isdecimal() else None,
                        "raw_image": path.relative_to(raw_root).as_posix(),
                        "polar_entry": "absent",
                        "status": "no_polar_entry",
                    }
                )
                counts[dataset]["no_polar_entry"] += 1
    return {
        "schema_version": 1,
        "frames": frames,
        "counts": {site: dict(value) for site, value in counts.items()},
    }


def _frame_key(image: dict) -> tuple[str, int]:
    dataset = image.get("dataset")
    frame = image.get("frame")
    if dataset is None or frame is None:
        name = Path(image["file_name"]).stem
        fallback_site, fallback_frame = name.rsplit("_", 1)
        dataset = fallback_site if dataset is None else dataset
        frame = int(fallback_frame) if frame is None else frame
    return str(dataset), int(frame)


def _index_coco(data: dict) -> tuple[dict, dict]:
    images = {}
    ids = {}
    for image in data["images"]:
        key = _frame_key(image)
        if key in images or image["id"] in ids:
            raise ValueError("COCO contains duplicate frame keys or image IDs")
        images[key] = image
        ids[image["id"]] = key
    annotations = defaultdict(list)
    annotation_ids = set()
    for annotation in data["annotations"]:
        if annotation["image_id"] not in ids or annotation["id"] in annotation_ids:
            raise ValueError("COCO contains an unknown image or duplicate annotation ID")
        annotation_ids.add(annotation["id"])
        annotations[ids[annotation["image_id"]]].append(annotation)
    return images, annotations


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _label_token(annotation: dict) -> str:
    payload = {key: value for key, value in annotation.items() if key not in {"id", "image_id"}}
    return json.dumps(_normalize_numbers(payload), sort_keys=True, allow_nan=False)


def _read_image(root: Path, image: dict) -> tuple[np.ndarray | None, str | None]:
    path = (root / image["file_name"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image filename escapes its directory: {image['file_name']}")
    if not path.is_file():
        return None, None
    data = path.read_bytes()
    pixels = (
        cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED) if data else None
    )
    return pixels, hashlib.sha256(data).hexdigest()


def compare_release(
    rebuilt: dict, rebuilt_images: Path, release_annotations: Path, release_images: Path
) -> dict:
    """Compare decoded pixels and label multisets keyed by site/frame, ignoring IDs."""
    release_bytes = release_annotations.read_bytes()
    release = json.loads(release_bytes)
    rebuild_by_key, rebuild_labels = _index_coco(rebuilt)
    release_by_key, release_labels = _index_coco(release)
    rows = []
    summary = Counter()
    for key in sorted(rebuild_by_key.keys() | release_by_key.keys()):
        row: dict[str, Any] = {"dataset": key[0], "frame": key[1], "pixels": None, "labels": None}
        if key not in rebuild_by_key:
            row["status"] = "missing_from_rebuild"
        elif key not in release_by_key:
            row["status"] = "missing_from_release"
        else:
            row["status"] = "compared"
            left_tokens = [_label_token(item) for item in rebuild_labels[key]]
            right_tokens = [_label_token(item) for item in release_labels[key]]
            left, right = Counter(left_tokens), Counter(right_tokens)
            row["labels"] = {
                "exact": left == right,
                "order_equal": left_tokens == right_tokens,
                "rebuild_count": len(left_tokens),
                "release_count": len(right_tokens),
                "matched_count": sum((left & right).values()),
                "rebuild_only": list((left - right).elements()),
                "release_only": list((right - left).elements()),
            }
            rebuilt_pixels, rebuilt_hash = _read_image(rebuilt_images, rebuild_by_key[key])
            released_pixels, released_hash = _read_image(release_images, release_by_key[key])
            both_readable = rebuilt_pixels is not None and released_pixels is not None
            same_shape = both_readable and rebuilt_pixels.shape == released_pixels.shape
            equal = bool(
                same_shape
                and rebuilt_pixels.dtype == released_pixels.dtype
                and np.array_equal(rebuilt_pixels, released_pixels)
            )
            pixels = {
                "equal": equal,
                "rebuild_readable": rebuilt_pixels is not None,
                "release_readable": released_pixels is not None,
                "rebuild_file_sha256": rebuilt_hash,
                "release_file_sha256": released_hash,
                "rebuild_shape": list(rebuilt_pixels.shape) if rebuilt_pixels is not None else None,
                "release_shape": list(released_pixels.shape)
                if released_pixels is not None
                else None,
            }
            if same_shape:
                pixels["different_values"] = int(
                    np.count_nonzero(rebuilt_pixels != released_pixels)
                )
                pixels["max_absolute_difference"] = int(
                    np.max(
                        np.abs(rebuilt_pixels.astype(np.int64) - released_pixels.astype(np.int64))
                    )
                )
            row["pixels"] = pixels
            summary["pixel_equal_frames"] += int(equal)
            summary["label_equal_frames"] += int(row["labels"]["exact"])
        summary[row["status"]] += 1
        rows.append(row)
    return {
        "schema_version": 1,
        "release_annotations_sha256": hashlib.sha256(release_bytes).hexdigest(),
        "comparison": "Decoded pixels; label multisets excluding id/image_id; integral floats equal integers.",
        "categories_equal": rebuilt.get("categories") == release.get("categories"),
        "frames": rows,
        "summary": dict(summary),
    }


def rebuild(
    raw_root: Path,
    output_root: Path,
    datasets: tuple[str, ...] = SITES,
    release_annotations: Path | None = None,
    release_images: Path | None = None,
) -> dict:
    """Write both historical render variants; any existing output root is refused."""
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    datasets = tuple(datasets)
    if output_root.exists():
        raise FileExistsError(f"Refusing existing output root: {output_root}")
    if output_root.is_relative_to(raw_root):
        raise ValueError("Output root must be separate from the raw input tree")
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(SITES):
        raise ValueError(f"Choose unique datasets from {SITES}")
    if tuple(site for site in SITES if site in datasets) != datasets:
        raise ValueError("Use historical dataset order DAAN, DARC, MANV (a subset is allowed)")
    for site in datasets:
        if not (raw_root / site / "polarMeas.json").is_file():
            raise FileNotFoundError(f"Missing {site}/polarMeas.json under {raw_root}")
        if not (raw_root / site / "radarImages").is_dir():
            raise FileNotFoundError(f"Missing {site}/radarImages under {raw_root}")
    if (release_annotations is None) != (release_images is None):
        raise ValueError("Supply both release_annotations and release_images, or neither")
    if release_annotations is not None and release_images is not None:
        release_annotations, release_images = Path(release_annotations), Path(release_images)
        _index_coco(json.loads(release_annotations.read_text()))
        if not release_images.is_dir():
            raise FileNotFoundError(release_images)
    source = _frozen_source()
    inventory = _inventory(raw_root, datasets)
    ledger = _frame_ledger(raw_root, datasets)
    output_root.mkdir(parents=True, exist_ok=False)
    _save(output_root / "raw_inventory.json", inventory)
    _save(output_root / "frame_ledger.json", ledger)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "recipe_version": RECIPE_VERSION,
        "status": "running",
        "interpretation": "historical_candidate_reconstruction",
        "scope": "Full-frame images and COCO annotations; no tiling, split assignment, or training.",
        "provenance_limit": "polarMeas.json association; independent AIS is not established. No submitted-pipeline or released-dataset equivalence is assumed.",
        "raw_root": str(raw_root),
        "datasets": list(datasets),
        "historical_source": source,
        "driver_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(VENDOR_ROOT / "SOURCE.json"),
        "raw_inventory_sha256": _sha256(output_root / "raw_inventory.json"),
        "frame_ledger_sha256": _sha256(output_root / "frame_ledger.json"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "settings": {
            "threshold_strict_gt": 25,
            "closing_ellipse_px": [5, 5],
            "component_area_inclusive": [10, 8000],
            "dbscan_eps_m": 200.0,
            "dbscan_min_samples": 5,
            "match_radius_strict_lt_px": 20,
            "include_unclaimed": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "clahe_order": "After site extraction and masking; labels use pre-CLAHE signal; no remask.",
        },
        "variants": {},
    }
    _save(output_root / "manifest.json", manifest)
    try:
        coco = historical.convert_all_datasets(
            raw_root,
            datasets=list(datasets),
            polar_eps_m=200.0,
            polar_min_samples=5,
            threshold=25,
            min_area=10,
            max_area=8000,
            include_unclaimed=False,
        )
        included = sum(row["status"] == "included" for row in ledger["frames"])
        if len(coco["images"]) != included:
            raise RuntimeError("Historical exporter coverage disagrees with the input ledger")
        for variant, use_clahe in VARIANTS.items():
            root = output_root / "variants" / variant
            image_dir = root / "images"
            image_dir.mkdir(parents=True)
            annotations = root / "annotations" / "all.json"
            _save(annotations, coco)
            images = []
            for info in coco["images"]:
                site, frame = _frame_key(info)
                path = raw_root / site / "radarImages" / f"{frame:03d}.png"
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise RuntimeError(f"Raw image became unreadable: {path}")
                signal = extract_radar_signal(bgr, dataset=site)
                mask = create_radar_mask(signal.shape[0], signal.shape[1])
                signal[mask == 0] = 0
                if use_clahe:
                    signal = apply_clahe(signal)
                destination = image_dir / info["file_name"]
                if not cv2.imwrite(str(destination), signal):
                    raise OSError(f"Could not write {destination}")
                images.append(
                    {
                        "dataset": site,
                        "frame": frame,
                        "file_name": info["file_name"],
                        "sha256": _sha256(destination),
                        "decoded_pixel_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
                        "shape": list(signal.shape),
                        "dtype": str(signal.dtype),
                    }
                )
            _save(root / "image_manifest.json", images)
            record = {
                "apply_clahe": use_clahe,
                "image_count": len(coco["images"]),
                "annotation_count": len(coco["annotations"]),
                "annotations_sha256": _sha256(annotations),
                "image_manifest_sha256": _sha256(root / "image_manifest.json"),
            }
            if release_annotations is not None and release_images is not None:
                comparison = compare_release(coco, image_dir, release_annotations, release_images)
                _save(root / "comparison.json", comparison)
                record["comparison_sha256"] = _sha256(root / "comparison.json")
                record["comparison_summary"] = comparison["summary"]
            manifest["variants"][variant] = record
        if _inventory(raw_root, datasets) != inventory:
            raise RuntimeError("Raw input inventory changed during reconstruction")
        manifest["raw_inventory_verified_after_rebuild"] = True
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _save(output_root / "manifest.json", manifest)
        raise
    _save(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=SITES, default=list(SITES))
    parser.add_argument("--release-annotations", type=Path)
    parser.add_argument("--release-images", type=Path)
    args = parser.parse_args()
    manifest = rebuild(
        args.raw_root,
        args.output_root,
        tuple(args.datasets),
        args.release_annotations,
        args.release_images,
    )
    print(json.dumps({"status": manifest["status"], "variants": manifest["variants"]}, indent=2))


if __name__ == "__main__":
    main()
