"""Compare released boxes with reconstructed image components, without association inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .protocol import save_json, sha256_file

HISTORICAL_EXPORTER = {
    "commit": "ec6d088018ff255e82cfdd629e2a458c0f7d4172",
    "git_blobs": {
        "scripts/preprocess.py": "e8a5b6a976dc25188592b501e8ad070448936d8d",
        "src/data/ais_anchored_gt.py": "a590e6db501f338b223a48d00da47f56e5e181f4",
        "src/data/radar_mask.py": "0da422ed782e394634196e5b7b954cbaae096416",
    },
}
DISC_CENTER = (531, 515)
DISC_RADIUS = 466
FIXED_THRESHOLD = 25
CLOSING_SIZE = (5, 5)
MIN_COMPONENT_AREA = 10
MAX_COMPONENT_AREA = 8000


def components(
    signal: np.ndarray, disc: np.ndarray, threshold: float
) -> list[tuple[tuple[int, ...], int]]:
    """Return each closed component's COCO box and peak on the masked signal."""
    masked = cv2.bitwise_and(signal, disc)
    binary = (masked > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    indices = np.flatnonzero((areas >= MIN_COMPONENT_AREA) & (areas <= MAX_COMPONENT_AREA)) + 1
    if len(indices) == 0:
        return []
    peaks = np.zeros(count, dtype=np.uint8)
    foreground = closed != 0
    np.maximum.at(peaks, labels[foreground], masked[foreground])
    return [
        (tuple(int(value) for value in stats[index, :4]), int(peaks[index])) for index in indices
    ]


def _compare_components(candidates: list, released: list[dict]) -> dict:
    # Allocate exact pairs before bbox-only matches so duplicates cannot consume
    # the candidate of a later label that also matches the peak.
    pair_remaining = Counter(candidates)
    box_remaining = Counter(bbox for bbox, _ in candidates)
    peak_remaining = Counter(peak for _, peak in candidates)
    exact_ids = []
    bbox_only_ids = []
    unmatched = []
    for annotation in released:
        bbox = tuple(annotation["bbox"])
        peak = annotation.get("blob_peak")
        pair = (bbox, peak)
        if pair_remaining[pair]:
            pair_remaining[pair] -= 1
            box_remaining[bbox] -= 1
            exact_ids.append(annotation["id"])
        else:
            unmatched.append(annotation)
    no_box_ids = []
    for annotation in unmatched:
        bbox = tuple(annotation["bbox"])
        if box_remaining[bbox]:
            box_remaining[bbox] -= 1
            bbox_only_ids.append(annotation["id"])
        else:
            no_box_ids.append(annotation["id"])
    isolated_peak_matches = peak_remaining & Counter(a.get("blob_peak") for a in released)
    return {
        "counts": {
            "candidate_components": len(candidates),
            "released_annotations": len(released),
            "bbox_exact": len(exact_ids) + len(bbox_only_ids),
            "peak_exact_ignoring_position": sum(isolated_peak_matches.values()),
            "bbox_and_peak_exact": len(exact_ids),
            "unreproduced_bboxes": len(no_box_ids),
            "unmatched_candidates_by_bbox": len(candidates) - len(exact_ids) - len(bbox_only_ids),
            "released_annotations_without_peak": sum(a.get("blob_peak") is None for a in released),
        },
        "bbox_and_peak_exact_annotation_ids": exact_ids,
        "bbox_only_annotation_ids": bbox_only_ids,
        "unreproduced_bbox_annotation_ids": no_box_ids,
    }


def _audit_frame(image: dict, annotations: list[dict], image_dir: Path, masks: dict) -> dict:
    image_path = (image_dir / image["file_name"]).resolve()
    if not image_path.is_relative_to(image_dir):
        raise ValueError(f"Unsafe image filename: {image['file_name']}")
    image_bytes = image_path.read_bytes()
    signal = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    dimensions = (image["height"], image["width"])
    if signal is None or signal.shape != dimensions or signal.dtype != np.uint8:
        raise ValueError(f"Expected a released 8-bit grayscale image of shape {dimensions}")
    if dimensions not in masks:
        disc = np.zeros(dimensions, np.uint8)
        cv2.circle(disc, DISC_CENTER, DISC_RADIUS, 255, thickness=-1)
        masks[dimensions] = (disc, disc != 0)
    disc, inside = masks[dimensions]
    if not inside.any():
        raise ValueError(f"Empty historical disc in image {image['id']}")
    otsu, _ = cv2.threshold(signal[inside], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fixed = components(signal, disc, FIXED_THRESHOLD)
    otsu_components = components(signal, disc, otsu)
    fixed_with_gate = [(bbox, peak) for bbox, peak in fixed if peak > otsu]
    known_peaks = [a for a in annotations if a.get("blob_peak") is not None]
    return {
        "image_id": image["id"],
        "dataset": image["dataset"],
        "file_name": image["file_name"],
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "masked_otsu_threshold": int(otsu),
        "released_annotations_at_or_below_otsu_ids": [
            a["id"] for a in known_peaks if a["blob_peak"] <= otsu
        ],
        "comparators": {
            "fixed_components": _compare_components(fixed, annotations),
            "otsu_components": _compare_components(otsu_components, annotations),
            "fixed_components_otsu_peak_gate": _compare_components(fixed_with_gate, annotations),
        },
    }


def audit(annotations_path: Path, image_dir: Path) -> dict:
    """Hash and compare the dataset as released, with per-frame reproducibility records."""
    annotations_bytes = annotations_path.read_bytes()
    data = json.loads(annotations_bytes)
    by_image = defaultdict(list)
    annotation_ids = set()
    for annotation in data["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError("Duplicate annotation ID")
        annotation_ids.add(annotation["id"])
        by_image[annotation["image_id"]].append(annotation)

    image_ids = {image["id"] for image in data["images"]}
    image_names = {image["file_name"] for image in data["images"]}
    if len(image_ids) != len(data["images"]) or len(image_names) != len(data["images"]):
        raise ValueError("Duplicate image ID or filename")
    if not by_image.keys() <= image_ids:
        raise ValueError("Annotation references an unknown image")

    image_dir = image_dir.resolve()
    masks = {}
    frames = [
        _audit_frame(image, by_image[image["id"]], image_dir, masks)
        for image in sorted(data["images"], key=lambda image: image["id"])
    ]
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["dataset"]].append(frame)
    return {
        "schema_version": 1,
        "annotations_sha256": hashlib.sha256(annotations_bytes).hexdigest(),
        "implementation_sha256": sha256_file(__file__),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "historical_exporter_comparator": HISTORICAL_EXPORTER,
        "settings": {
            "input": "released 8-bit grayscale PNG",
            "historical_disc_center_xy_px": DISC_CENTER,
            "historical_disc_radius_px": DISC_RADIUS,
            "fixed_threshold": FIXED_THRESHOLD,
            "otsu_population": "all intensities inside the historical disc, including zeros",
            "comparator": "strict greater-than for both segmentation and peak eligibility",
            "closing": {"shape": "ellipse", "size_px": CLOSING_SIZE},
            "connectivity": 8,
            "component_area_bounds_px": [MIN_COMPONENT_AREA, MAX_COMPONENT_AREA],
            "release_peak_field": "blob_peak",
        },
        "scope": [
            "Component comparisons use released frames and surviving annotations.",
            "Exact matches leave original selector, runtime, and rejected candidates unresolved.",
            "Unmatched image components have unknown vessel and association status.",
            "The historical code consumes polarMeas.json radar-derived candidate coordinates.",
            (
                "AIS referenceMeas.mat data and verified frame correspondence are required for "
                "an independent association and inventory audit."
            ),
        ],
        "total": _summarize(frames),
        "sites": {site: _summarize(items) for site, items in sorted(grouped.items())},
        "frames": frames,
    }


def _summarize(frames: list[dict]) -> dict:
    counts = defaultdict(Counter)
    for frame in frames:
        for name, comparator in frame["comparators"].items():
            counts[name].update(comparator["counts"])
    return {
        "images": len(frames),
        "released_annotations_at_or_below_otsu": sum(
            len(frame["released_annotations_at_or_below_otsu_ids"]) for frame in frames
        ),
        "comparators": {name: dict(totals) for name, totals in counts.items()},
        "otsu_threshold_counts": dict(Counter(frame["masked_otsu_threshold"] for frame in frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.ann.resolve() or args.output.resolve().is_relative_to(
        args.images.resolve()
    ):
        parser.error("Choose an output path outside the released annotations and image directory")
    cv2.setNumThreads(1)
    report = audit(args.ann, args.images)
    save_json(args.output, report)
    print(json.dumps(report["total"], indent=2))


if __name__ == "__main__":
    main()
