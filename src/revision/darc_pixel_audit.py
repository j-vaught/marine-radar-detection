"""Describe verified benchmark PNG pixels without fitting or evaluating detectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .protocol import _safe_image_path, save_json, sha256_file
from .provenance_audit import DISC_CENTER, DISC_RADIUS

QUANTILES = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1)
BOX_QUANTILES = (0.50, 0.90, 0.95, 0.99)
INNER_MARGIN = 4
OUTER_MARGIN = 12
SPLITS = ("train", "validation", "test", "validation_buffer", "test_buffer")


def quantile_key(q: float) -> str:
    return f"p{100 * q:g}"


def histogram_quantiles(histogram: np.ndarray) -> dict[str, float]:
    """Evaluate linear sample percentiles from a 256-bin integer histogram."""
    cumulative = np.cumsum(histogram)
    if cumulative[-1] == 0:
        return {}
    ranks = (int(cumulative[-1]) - 1) * np.asarray(QUANTILES)
    lower = np.floor(ranks).astype(int)
    lower_values = np.searchsorted(cumulative, lower, side="right")
    upper_values = np.searchsorted(cumulative, np.ceil(ranks).astype(int), side="right")
    values = lower_values + (upper_values - lower_values) * (ranks - lower)
    return dict(zip((quantile_key(q) for q in QUANTILES), values.tolist(), strict=True))


def summarize(samples: list[float]) -> dict:
    values = np.asarray(samples)
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "quantiles": (
            dict(
                zip(
                    map(quantile_key, QUANTILES),
                    np.quantile(values, QUANTILES).tolist(),
                    strict=True,
                )
            )
            if len(values)
            else {}
        ),
    }


@lru_cache(maxsize=8)
def disc_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, DISC_CENTER, DISC_RADIUS, 1, thickness=-1)
    return mask.astype(bool)


def box_bounds(bbox: list[float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    rows, columns = shape
    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate published bounding box: {bbox}")
    return (
        max(0, math.floor(x)),
        max(0, math.floor(y)),
        min(columns, math.ceil(x + width)),
        min(rows, math.ceil(y + height)),
    )


def box_samples(
    image: np.ndarray, inside: np.ndarray, annotations: list[dict]
) -> list[dict[str, float]]:
    bounds = [box_bounds(a["bbox"], image.shape) for a in annotations]
    occupied = np.zeros_like(inside)
    for x0, y0, x1, y1 in bounds:
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"Released box lies outside image: {(x0, y0, x1, y1)}")
        occupied[y0:y1, x0:x1] = True

    rows, columns = image.shape
    samples = []
    for annotation, (x0, y0, x1, y1) in zip(annotations, bounds, strict=True):
        box = image[y0:y1, x0:x1][inside[y0:y1, x0:x1]]
        values: dict[str, float] = {"in_disc_box_pixels": len(box)}
        if "blob_peak" in annotation:
            values["stored_blob_peak"] = annotation["blob_peak"]
        if not len(box):
            samples.append(values)
            continue
        values["box_max"] = int(box.max())
        if "stored_blob_peak" in values:
            values["box_max_minus_stored_peak"] = values["box_max"] - values["stored_blob_peak"]
        values["box_positive_fraction"] = float(np.count_nonzero(box) / len(box))
        values["box_255_fraction"] = float(np.count_nonzero(box == 255) / len(box))
        quantiles = np.quantile(box, BOX_QUANTILES)
        values.update(
            zip(("box_" + quantile_key(q) for q in BOX_QUANTILES), quantiles, strict=True)
        )

        left, top = max(0, x0 - OUTER_MARGIN), max(0, y0 - OUTER_MARGIN)
        right, bottom = min(columns, x1 + OUTER_MARGIN), min(rows, y1 + OUTER_MARGIN)
        annulus = inside[top:bottom, left:right] & ~occupied[top:bottom, left:right]
        guard_left, guard_top = max(left, x0 - INNER_MARGIN), max(top, y0 - INNER_MARGIN)
        guard_right, guard_bottom = min(right, x1 + INNER_MARGIN), min(bottom, y1 + INNER_MARGIN)
        annulus[guard_top - top : guard_bottom - top, guard_left - left : guard_right - left] = (
            False
        )
        background = image[top:bottom, left:right][annulus]
        values["annulus_pixels"] = len(background)
        if len(background):
            bg50, bg90 = np.quantile(background, (0.50, 0.90))
            values["annulus_p50"] = float(bg50)
            values["annulus_p90"] = float(bg90)
            values["box_p90_minus_annulus_p50"] = float(values["box_p90"] - bg50)
            values["box_p90_minus_annulus_p90"] = float(values["box_p90"] - bg90)
            background_positive = background[background > 0]
            box_positive = box[box > 0]
            values["annulus_positive_fraction"] = len(background_positive) / len(background)
            if len(background_positive) and len(box_positive):
                values["positive_box_p90_minus_positive_annulus_p90"] = float(
                    np.quantile(box_positive, 0.90) - np.quantile(background_positive, 0.90)
                )
        samples.append(values)
    return samples


@dataclass
class PixelGroup:
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    frame: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    box: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    counts: Counter = field(default_factory=Counter)

    def add(self, image: np.ndarray, annotations: list[dict]) -> None:
        inside = disc_mask(image.shape)
        histogram = np.bincount(image[inside], minlength=256)
        self.histogram += histogram
        positive = histogram.copy()
        positive[0] = 0
        total, n_positive = int(histogram.sum()), int(positive.sum())
        self.counts.update(frames=1, annotations=len(annotations))
        self.frame["positive_fraction"].append(n_positive / total)
        self.frame["saturated_fraction_all"].append(int(histogram[255]) / total)
        if n_positive:
            self.frame["saturated_fraction_positive"].append(int(positive[255]) / n_positive)
            for key, value in histogram_quantiles(positive).items():
                self.frame[f"positive_{key}"].append(value)
        for values in box_samples(image, inside, annotations):
            for key, value in values.items():
                self.box[key].append(float(value))
            difference = values.get("box_max_minus_stored_peak")
            if difference is not None:
                self.counts["boxes_with_peak_comparison"] += 1
                self.counts["box_max_equals_stored_peak"] += int(difference == 0)
                self.counts["box_max_below_stored_peak"] += int(difference < 0)
                self.counts["box_max_above_stored_peak"] += int(difference > 0)

    def report(self) -> dict:
        disc_total = int(self.histogram.sum())
        positive = self.histogram.copy()
        positive[0] = 0
        positive_total = int(positive.sum())
        saturated = int(self.histogram[255])
        counts = {
            **self.counts,
            "disc_pixels": disc_total,
            "disc_positive_pixels": positive_total,
            "disc_255_pixels": saturated,
        }
        distribution = {
            "disc_positive_fraction": positive_total / disc_total,
            "disc_255_fraction_all": saturated / disc_total,
            "disc_255_fraction_positive": saturated / positive_total if positive_total else None,
            "disc_all_quantiles": histogram_quantiles(self.histogram),
            "disc_positive_quantiles": histogram_quantiles(positive),
            "disc_all_histogram": self.histogram.tolist(),
        }
        return {
            "counts": counts,
            "pooled_in_disc": distribution,
            "per_frame": {key: summarize(values) for key, values in self.frame.items()},
            "per_box": {key: summarize(values) for key, values in self.box.items()},
        }


def verify_membership(manifest: dict, data: dict) -> tuple[dict, dict]:
    images = {row["id"]: row for row in data["images"]}
    indexed = {row["id"]: row for row in manifest["images"]}
    if (
        set(images) != set(indexed)
        or len(images) != len(data["images"])
        or len(indexed) != manifest["image_count"]
        or len(indexed) != len(manifest["images"])
        or len(data["annotations"]) != manifest["annotation_count"]
    ):
        raise ValueError("Manifest and released dataset disagree on IDs or counts")
    partitions = {}
    for site, site_info in manifest["sites"].items():
        for split, ids in site_info["partitions"].items():
            if split not in SPLITS:
                raise ValueError(f"Unrecognized manifest partition: {split}")
            for image_id in ids:
                row, metadata = indexed[image_id], images[image_id]
                if (
                    image_id in partitions
                    or (row["site"], row["frame"], row["file_name"])
                    != (metadata["dataset"], metadata["frame"], metadata["file_name"])
                    or site != row["site"]
                ):
                    raise ValueError(f"Inconsistent manifest identity or partition for {image_id}")
                partitions[image_id] = (site, split)
    if set(partitions) != set(images):
        raise ValueError("Manifest partitions do not cover the released image inventory")
    return images, partitions


def audit(
    manifest_path: Path,
    annotations_path: Path,
    image_dir: Path,
    sites: list[str] | None = None,
    splits: list[str] | None = None,
) -> dict:
    manifest_bytes, annotation_bytes = manifest_path.read_bytes(), annotations_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    if annotation_sha256 != manifest["annotation_sha256"]:
        raise ValueError("Annotation digest differs from the frozen manifest")
    data = json.loads(annotation_bytes)
    images, partitions = verify_membership(manifest, data)
    site_set, split_set = set(sites or manifest["sites"]), set(splits or SPLITS)
    if not site_set <= set(manifest["sites"]) or not split_set <= set(SPLITS):
        raise ValueError(f"Invalid site or split selection: {site_set} {split_set}")

    annotations = defaultdict(list)
    for row in data["annotations"]:
        if row["image_id"] not in images:
            raise ValueError(f"Unknown image for annotation {row['id']}")
        annotations[row["image_id"]].append(row)

    aggregates = {}
    checked = 0
    hash_list = hashlib.sha256()
    for row in manifest["images"]:
        site, split = partitions[row["id"]]
        if site not in site_set or split not in split_set:
            continue
        path = _safe_image_path(image_dir, row["file_name"])
        image_bytes = path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["sha256"]:
            raise ValueError(f"Image digest differs from the frozen manifest: {path}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        metadata = images[row["id"]]
        if (
            image is None
            or image.shape != (metadata["height"], metadata["width"])
            or image.dtype != np.uint8
        ):
            raise ValueError(f"Unexpected released PNG dimensions or datatype: {path}")
        key = (site, split)
        if key not in aggregates:
            aggregates[key] = PixelGroup()
        aggregates[key].add(image, annotations[row["id"]])
        checked += 1
        hash_list.update(f"{row['id']}:{row['sha256']}\n".encode())
    if not checked:
        raise ValueError("No frames in the requested site and split selection")

    return {
        "schema_version": 1,
        "scope": "Post-hoc description of released grayscale PNG digital numbers and annotations.",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "annotation_sha256": annotation_sha256,
        "implementation_sha256": sha256_file(__file__),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "selection": {"sites": sorted(site_set), "splits": sorted(split_set)},
        "image_hashes": {
            "verified": checked,
            "expected_for_selection": sum(
                site in site_set and split in split_set for site, split in partitions.values()
            ),
            "manifest_order_id_colon_sha256_newline_digest": hash_list.hexdigest(),
        },
        "method": {
            "disc_center_xy_px": list(DISC_CENTER),
            "disc_radius_px": DISC_RADIUS,
            "positive_pixel_rule": "digital number > 0",
            "saturated_pixel_rule": "digital number = 255",
            "quantile_method": "linear interpolation between ordered samples",
            "box_pixels": "clip an integer envelope of the COCO box to image bounds and disc",
            "box_peak_comparison": "in-disc PNG bounding-box maximum versus original blob_peak field",
            "annulus": (
                "axis-aligned rectangular band 4–12 px beyond the box, clipped to image and disc "
                "and excluding all labeled boxes in that frame"
            ),
            "contrast": "PNG intensity percentile differences in digital numbers, not radar SNR",
            "limitations": (
                "The annulus can include unlabeled targets and display clutter. Values are "
                "from rendered 8-bit images and do not identify sensor amplitude or noise power."
            ),
        },
        "groups": [
            {"site": site, "split": split, **group.report()}
            for (site, split), group in sorted(aggregates.items())
        ],
    }


def write_csv(result: dict, path: Path) -> None:
    def emit(writer, group, section, metric, statistic, count, value):
        writer.writerow((group["site"], group["split"], section, metric, statistic, count, value))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("site", "split", "section", "metric", "statistic", "n", "value"))
        for group in result["groups"]:
            for key, value in group["counts"].items():
                emit(writer, group, "counts", key, "count", None, value)
            for key, value in group["pooled_in_disc"].items():
                if isinstance(value, dict):
                    count_key = "disc_positive_pixels" if "positive" in key else "disc_pixels"
                    for stat, number in value.items():
                        emit(
                            writer,
                            group,
                            "pooled_in_disc",
                            key,
                            stat,
                            group["counts"][count_key],
                            number,
                        )
                elif not isinstance(value, list):
                    emit(writer, group, "pooled_in_disc", key, "fraction", None, value)
            for section in ("per_frame", "per_box"):
                for key, stats in group[section].items():
                    emit(writer, group, section, key, "mean", stats["n"], stats["mean"])
                    for stat, value in stats["quantiles"].items():
                        emit(writer, group, section, key, stat, stats["n"], value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--site", action="append")
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = audit(args.manifest, args.ann, args.images, args.site, args.split)
    save_json(args.output_json, result)
    write_csv(result, args.output_csv)
    print(f"Verified {result['image_hashes']['verified']} image digests; wrote {args.output_json}")


if __name__ == "__main__":
    main()
