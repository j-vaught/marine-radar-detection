"""Deterministic, versioned input transforms for the processed-image DARC study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import protocol

PROTOCOL_ID = "darc-processed-v1"
CENTER = (531.0, 515.0)
RADIUS = 466.0
SHAPE = (1024, 1050)


def disc_mask(shape=SHAPE):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2 <= RADIUS**2


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def hash_uniform(text):
    """Use the first 53 digest bits, interpreted as an unsigned big-endian integer."""
    return (int.from_bytes(_hash_text(text)[:7], "big") >> 3) / 2**53


def _image_ids(images):
    ids = [int(image["id"]) for image in images]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Image IDs must be nonempty and unique")
    return ids


def train_stream(coco_images, source, seed, presentations=6864):
    """Concatenate deterministic complete permutations before forming batches."""
    ids = _image_ids(coco_images)
    if not isinstance(presentations, int) or presentations < 0:
        raise ValueError("Presentation count must be a nonnegative integer")
    ordered = []
    cycle = 0
    while len(ordered) < presentations:
        ordered.extend(
            sorted(
                ids,
                key=lambda image_id: (
                    _hash_text(f"{PROTOCOL_ID}:train-order:{source}:{seed}:{cycle}:{image_id}"),
                    image_id,
                ),
            )
        )
        cycle += 1
    return list(enumerate(ordered[:presentations]))


def select_count_subset(coco_images, source, seed, count=316):
    """Select a count-matched subset from twelve contiguous frame-index strata."""
    _image_ids(coco_images)
    images = sorted(coco_images, key=lambda image: (int(image["frame"]), int(image["id"])))
    n = len(images)
    if not isinstance(count, int) or not 0 < count <= n:
        raise ValueError("Subset count must be positive and at most the source count")
    strata = [images[k * n // 12 : (k + 1) * n // 12] for k in range(12)]
    quotas = [count * len(stratum) // n for stratum in strata]
    remainders = [count * len(stratum) % n for stratum in strata]
    for k in sorted(range(12), key=lambda k: (-remainders[k], k))[: count - sum(quotas)]:
        quotas[k] += 1
    selected = set()
    for stratum, quota in zip(strata, quotas, strict=True):
        ranked = sorted(
            stratum,
            key=lambda image: (
                _hash_text(f"{PROTOCOL_ID}:count-subset:{source}:{seed}:{int(image['id'])}"),
                int(image["id"]),
            ),
        )
        selected.update(int(image["id"]) for image in ranked[:quota])
    if len(selected) != count:
        raise RuntimeError("Count-subset allocation did not preserve the declared count")
    return [int(image["id"]) for image in images if int(image["id"]) in selected]


def normalize_intensity(image, level="I0"):
    """Apply the frozen imagewise rule without using labels or pooled statistics."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Intensity input must be a two-dimensional uint8 raster")
    if level not in {"I0", "I1"}:
        raise ValueError(f"Unknown intensity level: {level}")
    mask = disc_mask(image.shape)
    support = image[mask & (image > 3)]
    count = len(support)
    high = float(np.percentile(support, 99.5, method="linear")) if count else None
    if count < 256:
        reason = "low_support"
    else:
        assert high is not None
        reason = "small_span" if high - 3 < 16 else None
    fallback = level == "I1" and reason is not None
    if level == "I0" or fallback:
        result = image.astype(np.float32) / np.float32(255)
    else:
        assert high is not None
        result = np.clip((image.astype(np.float64) - 3) / (high - 3), 0, 1).astype(np.float32)
    result[~mask] = 0
    return result, {
        "intensity": level,
        "intensity_fallback": fallback,
        "intensity_fallback_reason": reason if fallback else None,
        "high": high,
        "above_pedestal_count": count,
    }


def affine_boxes(boxes, scale):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    origin = np.asarray([*CENTER, *CENTER], dtype=np.float64)
    return (boxes - origin) * float(scale) + origin


def valid_boxes(boxes, shape=SHAPE, require_disc=True):
    boxes = np.asarray(boxes).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        return False
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        return False
    h, w = shape
    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [w, h]):
        return False
    if require_disc:
        corners = boxes[:, [0, 1, 2, 1, 0, 3, 2, 3]].reshape(-1, 2)
        return bool(np.all(np.sum((corners - CENTER) ** 2, axis=1) <= RADIUS**2))
    return True


def spatial_parameters(
    boxes_xyxy, source, seed, presentation_index, image_id, scale_level="S0", shape=SHAPE
):
    """Choose scale using only label geometry and a fixed hash-derived draw."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if not valid_boxes(boxes, shape, require_disc=False):
        raise ValueError("Original labels must be positive, finite and within the canvas")
    if scale_level not in {"S0", "S1"}:
        raise ValueError(f"Unknown scale level: {scale_level}")
    if scale_level == "S1" and (not isinstance(presentation_index, int) or presentation_index < 0):
        raise ValueError("Jitter requires a nonnegative integer presentation index")
    key = f"{PROTOCOL_ID}:zoom:{source}:{seed}:{presentation_index}:{image_id}"
    scale = 1.0
    reason = "baseline"
    attempted = []
    rejections = 0
    if scale_level == "S1":
        if hash_uniform(f"{key}:gate") < 0.5:
            reason = "gate"
        else:
            reason = "rejected"
            for tag in ("scale0", "scale1"):
                candidate = 2 ** (hash_uniform(f"{key}:{tag}") - 0.5)
                attempted.append(candidate)
                if valid_boxes(affine_boxes(boxes, candidate), shape):
                    scale, reason = candidate, None
                    break
                rejections += 1
    transformed = affine_boxes(boxes, scale)
    before = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    after = np.prod(transformed[:, 2:] - transformed[:, :2], axis=1)
    corners = boxes[:, [0, 1, 2, 1, 0, 3, 2, 3]].reshape(-1, 2)
    max_radius = (
        float(np.sqrt(np.sum((corners - CENTER) ** 2, axis=1)).max()) if len(boxes) else 0.0
    )
    return transformed, {
        "scale_level": scale_level,
        "scale": scale,
        "scale_identity_reason": reason,
        "scale_rejections": rejections,
        "attempted_scales": attempted,
        "object_count": len(boxes),
        "box_max_radius_px": max_radius,
        "box_areas_before": before.tolist(),
        "box_areas_after": after.tolist(),
        "native_size_hw": list(shape),
    }


def transform_training(
    image, boxes_xyxy, source, seed, presentation_index, image_id, scale_level="S0"
):
    """Apply geometry-only deterministic rejection, shared across intensity levels."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("Spatial input must be a finite float32 scalar raster")
    transformed, stats = spatial_parameters(
        boxes_xyxy, source, seed, presentation_index, image_id, scale_level, image.shape
    )
    scale = stats["scale"]
    if scale == 1:
        # Preserve an exact canonical raster for baseline and identity draws.
        result = image.copy()
    else:
        raster = image
        if scale < 1:
            sigma = 0.5 * np.sqrt(1 / scale**2 - 1)
            raster = cv2.GaussianBlur(
                raster, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT
            )
        matrix = np.array(
            [[scale, 0, (1 - scale) * CENTER[0]], [0, scale, (1 - scale) * CENTER[1]]],
            dtype=np.float64,
        )
        result = cv2.warpAffine(
            raster,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        result[~disc_mask(image.shape)] = 0
    return result, transformed.astype(np.float32), stats


class DARCData(Dataset):
    """Canonical evaluation items or an explicit deterministic training stream."""

    def __init__(
        self,
        coco,
        image_dir,
        intensity="I0",
        scale="S0",
        source="",
        seed=0,
        presentations=None,
        expected_hashes=None,
        cache_size=0,
    ):
        _image_ids(coco["images"])
        if scale == "S1" and presentations is None:
            raise ValueError("Scale jitter requires an explicit training-presentation stream")
        self.images = {int(image["id"]): image for image in coco["images"]}
        self.image_dir = Path(image_dir).resolve()
        self.intensity, self.scale, self.source, self.seed = intensity, scale, source, seed
        self.presentations = (
            [(None, int(image["id"])) for image in coco["images"]]
            if presentations is None
            else list(presentations)
        )
        if any(image_id not in self.images for _, image_id in self.presentations):
            raise ValueError("Presentation references an image outside the provided subset")
        self.expected_hashes = (
            None if expected_hashes is None else {int(k): v for k, v in expected_hashes.items()}
        )
        self.verified = set()
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError("Cache size cannot be negative")
        self.cache = OrderedDict()
        self.annotations = {}
        for annotation in coco["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in self.images:
                raise ValueError("Annotation references an image outside the provided subset")
            self.annotations.setdefault(image_id, []).append(annotation)

    def __len__(self):
        return len(self.presentations)

    def __getitem__(self, index):
        presentation_index, image_id = self.presentations[index]
        info = self.images[image_id]
        path = protocol._safe_image_path(self.image_dir, info["file_name"])
        if self.expected_hashes is not None and image_id not in self.verified:
            if image_id not in self.expected_hashes:
                raise ValueError(f"Image {image_id} has no expected digest")
            if protocol.sha256_file(path) != self.expected_hashes[image_id]:
                raise ValueError(f"Input image digest mismatch for image {image_id}")
            self.verified.add(image_id)
        cache_key = (
            PROTOCOL_ID,
            self.intensity,
            image_id,
            self.expected_hashes.get(image_id) if self.expected_hashes is not None else str(path),
        )
        cached = self.cache.pop(cache_key, None)
        if cached is None:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot decode image {path}")
            if image.shape != SHAPE or image.shape != (int(info["height"]), int(info["width"])):
                raise ValueError(f"Unexpected image dimensions for image {image_id}: {image.shape}")
            raster, intensity_stats = normalize_intensity(image, self.intensity)
        else:
            raster, intensity_stats = cached
        if self.cache_size:
            self.cache[cache_key] = (raster, intensity_stats)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        boxes = []
        for annotation in self.annotations.get(image_id, []):
            x, y, w, h = map(float, annotation["bbox"])
            boxes.append([x, y, x + w, y + h])
        raster, transformed, scale_stats = transform_training(
            raster, boxes, self.source, self.seed, presentation_index, image_id, self.scale
        )
        target = {
            "boxes": torch.from_numpy(transformed),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }
        stats = {
            "image_id": image_id,
            "presentation_index": presentation_index,
            **intensity_stats,
            **scale_stats,
        }
        return torch.from_numpy(np.repeat(raster[None], 3, axis=0)), target, stats


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _spatial_summary(records):
    scales = np.asarray([record["scale"] for record in records], dtype=np.float64)
    areas_before = [area for record in records for area in record["box_areas_before"]]
    areas_after = [area for record in records for area in record["box_areas_after"]]

    def quantiles(values):
        return np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist() if len(values) else []

    return {
        "presentations": len(records),
        "accepted_shrink": int(np.sum(scales < 1)),
        "accepted_grow": int(np.sum(scales > 1)),
        "identity_gate": sum(record["scale_identity_reason"] == "gate" for record in records),
        "identity_rejected": sum(
            record["scale_identity_reason"] == "rejected" for record in records
        ),
        "rejected_draws": sum(record["scale_rejections"] for record in records),
        "scale_min_p05_median_p95_max": quantiles(scales),
        "native_box_area_min_p05_median_p95_max": quantiles(areas_before),
        "warped_box_area_min_p05_median_p95_max": quantiles(areas_after),
    }


def source_spatial_audit(coco, split, presentations=6864):
    """Audit the exact source-training streams without reading destination labels."""
    images = {int(image["id"]): image for image in coco["images"]}
    results = []
    for source in ("DAAN", "DARC", "MANV"):
        parts = split["sites"][source]["partitions"]
        train_ids = {int(image_id) for image_id in parts["train"]}
        other_ids = {
            int(image_id) for key, ids in parts.items() if key != "train" for image_id in ids
        }
        if train_ids & other_ids:
            raise ValueError("Source training overlaps a nontraining partition")
        train_images = [images[image_id] for image_id in sorted(train_ids)]
        boxes = {image_id: [] for image_id in train_ids}
        for annotation in coco["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id in train_ids:
                x, y, w, h = map(float, annotation["bbox"])
                boxes[image_id].append([x, y, x + w, y + h])
        for seed in (0, 1, 2):
            stream = train_stream(train_images, source, seed, presentations)
            records = []
            for presentation_index, image_id in stream:
                _, stats = spatial_parameters(
                    boxes[image_id], source, seed, presentation_index, image_id, "S1"
                )
                records.append(stats)
            entry = {
                "source": source,
                "seed": seed,
                "train_ids": sorted(train_ids),
                "stream": stream,
                "stream_sha256": _json_digest(stream),
                "overall": _spatial_summary(records),
                "by_object_count": {
                    name: _spatial_summary(
                        [record for record in records if low <= record["object_count"] < high]
                    )
                    for name, low, high in (
                        ("0", 0, 1),
                        ("1", 1, 2),
                        ("2-3", 2, 4),
                        ("4+", 4, float("inf")),
                    )
                },
                "by_max_corner_radius": {
                    name: _spatial_summary(
                        [record for record in records if low <= record["box_max_radius_px"] < high]
                    )
                    for name, low, high in (
                        ("below_half_radius", 0, 233),
                        ("half_to_four_fifths", 233, 372.8),
                        ("outer_fifth", 372.8, float("inf")),
                    )
                },
            }
            if source in {"DAAN", "MANV"}:
                selected = select_count_subset(train_images, source, seed)
                reduced_stream = train_stream(
                    [images[image_id] for image_id in selected], source, seed, presentations
                )
                entry["reduced"] = {
                    "train_ids": selected,
                    "train_ids_sha256": _json_digest(selected),
                    "stream": reduced_stream,
                    "stream_sha256": _json_digest(reduced_stream),
                    "object_count": sum(len(boxes[image_id]) for image_id in selected),
                }
            results.append(entry)
    return {
        "scope": "Source-training geometry only, fixed stream; intensity-independent draws",
        "quantile_probabilities": [0, 0.05, 0.5, 0.95, 1],
        "runtime_gate": "The trainer records actual model-effective sizes; this audit uses native-raster geometry.",
        "runs": results,
    }


def source_pixel_smoke(coco, image_dir, split, expected_hashes):
    """Check old-loader identity and actual source-raster warp alignment before scoring."""
    from .torch_models import RadarDataset

    rows = []
    for source in ("DAAN", "DARC", "MANV"):
        ids = split["sites"][source]["partitions"]["train"]
        source_coco = protocol._subset(coco, ids)
        canonical = DARCData(source_coco, image_dir, expected_hashes=expected_hashes)
        legacy = RadarDataset(source_coco, Path(image_dir), "faster_rcnn", (640, 640))
        for index in (0, len(canonical) // 2, len(canonical) - 1):
            image, target, stats = canonical[index]
            old_image, old_target = legacy[index]
            if not torch.equal(image, old_image) or any(
                not torch.equal(target[key], old_target[key]) for key in target
            ):
                raise ValueError("Canonical I0 input differs from the historical detector loader")
            rows.append({"source": source, "image_id": stats["image_id"], "I0_bridge_exact": True})
        stream = train_stream(source_coco["images"], source, 0, 6864)
        jitter0 = DARCData(source_coco, image_dir, "I0", "S1", source, 0, stream, expected_hashes)
        jitter1 = DARCData(source_coco, image_dir, "I1", "S1", source, 0, stream, expected_hashes)
        directions = set()
        for index, (presentation_index, image_id) in enumerate(stream):
            boxes = []
            for annotation in jitter0.annotations.get(image_id, []):
                x, y, w, h = map(float, annotation["bbox"])
                boxes.append([x, y, x + w, y + h])
            _, stats = spatial_parameters(boxes, source, 0, presentation_index, image_id, "S1")
            direction = "shrink" if stats["scale"] < 1 else "grow" if stats["scale"] > 1 else None
            if direction is None or direction in directions:
                continue
            image0, target0, actual0 = jitter0[index]
            image1, target1, actual1 = jitter1[index]
            if actual0["scale"] != actual1["scale"] or not torch.equal(
                target0["boxes"], target1["boxes"]
            ):
                raise ValueError("Intensity levels received different geometry")
            if len(target0["boxes"]) != len(boxes):
                raise ValueError("Jitter changed the label inventory")
            for image in (image0, image1):
                if (
                    not torch.isfinite(image).all()
                    or float(image.min()) < 0
                    or float(image.max()) > 1
                ):
                    raise ValueError("Jitter produced an invalid pixel range")
                if bool(image[0][~torch.from_numpy(disc_mask())].any()):
                    raise ValueError("Jitter leaked pixels outside the fixed mask")
            inverse = affine_boxes(target0["boxes"].numpy(), 1 / actual0["scale"])
            error = float(np.max(np.abs(inverse - np.asarray(boxes)))) if boxes else 0.0
            if error > 1e-4:
                raise ValueError("Jitter box inverse exceeded float32 tolerance")
            rows.append(
                {
                    "source": source,
                    "image_id": image_id,
                    "presentation_index": presentation_index,
                    "direction": direction,
                    "scale": actual0["scale"],
                    "object_count": len(boxes),
                    "inverse_max_absolute_error_px": error,
                    "I0_float32_sha256": hashlib.sha256(
                        image0[0].numpy().astype("<f4", copy=False).tobytes()
                    ).hexdigest(),
                    "I1_float32_sha256": hashlib.sha256(
                        image1[0].numpy().astype("<f4", copy=False).tobytes()
                    ).hexdigest(),
                }
            )
            directions.add(direction)
            if directions == {"shrink", "grow"}:
                break
        if directions != {"shrink", "grow"}:
            raise ValueError(f"No realized shrink and grow smoke pair for {source}")
    return {"passed": True, "rows": rows}


def export_version(ann_path, image_dir, split_path, specification_path, output_dir):
    """Create a fresh portable ledger; transforms remain lossless float32 at load time."""
    ann_path, split_path, specification_path = map(Path, (ann_path, split_path, specification_path))
    specification = json.loads(specification_path.read_text())
    split = json.loads(split_path.read_text())
    coco = json.loads(ann_path.read_text())
    for actual, expected, name in (
        (
            protocol.sha256_file(ann_path),
            specification["inputs"]["annotations_sha256"],
            "annotations",
        ),
        (protocol.sha256_file(split_path), specification["inputs"]["split_sha256"], "split"),
    ):
        if actual != expected:
            raise ValueError(f"Frozen {name} digest does not match")
    if split["annotation_sha256"] != protocol.sha256_file(ann_path):
        raise ValueError("Split annotations do not match")
    if specification["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unsupported transform specification")
    expected = {int(record["id"]): record["sha256"] for record in split["images"]}
    if set(expected) != {int(image["id"]) for image in coco["images"]}:
        raise ValueError("Split and annotation image inventories differ")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for intensity in ("I0", "I1"):
        dataset = DARCData(coco, image_dir, intensity=intensity, expected_hashes=expected)
        for index in range(len(dataset)):
            image, _, stats = dataset[index]
            image_id = stats["image_id"]
            info = dataset.images[image_id]
            record = records.setdefault(
                image_id,
                {
                    "id": image_id,
                    "file_name": info["file_name"],
                    "site": info["dataset"],
                    "frame": info["frame"],
                    "source_sha256": expected[image_id],
                    "views": {},
                },
            )
            # Explicit little-endian scalar serialization, channels replicated at training load.
            array = image[0].numpy().astype("<f4", copy=False)
            record["views"][intensity] = {
                "scalar_float32_le_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "shape_hw": list(array.shape),
                "stats": stats,
            }
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "product": "Versioned float32 views of the released processed PNGs; no raw reconstruction",
        "storage": "Canonical input PNGs remain immutable. Apply the hashed loader online; no uint8 requantization.",
        "annotation_sha256": protocol.sha256_file(ann_path),
        "split_sha256": protocol.sha256_file(split_path),
        "specification_sha256": protocol.sha256_file(specification_path),
        "loader_sha256": protocol.sha256_file(Path(__file__)),
        "images": [records[key] for key in sorted(records)],
    }
    protocol.save_json(output_dir / "manifest.json", manifest)
    protocol.save_json(output_dir / "source_spatial_audit.json", source_spatial_audit(coco, split))
    protocol.save_json(
        output_dir / "source_pixel_smoke.json", source_pixel_smoke(coco, image_dir, split, expected)
    )
    (output_dir / "specification.json").write_bytes(specification_path.read_bytes())
    (output_dir / "split_manifest.json").write_bytes(split_path.read_bytes())
    (output_dir / "annotations.json").write_bytes(ann_path.read_bytes())
    (output_dir / "darc_data.py").write_bytes(Path(__file__).read_bytes())
    protocol.save_json(
        output_dir / "checksums.json",
        {path.name: protocol.sha256_file(path) for path in sorted(output_dir.iterdir())},
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_version(args.ann, args.images, args.split, args.specification, args.output)


if __name__ == "__main__":
    main()
