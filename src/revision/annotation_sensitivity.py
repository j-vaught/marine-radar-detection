"""Rescore frozen COCO predictions against 12 prespecified reference geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from . import evaluation
from .evaluation import evaluate_coco
from .protocol import save_json, sha256_file

AP_METRICS = ("ap50", "ap75", "ap")
CENTER_RADII = (2, 4, 8, 16)
CENTER_BUDGETS = (10, 30, 100)


def _clip_axis(start: float, extent: float, limit: float) -> tuple[float, float, bool]:
    end = start + extent
    low, high = max(0.0, start), min(limit, end)
    if low >= high:
        # A shift can move a thin edge box entirely outside the image.
        # Retain up to one pixel at the nearest edge and record the rescue.
        retained_extent = min(extent, limit, 1.0)
        return (0.0 if end <= 0.0 else limit - retained_extent), retained_extent, True
    return low, high - low, False


def _transform_bbox(
    bbox: list[float],
    image: dict,
    scale: float,
    dx: int,
    dy: int,
    fixed_dimensions: tuple[float, float] | None,
) -> tuple[list[float], bool, bool]:
    x, y, width, height = bbox
    new_width, new_height = fixed_dimensions or (width * scale, height * scale)
    x0 = x + dx + (width - new_width) / 2
    y0 = y + dy + (height - new_height) / 2
    image_width, image_height = image["width"], image["height"]
    clipped = x0 < 0 or y0 < 0 or x0 + new_width > image_width or y0 + new_height > image_height
    if not clipped:
        return [x0, y0, new_width, new_height], False, False
    out_x, out_w, rescued_x = _clip_axis(x0, new_width, image_width)
    out_y, out_h, rescued_y = _clip_axis(y0, new_height, image_height)
    return [out_x, out_y, out_w, out_h], True, rescued_x or rescued_y


def _validate_reference(data: dict) -> dict:
    images = {im["id"]: im for im in data["images"]}
    annotation_ids = {a["id"] for a in data["annotations"]}
    if len(images) != len(data["images"]) or len(annotation_ids) != len(data["annotations"]):
        raise ValueError("Duplicate image or annotation ID")
    for im in images.values():
        if any(not math.isfinite(size) or size <= 0 for size in (im["width"], im["height"])):
            raise ValueError(f"Invalid dimensions in image {im['id']}")
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        image = images[a["image_id"]]
        if (
            not all(math.isfinite(v) for v in (x, y, w, h))
            or min(w, h) <= 0
            or min(x, y) < 0
            or x + w > image["width"]
            or y + h > image["height"]
        ):
            raise ValueError(f"Annotation {a['id']} has an invalid released bbox")
        if a.get("segmentation"):
            raise ValueError("Reference geometry study requires bbox-only annotations")
    return images


def build_reference_views(
    references: dict, source_dimensions: tuple[float, float]
) -> dict[str, tuple[dict, dict]]:
    """Return each reference dataset and its clipping ledger, preserving every ID."""
    images = _validate_reference(references)
    if len(source_dimensions) != 2 or any(
        not math.isfinite(dimension) or dimension <= 0 for dimension in source_dimensions
    ):
        raise ValueError("Fixed source dimensions must be two positive finite lengths")
    treatments = [
        ("scale_0.8", 0.8, 0, 0, None),
        ("scale_1.2", 1.2, 0, 0, None),
    ]
    for pixels in (-4, -2, 2, 4):
        treatments.extend(
            [
                (f"shift_x_{pixels}", 1.0, pixels, 0, None),
                (f"shift_y_{pixels}", 1.0, 0, pixels, None),
            ]
        )
    treatments.append(("fixed_source_envelope", 1.0, 0, 0, source_dimensions))
    views = {"baseline": (deepcopy(references), _clip_ledger([], []))}
    for name, scale, dx, dy, fixed in treatments:
        transformed = deepcopy(references)
        clipped_ids = []
        rescued_ids = []
        for annotation in transformed["annotations"]:
            bbox, clipped, rescued = _transform_bbox(
                annotation["bbox"], images[annotation["image_id"]], scale, dx, dy, fixed
            )
            annotation["bbox"] = bbox
            annotation["area"] = bbox[2] * bbox[3]
            if clipped:
                clipped_ids.append(annotation["id"])
            if rescued:
                rescued_ids.append(annotation["id"])
        views[name] = (transformed, _clip_ledger(clipped_ids, rescued_ids))
    return views


def _clip_ledger(clipped_ids: list[int], rescued_ids: list[int]) -> dict:
    return {
        "clipped_annotations": len(clipped_ids),
        "fully_outside_on_an_axis": len(rescued_ids),
        "clipped_annotation_ids": clipped_ids,
        "edge_retained_annotation_ids": rescued_ids,
    }


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2, y + height / 2


def center_agreement(references: dict, predictions: list[dict]) -> dict:
    """Measure center recall under maximum-cardinality one-to-one matching."""
    gt = defaultdict(list)
    for annotation in references["annotations"]:
        if not annotation.get("iscrowd"):
            gt[(annotation["image_id"], annotation["category_id"])].append(
                _center(annotation["bbox"])
            )
    detections = defaultdict(list)
    for prediction in predictions:
        detections[(prediction["image_id"], prediction["category_id"])].append(prediction)

    totals = dict.fromkeys(((b, r) for b in CENTER_BUDGETS for r in CENTER_RADII), 0)
    for key, centers in gt.items():
        proposals = sorted(detections[key], key=lambda p: -p["score"])[: max(CENTER_BUDGETS)]
        if not proposals:
            continue
        gt_centers = np.array(centers)
        predicted_centers = np.array([_center(p["bbox"]) for p in proposals])
        squared_distances = ((predicted_centers[:, None] - gt_centers) ** 2).sum(axis=2)
        for budget, radius in totals:
            adjacency = csr_matrix(squared_distances[:budget] <= radius**2)
            matched = maximum_bipartite_matching(adjacency, perm_type="column") >= 0
            totals[budget, radius] += int(matched.sum())
    support = sum(map(len, gt.values()))
    return {
        "reference_annotations": support,
        "rows": [
            {
                "top_proposals_per_image_category": budget,
                "radius_px": radius,
                "matched_centers": count,
                "center_recall": count / support if support else None,
            }
            for (budget, radius), count in totals.items()
        ],
    }


def _content_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@lru_cache(maxsize=1)
def _scoring_backend() -> dict:
    return {
        "annotation_scorer_sha256": sha256_file(__file__),
        "coco_evaluator_sha256": sha256_file(evaluation.__file__),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pycocotools")},
    }


def _read_cached_score(path: Path, key: str) -> dict | None:
    try:
        record = json.loads(path.read_bytes())
        score = record["score"]
        if (
            record["key"] == key
            and isinstance(score, dict)
            and set(score) == {"coco", "center_agreement"}
            and record["score_sha256"] == _content_hash(score)
        ):
            return score
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


def _write_cached_score(path: Path, key: str, score: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as working:
        pending = Path(working) / path.name
        save_json(pending, {"key": key, "score": score, "score_sha256": _content_hash(score)})
        os.replace(pending, path)


def score_model_views(
    views: dict[str, tuple[dict, dict]],
    predictions: list[dict],
    *,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict], int]:
    """Score one frozen prediction set on each reference and return the cache hit count."""
    identity = {}
    if cache_dir is not None:
        identity = {
            "schema_version": 1,
            "prediction_sha256": _content_hash(predictions),
            "scoring_backend": _scoring_backend(),
            "center_budgets": CENTER_BUDGETS,
            "center_radii_px": CENTER_RADII,
        }
    scores = {}
    hits = 0
    for name, (references, _) in views.items():
        score = None
        path = None
        if cache_dir is not None:
            key = _content_hash({**identity, "reference_sha256": _content_hash(references)})
            path = cache_dir / key[:2] / f"{key}.json"
            score = _read_cached_score(path, key)
        if score is not None:
            hits += 1
        else:
            score = {
                "coco": evaluate_coco(references, predictions),
                "center_agreement": center_agreement(references, predictions),
            }
            if path is not None:
                _write_cached_score(path, path.stem, score)
        scores[name] = score
    return scores, hits


def score_reference_views(
    views: dict[str, tuple[dict, dict]],
    predictions_by_method: dict[str, list[dict]],
    *,
    cache_dir: Path | None = None,
) -> dict:
    """Score all methods on the same views for one fixed source and target."""
    if not predictions_by_method:
        raise ValueError("At least one frozen prediction set is required")
    scores = {
        method: score_model_views(views, predictions, cache_dir=cache_dir)[0]
        for method, predictions in predictions_by_method.items()
    }
    results: dict[str, dict] = {}
    for name, (_, clipping) in views.items():
        metrics = {method: method_scores[name].copy() for method, method_scores in scores.items()}
        rankings = {}
        for metric in AP_METRICS:
            values = {method: row["coco"][metric] for method, row in metrics.items()}
            rankings[metric] = {
                method: 1 + sum(other > value for other in values.values())
                for method, value in values.items()
            }
        results[name] = {
            "clipping": clipping,
            "methods": metrics,
            "ranks": rankings,
            "otsu_minus_comparators": {
                method: {
                    metric: metrics["otsu"]["coco"][metric] - metrics[method]["coco"][metric]
                    for metric in AP_METRICS
                }
                for method in metrics
                if "otsu" in metrics and method != "otsu"
            },
        }
    baseline = results["baseline"]
    for view in results.values():
        for method, result in view["methods"].items():
            result["ap_change_from_baseline"] = {
                metric: result["coco"][metric] - baseline["methods"][method]["coco"][metric]
                for metric in AP_METRICS
            }
            result["rank_change_from_baseline"] = {
                metric: view["ranks"][metric][method] - baseline["ranks"][metric][method]
                for metric in AP_METRICS
            }
        view["otsu_gap_change_from_baseline"] = {
            method: {
                metric: gap[metric] - baseline["otsu_minus_comparators"][method][metric]
                for metric in AP_METRICS
            }
            for method, gap in view["otsu_minus_comparators"].items()
        }
    return results


def _load_references(
    ann_path: Path, manifest: dict, source: str, target: str
) -> tuple[dict, set[int], dict]:
    if manifest["annotation_sha256"] != sha256_file(ann_path):
        raise ValueError("Annotation hash does not match the frozen split manifest")
    data = json.loads(ann_path.read_text())
    train_ids = set(manifest["sites"][source]["partitions"]["train"])
    test_ids = set(manifest["sites"][target]["partitions"]["test"])
    if not test_ids or train_ids & test_ids:
        raise ValueError("Require nonempty target test IDs disjoint from source train")
    images = {im["id"]: im for im in data["images"]}
    if any(images[i]["dataset"] != source for i in train_ids) or any(
        images[i]["dataset"] != target for i in test_ids
    ):
        raise ValueError("Split site and released image metadata disagree")
    source_annotations = [a for a in data["annotations"] if a["image_id"] in train_ids]
    if not source_annotations:
        raise ValueError("The source training partition has no labels for the fixed envelope")
    references = {
        **data,
        "images": [im for im in data["images"] if im["id"] in test_ids],
        "annotations": [a for a in data["annotations"] if a["image_id"] in test_ids],
    }
    envelope = {
        "source": source,
        "source_partition": "buffered train",
        "train_image_ids": sorted(train_ids),
        "source_annotations": len(source_annotations),
        "width_height_px": tuple(median(a["bbox"][i] for a in source_annotations) for i in (2, 3)),
        "comparison_scope": "same source, target holdout, and fixed envelope",
    }
    target_site_ids = {im["id"] for im in data["images"] if im["dataset"] == target}
    return references, target_site_ids, envelope


def _load_predictions(
    items: list[str], targets: set[int], test: set[int]
) -> tuple[dict[str, list[dict]], dict]:
    subsets = {}
    provenance = {}
    for item in items:
        method, separator, filename = item.partition("=")
        if not separator or not method or not filename or method in subsets:
            raise ValueError("Use --pred METHOD=PATH with a distinct method for each file")
        path = Path(filename)
        predictions = json.loads(path.read_text())
        if not isinstance(predictions, list):
            raise TypeError(f"Expected a COCO prediction array: {path}")
        if any(p["image_id"] not in targets for p in predictions):
            raise ValueError(f"Prediction image ID outside the declared target site: {path}")
        subset = [p for p in predictions if p["image_id"] in test]
        subsets[method] = subset
        provenance[method] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "prediction_rows": len(predictions),
            "test_prediction_rows": len(subset),
            "non_test_prediction_rows": len(predictions) - len(subset),
            "test_images_with_predictions": len({p["image_id"] for p in subset}),
        }
    return subsets, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--pred", metavar="METHOD=PATH", required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Reuse content-addressed model scores")
    args = parser.parse_args()
    inputs = {args.ann.resolve(), args.manifest.resolve()}
    inputs.update(Path(item.partition("=")[2]).resolve() for item in args.pred)
    if args.output.resolve() in inputs:
        parser.error("Choose an output path distinct from all input files")

    manifest = json.loads(args.manifest.read_text())
    references, target_ids, envelope = _load_references(
        args.ann, manifest, args.source, args.target
    )
    test_ids = {image["id"] for image in references["images"]}
    predictions, prediction_sources = _load_predictions(args.pred, target_ids, test_ids)
    views = build_reference_views(references, envelope["width_height_px"])
    report = {
        "schema_version": 1,
        "source": args.source,
        "target": args.target,
        "target_test_image_ids": sorted(test_ids),
        "test_reference_annotations": len(references["annotations"]),
        "annotations_sha256": manifest["annotation_sha256"],
        "split_manifest_sha256": sha256_file(args.manifest),
        "implementation_sha256": sha256_file(__file__),
        "predictions": prediction_sources,
        "fixed_source_envelope": envelope,
        "geometry": {
            "scale_factors": [0.8, 1.2],
            "one_axis_shifts_px": [-4, -2, 2, 4],
            "clipping": "Continuous COCO coordinates. An axis that leaves the image retains "
            "up to one pixel along the closest edge.",
        },
        "results": score_reference_views(views, predictions, cache_dir=args.cache_dir),
    }
    save_json(args.output, report)


if __name__ == "__main__":
    main()
