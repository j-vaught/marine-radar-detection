"""Prespecified source-validation wavelet neighbors and component diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from .classical import _native_predictions
from .evaluation import evaluate_coco
from .protocol import _safe_image_path, _subset, save_json, sha256_file


def neighbors(parent: dict) -> list[dict]:
    """Retain the parent at index zero and change one axis per neighbor."""
    result = [parent.copy()]
    for axis, values in (
        ("level", (1,)),
        ("threshold_sigma", (9.0, 10.5)),
        ("min_area", (100, 150)),
        ("box_padding", (1, 2)),
    ):
        for value in values:
            candidate = {**parent, axis: value}
            if candidate not in result:
                result.append(candidate)
    return result


def choose(candidates: list[dict]) -> dict:
    if not candidates or any(
        not math.isfinite(c["metrics"][key]) or c["metrics"][key] < 0
        for c in candidates
        for key in ("ap50", "ap")
    ):
        raise ValueError("All candidates must have defined finite validation AP and AP50")
    return min(candidates, key=lambda c: (-c["metrics"]["ap50"], -c["metrics"]["ap"], c["index"]))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def proposal_summary(coco, predictions):
    by_image = {}
    for pred in predictions:
        by_image.setdefault(pred["image_id"], []).append(pred)
    best = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) or ann["category_id"] != coco["categories"][0]["id"]:
            continue
        best.append(
            max(
                (iou(ann["bbox"], p["bbox"]) for p in by_image.get(ann["image_id"], [])),
                default=0.0,
            )
        )
    scores = Counter(p["score"] for p in predictions)
    return {
        "reference_objects": len(best),
        "proposal_count": len(predictions),
        "proposals_per_image": len(predictions) / len(coco["images"]),
        "mean_best_iou": sum(best) / len(best) if best else None,
        "recall_at_iou": {
            str(t): sum(v >= t for v in best) / len(best) if best else None
            for t in (0.25, 0.5, 0.75)
        },
        "unique_scores": len(scores),
        "tied_proposals": sum(n for n in scores.values() if n > 1),
        "largest_score_tie": max(scores.values(), default=0),
        "score_ties_scope": "pooled across images; scores are normalized per image",
    }


def _read(path):
    return json.loads(Path(path).read_text())


def validate_parent(parent: Path, ann: Path, manifest_path: Path, images: Path):
    completion = _read(parent / "completion.json")
    receipt = _read(parent / "receipt.json")
    job = _read(parent / "job.json")
    metrics = _read(parent / "metrics.json")
    for key in ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id"):
        if completion.get(key) is None or completion[key] != receipt.get(key):
            raise ValueError(f"Parent receipt mismatch: {key}")
    if (
        completion.get("state") != "succeeded"
        or completion.get("exit_code") != 0
        or completion.get("metrics_sha256") != sha256_file(parent / "metrics.json")
    ):
        raise ValueError("Parent completion or metrics hash invalid")
    if (
        job.get("job_id") != completion["job_id"]
        or job.get("attempt_nonce") != completion["nonce"]
        or metrics.get("job_id") != job["job_id"]
        or job.get("method") != "wavelet"
        or job.get("stage") != "core"
        or job.get("condition") != "buffered"
        or metrics.get("method") != "wavelet"
    ):
        raise ValueError("Parent is not a completed buffered core wavelet run")
    manifest = _read(manifest_path)
    ann_sha, manifest_sha = sha256_file(ann), sha256_file(manifest_path)
    if (
        manifest["annotation_sha256"] != ann_sha
        or metrics.get("annotation_sha256") != ann_sha
        or completion.get("manifest_sha256") != manifest_sha
        or job.get("protocol_sha256") != manifest_sha
        or metrics.get("paths", {}).get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("Annotation or split manifest hash mismatch")
    if (
        len(manifest["sites"]) != 3
        or set(job.get("eval_targets", [])) != set(manifest["sites"])
        or set(metrics["tests"]) != set(manifest["sites"])
    ):
        raise ValueError("Parent must use all three frozen holdouts")
    source = job["source"]
    selection = metrics["selected_config"]
    if selection.get("source") != source or _read(parent / "selected_config.json") != selection:
        raise ValueError("Selected configuration mismatch")
    data = _read(ann)
    names = {
        "validation": (source, "validation"),
        **{f"test_{s}": (s, "test") for s in manifest["sites"]},
    }
    subsets = {}
    records = {r["id"]: r for r in manifest["images"]}
    if len(records) != len(manifest["images"]):
        raise ValueError("Duplicate image IDs in manifest")
    seen_ids = set()
    for name, (site, partition) in names.items():
        ids = manifest["sites"][site]["partitions"][partition]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate partition {name}")
        if seen_ids.intersection(ids):
            raise ValueError("Validation and holdouts must be disjoint")
        seen_ids.update(ids)
        subset = _subset(data, ids)
        if {im["id"] for im in subset["images"]} != set(ids):
            raise ValueError(f"Missing images in {name}")
        path = parent / f"{name}_annotations.json"
        if _read(path) != subset or metrics["paths"]["subsets"][name][
            "annotations_sha256"
        ] != sha256_file(path):
            raise ValueError(f"Parent subset differs from frozen COCO: {name}")
        for im in subset["images"]:
            record = records[im["id"]]
            if (
                record["site"] != site
                or record["file_name"] != im["file_name"]
                or sha256_file(_safe_image_path(images, im["file_name"])) != record["sha256"]
            ):
                raise ValueError(f"Image identity or hash mismatch: {im['id']}")
        subsets[name] = subset
    return job, metrics, manifest, subsets


def run(
    parent: Path,
    ann: Path,
    images: Path,
    manifest_path: Path,
    binary: Path,
    out: Path,
    threads: int,
):
    if threads < 1 or not binary.is_file():
        raise ValueError("A native binary and positive thread count are required")
    parent, ann, images, manifest_path, binary, out = (
        p.resolve() for p in (parent, ann, images, manifest_path, binary, out)
    )
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    job, metrics, manifest, subsets = validate_parent(parent, ann, manifest_path, images)
    out.mkdir(parents=True)
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    started = time.monotonic()
    config = metrics["selected_config"]["config"]
    source = job["source"]
    val_path = out / "validation_annotations.json"
    save_json(val_path, subsets["validation"])
    provenance = {
        "parent_job_id": job["job_id"],
        "parent_completion_sha256": sha256_file(parent / "completion.json"),
        "parent_metrics_sha256": sha256_file(parent / "metrics.json"),
        "annotations_sha256": sha256_file(ann),
        "manifest_sha256": sha256_file(manifest_path),
        "binary_sha256": sha256_file(binary),
        "source": source,
        "threads": threads,
        "scope": "prespecified separate extension; source validation selection only",
    }
    save_json(out / "provenance.json", provenance)
    candidates = []
    for index, cfg in enumerate(neighbors(config)):
        tick = time.monotonic()
        path = out / f"candidate_{index:04d}_validation_predictions.json"
        preds = _native_predictions("wavelet", cfg, val_path, images, path, "cpu", binary)
        candidates.append(
            {
                "index": index,
                "config": cfg,
                "metrics": evaluate_coco(subsets["validation"], preds),
                "seconds": time.monotonic() - tick,
                "predictions_sha256": sha256_file(path),
            }
        )
        save_json(out / "candidates.json", candidates)
    selected = choose(candidates)
    # This durable record precedes any holdout predictions or metrics.
    save_json(
        out / "selection.json",
        {
            "criterion": "validation AP50, then AP, then lowest stable index",
            "winner": selected,
            "parent_index": 0,
            "candidate_count": len(candidates),
        },
    )
    selection_hash = sha256_file(out / "selection.json")
    max_area = max(im["width"] * im["height"] for im in subsets["validation"]["images"])
    diagnostics = {}
    for label, cfg in (("parent", config), ("selected", selected["config"])):
        for pruning, current in (
            ("pruned", cfg),
            ("unpruned", {**cfg, "min_area": 1, "max_area": max_area}),
        ):
            path = out / f"{label}_{pruning}_validation_predictions.json"
            tick = time.monotonic()
            preds = _native_predictions("wavelet", current, val_path, images, path, "cpu", binary)
            diagnostics[f"{label}_{pruning}"] = {
                **proposal_summary(subsets["validation"], preds),
                "config": current,
                "seconds": time.monotonic() - tick,
                "predictions_sha256": sha256_file(path),
            }
    diagnostics["score_provenance"] = (
        "Native detector clamps response ratio at 8, quantizes to 8-bit pixel scores, "
        "uses component peak, then divides by the maximum retained component peak per image. "
        "Pruning changes that normalization. No continuous-score ablation was performed."
    )
    save_json(out / "component_diagnostics.json", diagnostics)
    tests = {}
    for site in sorted(manifest["sites"]):
        subset = subsets[f"test_{site}"]
        path = out / f"test_{site}_annotations.json"
        save_json(path, subset)
        baseline_path = parent / f"test_{site}_predictions.json"
        baseline_pred = _read(baseline_path)
        baseline = evaluate_coco(subset, baseline_pred)
        recorded = metrics["tests"][site]
        if any(
            not math.isclose(baseline[k], recorded[k], rel_tol=0, abs_tol=1e-10) for k in baseline
        ):
            raise ValueError(
                f"Parent prediction metrics do not match receipt-bound metrics for {site}"
            )
        tick = time.monotonic()
        pred_path = out / f"test_{site}_predictions.json"
        preds = _native_predictions(
            "wavelet", selected["config"], path, images, pred_path, "cpu", binary
        )
        tests[site] = {
            "baseline": baseline,
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "selected": evaluate_coco(subset, preds),
            "predictions_sha256": sha256_file(pred_path),
            "annotations_sha256": sha256_file(path),
            "seconds": time.monotonic() - tick,
        }
    result = {
        "selection_sha256": selection_hash,
        "tests": tests,
        "seconds": time.monotonic() - started,
        "interpretation": "Separate source-only extension; holdouts did not select the configuration.",
    }
    save_json(out / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "ann", "images", "manifest", "binary", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    run(args.parent, args.ann, args.images, args.manifest, args.binary, args.out, args.threads)


if __name__ == "__main__":
    main()
