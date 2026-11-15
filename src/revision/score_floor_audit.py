"""Compare score floors using a frozen checkpoint and source validation only."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import evaluation, protocol
from .ultralytics_models import _predict, stage_split


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def source_validation_inputs(run: Path, ann: Path, images: Path, manifest: Path) -> dict:
    """Verify the frozen run and reconstruct only its source validation subset."""
    job, metrics, completion = (
        _load(run / name) for name in ("job.json", "metrics.json", "completion.json")
    )
    if completion.get("state") != "succeeded" or completion.get("exit_code") != 0:
        raise ValueError("run must have successfully completed")
    if completion.get("job_id") != job.get("job_id") or metrics.get("job_id") != job.get("job_id"):
        raise ValueError("run identity mismatch")
    if completion.get("metrics_sha256") != protocol.sha256_file(run / "metrics.json"):
        raise ValueError("metrics checksum mismatch")
    if job.get("method") not in ("yolo", "rtdetr"):
        raise ValueError("only YOLO and RT-DETR are supported")
    if (
        job["method"] == "yolo"
        and metrics.get("evaluation_recipe_version") != "yolo-exact-letterbox-v1"
    ):
        raise ValueError("YOLO checkpoint must have corrected prediction artifacts")
    manifest_hash, ann_hash = protocol.sha256_file(manifest), protocol.sha256_file(ann)
    if manifest_hash != metrics.get("manifest_sha256") or manifest_hash != completion.get(
        "manifest_sha256"
    ):
        raise ValueError("manifest checksum mismatch")
    if ann_hash != metrics.get("annotation_sha256"):
        raise ValueError("annotation checksum mismatch")
    checkpoint = run / "fit/weights/best.pt"
    if protocol.sha256_file(checkpoint) != metrics.get("best_sha256"):
        raise ValueError("checkpoint checksum mismatch")
    # Reconstructing with an empty target list prevents any test subset access.
    source_job = {
        **job,
        "ann": str(ann),
        "images": str(images),
        "manifest": str(manifest),
        "eval_targets": [],
    }
    prepared = protocol.prepare_job(source_job)
    if prepared["tests"]:
        raise ValueError("source-only audit cannot contain test subsets")
    gt = prepared["validation"]
    if gt != _load(run / "validation_gt.json"):
        raise ValueError("source validation does not match the frozen run")
    records = {record["id"]: record for record in _load(manifest)["images"]}
    for image in gt["images"]:
        record = records.get(image["id"])
        path = (images / image["file_name"]).resolve()
        if (
            not record
            or record["site"] != job["source"]
            or not path.is_relative_to(images.resolve())
        ):
            raise ValueError("source validation image is not in the source manifest")
        if protocol.sha256_file(path) != record["sha256"]:
            raise ValueError("source validation image checksum mismatch")
    reference_path = run / "validation_raw.json"
    return {
        "job": job,
        "metrics": metrics,
        "gt": gt,
        "checkpoint": checkpoint,
        "reference_path": reference_path,
        "reference": _load(reference_path),
        "manifest_sha256": manifest_hash,
        "annotation_sha256": ann_hash,
    }


def run_audit(
    run: Path,
    ann: Path,
    images: Path,
    manifest: Path,
    output: Path,
    low_floor: float = 1e-5,
    threads: int = 4,
) -> dict:
    """Reinfer source validation once and compare floors on identical predictions."""
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import RTDETR, YOLO

    run, output = run.resolve(), output.resolve()
    if run == output or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("output must be separate from the frozen run")
    if (output / "audit.json").exists() or (output / "low_floor_raw.json").exists():
        raise FileExistsError("audit output already exists")
    if threads < 1:
        raise ValueError("threads must be positive")
    source = source_validation_inputs(run, ann.resolve(), images.resolve(), manifest.resolve())
    job, metrics, gt = source["job"], source["metrics"], source["gt"]
    floor = float(metrics["config"]["postprocessing"]["score_floor"])
    if not 0 < low_floor < floor:
        raise ValueError("low floor must be positive and below the run score floor")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    directory = stage_split(gt, images.resolve(), output / "data/val")
    model = (YOLO if job["method"] == "yolo" else RTDETR)(str(source["checkpoint"]))
    start = time.perf_counter()
    raw = _predict(
        model,
        directory,
        gt,
        floor=low_floor,
        device="cpu",
        imgsz=int(metrics["config"]["training"]["imgsz"]),
        method=job["method"],
    )
    seconds = time.perf_counter() - start
    filtered = [p for p in raw if p["score"] >= floor]
    by_image = defaultdict(list)
    for prediction in raw:
        by_image[prediction["image_id"]].append(prediction)
    counts = Counter(p["image_id"] for p in raw)
    added_top100 = sum(
        sum(
            p["score"] < floor
            for p in sorted(by_image[image["id"]], key=lambda x: -x["score"])[:100]
        )
        for image in gt["images"]
    )
    scores = np.asarray([p["score"] for p in raw])
    raw_path = output / "low_floor_raw.json"
    protocol.save_json(raw_path, raw)
    result = {
        "scope": "source-validation-only",
        "source": job["source"],
        "method": job["method"],
        "seed": job["seed"],
        "parent_run": str(run),
        "checkpoint_sha256": protocol.sha256_file(source["checkpoint"]),
        "manifest_sha256": source["manifest_sha256"],
        "annotation_sha256": source["annotation_sha256"],
        "source_validation_gt_sha256": protocol.sha256_file(run / "validation_gt.json"),
        "original_raw_sha256": protocol.sha256_file(source["reference_path"]),
        "low_raw_path": str(raw_path),
        "low_raw_sha256": protocol.sha256_file(raw_path),
        "image_count": len(gt["images"]),
        "image_hash_mismatches": 0,
        "evaluation_device": "cpu",
        "torch_threads": threads,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "inference_seconds": seconds,
        "low_floor": low_floor,
        "reference_floor": floor,
        "predictor_max_detections": 300,
        "coco_max_detections": 100,
        "original_reference_metrics": evaluation.evaluate_coco(gt, source["reference"]),
        "same_cpu_reference_metrics": evaluation.evaluate_coco(gt, filtered),
        "same_cpu_low_floor_metrics": evaluation.evaluate_coco(gt, raw),
        "added_predictions": len(raw) - len(filtered),
        "added_predictions_in_top100": added_top100,
        "images_above_100_predictions": sum(counts[i["id"]] > 100 for i in gt["images"]),
        "images_at_300_predictions": sum(counts[i["id"]] >= 300 for i in gt["images"]),
        "score_percentiles_0_1_10_50_90_99_100": np.percentile(
            scores, [0, 1, 10, 50, 90, 99, 100]
        ).tolist()
        if len(scores)
        else [],
    }
    protocol.save_json(output / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--low-floor", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run_audit(
        args.run_dir,
        args.ann,
        args.images,
        args.manifest,
        args.output,
        args.low_floor,
        args.threads,
    )
    print(
        json.dumps(
            {
                "audit": str(args.output / "audit.json"),
                "source": result["source"],
                "method": result["method"],
            }
        )
    )


if __name__ == "__main__":
    main()
