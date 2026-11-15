"""Re-evaluate a verified YOLO fit with exact letterbox coordinates."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import evaluation, protocol
from .ultralytics_models import _predict, stage_split


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_parent(job: dict) -> tuple[Path, dict, dict]:
    """Verify run identity and its completed artifacts without modifying them."""
    parent = Path(job["evaluation_from"]).resolve(strict=True)
    out = Path(job["out_dir"]).resolve()
    if out == parent or out.is_relative_to(parent) or parent.is_relative_to(out):
        raise ValueError("repair output must be separate from the parent attempt")
    original_job = _read_json(parent / "job.json")
    receipt = _read_json(parent / "receipt.json")
    completion = _read_json(parent / "completion.json")
    metrics_path = parent / "metrics.json"
    metrics = _read_json(metrics_path)
    if completion.get("state") != "succeeded" or completion.get("exit_code") != 0:
        raise ValueError("parent has not successfully completed")
    for key in ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id"):
        if receipt.get(key) != completion.get(key) or receipt.get(key) is None:
            raise ValueError(f"parent receipt/completion mismatch: {key}")
    if (
        receipt["job_id"] != original_job["job_id"]
        or receipt["nonce"] != original_job["attempt_nonce"]
    ):
        raise ValueError("parent receipt/job identity mismatch")
    if completion.get("metrics_sha256") != protocol.sha256_file(metrics_path):
        raise ValueError("parent metrics digest mismatch")
    if completion.get("checkpoint_storage", {}).get("status") == "archive_failed":
        raise ValueError("parent checkpoint archival failed")
    if job["method"] != "yolo":
        raise ValueError("evaluation-only repair is specific to YOLO")
    for key in ("job_id", "method", "source", "seed", "condition", "stage", "initialization"):
        if original_job.get(key) != job.get(key) or metrics.get(key) != job.get(key):
            raise ValueError(f"parent identity mismatch: {key}")
    if original_job.get("eval_targets", []) != job.get("eval_targets", []):
        raise ValueError("parent evaluation targets differ")
    if original_job.get("train_limit") != job.get("train_limit") or original_job.get(
        "overfit", False
    ) != job.get("overfit", False):
        raise ValueError("parent split setup differs")
    manifest_hash = protocol.sha256_file(Path(job["manifest"]))
    annotation_hash = protocol.sha256_file(Path(job["ann"]))
    if (
        manifest_hash != metrics.get("manifest_sha256")
        or manifest_hash != completion.get("manifest_sha256")
        or manifest_hash != job.get("protocol_sha256")
    ):
        raise ValueError("manifest digest mismatch")
    if annotation_hash != metrics.get("annotation_sha256"):
        raise ValueError("annotation digest mismatch")
    best = Path(metrics["best_checkpoint"]).resolve(strict=True)
    if not best.is_relative_to(parent) or best != (parent / "fit/weights/best.pt").resolve():
        raise ValueError("parent best checkpoint outside expected location")
    if protocol.sha256_file(best) != metrics["best_sha256"]:
        raise ValueError("parent checkpoint digest mismatch")
    if job.get("config_overrides", {}) != original_job.get("config_overrides", {}):
        raise ValueError("parent configuration overrides differ")
    return parent, metrics, completion


def evaluate_from_parent(job: dict) -> dict:
    """Re-run source calibration and requested tests from the same frozen weights."""
    import ultralytics
    from ultralytics import YOLO

    parent, old, completion = verify_parent(job)
    out = Path(job["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / x).exists() for x in ("fit", "resume", "initial_state.pt", "metrics.json")):
        raise ValueError("repair output already contains training artifacts")
    prepared = protocol.prepare_job(job)
    val = prepared["validation"]
    if val != _read_json(parent / "validation_gt.json"):
        raise ValueError("reconstructed source validation differs from parent")
    # Copies prevent later changes to a repair attempt from mutating its parent.
    for folder in ("fit", "resume"):
        shutil.copytree(parent / folder, out / folder)
    shutil.copy2(parent / "initial_state.pt", out / "initial_state.pt")
    best, last = out / "fit/weights/best.pt", out / "fit/weights/last.pt"
    if (
        protocol.sha256_file(best) != old["best_sha256"]
        or protocol.sha256_file(last) != old["last_sha256"]
    ):
        raise ValueError("copied checkpoint digest mismatch")
    if protocol.sha256_file(out / "initial_state.pt") != old["initial_state_sha256"]:
        raise ValueError("copied initialization digest mismatch")
    config = old["config"]
    floor = float(config["postprocessing"]["score_floor"])
    thresholds = sorted(
        set(
            map(
                float,
                config["postprocessing"].get(
                    "score_threshold_grid", [config["postprocessing"]["score_threshold"]]
                ),
            )
        )
    )
    if not thresholds or min(thresholds) < floor:
        raise ValueError("invalid score threshold grid")
    if str(job["device"]) != "cpu":
        raise ValueError("evaluation-only repair requires CPU")
    model = YOLO(str(best))
    categories = sorted(prepared["train"]["categories"], key=lambda c: c["id"])
    val_path = out / "validation_gt.json"
    protocol.save_json(val_path, val)
    val_dir = stage_split(val, Path(prepared["image_dir"]), out / "data/val")
    t0 = time.perf_counter()
    val_raw = _predict(
        model,
        val_dir,
        val,
        floor=floor,
        device="cpu",
        imgsz=int(config["training"]["imgsz"]),
        method="yolo",
    )
    val_seconds = time.perf_counter() - t0
    protocol.save_json(out / "validation_raw.json", val_raw)
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        filtered = [p for p in val_raw if p["score"] >= threshold]
        protocol.save_json(out / f"validation_threshold_{threshold:.8g}.json", filtered)
        sweep.append({"threshold": threshold, "metrics": evaluation.evaluate_coco(val, filtered)})
    choice = max(sweep, key=lambda x: (x["metrics"]["ap50"], x["metrics"]["ap"]))
    results = {}
    for target in job.get("eval_targets", []):
        coco = prepared["tests"][target]
        parent_gt = parent / f"{target}_gt.json"
        if (
            coco != _read_json(parent_gt)
            or protocol.sha256_file(parent_gt) != old["tests"][target]["ground_truth_sha256"]
        ):
            raise ValueError(f"reconstructed test reference differs from parent: {target}")
        if sorted(c["id"] for c in coco["categories"]) != [c["id"] for c in categories]:
            raise ValueError("test categories differ from training categories")
        gt_path = out / f"{target}_gt.json"
        protocol.save_json(gt_path, coco)
        test_dir = stage_split(coco, Path(prepared["image_dir"]), out / "data/test" / target)
        started = time.perf_counter()
        raw = _predict(
            model,
            test_dir,
            coco,
            floor=floor,
            device="cpu",
            imgsz=int(config["training"]["imgsz"]),
            method="yolo",
        )
        elapsed = time.perf_counter() - started
        filtered = [p for p in raw if p["score"] >= choice["threshold"]]
        raw_path, op_path = out / f"{target}_raw.json", out / f"{target}_operating.json"
        protocol.save_json(raw_path, raw)
        protocol.save_json(op_path, filtered)
        results[target] = {
            "ground_truth_path": str(gt_path),
            "ground_truth_sha256": protocol.sha256_file(gt_path),
            "checkpoint_sha256": old["best_sha256"],
            "raw_metrics": evaluation.evaluate_coco(coco, raw),
            "operating_metrics": evaluation.evaluate_coco(coco, filtered),
            "inference_seconds": elapsed,
            "images": len(coco["images"]),
            "raw_path": str(raw_path),
            "operating_path": str(op_path),
            "raw_count": len(raw),
            "operating_count": len(filtered),
            "scores": [p["score"] for p in raw],
        }
    result = dict(old)
    result.update(
        {
            "job_id": job["job_id"],
            "job": job,
            "evaluation_recipe": "yolo-exact-letterbox-v1",
            "evaluation_recipe_version": "yolo-exact-letterbox-v1",
            "evaluation_code_commit": job.get("code_commit"),
            "evaluation_ultralytics_version": ultralytics.__version__,
            "evaluation_device": "cpu",
            "validation_inference_seconds": val_seconds,
            "training_parent": {
                "directory": str(parent),
                "receipt_sha256": protocol.sha256_file(parent / "receipt.json"),
                "completion_sha256": protocol.sha256_file(parent / "completion.json"),
                "metrics_sha256": protocol.sha256_file(parent / "metrics.json"),
                "checkpoint_sha256": old["best_sha256"],
                "code_commit": completion.get("code_commit"),
                "runtime_provenance": completion.get("runtime_provenance"),
            },
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "training_log": str(out / "fit/results.csv"),
            "initial_state_path": str(out / "initial_state.pt"),
            "resume_best_path": str(out / "resume/best.pt"),
            "resume_last_path": str(out / "resume/last.pt"),
            "resume_best_training_state": str(out / "resume/best_training_state.pt"),
            "resume_last_training_state": str(out / "resume/last_training_state.pt"),
            "validation_gt_path": str(val_path),
            "validation_gt_sha256": protocol.sha256_file(val_path),
            "validation_raw_path": str(out / "validation_raw.json"),
            "validation_raw_count": len(val_raw),
            "validation_scores": [p["score"] for p in val_raw],
            "validation_sweep": sweep,
            "selected_threshold": choice["threshold"],
            "validation": {
                "raw_metrics": evaluation.evaluate_coco(val, val_raw),
                "selected": choice,
            },
            "tests": results,
            "targets": results,
            "metrics_path": str(out / "metrics.json"),
        }
    )
    protocol.save_json(out / "metrics.json", result)
    return result
