"""Compare a frozen YOLO CPU evaluation with exact-checkpoint GPU inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import evaluation, protocol, ultralytics_models


def _load(path: Path):
    return json.loads(path.read_text())


def _iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    a_end, b_end = a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:]
    intersection = np.maximum(
        0,
        np.minimum(a_end[:, None, :], b_end[None, :, :])
        - np.maximum(a[:, None, :2], b[None, :, :2]),
    ).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prediction_comparison(cpu: list[dict], gpu: list[dict]) -> dict:
    """Match per-image boxes by maximum IoU and summarize paired differences."""
    by_cpu, by_gpu = defaultdict(list), defaultdict(list)
    for p in cpu:
        by_cpu[(p["image_id"], p["category_id"])].append(p)
    for p in gpu:
        by_gpu[(p["image_id"], p["category_id"])].append(p)
    diffs, score_diffs, ious = [], [], []
    for key in by_cpu.keys() & by_gpu.keys():
        x, y = by_cpu[key], by_gpu[key]
        matrix = _iou([p["bbox"] for p in x], [p["bbox"] for p in y])
        left, right = linear_sum_assignment(-matrix)
        for i, j in zip(left, right, strict=True):
            if matrix[i, j] < 0.5:
                continue
            diffs.append(np.asarray(y[j]["bbox"]) - np.asarray(x[i]["bbox"]))
            score_diffs.append(y[j]["score"] - x[i]["score"])
            ious.append(matrix[i, j])
    absolute = np.abs(np.asarray(diffs)).reshape(-1, 4)
    score_absolute = np.abs(np.asarray(score_diffs))
    return {
        "cpu_count": len(cpu),
        "gpu_count": len(gpu),
        "matched_iou_at_least_0_5": len(ious),
        "unmatched_cpu": len(cpu) - len(ious),
        "unmatched_gpu": len(gpu) - len(ious),
        "absolute_xywh_delta_median_p95_max": np.percentile(
            absolute, [50, 95, 100], axis=0
        ).tolist()
        if len(absolute)
        else [],
        "absolute_score_delta_median_p95_max": np.percentile(score_absolute, [50, 95, 100]).tolist()
        if len(score_absolute)
        else [],
        "match_iou_min_median": [float(min(ious)), float(np.median(ious))] if ious else [],
    }


def verify_cpu_metrics(expected: dict, actual: dict) -> None:
    for key in ("ap", "ap50", "ap75", "ar100", "predictions", "preds_per_image"):
        if (
            key not in expected
            or key not in actual
            or abs(float(expected[key]) - float(actual[key])) > 1e-12
        ):
            raise ValueError(f"CPU primary metric replay mismatch: {key}")


def _verify_idle_gpu(uuid: str) -> None:
    data = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in data.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].lower() == uuid.lower() and parts[1] != str(os.getpid()):
            raise RuntimeError(f"assigned GPU is in use by process {parts[1]}")


def run_study(run: Path, out: Path, gpu_uuid: str, code_commit: str) -> dict:
    """Replay fixed target sets on a matching GPU without modifying primary artifacts."""
    run, out = run.resolve(strict=True), out.resolve()
    if run == out or run.is_relative_to(out) or out.is_relative_to(run):
        raise ValueError("study output must be separate from primary attempt")
    if out.exists() and any(out.iterdir()):
        raise ValueError("study output must be empty")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or not gpu_uuid.startswith("GPU-"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be the assigned single GPU UUID")
    _verify_idle_gpu(gpu_uuid)
    job, primary, completion = (
        _load(run / name) for name in ("job.json", "metrics.json", "completion.json")
    )
    if completion.get("state") != "succeeded" or completion.get("exit_code") != 0:
        raise ValueError("primary evaluation did not succeed")
    if completion.get("metrics_sha256") != protocol.sha256_file(run / "metrics.json"):
        raise ValueError("primary metrics digest mismatch")
    if (
        job.get("method") != "yolo"
        or primary.get("evaluation_recipe_version") != "yolo-exact-letterbox-v1"
    ):
        raise ValueError("primary is not a corrected YOLO evaluation")
    if primary.get("evaluation_device") != "cpu" or set(primary["tests"]) != {
        "DAAN",
        "DARC",
        "MANV",
    }:
        raise ValueError("primary CPU evaluation must contain the three fixed holdouts")
    if set(job.get("eval_targets", [])) != set(primary["tests"]):
        raise ValueError("primary job target inventory mismatch")
    manifest = Path(job["manifest"])
    if protocol.sha256_file(manifest) != primary["manifest_sha256"]:
        raise ValueError("protocol manifest checksum mismatch")
    ann = Path(job["ann"])
    if protocol.sha256_file(ann) != primary["annotation_sha256"]:
        raise ValueError("annotation checksum mismatch")
    checkpoint = run / "fit/weights/best.pt"
    checkpoint_sha = protocol.sha256_file(checkpoint)
    if (
        checkpoint_sha != primary["best_sha256"]
        or checkpoint_sha != primary["training_parent"]["checkpoint_sha256"]
    ):
        raise ValueError("selected checkpoint checksum mismatch")
    import torch
    import torch.version
    import ultralytics
    from ultralytics import YOLO

    if torch.cuda.device_count() != 1:
        raise ValueError("must expose exactly one GPU")
    props = torch.cuda.get_device_properties(0)
    if ("GPU-" + str(props.uuid)).lower() != gpu_uuid.lower():
        raise ValueError("visible GPU UUID mismatch")
    original = primary["training_parent"]["runtime_provenance"]
    if not original or props.name != original["gpu"]["model"]:
        raise ValueError("GPU model differs from original fit")
    expected_flags = {
        "cudnn_allow_tf32": True,
        "matmul_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
    }
    if any(original["torch"].get(key) != value for key, value in expected_flags.items()):
        raise ValueError("original backend flags differ from study settings")
    if primary.get("resolved_train_args", {}).get("half") is not False:
        raise ValueError("original model arguments did not record float32 inference")
    original_threads = original["thread_environment"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        if os.environ.get(name) != original_threads.get(name):
            raise ValueError(f"{name} differs from original fit")
    torch.set_num_threads(int(original_threads["OMP_NUM_THREADS"]))
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    if (
        original["torch"]["version"] != torch.__version__
        or original["torch"]["cuda_version"] != torch.version.cuda
    ):
        raise ValueError("Torch or CUDA version differs from original fit")
    out.mkdir(parents=True)
    model = YOLO(str(checkpoint))
    threshold = float(primary["selected_threshold"])
    floor = float(primary["config"]["postprocessing"]["score_floor"])
    imgsz = int(primary["config"]["training"]["imgsz"])
    reports = {}
    for target in ("DAAN", "DARC", "MANV"):
        old = primary["tests"][target]
        gt_path, cpu_path = Path(old["ground_truth_path"]), Path(old["raw_path"])
        if not gt_path.resolve().is_relative_to(run) or not cpu_path.resolve().is_relative_to(run):
            raise ValueError("primary artifact path is outside attempt")
        if protocol.sha256_file(gt_path) != old["ground_truth_sha256"]:
            raise ValueError("ground truth digest mismatch")
        gt, cpu = _load(gt_path), _load(cpu_path)
        cpu_metrics = evaluation.evaluate_coco(gt, cpu)
        verify_cpu_metrics(old["raw_metrics"], cpu_metrics)
        image_dir = run / "data/test" / target / "images"
        image_hashes = {}
        for image in gt["images"]:
            path = image_dir / f"{image['id']}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            image_hashes[str(image["id"])] = protocol.sha256_file(path)
        if len(list(image_dir.glob("*.png"))) != len(gt["images"]):
            raise ValueError("staged input images differ from GT inventory")
        gpu = ultralytics_models._predict(
            model, image_dir, gt, floor=floor, device="", imgsz=imgsz, method="yolo"
        )
        if model.predictor is None:
            raise RuntimeError("predictor was not initialized")
        if model.predictor.args.half:
            raise ValueError("inference did not run in float32")
        raw_path = out / f"{target}_gpu_raw.json"
        protocol.save_json(raw_path, gpu)
        gpu_metrics = evaluation.evaluate_coco(gt, gpu)
        cpu_operating = [p for p in cpu if p["score"] >= threshold]
        gpu_operating = [p for p in gpu if p["score"] >= threshold]
        reports[target] = {
            "images": len(gt["images"]),
            "ground_truth_sha256": protocol.sha256_file(gt_path),
            "cpu_raw_sha256": protocol.sha256_file(cpu_path),
            "gpu_raw_path": str(raw_path),
            "gpu_raw_sha256": protocol.sha256_file(raw_path),
            "input_image_sha256": image_hashes,
            "cpu_raw_metrics": cpu_metrics,
            "gpu_raw_metrics": gpu_metrics,
            "gpu_minus_cpu_raw_metrics": {
                k: gpu_metrics[k] - cpu_metrics[k] for k in ("ap", "ap50", "ap75", "ar100")
            },
            "cpu_operating_metrics": evaluation.evaluate_coco(gt, cpu_operating),
            "gpu_operating_metrics": evaluation.evaluate_coco(gt, gpu_operating),
            "prediction_comparison": prediction_comparison(cpu, gpu),
            "images_with_count_difference": sum(
                a != b
                for a, b in zip(
                    (Counter(p["image_id"] for p in cpu)[i["id"]] for i in gt["images"]),
                    (Counter(p["image_id"] for p in gpu)[i["id"]] for i in gt["images"]),
                    strict=True,
                )
            ),
        }
    if model.predictor is None:
        raise RuntimeError("predictor was not initialized")
    package = Path(__file__).parent
    result = {
        "scope": "fixed-checkpoint backend sensitivity; no selection",
        "job_id": job["job_id"],
        "primary_run": str(run),
        "primary_metrics_sha256": protocol.sha256_file(run / "metrics.json"),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": primary["manifest_sha256"],
        "annotation_sha256": primary["annotation_sha256"],
        "evaluation_recipe_version": primary["evaluation_recipe_version"],
        "code_commit": code_commit,
        "code_sha256": {
            name: protocol.sha256_file(package / name)
            for name in (
                "yolo_backend_study.py",
                "ultralytics_models.py",
                "evaluation.py",
                "protocol.py",
            )
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "gpu_uuid": gpu_uuid,
        "gpu_model": props.name,
        "torch_threads": torch.get_num_threads(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32": not model.predictor.args.half,
        "score_floor": floor,
        "selected_threshold": threshold,
        "imgsz": imgsz,
        "batch": 1,
        "targets": reports,
    }
    protocol.save_json(out / "study.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result = run_study(args.run_dir, args.output, args.gpu_uuid, args.code_commit)
    print(json.dumps({"study": str(args.output / "study.json"), "job_id": result["job_id"]}))


if __name__ == "__main__":
    main()
