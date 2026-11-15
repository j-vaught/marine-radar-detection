"""Time the four selected DAAN seed-zero checkpoints on shared validation frames."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from . import protocol

METHODS = ("faster_rcnn", "yolo", "unet", "rtdetr")
FRAME_COUNT = 32
TIMED_PASSES = 3


@dataclass
class SelectedRun:
    checkpoint: Path
    config: dict
    initialization: str
    pixel_threshold: float | None
    provenance: dict


def read_run(runs: Path, method: str, manifest_hash: str) -> SelectedRun:
    attempt = 2 if method == "yolo" else 1
    job_id = f"core-{method}-DAAN-s0"
    run = runs / job_id / str(attempt)
    metrics_path = run / "metrics.json"
    completion_path = run / "completion.json"
    completion = json.loads(completion_path.read_text())
    metrics_hash = protocol.sha256_file(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    if (
        completion["state"] != "succeeded"
        or completion["attempt"] != attempt
        or completion["job_id"] != job_id
        or completion["metrics_sha256"] != metrics_hash
        or completion["manifest_sha256"] != manifest_hash
        or metrics["manifest_sha256"] != manifest_hash
        or (metrics["job_id"], metrics["method"], metrics["source"], metrics["seed"])
        != (job_id, method, "DAAN", 0)
        or metrics["stage"] != "core"
        or metrics["condition"] != "buffered"
    ):
        raise ValueError(f"Invalid selected completion or metrics: {run}")
    torch_method = method in ("faster_rcnn", "unet")
    checkpoint = run / ("best.pt" if torch_method else "fit/weights/best.pt")
    checkpoint_hash = protocol.sha256_file(checkpoint)
    if torch_method:
        config_path = run / "resolved_config.json"
        config_hash = protocol.sha256_file(config_path)
        if config_hash != metrics["resolved_config_sha256"]:
            raise ValueError(f"Resolved configuration digest mismatch: {run}")
        config = json.loads(config_path.read_text())
        expected_hash = metrics["best_weights_sha256"]
    else:
        config, config_hash = metrics, None
        expected_hash = metrics["best_sha256"]
    if checkpoint_hash != expected_hash:
        raise ValueError(f"Selected checkpoint digest mismatch: {run}")
    if method == "yolo" and (
        metrics.get("evaluation_recipe_version") != "yolo-exact-letterbox-v1"
        or metrics.get("training_parent", {}).get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("Corrected YOLO evaluation must attest the original training checkpoint")
    pixel = metrics.get("pixel_threshold")
    if method == "unet" and (pixel is None or not 0 < pixel < 1):
        raise ValueError("U-Net must have a frozen source validation pixel threshold")
    return SelectedRun(
        checkpoint,
        config["config"],
        config["initialization"],
        pixel,
        {
            "job_id": job_id,
            "attempt": attempt,
            "training_or_repair_code_commit": completion["code_commit"],
            "checkpoint_sha256": checkpoint_hash,
            "metrics_sha256": metrics_hash,
            "completion_sha256": protocol.sha256_file(completion_path),
            "resolved_config_sha256": config_hash,
            "initialization": config["initialization"],
        },
    )


def validation_sample(manifest: dict) -> list[dict]:
    ids = manifest["sites"]["DAAN"]["partitions"]["validation"][:FRAME_COUNT]
    records = {row["id"]: row for row in manifest["images"]}
    selection = [records[image_id] for image_id in ids]
    ordered_frames = [row["frame"] for row in selection]
    if (
        len(set(ids)) != FRAME_COUNT
        or any(row["site"] != "DAAN" for row in selection)
        or ordered_frames != sorted(set(ordered_frames))
    ):
        raise ValueError("Expected 32 distinct DAAN validation frames in manifest order")
    return selection


def stage_sample(manifest: dict, annotation: Path, images: Path, staging: Path):
    from .ultralytics_models import stage_split

    if protocol.sha256_file(annotation) != manifest["annotation_sha256"]:
        raise ValueError("Image metadata does not match the frozen annotation digest")
    metadata = {row["id"]: row for row in json.loads(annotation.read_text())["images"]}
    sample = validation_sample(manifest)
    coco = {"images": [], "categories": [{"id": 1}], "annotations": []}
    for row in sample:
        path = protocol._safe_image_path(images, row["file_name"])
        if protocol.sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source PNG digest mismatch: {path}")
        info = metadata[row["id"]]
        if info["file_name"] != row["file_name"]:
            raise ValueError(f"Image metadata filename mismatch: {path}")
        coco["images"].append(info)
    if len({(info["height"], info["width"]) for info in coco["images"]}) != 1:
        raise ValueError("Validation frames must share a native raster size")
    directory = stage_split(coco, images, staging)
    # The native directory loader visits names lexicographically. Match it for Torch.
    coco["images"].sort(key=lambda info: f"{info['id']}.png")
    frames = [
        {**row, "staged_png_sha256": protocol.sha256_file(directory / f"{row['id']}.png")}
        for row in sample
    ]
    return directory, coco, frames


def make_predictor(method: str, selected: SelectedRun, directory: Path, coco: dict, device):
    import torch

    config, post = selected.config, selected.config["postprocessing"]
    if method in ("yolo", "rtdetr"):
        from ultralytics import RTDETR, YOLO

        from .ultralytics_models import _predict

        model = (YOLO if method == "yolo" else RTDETR)(str(selected.checkpoint))
        network = model.model
        if not isinstance(network, torch.nn.Module):
            raise TypeError(f"{method} checkpoint did not load a PyTorch module")
        network.to(device)
        floor = float(post["score_floor"])
        if floor != 0.001:
            raise ValueError(f"Unexpected {method} floor: {floor}")
        imgsz = int(config["training"]["imgsz"])
        expected_names = [f"{image['id']}.png" for image in coco["images"]]

        def predict():
            raw = _predict(
                model, directory, coco, floor=floor, device=str(device), imgsz=imgsz, method=method
            )
            predictor = model.predictor
            if (
                predictor.args.batch != 1
                or predictor.dataset.bs != 1
                or predictor.model.fp16
                or predictor.seen != len(expected_names)
                or [Path(path).name for path in predictor.dataset.files] != expected_names
            ):
                raise RuntimeError(f"{method} did not visit the shared PNGs with batch size one")
            return raw

        return predict, {
            "imgsz": imgsz,
            "score_floor": floor,
            "precision": "float32",
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in network.parameters()}
            ),
        }

    import cv2
    import numpy as np

    from .torch_models import _append_predictions

    preprocess: dict[str, object]
    if method == "faster_rcnn":
        from ..faster_rcnn.model import build_model

        pretrained = selected.initialization == "pretrained"
        if pretrained:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            url = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url
            cached = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
            if not cached.is_file():
                raise FileNotFoundError(
                    f"Native Faster R-CNN bootstrap weights must be cached: {cached}"
                )
        model = build_model(
            config,
            pretrained=pretrained,
            box_score_thresh=0.0,
            box_detections_per_img=int(config["model"].get("eval_box_detections_per_img", 300)),
        )
        preprocess = {
            "transform_min_size": list(model.transform.min_size),
            "transform_max_size": model.transform.max_size,
            "score_floor": 0.0,
            "precision": "float32",
        }

        def predict_image(image):
            tensor = torch.from_numpy(np.stack([image] * 3)).float() / 255
            with torch.no_grad():
                result = model([tensor.to(device)])[0]
            return result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()

    else:
        from ..unet.model import UNet, predict_prob_map
        from ..unet.utils import binary_mask_to_bboxes_with_scores

        model = UNet(
            **{key: config["model"][key] for key in ("in_channels", "num_classes", "base_channels")}
        )
        size = tuple(config["training"]["input_size"])
        pixel = selected.pixel_threshold
        if pixel is None:
            raise ValueError("Missing source-selected U-Net threshold")
        first_image = coco["images"][0]
        yy, xx = np.ogrid[: first_image["height"], : first_image["width"]]
        roi = (xx - 531) ** 2 + (yy - 515) ** 2 <= 466**2
        component_options = {
            key: int(post[key]) for key in ("min_area", "max_area", "dilation", "box_padding")
        }
        preprocess = {
            "input_size": size,
            "pixel_threshold": pixel,
            "score_floor": 0.0,
            "precision": "native CUDA float16 autocast",
            **component_options,
        }

        def predict_image(image, threshold: float = pixel):
            prob = predict_prob_map(model, image, device=str(device), input_size=size)
            prob[~roi] = 0
            return binary_mask_to_bboxes_with_scores(
                (prob >= threshold).astype(np.uint8),
                prob,
                min_area=component_options["min_area"],
                max_area=component_options["max_area"],
                dilation=component_options["dilation"],
                box_padding=component_options["box_padding"],
            )

    model.load_state_dict(torch.load(selected.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    preprocess["parameter_dtypes"] = sorted(
        {str(parameter.dtype) for parameter in model.parameters()}
    )

    def predict():
        raw = []
        for info in coco["images"]:
            path = directory / f"{info['id']}.png"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (info["height"], info["width"]):
                raise ValueError(f"Cannot decode staged PNG: {path}")
            boxes, scores = predict_image(image)
            _append_predictions(raw, info, boxes, scores)
        return raw

    return predict, preprocess


def check_predictions(raw: list[dict], image_ids: list[int]) -> dict:
    counts = dict.fromkeys(image_ids, 0)
    degenerate = 0
    for prediction in raw:
        image_id = prediction["image_id"]
        box, score = prediction["bbox"], prediction["score"]
        if (
            image_id not in counts
            or prediction["category_id"] != 1
            or len(box) != 4
            or not all(math.isfinite(value) for value in (*box, score))
            or min(box[2:]) < 0
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid raw prediction: {prediction}")
        counts[image_id] += 1
        degenerate += int(min(box[2:]) == 0)
    return {
        "boxes": len(raw),
        "boxes_per_image": list(counts.values()),
        "degenerate_boxes": degenerate,
    }


def measure_pass(
    predict: Callable[[], list[dict]], synchronize: Callable[[], None], image_ids: list[int]
) -> dict:
    synchronize()
    started = time.perf_counter()
    raw = predict()
    synchronize()
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "ms_per_frame": elapsed * 1000 / len(image_ids),
        "fps": len(image_ids) / elapsed,
        **check_predictions(raw, image_ids),
    }


def gpu_snapshot(uuid: str, expected_pids: set[int]) -> dict:
    def query(flag: str) -> str:
        return subprocess.check_output(
            ["nvidia-smi", flag, "--format=csv,noheader,nounits"], text=True, timeout=20
        ).strip()

    inventory = {}
    for row in query(
        "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu"
    ).splitlines():
        gpu_uuid, *fields = (field.strip() for field in row.split(","))
        inventory[gpu_uuid] = fields
    if uuid not in inventory:
        raise RuntimeError(f"Assigned GPU is missing from NVIDIA inventory: {uuid}")
    name, driver, memory, utilization = inventory[uuid]
    pids = {
        int(parts[1])
        for line in query("--query-compute-apps=gpu_uuid,pid").splitlines()
        if (parts := [field.strip() for field in line.split(",")])[0] == uuid
    }
    if pids != expected_pids or "RTX 6000 Ada Generation" not in name:
        raise RuntimeError(
            f"Assigned Ada GPU has unexpected compute processes: {uuid}, {sorted(pids)}"
        )
    return {
        "uuid": uuid,
        "name": name,
        "driver": driver,
        "memory_used_mib": int(memory),
        "utilization_percent": int(utilization),
    }


def run(args: argparse.Namespace) -> None:
    import cv2
    import torch
    from torch.backends import cuda, cudnn
    from torch.version import cuda as torch_cuda

    from .fleet import cuda_uuid

    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly the assigned GPU UUID")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_hash = protocol.sha256_file(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    selected = {method: read_run(args.runs, method, manifest_hash) for method in METHODS}
    hardware = gpu_snapshot(args.gpu_uuid, set())
    if torch.cuda.device_count() != 1 or cuda_uuid() != args.gpu_uuid:
        raise RuntimeError("Logical CUDA device zero did not match the assigned UUID")
    device = torch.device("cuda:0")
    torch.set_num_interop_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def configure_threads():
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)

    def execution_settings():
        settings = {
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "cudnn_version": cudnn.version(),
            "cudnn_enabled": cudnn.enabled,
            "cudnn_benchmark": cudnn.benchmark,
            "cudnn_deterministic": cudnn.deterministic,
            "cudnn_allow_tf32": cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if settings["torch_threads"] != args.threads or settings["opencv_threads"] != args.threads:
            raise RuntimeError(f"Native runtime thread settings changed: {settings}")
        return settings

    def synchronize():
        torch.cuda.synchronize(device)

    configure_threads()

    with tempfile.TemporaryDirectory(prefix="timing-png-", dir=args.output.parent) as folder:
        started = time.perf_counter()
        directory, coco, frames = stage_sample(manifest, args.ann, args.images, Path(folder))
        staging_seconds = time.perf_counter() - started
        image_ids = [info["id"] for info in coco["images"]]
        timing = {}
        for method, model_run in selected.items():
            before = gpu_snapshot(args.gpu_uuid, {os.getpid()})
            synchronize()
            started = time.perf_counter()
            predict, preprocess = make_predictor(method, model_run, directory, coco, device)
            synchronize()
            load_seconds = time.perf_counter() - started
            configure_threads()
            torch.cuda.reset_peak_memory_stats(device)
            warmup = measure_pass(predict, synchronize, image_ids)
            warmup_peak = torch.cuda.max_memory_allocated(device)
            # The framework may change OpenCV settings during import or lazy predictor setup.
            configure_threads()
            runtime = execution_settings()
            torch.cuda.reset_peak_memory_stats(device)
            passes = [measure_pass(predict, synchronize, image_ids) for _ in range(TIMED_PASSES)]
            timed_peak = torch.cuda.max_memory_allocated(device)
            if execution_settings() != runtime:
                raise RuntimeError(f"Native runtime settings changed during timed passes: {method}")
            after = gpu_snapshot(args.gpu_uuid, {os.getpid()})
            timing[method] = {
                **model_run.provenance,
                "preprocessing_and_raw_floor": preprocess,
                "before_gpu": before,
                "after_gpu": after,
                "model_load_seconds": load_seconds,
                "runtime_settings": runtime,
                "warmup_pass": warmup,
                "timed_passes": passes,
                "warmup_peak_torch_allocated_bytes": warmup_peak,
                "timed_peak_torch_allocated_bytes": timed_peak,
                "median_ms_per_frame": statistics.median(p["ms_per_frame"] for p in passes),
                "median_fps": statistics.median(p["fps"] for p in passes),
                "counts_match_across_passes": all(
                    row["boxes_per_image"] == warmup["boxes_per_image"] for row in passes
                ),
            }
            print(f"{method}: {timing[method]['median_ms_per_frame']:.3f} ms/frame", flush=True)
            del predict
            gc.collect()
            torch.cuda.empty_cache()
        protocol.save_json(
            args.output,
            {
                "source": "DAAN",
                "partition": "validation",
                "seed": 0,
                "batch_size": 1,
                "manifest_sha256": manifest_hash,
                "annotation_sha256": manifest["annotation_sha256"],
                "host": platform.node(),
                "gpu": hardware,
                "torch_cuda": torch_cuda,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "cpu_threads": args.threads,
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
                },
                "environment": {
                    package: version(package)
                    for package in ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")
                },
                "python": sys.version.split()[0],
                "code_sha256": {
                    name: protocol.sha256_file(Path(__file__).with_name(name))
                    for name in ("inference_timing.py", "torch_models.py", "ultralytics_models.py")
                },
                "source_frames": frames,
                "processing_image_ids": image_ids,
                "staging_seconds": staging_seconds,
                "method_order": list(selected),
                "timings": timing,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "manifest", "ann", "images", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
