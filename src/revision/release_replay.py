"""Replay a portable selected checkpoint through the frozen revision predictors.

Use a packaged test COCO file for --annotations. The output contains raw COCO
predictions, their source-selected operating subset, input digests, and optional
reference parity. Original checkpoint and image files are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from . import protocol

BOOTSTRAP_SHA256 = "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384"
METHODS = {"faster_rcnn", "unet", "yolo", "rtdetr"}
SCOPES = {"primary", "adjacency", "darc", "adaptation"}


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {label} SHA-256")
    return value


def _relative_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Artifact paths must be nonempty relative POSIX paths")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe relative path: {name}")
    return protocol._safe_image_path(root, name)


def read_run(path):
    path = Path(path)
    run = json.loads(path.read_text())
    if run.get("schema_version") != 1 or run.get("scope") not in SCOPES:
        raise ValueError("Unsupported release run schema or scope")
    if run.get("method") not in METHODS or run.get("initialization") not in {
        "pretrained",
        "scratch",
    }:
        raise ValueError("Unsupported method or initialization")
    if not isinstance(run.get("job_id"), str) or not run["job_id"]:
        raise ValueError("Missing job identifier")
    if not isinstance(run.get("seed"), int) or isinstance(run["seed"], bool):
        raise TypeError("Seed must be an integer")
    if not isinstance(run.get("config"), dict) or not isinstance(
        run.get("numerical_policy", {}), dict
    ):
        raise TypeError("Configuration and numerical policy must be objects")
    if run["scope"] in {"darc", "adaptation"}:
        if run["method"] != "faster_rcnn" or run["initialization"] != "pretrained":
            raise ValueError("DARC and adaptation require pretrained Faster R-CNN")
        if run.get("intensity") not in {"I0", "I1"}:
            raise ValueError("DARC intensity must be I0 or I1")
        if run["scope"] == "adaptation" and run["intensity"] != "I0":
            raise ValueError("The frozen adaptation protocol uses I0")
    if run["method"] == "unet":
        pixel = run.get("pixel_threshold")
        if not isinstance(pixel, (int, float)) or not 0 < pixel < 1:
            raise ValueError("U-Net requires its frozen source-selected pixel threshold")
    threshold = run.get("threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid source-selected operating threshold")
    checkpoint = _relative_file(path.parent, run.get("checkpoint"))
    expected = _digest(run.get("checkpoint_sha256"), "checkpoint")
    if protocol.sha256_file(checkpoint) != expected:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if run.get("tensor_state_sha256") is not None:
        _digest(run["tensor_state_sha256"], "tensor state")
    return run, checkpoint


def prepare_inputs(annotations, images, split_manifest=None, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("Image limit must be positive")
    coco = json.loads(Path(annotations).read_text())
    records = coco.get("images", [])
    if not records:
        raise ValueError("Replay requires at least one image")
    ids = [row["id"] for row in records]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
        raise ValueError("Image identifiers must be integers")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image identifiers")
    for row in records:
        if any(not isinstance(row[key], int) or row[key] <= 0 for key in ("width", "height")):
            raise ValueError("Invalid native image dimensions")
    records = sorted(records, key=lambda row: f"{row['id']}.png")[:limit]
    selected_ids = {row["id"] for row in records}
    all_ids = set(ids)
    if any(row["image_id"] not in all_ids for row in coco.get("annotations", [])):
        raise ValueError("Annotation references an unknown image")
    coco = {
        **coco,
        "images": records,
        "annotations": [
            row for row in coco.get("annotations", []) if row["image_id"] in selected_ids
        ],
    }
    expected = None
    if split_manifest is not None:
        manifest = json.loads(Path(split_manifest).read_text())
        expected = {row["id"]: row for row in manifest["images"]}
        if len(expected) != len(manifest["images"]):
            raise ValueError("Duplicate split-manifest image identifiers")
    observed = []
    for row in records:
        path = _relative_file(images, row["file_name"])
        digest = protocol.sha256_file(path)
        if expected is not None:
            match = expected.get(row["id"])
            if match is None or match["file_name"] != row["file_name"]:
                raise ValueError(f"Split inventory mismatch for image {row['id']}")
            if digest != _digest(match["sha256"], "image"):
                raise ValueError(f"Input image SHA-256 mismatch for image {row['id']}")
        observed.append(
            {
                "image_id": row["id"],
                "sha256": digest,
                "width": row["width"],
                "height": row["height"],
            }
        )
    return coco, observed


def tensor_digest(checkpoint, method):
    """Hash native saved tensors, before a framework converts their precision."""
    import torch

    from .darc_adaptation import _state_digest

    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=method not in {"yolo", "rtdetr"}
    )
    if method in {"yolo", "rtdetr"}:
        model = payload.get("ema")
        if model is None:
            model = payload["model"]
        state = model.state_dict()
    else:
        state = payload
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        )
    ):
        raise ValueError("Checkpoint does not contain a tensor state dictionary")
    return _state_digest(state)


def configure_runtime(run, device_name):
    import cv2
    import torch

    if device_name not in {"cpu", "cuda:0"}:
        raise ValueError("Replay device must be cpu or cuda:0")
    if device_name == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = run.get("numerical_policy", {})
    darc = run["scope"] in {"darc", "adaptation"}
    threads = int(policy.get("torch_intraop_threads", 8 if darc else 2))
    interop = int(policy.get("torch_interop_threads", 1))
    cv_threads = int(policy.get("opencv_threads", 1))
    if min(threads, interop, cv_threads) < 1:
        raise ValueError("Thread counts must be positive")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != interop:
        torch.set_num_interop_threads(interop)
    cv2.setNumThreads(cv_threads)
    precision = policy.get("float32_matmul_precision", "highest")
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("Unsupported matmul precision")
    if darc and (
        precision != "highest" or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"
    ):
        raise ValueError("DARC replay requires FP32 without a TF32 override")
    torch.set_float32_matmul_precision(precision)
    deterministic = policy.get(
        "deterministic_algorithms", torch.are_deterministic_algorithms_enabled()
    )
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic_algorithms must be boolean")
    torch.use_deterministic_algorithms(deterministic)
    if "cuda_matmul_allow_tf32" in policy:
        if (
            policy.get("matmul_allow_tf32", policy["cuda_matmul_allow_tf32"])
            != policy["cuda_matmul_allow_tf32"]
        ):
            raise ValueError("Conflicting matmul TF32 policies")
        policy = {**policy, "matmul_allow_tf32": policy["cuda_matmul_allow_tf32"]}
    backend_options = {
        "matmul_allow_tf32": (torch.backends.cuda.matmul, "allow_tf32"),
        "cudnn_allow_tf32": (torch.backends.cudnn, "allow_tf32"),
        "cudnn_deterministic": (torch.backends.cudnn, "deterministic"),
        "cudnn_benchmark": (torch.backends.cudnn, "benchmark"),
    }
    for key, (backend, attribute) in backend_options.items():
        value = policy.get(key, False if darc else getattr(backend, attribute))
        if not isinstance(value, bool) or (darc and key.endswith("allow_tf32") and value):
            raise ValueError(f"Invalid numerical policy: {key}")
        setattr(backend, attribute, value)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Replay requires float32 default tensors")
    observed = {key: getattr(backend, attr) for key, (backend, attr) in backend_options.items()}
    observed.update(
        torch_intraop_threads=threads,
        torch_interop_threads=interop,
        opencv_threads=cv_threads,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=precision,
        device=device_name,
        torch_version=torch.__version__,
    )
    return torch.device(device_name), observed


def _verify_bootstrap(run):
    if run["method"] == "faster_rcnn" and run["initialization"] == "pretrained":
        from .darc_train import _verified_pretrained_asset

        try:
            _verified_pretrained_asset(BOOTSTRAP_SHA256)
        except ValueError as error:
            raise ValueError(
                "Cache fasterrcnn_resnet50_fpn_coco-258fb6c6.pth under "
                "$TORCH_HOME/hub/checkpoints with SHA-256 " + BOOTSTRAP_SHA256
            ) from error


def predict(run, checkpoint, coco, images, observed, device, staging):
    _verify_bootstrap(run)
    if run["scope"] in {"darc", "adaptation"}:
        import torch

        from ..faster_rcnn.model import build_model
        from .darc_data import DARCData
        from .darc_train import _predictions

        model = build_model(
            run["config"], pretrained=True, box_score_thresh=0.0, box_detections_per_img=300
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.to(device).eval()
        dataset = DARCData(
            coco,
            images,
            intensity=run["intensity"],
            scale="S0",
            source=run["source"],
            seed=run["seed"],
            expected_hashes={row["image_id"]: row["sha256"] for row in observed},
        )
        predictions, _audits = _predictions(model, dataset, device)
        return predictions, {
            "intensity": run["intensity"],
            "scale": "S0",
            "precision": "float32",
            "score_floor": 0.0,
        }
    from .inference_timing import SelectedRun, make_predictor
    from .ultralytics_models import stage_split

    directory = stage_split(coco, Path(images), staging)
    selected = SelectedRun(
        checkpoint, run["config"], run["initialization"], run.get("pixel_threshold"), {}
    )
    predictor, preprocessing = make_predictor(run["method"], selected, directory, coco, device)
    if run["method"] == "unet" and device.type == "cpu":
        preprocessing["precision"] = "float32 (CPU, no CUDA autocast)"
    return predictor(), preprocessing


def validate_predictions(predictions, coco):
    images = {row["id"]: row for row in coco["images"]}
    categories = {row["id"] for row in coco["categories"]}
    degenerate = 0
    for row in predictions:
        info = images.get(row["image_id"])
        if info is None or row["category_id"] not in categories or len(row["bbox"]) != 4:
            raise ValueError("Prediction has an unknown image/category or invalid box")
        x, y, width, height = row["bbox"]
        if not all(math.isfinite(value) for value in [x, y, width, height, row["score"]]):
            raise ValueError("Nonfinite prediction")
        if (
            min(x, y) < -1e-4
            or min(width, height) < 0
            or x + width > info["width"] + 1e-4
            or y + height > info["height"] + 1e-4
            or not 0 <= row["score"] <= 1
        ):
            raise ValueError("Prediction is outside its native raster or score range")
        degenerate += int(min(width, height) == 0)
    return {"prediction_count": len(predictions), "degenerate_boxes": degenerate}


def compare_predictions(predictions, reference, image_ids, atol=0.0, rtol=0.0):
    if min(atol, rtol) < 0 or not all(map(math.isfinite, (atol, rtol))):
        raise ValueError("Parity tolerances must be finite and nonnegative")
    selected = set(image_ids)
    reference = [row for row in reference if row["image_id"] in selected]
    counts = lambda rows: Counter((row["image_id"], row["category_id"]) for row in rows)
    result = {
        "reference_prediction_count": len(reference),
        "prediction_count": len(predictions),
        "counts_equal": counts(predictions) == counts(reference),
        "ordering": "image, category, descending score, bbox",
        "atol": atol,
        "rtol": rtol,
        "exact": False,
        "within_tolerance": False,
    }
    if not result["counts_equal"]:
        return result
    key = lambda row: (row["image_id"], row["category_id"], -row["score"], *row["bbox"])
    pairs = list(zip(sorted(predictions, key=key), sorted(reference, key=key), strict=True))
    deltas, scores, close = [], [], True
    exact = True
    for actual, expected in pairs:
        for index, (left, right) in enumerate(
            zip(
                [*actual["bbox"], actual["score"]],
                [*expected["bbox"], expected["score"]],
                strict=True,
            )
        ):
            if not math.isfinite(right):
                raise ValueError("Nonfinite reference prediction")
            (scores if index == 4 else deltas).append(abs(left - right))
            exact = exact and left == right
            close = close and abs(left - right) <= atol + rtol * abs(right)
    return {
        **result,
        "exact": exact,
        "within_tolerance": close,
        "max_abs_box_difference": max(deltas, default=0.0),
        "max_abs_score_difference": max(scores, default=0.0),
    }


def run(args):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Replay output must be a fresh file")
    manifest, checkpoint = read_run(args.run)
    coco, observed = prepare_inputs(args.annotations, args.images, args.split_manifest, args.limit)
    state_digest = tensor_digest(checkpoint, manifest["method"])
    if manifest.get("tensor_state_sha256") not in (None, state_digest):
        raise ValueError("Checkpoint tensor-state SHA-256 mismatch")
    device, numerical = configure_runtime(manifest, args.device)
    protocol.set_seed(manifest["seed"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-replay-", dir=output.parent) as temporary:
        predictions, preprocessing = predict(
            manifest, checkpoint, coco, args.images, observed, device, Path(temporary)
        )
    prediction_validation = validate_predictions(predictions, coco)
    result = {
        "schema_version": 1,
        "job_id": manifest["job_id"],
        "source": manifest["source"],
        "seed": manifest["seed"],
        "method": manifest["method"],
        "scope": manifest["scope"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "tensor_state_sha256": state_digest,
        "tensor_state_verified": manifest.get("tensor_state_sha256") is not None,
        "run_sha256": protocol.sha256_file(args.run),
        "annotations_sha256": protocol.sha256_file(args.annotations),
        "split_manifest_sha256": protocol.sha256_file(args.split_manifest)
        if args.split_manifest
        else None,
        "input_images_verified_against_split": args.split_manifest is not None,
        "images": observed,
        "numerical_policy": numerical,
        "preprocessing": preprocessing,
        "threshold": manifest.get("threshold"),
        "predictions": predictions,
        "prediction_validation": prediction_validation,
    }
    if manifest.get("threshold") is not None:
        result["operating_predictions"] = [
            row for row in predictions if row["score"] >= manifest["threshold"]
        ]
    if args.reference_predictions is not None:
        reference = json.loads(Path(args.reference_predictions).read_text())
        if isinstance(reference, dict):
            reference = reference["predictions"]
        result["parity"] = compare_predictions(
            predictions, reference, [row["id"] for row in coco["images"]], args.atol, args.rtol
        )
        result["reference_predictions_sha256"] = protocol.sha256_file(args.reference_predictions)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "annotations", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "job_id": result["job_id"],
                "image_count": len(result["images"]),
                "prediction_count": len(result["predictions"]),
                "parity": result.get("parity"),
            }
        )
    )


if __name__ == "__main__":
    main()
