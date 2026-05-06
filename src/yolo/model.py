"""Ultralytics YOLO helpers for marine radar detection."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str | Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def resolve_device(device: str | int | None) -> str:
    if device is None:
        return "0"
    if isinstance(device, int):
        return str(device)
    return str(device)


def resolve_accelerator(device: str | int | None) -> str:
    resolved = resolve_device(device)
    return f"cuda:{resolved}" if resolved.isdigit() else resolved


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "outofmemoryerror" in msg
        or "cublas_status_alloc_failed" in msg
    )


def train_yolo(
    data_yaml: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    cfg: dict | None = None,
):
    """Train an Ultralytics YOLO model and return the resolved save paths."""
    from ultralytics import YOLO
    import torch

    cfg = cfg or load_config(config_path)
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    device = resolve_device(train_cfg.get("device"))
    run_name = f"yolo_seed{seed}"
    batch_size = int(train_cfg["batch_size"])
    min_batch_size = int(train_cfg.get("min_batch_size", 2))

    while True:
        model = YOLO(cfg["model"]["variant"] + ".pt")
        try:
            model.train(
                data=str(data_yaml),
                epochs=train_cfg["epochs"],
                batch=batch_size,
                imgsz=train_cfg["imgsz"],
                optimizer=train_cfg["optimizer"],
                lr0=train_cfg["lr0"],
                lrf=train_cfg["lrf"],
                warmup_epochs=train_cfg["warmup_epochs"],
                patience=train_cfg["patience"],
                device=device,
                workers=train_cfg["workers"],
                amp=bool(train_cfg.get("amp", True)),
                cache=train_cfg.get("cache", False),
                seed=seed,
                project=str(output_dir),
                name=run_name,
                hsv_h=aug_cfg.get("hsv_h", 0.015),
                hsv_s=aug_cfg.get("hsv_s", 0.7),
                hsv_v=aug_cfg.get("hsv_v", 0.4),
                flipud=aug_cfg.get("flipud", 0.5),
                fliplr=aug_cfg.get("fliplr", 0.5),
                mosaic=aug_cfg.get("mosaic", 0.0),
                mixup=aug_cfg.get("mixup", 0.0),
            )
            break
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or batch_size <= min_batch_size:
                raise
            next_batch_size = max(min_batch_size, batch_size // 2)
            print(f"[yolo] CUDA OOM at batch={batch_size}; retrying with batch={next_batch_size}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = next_batch_size

    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) is not None else Path(output_dir) / run_name
    return {
        "model": model,
        "save_dir": save_dir,
        "weights": save_dir / "weights" / "best.pt",
        "batch_size": batch_size,
    }


def predict_yolo(
    model_path: str | Path,
    image: np.ndarray,
    conf: float = 0.25,
    device: str | int = "0",
) -> np.ndarray:
    """Run YOLO inference on a single image."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(image, conf=conf, device=resolve_device(device), verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
    return np.hstack([boxes, confs])


def predict_sahi(
    model_path: str | Path,
    image_path: str | Path,
    config_path: str | Path,
    device: str | int = "0",
    cfg: dict | None = None,
) -> list[dict]:
    """Run SAHI tiled inference with an Ultralytics YOLO checkpoint."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    cfg = cfg or load_config(config_path)
    sahi_cfg = cfg["sahi"]
    accelerator = resolve_accelerator(device)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=float(sahi_cfg.get("confidence_threshold", 0.25)),
        device=accelerator,
    )

    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=sahi_cfg["slice_height"],
        slice_width=sahi_cfg["slice_width"],
        overlap_height_ratio=sahi_cfg["overlap_height_ratio"],
        overlap_width_ratio=sahi_cfg["overlap_width_ratio"],
        postprocess_type=sahi_cfg["postprocess_type"],
        postprocess_match_threshold=sahi_cfg["postprocess_match_threshold"],
    )

    predictions = []
    for pred in result.object_prediction_list:
        predictions.append(
            {
                "bbox": pred.bbox.to_xyxy(),
                "score": pred.score.value,
                "category": pred.category.name,
            }
        )

    return predictions
