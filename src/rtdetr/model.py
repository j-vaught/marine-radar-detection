"""RT-DETR: Real-Time DEtection TRansformer for marine radar (NMS-free)."""

from __future__ import annotations

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


def train_rt_detr(
    data_yaml: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    cfg: dict | None = None,
):
    """Train RT-DETR model using Ultralytics.

    Args:
        data_yaml: Path to YOLO-format data config.
        config_path: Path to model config YAML.
        output_dir: Directory for training outputs.
        seed: Random seed.
    """
    from ultralytics import RTDETR

    cfg = cfg or load_config(config_path)

    model = RTDETR(cfg["model"]["variant"] + ".pt")

    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    device = resolve_device(train_cfg.get("device"))
    run_name = f"rtdetr_seed{seed}"

    model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["imgsz"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr"],
        warmup_epochs=train_cfg["warmup_epochs"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        device=device,
        workers=train_cfg["workers"],
        patience=train_cfg.get("patience", 100),
        amp=bool(train_cfg.get("amp", True)),
        cache=train_cfg.get("cache", False),
        seed=seed,
        project=str(output_dir),
        name=run_name,
        flipud=aug_cfg.get("flipud", 0.5),
        fliplr=aug_cfg.get("fliplr", 0.5),
        mosaic=aug_cfg.get("mosaic", 0.5),
    )
    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) is not None else Path(output_dir) / run_name
    return {
        "model": model,
        "save_dir": save_dir,
        "weights": save_dir / "weights" / "best.pt",
    }


def predict_rt_detr(
    model_path: str | Path,
    image: np.ndarray,
    conf: float = 0.25,
    device: str | int = "0",
) -> np.ndarray:
    """Run RT-DETR inference.

    Returns:
        (N, 5) array of [x1, y1, x2, y2, confidence].
    """
    from ultralytics import RTDETR

    model = RTDETR(str(model_path))
    results = model.predict(image, conf=conf, device=resolve_device(device), verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
    return np.hstack([boxes, confs])
