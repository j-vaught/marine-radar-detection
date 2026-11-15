"""Source-only training and fixed-holdout evaluation for revision jobs.

The segmentation detector uses one fixed pixel threshold to form components and
selects a component-score threshold using source validation only.
"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from . import evaluation, protocol


class RadarDataset(Dataset):
    def __init__(self, coco: dict, image_dir: Path, method: str, size: tuple[int, int]):
        self.images = coco["images"]
        self.image_dir = image_dir
        self.method = method
        self.size = size
        self.annotations = {}
        for annotation in coco["annotations"]:
            self.annotations.setdefault(annotation["image_id"], []).append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.image_dir / info["file_name"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape
        if (height, width) != (int(info["height"]), int(info["width"])):
            raise ValueError(f"Image dimensions do not match annotations: {path}")
        yy, xx = np.ogrid[:height, :width]
        roi = (xx - 531) ** 2 + (yy - 515) ** 2 <= 466**2
        image = np.where(roi, image, 0).astype(np.uint8)
        anns = self.annotations.get(info["id"], [])
        boxes = []
        mask = np.zeros((height, width), np.uint8)
        for ann in anns:
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid bounding box in image {info['id']}")
            boxes.append([x, y, x + w, y + h])
            x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
            x2, y2 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
            mask[y1:y2, x1:x2] = 1
        if self.method == "unet":
            mask *= roi
            out_h, out_w = self.size
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(image.copy()).unsqueeze(0).float() / 255, torch.from_numpy(
                mask.copy()
            ).unsqueeze(0).float()
        image_tensor = torch.from_numpy(np.stack([image] * 3)).float() / 255
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return image_tensor, {
            "boxes": box_tensor,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(int(info["id"])),
        }


def _merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _loader(coco, image_dir, method, size, cfg, shuffle):
    ds = RadarDataset(coco, image_dir, method, size)
    if not len(ds):
        raise ValueError("Training and evaluation subsets must contain images")
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["workers"]),
        collate_fn=(lambda batch: tuple(zip(*batch))) if method == "faster_rcnn" else None,
    )


def _loss(model, batch, device, method, criterion):
    images, targets = batch
    if method == "unet":
        return criterion(model(images.to(device)), targets.to(device))
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    return sum(model(images, targets).values())


def _predictions(model, coco, image_dir, method, size, device, post):
    from ..unet.model import predict_prob_map
    from ..unet.utils import binary_mask_to_bboxes_with_scores

    dataset = RadarDataset(coco, image_dir, method, size)
    result = []
    model.eval()
    for index, info in enumerate(coco["images"]):
        if method == "unet":
            path = image_dir / info["file_name"]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape
            yy, xx = np.ogrid[:h, :w]
            roi = (xx - 531) ** 2 + (yy - 515) ** 2 <= 466**2
            prob = predict_prob_map(
                model, np.where(roi, image, 0).astype(np.uint8), device=str(device), input_size=size
            )
            prob[~roi] = 0
            boxes, scores = binary_mask_to_bboxes_with_scores(
                (prob >= float(post["threshold"])).astype(np.uint8),
                prob,
                min_area=int(post["min_area"]),
                max_area=int(post["max_area"]),
                dilation=int(post["dilation"]),
                box_padding=int(post["box_padding"]),
            )
        else:
            image, _ = dataset[index]
            with torch.no_grad():
                output = model([image.to(device)])[0]
            boxes, scores = output["boxes"].cpu().numpy(), output["scores"].cpu().numpy()
        for box, score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = map(float, box)
            if x2 > x1 and y2 > y1:
                result.append(
                    {
                        "image_id": int(info["id"]),
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return result


def run_job(job: dict) -> dict:
    """Train one source model, select on source validation, evaluate fixed targets."""
    from ..faster_rcnn.model import build_model, resolve_device
    from ..unet.model import DiceBCELoss, UNet

    method = job["method"]
    if method not in {"unet", "faster_rcnn"}:
        raise ValueError(f"Unsupported method: {method}")
    init = job.get("initialization", "scratch")
    if init not in {"pretrained", "scratch"}:
        raise ValueError(f"Unsupported initialization: {init}")
    if method == "unet" and init != "scratch":
        raise ValueError("U-Net does not provide pretrained weights")
    if job.get("stage") == "diagnostic" and job.get("eval_targets"):
        raise ValueError("Diagnostic jobs cannot evaluate target holdouts")
    protocol.set_seed(int(job["seed"]))
    prepared = protocol.prepare_job(job)
    output = Path(job["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(__file__).parents[1] / method / "config.yaml"
    cfg = _merge(yaml.safe_load(cfg_path.read_text()), job.get("config_overrides", {}))
    train_cfg = cfg["training"]
    train_cfg.update(epochs=int(job["epochs"]), patience=int(job["patience"]), device=job["device"])
    if method == "unet":
        train_cfg["T_max"] = int(train_cfg["epochs"])
    size = tuple(train_cfg.get("input_size", (640, 640)))
    device = resolve_device(job["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = (
        UNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            base_channels=cfg["model"]["base_channels"],
        )
        if method == "unet"
        else build_model(
            cfg,
            pretrained=init == "pretrained",
            box_score_thresh=0.0,
            box_detections_per_img=int(cfg["model"].get("eval_box_detections_per_img", 300)),
        )
    )
    initial_hash = _state_hash(model)
    model.to(device)
    train_loader = _loader(prepared["train"], prepared["image_dir"], method, size, train_cfg, True)
    val_loader = _loader(
        prepared["validation"], prepared["image_dir"], method, size, train_cfg, False
    )
    criterion = (
        DiceBCELoss(**{k: cfg["loss"][k] for k in ("dice_weight", "bce_weight")})
        if method == "unet"
        else None
    )
    if method == "unet":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg["T_max"])
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            momentum=float(train_cfg["momentum"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(train_cfg["lr_step_size"]), gamma=float(train_cfg["lr_gamma"])
        )
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best_epoch, best_loss, history = 0, 0, float("inf"), []
    resume = job.get("resume")
    if resume:
        checkpoint = torch.load(
            str(resume) if isinstance(resume, str) else output / "resume.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        start, best_epoch, best_loss, history = (
            checkpoint["epoch"],
            checkpoint["best_epoch"],
            checkpoint["best_loss"],
            checkpoint["history"],
        )
        initial_hash = checkpoint["initial_hash"]
    protocol.save_json(
        output / "resolved_config.json",
        {
            "job": job,
            "config": cfg,
            "initialization": init,
            "initialized_state_sha256": initial_hash,
            "roi": {"center": [531, 515], "radius": 466},
            "input_size": list(size) if method == "unet" else "native",
        },
    )
    started = time.perf_counter()
    for epoch in range(start, int(train_cfg["epochs"])):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = _loss(model, batch, device, method, criterion)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
        model.train(method == "faster_rcnn")
        buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
        val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = _loss(model, batch, device, method, criterion)
                val += float(loss.item())
        for name, buffer in model.named_buffers():
            buffer.copy_(buffers[name])
        val /= len(val_loader)
        if not np.isfinite(val):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}")
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train_loader),
                "val_loss": val,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val < best_loss:
            best_loss, best_epoch = val, epoch + 1
            torch.save(model.state_dict(), output / "best.pt")
        torch.save(model.state_dict(), output / "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "history": history,
                "initial_hash": initial_hash,
            },
            output / "resume.pt",
        )
        protocol.save_json(output / "history.json", history)
        if int(train_cfg["patience"]) > 0 and epoch + 1 - best_epoch >= int(train_cfg["patience"]):
            break
    train_seconds = time.perf_counter() - started
    if not best_epoch or not (output / "best.pt").exists():
        raise RuntimeError("No best checkpoint available")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    post = cfg["postprocessing"]
    val_gt = prepared["validation"]
    val_raw = _predictions(model, val_gt, prepared["image_dir"], method, size, device, post)
    protocol.save_json(output / "val_raw_predictions.json", val_raw)
    grid = [float(x) for x in post.get("score_threshold_grid", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])]
    choices = []
    for threshold in grid:
        preds = [p for p in val_raw if p["score"] >= threshold]
        metrics = evaluation.evaluate_coco(val_gt, preds)
        choices.append((metrics["ap50"], metrics["ap"], -threshold, threshold, metrics))
    if not choices:
        raise ValueError("Score threshold grid must not be empty")
    _, _, _, threshold, val_metrics = max(choices, key=lambda x: x[:3])
    protocol.save_json(
        output / "val_predictions.json", [p for p in val_raw if p["score"] >= threshold]
    )
    targets = {}
    eval_started = time.perf_counter()
    for site in job.get("eval_targets", []):
        gt = prepared["tests"][site]
        raw = _predictions(model, gt, prepared["image_dir"], method, size, device, post)
        filtered = [p for p in raw if p["score"] >= threshold]
        protocol.save_json(output / f"{site}_raw_predictions.json", raw)
        protocol.save_json(output / f"{site}_predictions.json", filtered)
        targets[site] = {
            "raw": evaluation.evaluate_coco(gt, raw),
            "filtered": evaluation.evaluate_coco(gt, filtered),
            "raw_count": len(raw),
            "filtered_count": len(filtered),
        }
    result = {
        "job_id": job["job_id"],
        "method": method,
        "source": job["source"],
        "seed": job["seed"],
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "threshold": threshold,
        "validation": val_metrics,
        "val_raw_count": len(val_raw),
        "tests": targets,
        "checkpoint": str(output / "best.pt"),
        "train_seconds": train_seconds,
        "target_eval_seconds": time.perf_counter() - eval_started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "initialized_state_sha256": initial_hash,
        "best_weights_sha256": protocol.sha256_file(output / "best.pt"),
        "resolved_config_sha256": protocol.sha256_file(output / "resolved_config.json"),
    }
    protocol.save_json(output / "metrics.json", result)
    return result
