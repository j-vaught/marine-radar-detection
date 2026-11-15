"""Marine Faster R-CNN training and inference helpers."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def resolve_device(device: str | int | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    raw = str(device).strip()
    if raw.isdigit():
        return torch.device(f"cuda:{raw}" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_model(
    cfg: dict,
    *,
    pretrained: bool = True,
    box_score_thresh: float | None = None,
    box_detections_per_img: int | None = None,
) -> nn.Module:
    """Build Faster R-CNN with radar-oriented anchors."""
    from torchvision.models.detection.rpn import AnchorGenerator

    model_cfg = cfg["model"]
    anchor_sizes = tuple(tuple(s) for s in model_cfg["anchor_sizes"])
    anchor_ratios = tuple(tuple(r) for r in model_cfg["anchor_ratios"]) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_ratios)

    score_thresh = (
        float(box_score_thresh)
        if box_score_thresh is not None
        else float(model_cfg["box_score_thresh"])
    )
    detections_per_img = (
        int(box_detections_per_img)
        if box_detections_per_img is not None
        else int(model_cfg["box_detections_per_img"])
    )

    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=int(model_cfg["rpn_pre_nms_top_n_train"]),
        rpn_post_nms_top_n_train=int(model_cfg["rpn_post_nms_top_n_train"]),
        rpn_pre_nms_top_n_test=int(model_cfg["rpn_pre_nms_top_n_test"]),
        rpn_post_nms_top_n_test=int(model_cfg["rpn_post_nms_top_n_test"]),
        rpn_nms_thresh=float(model_cfg["rpn_nms_thresh"]),
        rpn_fg_iou_thresh=float(model_cfg["rpn_fg_iou_thresh"]),
        rpn_bg_iou_thresh=float(model_cfg["rpn_bg_iou_thresh"]),
        box_score_thresh=score_thresh,
        box_nms_thresh=float(model_cfg["box_nms_thresh"]),
        box_detections_per_img=detections_per_img,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(model_cfg["num_classes"]))
    return model


def _maybe_autocast(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def _average_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> float:
    was_training = model.training
    model.train()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with _maybe_autocast(device, amp):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            total_loss += float(losses.item())
            steps += 1

    if not was_training:
        model.eval()
    return total_loss / max(1, steps)


def train_faster_rcnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Train Faster R-CNN and checkpoint the best validation-loss epoch."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    device = resolve_device(train_cfg.get("device"))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg["lr_step_size"]),
        gamma=float(train_cfg["lr_gamma"]),
    )

    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(train_cfg.get("patience", train_cfg["epochs"]))
    best_val_loss = float("inf")
    best_epoch = -1
    stagnant_epochs = 0
    history: list[dict] = []

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with _maybe_autocast(device, use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

            if use_amp:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            loss_value = float(losses.item())
            running_loss += loss_value
            pbar.set_postfix(loss=loss_value, lr=float(optimizer.param_groups[0]["lr"]))

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))
        val_loss = _average_loss(model, val_loader, device, use_amp)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        state_dict = model.state_dict()
        torch.save(state_dict, output_dir / f"last_seed{seed}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            stagnant_epochs = 0
            torch.save(state_dict, output_dir / f"best_seed{seed}.pth")
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def predict_faster_rcnn(
    model: nn.Module,
    image: torch.Tensor,
    device: str | int | torch.device = "cuda:0",
    score_thresh: float = 0.5,
) -> np.ndarray:
    """Run inference and return [x1, y1, x2, y2, score]."""
    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)

    with torch.no_grad(), _maybe_autocast(resolved_device, resolved_device.type == "cuda"):
        prediction = model([image.to(resolved_device)])[0]

    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep].reshape(-1, 1)
    return np.hstack([boxes, scores]) if len(boxes) > 0 else np.zeros((0, 5), dtype=np.float32)
