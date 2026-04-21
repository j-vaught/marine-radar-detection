"""U-Net segmentation model for marine radar detection."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import binary_mask_to_bboxes_with_scores


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        channels = [base_channels * (2 ** i) for i in range(5)]

        # Encoder
        self.enc1 = DoubleConv(in_channels, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])

        self.bottleneck = DoubleConv(channels[3], channels[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0])

        self.out_conv = nn.Conv2d(channels[0], num_classes, 1)
        self.total_stride = 16

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (-orig_h) % self.total_stride
        pad_w = (-orig_w) % self.total_stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self._match_size(d4, e4), e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self._match_size(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self._match_size(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._match_size(d1, e1), e1], dim=1))

        out = self.out_conv(d1)
        return out[..., :orig_h, :orig_w]


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)

        # Dice
        smooth = 1.0
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        # BCE
        bce = F.binary_cross_entropy_with_logits(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_raw: str | int) -> torch.device:
    if isinstance(device_raw, int):
        device_raw = str(device_raw)
    if isinstance(device_raw, str) and device_raw.isdigit():
        return torch.device(f"cuda:{device_raw}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_raw)


def _make_autocast(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def train_unet(
    model: UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Train U-Net segmentation model."""
    _seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    device = _resolve_device(train_cfg["device"])
    model.to(device)

    criterion = DiceBCELoss(
        dice_weight=loss_cfg["dice_weight"],
        bce_weight=loss_cfg["bce_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["T_max"])
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)
    patience = int(train_cfg.get("patience", 0))

    best_val_loss = float("inf")
    best_epoch = -1
    train_history: list[dict] = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _make_autocast(device, amp_enabled):
                pred = model(images)
                loss = criterion(pred, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with _make_autocast(device, amp_enabled):
                    pred = model(images)
                    loss = criterion(pred, masks)
                val_loss_sum += float(loss.item())

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / f"best_seed{seed}.pth")
        if patience > 0 and (epoch + 1 - best_epoch) >= patience:
            break

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": train_history,
    }


def predict_prob_map(
    model: UNet,
    image: np.ndarray | torch.Tensor,
    device: str = "cuda:0",
    input_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Predict a probability map and return it at the original image size."""
    model.eval()
    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    if isinstance(image, torch.Tensor):
        image_np = image.squeeze().cpu().numpy()
    else:
        image_np = image
    image_np = image_np.astype(np.float32)
    orig_h, orig_w = image_np.shape[-2:]

    if input_size is not None:
        resize_h, resize_w = input_size
        resized = cv2.resize(image_np, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image_np

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    with torch.no_grad():
        with _make_autocast(resolved_device, resolved_device.type == "cuda"):
            pred = model(tensor.to(resolved_device, non_blocking=True))
            prob = torch.sigmoid(pred)
        if prob.shape[-2:] != (orig_h, orig_w):
            prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return prob.squeeze().float().cpu().numpy()


def predict_unet(
    model: UNet,
    image: np.ndarray | torch.Tensor,
    device: str = "cuda:0",
    threshold: float = 0.5,
    min_area: int = 15,
    max_area: int = 50000,
    dilation: int = 0,
    box_padding: int = 0,
    input_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run U-Net inference and extract bounding boxes from predicted mask.

    Returns:
        Tuple of (bboxes, predicted_mask, scores).
    """
    prob_map = predict_prob_map(model, image, device=device, input_size=input_size)
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    bboxes, scores = binary_mask_to_bboxes_with_scores(
        pred_mask,
        prob_map,
        min_area=min_area,
        max_area=max_area,
        dilation=dilation,
        box_padding=box_padding,
    )
    return bboxes, pred_mask, scores
