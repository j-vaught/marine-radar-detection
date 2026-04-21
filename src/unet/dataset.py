"""Segmentation datasets and split helpers for radar U-Net experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarSegmentationDataset(Dataset):
    """Load paired tile + binary-mask images for U-Net training."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transforms=None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return image, mask


@dataclass(frozen=True)
class SegmentationItem:
    image_id: int
    file_name: str
    width: int
    height: int
    site: str
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]


def site_from_file_name(file_name: str) -> str:
    return file_name.split("_", 1)[0].upper()


def load_segmentation_items(ann_path: str | Path) -> list[SegmentationItem]:
    ann = json.loads(Path(ann_path).read_text())
    by_img: dict[int, list[tuple[float, float, float, float]]] = {}
    for entry in ann["annotations"]:
        x, y, w, h = entry["bbox"]
        by_img.setdefault(entry["image_id"], []).append((x, y, x + w, y + h))

    items: list[SegmentationItem] = []
    for info in ann["images"]:
        items.append(
            SegmentationItem(
                image_id=int(info["id"]),
                file_name=str(info["file_name"]),
                width=int(info["width"]),
                height=int(info["height"]),
                site=site_from_file_name(str(info["file_name"])),
                boxes_xyxy=tuple(by_img.get(int(info["id"]), ())),
            )
        )
    return items


def build_box_mask(
    height: int,
    width: int,
    boxes_xyxy: tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        ix1 = max(0, min(width, int(np.floor(x1))))
        iy1 = max(0, min(height, int(np.floor(y1))))
        ix2 = max(0, min(width, int(np.ceil(x2))))
        iy2 = max(0, min(height, int(np.ceil(y2))))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 255
    return mask


def subset_items(items: list[SegmentationItem], indices: list[int]) -> list[SegmentationItem]:
    return [items[idx] for idx in indices]


def subset_coco(ann_path: str | Path, image_ids: set[int]) -> dict:
    ann = json.loads(Path(ann_path).read_text())
    images = [img for img in ann["images"] if int(img["id"]) in image_ids]
    annotations = [entry for entry in ann["annotations"] if int(entry["image_id"]) in image_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": ann["categories"],
    }


def pooled_split(
    n_items: int,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[list[int], list[int], list[int]]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    order = np.arange(n_items)
    rng.shuffle(order)
    n_train = int(round(n_items * ratios[0]))
    n_val = int(round(n_items * ratios[1]))
    train_idx = order[:n_train].tolist()
    val_idx = order[n_train:n_train + n_val].tolist()
    test_idx = order[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def indices_for_site(items: list[SegmentationItem], site: str) -> list[int]:
    site = site.upper()
    return [idx for idx, item in enumerate(items) if item.site == site]


def site_internal_split(
    items: list[SegmentationItem],
    site: str,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[list[int], list[int], list[int]]:
    site_idx = indices_for_site(items, site)
    train_local, val_local, test_local = pooled_split(len(site_idx), seed=seed, ratios=ratios)
    return (
        [site_idx[idx] for idx in train_local],
        [site_idx[idx] for idx in val_local],
        [site_idx[idx] for idx in test_local],
    )


def site_transfer_split(
    items: list[SegmentationItem],
    train_site: str,
    test_site: str,
    seed: int = 0,
    train_val_ratio: tuple[float, float] = (0.85, 0.15),
) -> tuple[list[int], list[int], list[int]]:
    src_idx = indices_for_site(items, train_site)
    rng = np.random.default_rng(seed)
    order = np.array(src_idx)
    rng.shuffle(order)
    n_train = int(round(len(order) * train_val_ratio[0]))
    train_idx = order[:n_train].tolist()
    val_idx = order[n_train:].tolist()
    test_idx = indices_for_site(items, test_site)
    return train_idx, val_idx, test_idx


class RadarSegmentationItemsDataset(Dataset):
    """Build segmentation masks directly from COCO boxes."""

    def __init__(
        self,
        image_dir: str | Path,
        items: list[SegmentationItem],
        input_size: tuple[int, int] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.items = items
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.image_dir / item.file_name
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(img_path)
        mask = build_box_mask(item.height, item.width, item.boxes_xyxy)

        if self.input_size is not None:
            out_h, out_w = self.input_size
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return image_tensor, mask_tensor
