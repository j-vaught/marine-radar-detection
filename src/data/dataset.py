"""PyTorch datasets for marine radar detection."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarDetectionDataset(Dataset):
    """Dataset for object detection on radar tiles (Faster R-CNN, RT-DETR)."""

    def __init__(
        self,
        image_dir: str | Path,
        annotations: list[dict],
        transforms=None,
    ):
        """
        Args:
            image_dir: Directory containing tile images.
            annotations: List of dicts with 'file_name', 'boxes' (N,4), 'labels' (N,).
            transforms: Albumentations or torchvision transforms.
        """
        self.image_dir = Path(image_dir)
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = self.image_dir / ann["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = np.stack([image] * 3, axis=-1)  # grayscale -> 3-channel

        boxes = np.array(ann["boxes"], dtype=np.float32)
        labels = np.array(ann["labels"], dtype=np.int64)
        # torchvision Faster R-CNN requires boxes of shape (N, 4) even when N=0
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["labels"], dtype=np.int64)
            if boxes.size == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        return image, target


class RadarSegmentationDataset(Dataset):
    """Dataset for U-Net segmentation on radar tiles."""

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
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask
