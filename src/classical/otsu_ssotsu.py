"""Otsu and SSOTSU (Speed-up Seed Point Otsu) for small-target detection."""

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .utils import binary_mask_to_bboxes


def detect_otsu(
    image: np.ndarray,
    min_area: int = 10,
    max_area: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Basic Otsu thresholding + connected component detection.

    Args:
        image: Grayscale input image.
        min_area: Minimum connected component area.
        max_area: Maximum connected component area.

    Returns:
        Tuple of (bboxes, binary_mask).
    """
    if image.dtype in (np.float32, np.float64):
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    thresh = threshold_otsu(img_u8)
    binary = (img_u8 > thresh).astype(np.uint8)
    bboxes = binary_mask_to_bboxes(binary, min_area=min_area, max_area=max_area)
    return bboxes, binary


def detect_ssotsu(
    image: np.ndarray,
    block_size: int = 64,
    seed_threshold_factor: float = 1.5,
    region_grow_threshold: float = 0.8,
    min_area: int = 5,
    max_area: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """SSOTSU: Speed-up Seed Point Otsu method for small target detection.

    Divides image into blocks, applies local Otsu to find seed points,
    then grows regions from seeds using intensity similarity.

    Args:
        image: Grayscale input image.
        block_size: Local block size for seed detection.
        seed_threshold_factor: Factor above local Otsu for seed qualification.
        region_grow_threshold: Relative intensity threshold for region growing.
        min_area: Minimum component area.
        max_area: Maximum component area.

    Returns:
        Tuple of (bboxes, binary_mask).
    """
    if image.dtype in (np.float32, np.float64):
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    h, w = img_u8.shape
    seed_mask = np.zeros((h, w), dtype=np.uint8)

    # Phase 1: Find seed points via local Otsu
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_u8[y: min(y + block_size, h), x: min(x + block_size, w)]
            if block.std() < 1:
                continue
            try:
                local_thresh = threshold_otsu(block)
            except ValueError:
                continue
            seed_thresh = local_thresh * seed_threshold_factor
            local_seeds = block > seed_thresh
            seed_mask[y: y + block.shape[0], x: x + block.shape[1]] = local_seeds.astype(np.uint8)

    # Phase 2: Region growing from seed points
    result_mask = np.zeros_like(seed_mask)
    seed_points = np.argwhere(seed_mask > 0)

    visited = np.zeros_like(seed_mask, dtype=bool)

    for sy, sx in seed_points:
        if visited[sy, sx]:
            continue

        seed_intensity = float(img_u8[sy, sx])
        grow_thresh = seed_intensity * region_grow_threshold

        stack = [(sy, sx)]
        region_pixels = []

        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            if visited[cy, cx]:
                continue
            if img_u8[cy, cx] < grow_thresh:
                continue

            visited[cy, cx] = True
            region_pixels.append((cy, cx))
            result_mask[cy, cx] = 1

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    stack.append((ny, nx))

    bboxes = binary_mask_to_bboxes(result_mask, min_area=min_area, max_area=max_area)
    return bboxes, result_mask
