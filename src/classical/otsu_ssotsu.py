"""Otsu and SSOTSU (Speed-up Seed Point Otsu) for small-target detection."""

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .utils import binary_mask_to_bboxes, binary_mask_to_bboxes_with_scores


def detect_otsu(
    image: np.ndarray,
    min_area: int = 10,
    max_area: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basic Otsu thresholding + connected component detection.

    Args:
        image: Grayscale input image.
        min_area: Minimum connected component area.
        max_area: Maximum connected component area.

    Returns:
        Tuple of (bboxes, binary_mask, scores) where scores are normalised
        peak-intensity-to-Otsu-threshold ratios in [0, 1].
    """
    if image.dtype in (np.float32, np.float64):
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    thresh = threshold_otsu(img_u8)
    binary = (img_u8 > thresh).astype(np.uint8)
    bboxes = binary_mask_to_bboxes(binary, min_area=min_area, max_area=max_area)

    threshold_map = np.full(binary.shape, float(thresh), dtype=np.float32)
    _, scores = binary_mask_to_bboxes_with_scores(
        binary,
        img_u8.astype(np.float32),
        threshold_map=threshold_map,
        min_area=min_area,
        max_area=max_area,
    )
    return bboxes, binary, scores


def detect_ssotsu(
    image: np.ndarray,
    block_size: int = 64,
    seed_threshold_factor: float = 1.5,
    region_grow_threshold: float = 0.8,
    min_area: int = 5,
    max_area: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple of (bboxes, binary_mask, scores) where scores are normalised
        peak-intensity-to-seed-threshold ratios in [0, 1].
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

    # Build a threshold map from the per-block local Otsu values scaled by
    # seed_threshold_factor, so score = peak / local_seed_threshold per component.
    # Recompute block thresholds to populate the map.
    thresh_map = np.zeros(img_u8.shape, dtype=np.float32)
    h, w = img_u8.shape
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_u8[y: min(y + block_size, h), x: min(x + block_size, w)]
            if block.std() < 1:
                local_t = float(img_u8.mean())
            else:
                try:
                    local_t = float(threshold_otsu(block)) * seed_threshold_factor
                except ValueError:
                    local_t = float(img_u8.mean())
            thresh_map[y: y + block.shape[0], x: x + block.shape[1]] = max(local_t, 1.0)

    _, scores = binary_mask_to_bboxes_with_scores(
        result_mask,
        img_u8.astype(np.float32),
        threshold_map=thresh_map,
        min_area=min_area,
        max_area=max_area,
    )
    return bboxes, result_mask, scores
