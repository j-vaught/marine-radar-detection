"""Shared utilities for classical detection methods."""

import cv2
import numpy as np
from skimage import measure


def binary_mask_to_bboxes(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 50000,
) -> np.ndarray:
    """Convert a binary mask to bounding boxes via connected components.

    Args:
        mask: Binary mask (0/1 or bool).
        min_area: Minimum component area to keep.
        max_area: Maximum component area to keep.

    Returns:
        (N, 4) array of [x1, y1, x2, y2] bounding boxes.
    """
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels)

    bboxes = []
    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue
        # regionprops bbox is (min_row, min_col, max_row, max_col)
        y1, x1, y2, x2 = region.bbox
        bboxes.append([x1, y1, x2, y2])

    return np.array(bboxes, dtype=np.float32).reshape(-1, 4)


def binary_mask_to_bboxes_with_props(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 50000,
    min_eccentricity: float = 0.0,
    max_eccentricity: float = 0.99,
) -> tuple[np.ndarray, list[dict]]:
    """Convert binary mask to bounding boxes with region properties.

    Returns:
        Tuple of (bboxes, properties) where properties include area, eccentricity, etc.
    """
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels)

    bboxes = []
    props = []
    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue
        if region.eccentricity < min_eccentricity or region.eccentricity > max_eccentricity:
            continue

        y1, x1, y2, x2 = region.bbox
        bboxes.append([x1, y1, x2, y2])
        props.append({
            "area": region.area,
            "eccentricity": region.eccentricity,
            "centroid": region.centroid,
            "orientation": region.orientation,
        })

    return np.array(bboxes, dtype=np.float32).reshape(-1, 4), props


def compute_local_stats(
    image: np.ndarray,
    center: tuple[int, int],
    guard: tuple[int, int],
    ref: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference window pixels for CFAR, split into leading/lagging halves.

    Args:
        image: Input intensity image.
        center: (row, col) of the cell under test.
        guard: (guard_rows, guard_cols) half-size of guard band.
        ref: (ref_rows, ref_cols) half-size of reference window.

    Returns:
        Tuple of (leading_pixels, lagging_pixels) arrays.
    """
    r, c = center
    gr, gc = guard
    rr, rc = ref
    h, w = image.shape

    # Full reference window
    r_min = max(0, r - rr)
    r_max = min(h, r + rr + 1)
    c_min = max(0, c - rc)
    c_max = min(w, c + rc + 1)

    # Guard window
    g_r_min = max(0, r - gr)
    g_r_max = min(h, r + gr + 1)
    g_c_min = max(0, c - gc)
    g_c_max = min(w, c + gc + 1)

    ref_window = image[r_min:r_max, c_min:c_max].copy()

    # Zero out guard region in reference window
    local_g_r_min = g_r_min - r_min
    local_g_r_max = g_r_max - r_min
    local_g_c_min = g_c_min - c_min
    local_g_c_max = g_c_max - c_min
    ref_window[local_g_r_min:local_g_r_max, local_g_c_min:local_g_c_max] = 0

    # Split into leading (above CUT) and lagging (below CUT)
    cut_local_r = r - r_min
    leading = ref_window[:cut_local_r].flatten()
    lagging = ref_window[cut_local_r + 1:].flatten()

    leading = leading[leading > 0]
    lagging = lagging[lagging > 0]

    return leading, lagging
