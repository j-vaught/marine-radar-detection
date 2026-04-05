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


def binary_mask_to_bboxes_with_scores(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    threshold_map: np.ndarray | None = None,
    min_area: int = 10,
    max_area: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a binary mask to bounding boxes with per-detection confidence scores.

    For each connected component the score is derived from how strongly the
    detected pixels exceed the background level:

    - With threshold_map: score = peak_intensity / threshold_at_centroid,
      clamped to [0, 1] after global normalisation across all detections.
    - Without threshold_map: score = mean_intensity_in_component / 255.0
      (or / 1.0 for float images), normalised to the per-image maximum so
      scores span [0, 1].

    Args:
        mask: Binary mask (0/1 or bool).
        intensity_image: Original intensity image aligned with mask.
        threshold_map: Optional per-pixel CFAR / Otsu threshold map.
        min_area: Minimum component area to keep.
        max_area: Maximum component area to keep.

    Returns:
        Tuple of ((N, 4) bboxes in [x1, y1, x2, y2], (N,) scores in [0, 1]).
    """
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels, intensity_image=intensity_image.astype(np.float32))

    bboxes = []
    raw_scores = []

    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue

        y1, x1, y2, x2 = region.bbox
        bboxes.append([x1, y1, x2, y2])

        if threshold_map is not None:
            # Peak intensity of the component
            peak = float(region.intensity_image.max())
            # Threshold at the centroid pixel
            cy, cx = int(round(region.centroid[0])), int(round(region.centroid[1]))
            cy = np.clip(cy, 0, threshold_map.shape[0] - 1)
            cx = np.clip(cx, 0, threshold_map.shape[1] - 1)
            local_thresh = float(threshold_map[cy, cx])
            score = peak / (local_thresh + 1e-9)
        else:
            # Normalise mean intensity to image scale
            scale = 255.0 if intensity_image.max() > 1.0 else 1.0
            score = float(region.mean_intensity) / scale

        raw_scores.append(score)

    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    bboxes_arr = np.array(bboxes, dtype=np.float32)
    scores_arr = np.array(raw_scores, dtype=np.float32)

    # Normalise to [0, 1] across detections in this image
    s_max = scores_arr.max()
    if s_max > 0:
        scores_arr = scores_arr / s_max
    scores_arr = np.clip(scores_arr, 0.0, 1.0)

    return bboxes_arr, scores_arr


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
