"""Connected-component utilities used by the U-Net detector."""

import cv2
import numpy as np
from math import atan2, sqrt


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _component_bbox(stats_row: np.ndarray) -> list[float]:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return [float(x), float(y), float(x + w), float(y + h)]


def _component_moments(component_mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(component_mask)
    area = float(len(xs))
    if area <= 1.0:
        return 0.0, 0.0, 0.0

    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    dx = xs.astype(np.float64) - x_mean
    dy = ys.astype(np.float64) - y_mean
    mu20 = float(np.dot(dx, dx))
    mu02 = float(np.dot(dy, dy))
    mu11 = float(np.dot(dx, dy))
    return mu20, mu02, mu11


def _component_shape_props(component_mask: np.ndarray) -> tuple[float, float]:
    mu20, mu02, mu11 = _component_moments(component_mask)
    trace = mu20 + mu02
    det_term = (mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2)
    root = sqrt(max(det_term, 0.0))
    major = 0.5 * (trace + root)
    minor = 0.5 * (trace - root)
    if major <= 1e-12:
        eccentricity = 0.0
    else:
        eccentricity = sqrt(max(0.0, 1.0 - minor / major))
    orientation = 0.5 * atan2(2.0 * mu11, mu20 - mu02)
    return eccentricity, orientation


def dilate_mask(mask: np.ndarray, dilation: int = 0) -> np.ndarray:
    """Dilate a binary mask to grow detected regions."""
    if dilation <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def pad_bboxes(bboxes: np.ndarray, padding: int = 0, img_shape: tuple | None = None) -> np.ndarray:
    """Pad bounding boxes by a fixed number of pixels on each side."""
    if padding <= 0 or len(bboxes) == 0:
        return bboxes
    bboxes = bboxes.copy()
    bboxes[:, 0] -= padding  # x1
    bboxes[:, 1] -= padding  # y1
    bboxes[:, 2] += padding  # x2
    bboxes[:, 3] += padding  # y2
    if img_shape is not None:
        h, w = img_shape[:2]
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)
    return bboxes


def binary_mask_to_bboxes(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 50000,
    dilation: int = 0,
    box_padding: int = 0,
) -> np.ndarray:
    """Convert a binary mask to bounding boxes via connected components.

    Args:
        mask: Binary mask (0/1 or bool).
        min_area: Minimum component area to keep.
        max_area: Maximum component area to keep.
        dilation: Dilate the mask by this many pixels before extraction.
        box_padding: Pad each output box by this many pixels on each side.

    Returns:
        (N, 4) array of [x1, y1, x2, y2] bounding boxes.
    """
    if dilation > 0:
        mask = dilate_mask(mask, dilation)

    num_labels, labels, stats, _ = _connected_components(mask)

    bboxes = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bboxes.append(_component_bbox(stats[label]))

    result = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    if box_padding > 0 and len(result) > 0:
        result = pad_bboxes(result, box_padding, mask.shape)
    return result


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
    num_labels, labels, stats, centroids = _connected_components(mask)

    bboxes = []
    props = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        component_mask = labels == label
        eccentricity, orientation = _component_shape_props(component_mask)
        if eccentricity < min_eccentricity or eccentricity > max_eccentricity:
            continue

        bboxes.append(_component_bbox(stats[label]))
        props.append({
            "area": area,
            "eccentricity": eccentricity,
            "centroid": (float(centroids[label][1]), float(centroids[label][0])),
            "orientation": orientation,
        })

    return np.array(bboxes, dtype=np.float32).reshape(-1, 4), props


def binary_mask_to_bboxes_with_scores(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    threshold_map: np.ndarray | None = None,
    min_area: int = 10,
    max_area: int = 50000,
    dilation: int = 0,
    box_padding: int = 0,
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
    if dilation > 0:
        mask = dilate_mask(mask, dilation)

    intensity_image = intensity_image.astype(np.float32)
    num_labels, labels, stats, centroids = _connected_components(mask)

    bboxes = []
    raw_scores = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        component_mask = labels == label
        bboxes.append(_component_bbox(stats[label]))
        component_pixels = intensity_image[component_mask]

        if threshold_map is not None:
            # Peak intensity of the component
            peak = float(component_pixels.max()) if component_pixels.size else 0.0
            # Threshold at the centroid pixel
            cy = int(round(float(centroids[label][1])))
            cx = int(round(float(centroids[label][0])))
            cy = np.clip(cy, 0, threshold_map.shape[0] - 1)
            cx = np.clip(cx, 0, threshold_map.shape[1] - 1)
            local_thresh = float(threshold_map[cy, cx])
            score = peak / (local_thresh + 1e-9)
        else:
            # Normalise mean intensity to image scale
            scale = 255.0 if intensity_image.max() > 1.0 else 1.0
            score = (float(component_pixels.mean()) / scale) if component_pixels.size else 0.0

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

    if box_padding > 0 and len(bboxes_arr) > 0:
        bboxes_arr = pad_bboxes(bboxes_arr, box_padding, mask.shape)
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
