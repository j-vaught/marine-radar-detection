"""Radar PPI circular mask to exclude UI elements from detection."""

import cv2
import numpy as np


# Default radar circle parameters (measured directly from DLR PPI captures
# via the interactive crop tuner).  Note that ``polar_to_pixel`` in
# ``polar_to_coco.py`` uses the naive image-centred formula
# ``(min(W, H) / 2) * 0.92`` which gives (525, 512, 471); the measured display
# is offset a few pixels to the lower-right, so GT bounding boxes produced by
# that formula sit a few pixels off the real radar returns. Using the measured
# values here ensures the mask matches the actual display.
DEFAULT_CENTER_X = 531
DEFAULT_CENTER_Y = 515
DEFAULT_RADIUS   = 466


def create_radar_mask(
    height: int = 1024,
    width: int = 1050,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    """Create a circular mask for the radar PPI display area.

    Args:
        height: Image height.
        width: Image width.
        center: (x, y) center of the radar circle. Defaults to image center.
        radius: Radius of the radar circle. Defaults to min(w,h)/2 * 0.92.

    Returns:
        Binary mask (uint8, 0 or 255).
    """
    if center is None:
        center = (DEFAULT_CENTER_X, DEFAULT_CENTER_Y)
    if radius is None:
        radius = DEFAULT_RADIUS

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def apply_radar_mask(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Zero out pixels outside the radar circle.

    Args:
        image: Input image (grayscale or BGR).
        mask: Pre-computed mask. If None, creates one from image dimensions.

    Returns:
        Masked image.
    """
    if mask is None:
        h, w = image.shape[:2]
        mask = create_radar_mask(h, w)

    if image.ndim == 3:
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        return cv2.bitwise_and(image, mask)


def filter_boxes_by_mask(
    bboxes: np.ndarray,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
    img_width: int = 1050,
    img_height: int = 1024,
) -> np.ndarray:
    """Filter bounding boxes to only keep those whose center is inside the radar circle.

    Args:
        bboxes: (N, 4+) array with at least [x1, y1, x2, y2] or COCO [x, y, w, h].
        center: Radar circle center (x, y).
        radius: Radar circle radius.
        img_width: Image width for default center/radius.
        img_height: Image height for default center/radius.

    Returns:
        Boolean mask of which boxes to keep.
    """
    if center is None:
        center = (DEFAULT_CENTER_X, DEFAULT_CENTER_Y)
    if radius is None:
        radius = DEFAULT_RADIUS

    if len(bboxes) == 0:
        return np.array([], dtype=bool)

    # Compute box centers
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # works for both xyxy and xywh
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

    # Distance from radar center
    dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)

    return dist <= radius
