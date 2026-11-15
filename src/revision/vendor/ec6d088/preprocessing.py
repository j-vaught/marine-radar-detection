"""Data preprocessing for DLR DAAN and HAXR marine radar datasets."""

import json
from pathlib import Path

import cv2
import numpy as np


def load_ppi_image(path: str | Path, normalize: bool = True) -> np.ndarray:
    """Load a PPI radar image as grayscale.

    Args:
        path: Path to the image file.
        normalize: If True, normalize to [0, 1] float32.

    Returns:
        Grayscale image array.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img


def apply_log_transform(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-scale intensity compression for radar imagery."""
    return np.log1p(image / eps).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def polar_to_cartesian(
    ppi: np.ndarray, output_size: int | None = None
) -> np.ndarray:
    """Convert polar PPI scan to Cartesian coordinates.

    Args:
        ppi: Input PPI image (rows=range, cols=azimuth or vice versa).
        output_size: Size of the output square image.

    Returns:
        Cartesian-mapped image.
    """
    h, w = ppi.shape[:2]
    if output_size is None:
        output_size = max(h, w)
    center = (output_size // 2, output_size // 2)
    max_radius = output_size // 2

    cartesian = np.zeros((output_size, output_size), dtype=ppi.dtype)
    for y in range(output_size):
        for x in range(output_size):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx) % (2 * np.pi)

            ri = int(r / max_radius * h)
            ti = int(theta / (2 * np.pi) * w)

            if 0 <= ri < h and 0 <= ti < w:
                cartesian[y, x] = ppi[ri, ti]

    return cartesian


def load_coco_annotations(ann_path: str | Path) -> dict:
    """Load COCO-format annotations."""
    with open(ann_path) as f:
        return json.load(f)
