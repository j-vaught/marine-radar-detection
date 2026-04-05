"""Tile PPI frames into fixed-size patches for training and inference."""

from pathlib import Path

import cv2
import numpy as np


def tile_image(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[dict]:
    """Split an image into overlapping tiles.

    Args:
        image: Input image (H, W) or (H, W, C).
        tile_size: Size of each square tile.
        overlap: Fractional overlap between adjacent tiles.

    Returns:
        List of dicts with keys: 'tile', 'x_offset', 'y_offset'.
    """
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = image[y_start:y_end, x_start:x_end]

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                if image.ndim == 3:
                    padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                else:
                    padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            tiles.append({
                "tile": tile,
                "x_offset": x_start,
                "y_offset": y_start,
            })

    return tiles


def clip_boxes_to_tile(
    boxes: np.ndarray,
    x_offset: int,
    y_offset: int,
    tile_size: int,
    min_visibility: float = 0.3,
) -> np.ndarray:
    """Clip bounding boxes to a tile and filter by visibility.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] in full-image coordinates.
        x_offset: Tile x offset in the full image.
        y_offset: Tile y offset in the full image.
        tile_size: Size of the tile.
        min_visibility: Minimum fraction of box area that must remain visible.

    Returns:
        (M, 4) array of boxes in tile-local coordinates.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes.copy().astype(np.float32)
    original_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Shift to tile coordinates
    boxes[:, [0, 2]] -= x_offset
    boxes[:, [1, 3]] -= y_offset

    # Clip
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tile_size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, tile_size)

    # Filter by visibility
    clipped_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (clipped_areas / (original_areas + 1e-6)) >= min_visibility
    keep &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    return boxes[keep]


def save_tiles(
    tiles: list[dict],
    output_dir: str | Path,
    frame_id: str,
    fmt: str = "png",
) -> list[Path]:
    """Save tiles to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(tiles):
        path = output_dir / f"{frame_id}_tile{i:04d}.{fmt}"
        cv2.imwrite(str(path), t["tile"])
        paths.append(path)
    return paths
