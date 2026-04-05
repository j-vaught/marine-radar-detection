"""Qualitative visualization of detection results overlaid on radar imagery."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .plots import GARNET, ATLANTIC, ROSE, CONGAREE, BLACK, WHITE


def draw_detections(
    image: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray | None = None,
    color: tuple = (115, 0, 10),  # Garnet in BGR
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: Input image (grayscale or BGR).
        bboxes: (N, 4) array of [x1, y1, x2, y2].
        scores: Optional confidence scores.
        color: BGR color tuple.
        thickness: Line thickness.
        label: Optional method label.

    Returns:
        Image with drawn detections.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        if scores is not None and i < len(scores):
            text = f"{scores[i]:.2f}"
            cv2.putText(vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if label:
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return vis


def comparison_grid(
    image: np.ndarray,
    detections: dict[str, tuple[np.ndarray, np.ndarray | None]],
    gt_boxes: np.ndarray | None = None,
    output_path: str | Path | None = None,
    cols: int = 4,
):
    """Create a grid comparing detections from multiple methods.

    Args:
        image: Original radar image.
        detections: Dict mapping method name -> (bboxes, scores).
        gt_boxes: Optional ground truth boxes.
        output_path: Path to save figure.
        cols: Number of columns in the grid.
    """
    n_methods = len(detections)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    method_colors = [
        (115, 0, 10),    # Garnet
        (204, 46, 64),   # Rose
        (70, 106, 159),  # Atlantic
        (31, 65, 77),    # Congaree
        (101, 120, 11),  # Horseshoe
        (164, 145, 55),  # Honeycomb
        (54, 54, 54),    # 90% Black
        (92, 92, 92),    # 70% Black
        (206, 211, 24),  # Grass
        (103, 97, 86),   # Warm Grey
        (0, 0, 0),       # Black
    ]

    for i, (method_name, (bboxes, scores)) in enumerate(detections.items()):
        vis = draw_detections(
            image, bboxes, scores,
            color=method_colors[i % len(method_colors)],
            label=method_name,
        )

        if gt_boxes is not None:
            for gt in gt_boxes:
                x1, y1, x2, y2 = gt.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(method_name, fontsize=10)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
