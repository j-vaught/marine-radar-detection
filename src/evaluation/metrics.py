"""COCO-style evaluation metrics for object detection."""

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(
    gt_annotations: dict | str | Path,
    predictions: list[dict],
    iou_type: str = "bbox",
) -> dict:
    """Evaluate predictions using COCO metrics.

    Args:
        gt_annotations: COCO ground truth (dict or path to JSON).
        predictions: List of dicts with keys:
            'image_id', 'category_id', 'bbox' [x,y,w,h], 'score'.
        iou_type: 'bbox' or 'segm'.

    Returns:
        Dict with AP@0.5, AP@0.75, AP@[0.50:0.95], AR metrics by size.
    """
    if isinstance(gt_annotations, (str, Path)):
        coco_gt = COCO(str(gt_annotations))
    else:
        # In-memory
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_annotations, f)
            tmp_path = f.name
        coco_gt = COCO(tmp_path)

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "AP@0.50:0.95": coco_eval.stats[0],
        "AP@0.50": coco_eval.stats[1],
        "AP@0.75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    return results


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h] format."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def format_predictions_coco(
    image_id: int,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray | None = None,
    category_id: int = 1,
) -> list[dict]:
    """Format detection results as COCO prediction entries."""
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    preds = []
    for i in range(len(boxes_xywh)):
        preds.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": boxes_xywh[i].tolist(),
            "score": float(scores[i]) if scores is not None else 1.0,
        })
    return preds
