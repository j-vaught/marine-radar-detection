"""Common COCO bounding-box evaluation for validation and fixed test sets."""

import contextlib
import copy
import io


def evaluate_coco(gt: dict, predictions: list[dict]) -> dict:
    """Evaluate a single foreground class with COCO maxDets of 100.

    Counts include all raw predictions. Undefined COCO metrics are reported as -1.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(gt["categories"]) != 1:
        raise ValueError("Expected exactly one foreground category")
    category = gt["categories"][0]["id"]
    image_ids = {im["id"] for im in gt["images"]}
    if len(image_ids) != len(gt["images"]):
        raise ValueError("Duplicate image IDs")
    if any(p["image_id"] not in image_ids or p["category_id"] != category for p in predictions):
        raise ValueError("Prediction refers to an unknown image or category")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])
        coco_gt.createIndex()
        if predictions:
            coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": copy.deepcopy(gt["images"]),
                "categories": copy.deepcopy(gt["categories"]),
                "annotations": [],
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = [category]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    count = len(predictions)
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ar100": float(evaluator.stats[8]),
        "predictions": count,
        "preds_per_image": count / len(image_ids) if image_ids else 0.0,
    }
