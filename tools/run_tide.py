#!/usr/bin/env python3
"""Run TIDE on bbox-only COCO ground truth and prediction JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tidecv import (
    BackgroundError,
    BoxError,
    ClassError,
    Data,
    DuplicateError,
    MissedError,
    OtherError,
    TIDE,
)

MAIN_ERROR_TYPES = [
    ClassError,
    BoxError,
    OtherError,
    DuplicateError,
    BackgroundError,
    MissedError,
]


def load_bbox_coco_gt(path: Path, max_dets: int = 100) -> tuple[Data, dict[str, Any]]:
    with path.open() as f:
        coco = json.load(f)

    data = Data(str(path), max_dets=max_dets)

    for category in coco.get("categories", []):
        data.add_class(int(category["id"]), category.get("name", f"class_{category['id']}"))

    for image in coco.get("images", []):
        data.add_image(int(image["id"]), image.get("file_name", str(image["id"])))

    for ann in coco.get("annotations", []):
        data.add_ground_truth(
            int(ann["image_id"]),
            int(ann["category_id"]),
            box=ann["bbox"],
        )

    return data, coco


def load_bbox_coco_predictions(
    path: Path, max_dets: int = 100
) -> tuple[Data, list[dict[str, Any]]]:
    with path.open() as f:
        preds = json.load(f)

    data = Data(str(path), max_dets=max_dets)

    for pred in preds:
        data.add_detection(
            int(pred["image_id"]),
            int(pred["category_id"]),
            float(pred["score"]),
            box=pred["bbox"],
        )

    return data, preds


def build_summary(
    gt_path: Path,
    pred_path: Path,
    name: str,
    max_dets: int = 100,
) -> tuple[TIDE, dict[str, Any]]:
    gt_data, gt_json = load_bbox_coco_gt(gt_path, max_dets=max_dets)
    pred_data, pred_json = load_bbox_coco_predictions(pred_path, max_dets=max_dets)

    tide = TIDE()
    tide.evaluate_range(gt_data, pred_data, name=name)

    run = tide.runs[name]
    threshold_runs = tide.run_thresholds[name]

    main_errors = tide.get_main_errors()[name]
    special_errors = tide.get_special_errors()[name]

    summary = {
        "name": name,
        "gt_path": str(gt_path.resolve()),
        "pred_path": str(pred_path.resolve()),
        "max_dets": max_dets,
        "images": len(gt_json.get("images", [])),
        "gt_boxes": len(gt_json.get("annotations", [])),
        "pred_boxes": len(pred_json),
        "ap_mean": sum(th.ap for th in threshold_runs) / len(threshold_runs),
        "ap_by_threshold": {str(int(th.pos_thresh * 100)): th.ap for th in threshold_runs},
        "main_errors": main_errors,
        "special_errors": special_errors,
        "main_error_counts": {err.short_name: len(run.error_dict[err]) for err in MAIN_ERROR_TYPES},
    }

    return tide, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, type=Path, help="COCO ground-truth JSON")
    ap.add_argument("--pred", required=True, type=Path, help="COCO predictions JSON")
    ap.add_argument("--name", required=True, help="Run label for TIDE outputs")
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="Directory for summary.json and TIDE plot(s)",
    )
    ap.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="Max detections per image to evaluate (default: 100, COCO-style)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the TIDE summary plot",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tide, summary = build_summary(
        gt_path=args.gt,
        pred_path=args.pred,
        name=args.name,
        max_dets=args.max_dets,
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        if not args.no_plot:
            tide.plot(str(args.out_dir))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
