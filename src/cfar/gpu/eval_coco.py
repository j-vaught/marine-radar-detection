"""COCO eval wrappers.

``fast_ap50`` is a numpy-only AP50 used inside the grid-search inner loop.
``evaluate`` wraps pycocotools and is reserved for final test scoring.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .postproc import PackedPreds


def build_gt_index(full_coco: dict) -> dict[int, np.ndarray]:
    """Return {image_id: (K, 4) xyxy array} for fast per-image GT lookup."""
    gt: dict[int, list[list[float]]] = {}
    for a in full_coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(int(a["image_id"]), []).append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=np.float32) if v else np.zeros((0, 4), dtype=np.float32)
            for k, v in gt.items()}


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (M, 4) and (N, 4) xyxy arrays → (M, N)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def fast_ap50(
    preds: list[dict] | PackedPreds,
    gt_index: dict[int, np.ndarray],
    image_ids: np.ndarray,
    iou_thr: float = 0.5,
) -> float:
    """Numpy AP@IoU=0.5 (COCO-style 101-point interpolation), per-image greedy match."""
    image_ids = set(int(i) for i in image_ids)

    packed: PackedPreds | None = preds if isinstance(preds, PackedPreds) else None
    by_img: dict[int, list[dict]] = {}
    if packed is None:
        for p in preds:
            iid = int(p["image_id"])
            if iid in image_ids:
                by_img.setdefault(iid, []).append(p)

    total_gt = sum(gt_index.get(i, np.zeros((0, 4))).shape[0] for i in image_ids)
    if total_gt == 0:
        return 0.0

    all_scores: list[float] = []
    all_tp: list[int] = []
    all_fp: list[int] = []
    for iid in image_ids:
        gt = gt_index.get(iid, np.zeros((0, 4), dtype=np.float32))
        if packed is None:
            dets = by_img.get(iid, [])
            if not dets:
                continue
            scores = np.array([d["score"] for d in dets], dtype=np.float32)
            xywh = np.array([d["bbox"] for d in dets], dtype=np.float32)
        else:
            keep = packed.image_ids == iid
            if not keep.any():
                continue
            scores = packed.scores[keep]
            xywh = packed.boxes_xywh[keep]
        xyxy = np.stack([xywh[:, 0], xywh[:, 1],
                         xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], axis=1)
        order = np.argsort(-scores)
        xyxy = xyxy[order]; scores = scores[order]
        ious = _pairwise_iou(xyxy, gt)           # [M, G]
        matched = np.zeros(gt.shape[0], dtype=bool)
        for i in range(xyxy.shape[0]):
            if gt.shape[0] == 0:
                all_tp.append(0); all_fp.append(1); all_scores.append(float(scores[i])); continue
            row = ious[i].copy()
            row[matched] = 0.0
            j = int(row.argmax())
            if row[j] >= iou_thr:
                matched[j] = True
                all_tp.append(1); all_fp.append(0)
            else:
                all_tp.append(0); all_fp.append(1)
            all_scores.append(float(scores[i]))

    if not all_scores:
        return 0.0
    scores = np.array(all_scores); tp = np.array(all_tp); fp = np.array(all_fp)
    order = np.argsort(-scores)
    tp = np.cumsum(tp[order]); fp = np.cumsum(fp[order])
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    rec_thrs = np.linspace(0.0, 1.0, 101)
    # For each recall threshold, take max precision at recall >= threshold
    ap = 0.0
    for t in rec_thrs:
        mask = recall >= t
        ap += precision[mask].max() if mask.any() else 0.0
    return float(ap / 101.0)


def _subset_gt(full_coco: dict, image_ids: np.ndarray) -> dict:
    keep_ids = set(int(i) for i in image_ids)
    out = {
        "info": full_coco.get("info", {}),
        "licenses": full_coco.get("licenses", []),
        "categories": full_coco["categories"],
        "images": [im for im in full_coco["images"] if int(im["id"]) in keep_ids],
        "annotations": [a for a in full_coco["annotations"] if int(a["image_id"]) in keep_ids],
    }
    return out


def evaluate(
    preds: list[dict] | PackedPreds,
    full_coco: dict,
    image_ids: np.ndarray,
) -> dict:
    """Score preds against the subset of the GT restricted to ``image_ids``."""
    gt_json = _subset_gt(full_coco, image_ids)

    if isinstance(preds, PackedPreds):
        preds = preds.to_coco()

    n_images = len(image_ids)
    n_pred = len(preds)

    # pycocotools is chatty; silence it.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gt_path = td / "gt.json"
        pred_path = td / "pred.json"
        gt_path.write_text(json.dumps(gt_json))
        pred_path.write_text(json.dumps(preds if preds else [{
            "image_id": int(image_ids[0]), "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.0,
        }]))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gt = COCO(str(gt_path))
            dt = gt.loadRes(str(pred_path))
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = [int(i) for i in image_ids]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()

    # COCOeval stats order:
    # 0: AP IoU=.50:.95  1: AP50  2: AP75
    # 3: APsmall  4: APmedium  5: APlarge
    # 6: AR@1  7: AR@10  8: AR@100  9-11: ARsmall/med/large
    ap = float(ev.stats[0]); ap50 = float(ev.stats[1]); ap75 = float(ev.stats[2])

    # recall at precision >= 0.9 on IoU=0.5 curve
    # ev.eval['precision'] shape: [T, R, K, A, M] = [10, 101, 1, 4, 3] for us
    precision = ev.eval["precision"]    # iouThr x recThr x cls x areaRng x maxDet
    recall_grid = np.array(ev.params.recThrs)
    # IoU=0.5 is index 0; class 0; areaRng 'all' is index 0; maxDet=100 is last
    p_curve = precision[0, :, 0, 0, -1]    # [101]
    valid = p_curve >= 0.9
    recall_at_p90 = float(recall_grid[valid].max()) if valid.any() else 0.0

    # FA per image: false positives at matched top-100 detections
    # Use the "unused" dt entries from ev.evalImgs under IoU=0.5.
    # Simpler: TP at IoU=0.5 is the recall integrated times total GT — but we
    # don't want a rate.  Define FA/image = (total dets - matched dets) / n_images.
    matched = 0
    total = 0
    for e in ev.evalImgs:
        if e is None: continue
        if e["aRng"] != ev.params.areaRng[0]: continue  # only 'all'
        dt_scores = np.array(e["dtScores"])
        dt_matches = np.array(e["dtMatches"][0])         # IoU=0.5 row
        total += len(dt_scores)
        matched += int((dt_matches > 0).sum())
    fa_per_image = float((total - matched) / max(n_images, 1))

    return {
        "n_images": int(n_images),
        "n_pred": int(n_pred),
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "recall_at_p90": recall_at_p90,
        "fa_per_image": fa_per_image,
    }
