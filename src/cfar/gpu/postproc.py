"""Mask → bboxes → scores via cv2.connectedComponentsWithStats (much faster
than scipy.ndimage.label).  Scoring replicates the CPU reference:
score = peak_intensity / threshold_at_centroid, normalized per image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_CCL_ALGO = getattr(cv2, "CCL_BBDT", getattr(cv2, "CCL_DEFAULT", -1))
_HAS_CC_ALGO = hasattr(cv2, "connectedComponentsWithStatsWithAlgorithm")


@dataclass
class PackedPreds:
    image_ids: np.ndarray
    boxes_xywh: np.ndarray
    scores: np.ndarray

    def to_coco(self) -> list[dict]:
        if self.image_ids.size == 0:
            return []
        return [
            {
                "image_id": int(self.image_ids[i]),
                "category_id": 1,
                "bbox": [
                    float(self.boxes_xywh[i, 0]),
                    float(self.boxes_xywh[i, 1]),
                    float(self.boxes_xywh[i, 2]),
                    float(self.boxes_xywh[i, 3]),
                ],
                "score": float(self.scores[i]),
            }
            for i in range(self.image_ids.shape[0])
        ]


def empty_packed_preds() -> PackedPreds:
    return PackedPreds(
        image_ids=np.zeros(0, dtype=np.int64),
        boxes_xywh=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros(0, dtype=np.float32),
    )


def concat_packed_preds(chunks: list[PackedPreds]) -> PackedPreds:
    if not chunks:
        return empty_packed_preds()
    non_empty = [c for c in chunks if c.image_ids.size > 0]
    if not non_empty:
        return empty_packed_preds()
    return PackedPreds(
        image_ids=np.concatenate([c.image_ids for c in non_empty], axis=0),
        boxes_xywh=np.concatenate([c.boxes_xywh for c in non_empty], axis=0),
        scores=np.concatenate([c.scores for c in non_empty], axis=0),
    )


def mask_to_pred_arrays(
    mask_np: np.ndarray,
    intensity_np: np.ndarray,
    threshold_np: np.ndarray | None,
    image_id: int,
    min_areas: tuple[int, ...] = (20, 50, 100),
    *,
    stat_np: np.ndarray | None = None,
    alpha: float | None = None,
) -> dict[int, PackedPreds]:
    """Return packed predictions per ``min_area`` from one image's mask."""
    mask_np = mask_np.astype(np.uint8, copy=False)
    if _HAS_CC_ALGO:
        n_comp, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
            mask_np,
            8,
            cv2.CV_32S,
            _CCL_ALGO,
        )
    else:
        n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if n_comp <= 1:
        return {m: empty_packed_preds() for m in min_areas}

    areas = stats[1:, cv2.CC_STAT_AREA]
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    # Per-component peak intensity via one vectorized pass over the label image.
    peaks = np.zeros(n_comp, dtype=np.float32)
    flat_labels = labels.reshape(-1)
    valid = flat_labels > 0
    np.maximum.at(peaks, flat_labels[valid], intensity_np.reshape(-1)[valid])
    peaks = peaks[1:]

    cx = np.clip(centroids[1:, 0].round().astype(np.int32), 0, intensity_np.shape[1] - 1)
    cy = np.clip(centroids[1:, 1].round().astype(np.int32), 0, intensity_np.shape[0] - 1)
    if threshold_np is not None:
        local_thr = threshold_np[cy, cx]
    else:
        if stat_np is None or alpha is None:
            raise ValueError("Pass threshold_np or both stat_np and alpha.")
        local_thr = alpha * stat_np[cy, cx]
    raw_score = peaks / (local_thr + 1e-9)

    out: dict[int, PackedPreds] = {}
    for m in min_areas:
        keep = areas >= m
        if not keep.any():
            out[m] = empty_packed_preds()
            continue
        scores = raw_score[keep]
        s_max = scores.max()
        if s_max > 0:
            scores = np.clip(scores / s_max, 0.0, 1.0)
        else:
            scores = np.clip(scores, 0.0, 1.0)
        boxes = np.stack([xs[keep], ys[keep], ws[keep], hs[keep]], axis=1).astype(np.float32, copy=False)
        image_ids = np.full(int(keep.sum()), image_id, dtype=np.int64)
        out[m] = PackedPreds(image_ids=image_ids, boxes_xywh=boxes, scores=scores.astype(np.float32, copy=False))
    return out


def mask_to_preds(
    mask_np: np.ndarray,
    intensity_np: np.ndarray,
    threshold_np: np.ndarray | None,
    image_id: int,
    min_areas: tuple[int, ...] = (20, 50, 100),
    *,
    stat_np: np.ndarray | None = None,
    alpha: float | None = None,
) -> dict[int, list[dict]]:
    """Return COCO-style predictions per ``min_area`` from one image's mask."""
    packed = mask_to_pred_arrays(
        mask_np,
        intensity_np,
        threshold_np,
        image_id,
        min_areas,
        stat_np=stat_np,
        alpha=alpha,
    )
    return {m: packed[m].to_coco() for m in min_areas}
