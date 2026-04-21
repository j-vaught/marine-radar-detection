use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::image_io::{CocoDataset, CocoPrediction};

pub type GtIndex = HashMap<u64, Vec<[f32; 4]>>;

const IOU_THRESHOLDS: [f32; 10] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub n_images: usize,
    pub n_pred: usize,
    pub ap: f32,
    pub ap50: f32,
    pub ap75: f32,
    pub recall_at_p90: f32,
    pub fa_per_image: f32,
}

#[derive(Clone, Copy, Debug)]
struct EvalPoint {
    score: f32,
    tp: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThresholdMetrics {
    ap: f32,
    recall_at_p90: f32,
    matched: usize,
    detections: usize,
}

pub fn build_gt_index(dataset: &CocoDataset) -> GtIndex {
    let mut gt = HashMap::<u64, Vec<[f32; 4]>>::new();
    for ann in &dataset.annotations {
        let [x, y, w, h] = ann.bbox;
        gt.entry(ann.image_id)
            .or_default()
            .push([x, y, x + w, y + h]);
    }
    gt
}

pub fn evaluate_predictions(
    preds: &[CocoPrediction],
    dataset: &CocoDataset,
    image_ids: &[u64],
) -> Metrics {
    let gt_index = build_gt_index(dataset);
    evaluate_predictions_with_index(preds, &gt_index, image_ids)
}

pub fn evaluate_predictions_with_index(
    preds: &[CocoPrediction],
    gt_index: &GtIndex,
    image_ids: &[u64],
) -> Metrics {
    let n_images = image_ids.len();
    if n_images == 0 {
        return Metrics::default();
    }

    let mut ap_sum = 0.0f32;
    let mut ap50 = 0.0f32;
    let mut ap75 = 0.0f32;
    let mut recall_at_p90 = 0.0f32;
    let mut fa_per_image = 0.0f32;

    for &thr in &IOU_THRESHOLDS {
        let tm = evaluate_threshold(preds, gt_index, image_ids, thr);
        ap_sum += tm.ap;
        if (thr - 0.50).abs() < 1.0e-6 {
            ap50 = tm.ap;
            recall_at_p90 = tm.recall_at_p90;
            fa_per_image = (tm.detections.saturating_sub(tm.matched)) as f32 / n_images as f32;
        }
        if (thr - 0.75).abs() < 1.0e-6 {
            ap75 = tm.ap;
        }
    }

    Metrics {
        n_images,
        n_pred: preds
            .iter()
            .filter(|pred| image_ids.contains(&pred.image_id))
            .count(),
        ap: ap_sum / IOU_THRESHOLDS.len() as f32,
        ap50,
        ap75,
        recall_at_p90,
        fa_per_image,
    }
}

pub fn fast_ap50(preds: &[CocoPrediction], gt_index: &GtIndex, image_ids: &[u64]) -> f32 {
    evaluate_threshold(preds, gt_index, image_ids, 0.50).ap
}

fn evaluate_threshold(
    preds: &[CocoPrediction],
    gt_index: &GtIndex,
    image_ids: &[u64],
    iou_thr: f32,
) -> ThresholdMetrics {
    let keep_ids = image_ids.iter().copied().collect::<HashSet<_>>();
    let total_gt = image_ids
        .iter()
        .map(|image_id| gt_index.get(image_id).map_or(0usize, Vec::len))
        .sum::<usize>();
    if total_gt == 0 {
        return ThresholdMetrics::default();
    }

    let mut by_image = HashMap::<u64, Vec<&CocoPrediction>>::new();
    for pred in preds {
        if keep_ids.contains(&pred.image_id) {
            by_image.entry(pred.image_id).or_default().push(pred);
        }
    }
    for entries in by_image.values_mut() {
        entries.sort_by(|a, b| compare_desc(a.score, b.score));
    }

    let mut points = Vec::<EvalPoint>::new();
    let mut matched_total = 0usize;
    for &image_id in image_ids {
        let gt = gt_index.get(&image_id).map_or(&[][..], Vec::as_slice);
        let mut matched = vec![false; gt.len()];
        let detections = by_image.get(&image_id).map_or(&[][..], Vec::as_slice);
        for pred in detections {
            let pred_xyxy = bbox_xyxy(pred.bbox);
            let mut best_iou = 0.0f32;
            let mut best_gt = None::<usize>;
            for (gt_idx, gt_box) in gt.iter().enumerate() {
                if matched[gt_idx] {
                    continue;
                }
                let iou = pair_iou(pred_xyxy, *gt_box);
                if iou > best_iou {
                    best_iou = iou;
                    best_gt = Some(gt_idx);
                }
            }
            let is_tp = best_iou >= iou_thr && best_gt.is_some();
            if let Some(gt_idx) = best_gt.filter(|_| is_tp) {
                matched[gt_idx] = true;
                matched_total += 1;
            }
            points.push(EvalPoint {
                score: pred.score,
                tp: is_tp,
            });
        }
    }

    if points.is_empty() {
        return ThresholdMetrics::default();
    }
    points.sort_by(|a, b| compare_desc(a.score, b.score));

    let mut precision = Vec::<f32>::with_capacity(points.len());
    let mut recall = Vec::<f32>::with_capacity(points.len());
    let mut tp_cum = 0usize;
    let mut fp_cum = 0usize;
    for point in &points {
        if point.tp {
            tp_cum += 1;
        } else {
            fp_cum += 1;
        }
        let denom = (tp_cum + fp_cum).max(1) as f32;
        precision.push(tp_cum as f32 / denom);
        recall.push(tp_cum as f32 / total_gt as f32);
    }

    let mut ap = 0.0f32;
    for step in 0..=100 {
        let rec_thr = step as f32 / 100.0;
        let mut max_precision = 0.0f32;
        for (&p, &r) in precision.iter().zip(recall.iter()) {
            if r >= rec_thr && p > max_precision {
                max_precision = p;
            }
        }
        ap += max_precision;
    }
    ap /= 101.0;

    let mut recall_at_p90 = 0.0f32;
    for (&p, &r) in precision.iter().zip(recall.iter()) {
        if p >= 0.90 && r > recall_at_p90 {
            recall_at_p90 = r;
        }
    }

    ThresholdMetrics {
        ap,
        recall_at_p90,
        matched: matched_total,
        detections: points.len(),
    }
}

#[inline(always)]
fn bbox_xyxy(bbox_xywh: [f32; 4]) -> [f32; 4] {
    [
        bbox_xywh[0],
        bbox_xywh[1],
        bbox_xywh[0] + bbox_xywh[2],
        bbox_xywh[1] + bbox_xywh[3],
    ]
}

#[inline(always)]
fn pair_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

#[inline(always)]
fn compare_desc(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}
