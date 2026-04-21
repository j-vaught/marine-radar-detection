"""Correctness gate: GPU CA / OS must match the CPU reference on bboxes + AP50.

Run with ``python -m src.cfar.gpu.test_oracle``.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..detector import detect_cfar
from .ca import ca_background, ca_alpha
from .data import load_corpus
from .eval_coco import evaluate
from .os_cfar import os_rank_maps, os_alpha
from .postproc import mask_to_preds


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_sets(gpu_preds, cpu_preds):
    if not gpu_preds and not cpu_preds:
        return 1.0
    if not gpu_preds or not cpu_preds:
        return 0.0
    matched = 0
    used = set()
    ious = []
    for g in gpu_preds:
        best, best_i = -1.0, -1
        for i, c in enumerate(cpu_preds):
            if i in used: continue
            iou = _bbox_iou(g["bbox"], c["bbox"])
            if iou > best:
                best, best_i = iou, i
        if best_i >= 0 and best >= 0.5:
            used.add(best_i); matched += 1; ious.append(best)
    return (matched / max(len(gpu_preds), len(cpu_preds)), float(np.mean(ious)) if ious else 0.0)


def run(n_images: int, seed: int, method: str) -> None:
    corpus = load_corpus(device="cuda:1")
    rng = np.random.default_rng(seed)
    idx = rng.choice(corpus.N, size=n_images, replace=False)

    guard, ref, pfa, min_area, k_frac = 15, 30, 1e-3, 50, 0.75

    gpu_preds: list[dict] = []
    cpu_preds: list[dict] = []

    total_match = 0
    total_count = 0

    for i in idx:
        img_gpu = corpus.images[i:i + 1]                     # [1, 1, H, W]
        circle = corpus.circle_mask
        intensity_np = (img_gpu * circle).squeeze().cpu().numpy()

        if method == "ca":
            bg = ca_background(img_gpu, guard, ref)
            thr_gpu = ca_alpha(pfa, guard, ref) * bg
        else:
            rank = os_rank_maps(img_gpu, guard, ref, k_fracs=(k_frac,))
            thr_gpu = os_alpha(pfa, guard, ref) * rank[0:1]  # [1, 1, H, W]

        mask_gpu = ((img_gpu > thr_gpu) & (img_gpu > 0)).squeeze(0).squeeze(0)
        mask_np = mask_gpu.cpu().numpy().astype(np.uint8)
        thr_np = thr_gpu.squeeze().cpu().numpy()

        preds_by_min = mask_to_preds(mask_np, intensity_np, thr_np,
                                     image_id=int(corpus.image_ids[i]),
                                     min_areas=(min_area,))
        these_gpu = preds_by_min[min_area]

        # CPU reference
        bboxes, _, scores = detect_cfar(
            intensity_np.astype(np.float32),
            (guard, guard), (ref, ref), pfa,
            mode="CA" if method == "ca" else "OS",
            k_os=k_frac, min_area=min_area,
        )
        these_cpu = []
        for b, s in zip(bboxes, scores):
            these_cpu.append({
                "image_id": int(corpus.image_ids[i]), "category_id": 1,
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(s),
            })

        m = _match_sets(these_gpu, these_cpu)
        if isinstance(m, tuple):
            rate, mean_iou = m
        else:
            rate, mean_iou = m, 1.0
        total_match += len(these_gpu)
        total_count += len(these_cpu)
        print(f"  img {int(corpus.image_ids[i]):4d}  gpu={len(these_gpu):3d}  cpu={len(these_cpu):3d}  "
              f"match={rate:.3f}  meanIoU={mean_iou:.3f}")

        gpu_preds.extend(these_gpu)
        cpu_preds.extend(these_cpu)

    ids = corpus.image_ids[idx]
    gpu_m = evaluate(gpu_preds, corpus.coco, ids)
    cpu_m = evaluate(cpu_preds, corpus.coco, ids)
    print(f"\n== {method.upper()} on {n_images} images ==")
    print(f"GPU AP50={gpu_m['AP50']:.4f}  FA/img={gpu_m['fa_per_image']:.2f}")
    print(f"CPU AP50={cpu_m['AP50']:.4f}  FA/img={cpu_m['fa_per_image']:.2f}")
    print(f"ΔAP50={abs(gpu_m['AP50'] - cpu_m['AP50']):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["ca", "os"], default="ca")
    args = ap.parse_args()
    run(args.n, args.seed, args.method)


if __name__ == "__main__":
    main()
