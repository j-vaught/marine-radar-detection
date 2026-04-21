"""Benchmark baseline vs optimized CA- and OS-CFAR GPU kernels.

Example:
    .venv/bin/python -m src.cfar.gpu.benchmark --device cuda:0
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from .ca import ca_alpha, ca_background
from .data import load_corpus
from .os_cfar import os_alpha, os_rank_maps, os_rank_maps_hist
from .postproc import concat_packed_preds, mask_to_pred_arrays, mask_to_preds
from .triton_mask import build_threshold_masks

PFAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
OS_KS = (0.5, 0.75, 0.9)
CA_PIPELINE_IMAGES = 8
OS_PIPELINE_IMAGES = 4


def _timeit(fn, warmup: int = 1, repeat: int = 3) -> tuple[float, object]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return min(times), out


def _old_mask_to_preds(
    mask_np: np.ndarray,
    intensity_np: np.ndarray,
    stat_np: np.ndarray,
    alpha: float,
    image_id: int,
    min_areas: tuple[int, ...] = (20, 50, 100),
) -> dict[int, list[dict]]:
    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8), connectivity=8,
    )
    if n_comp <= 1:
        return {m: [] for m in min_areas}

    areas = stats[1:, cv2.CC_STAT_AREA]
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    peaks = np.zeros(n_comp - 1, dtype=np.float32)
    for i in range(n_comp - 1):
        x, y, w, h = xs[i], ys[i], ws[i], hs[i]
        sub_lab = labels[y:y + h, x:x + w]
        sub_int = intensity_np[y:y + h, x:x + w]
        peaks[i] = sub_int[sub_lab == (i + 1)].max() if (sub_lab == (i + 1)).any() else 0.0

    cx = np.clip(centroids[1:, 0].round().astype(np.int32), 0, intensity_np.shape[1] - 1)
    cy = np.clip(centroids[1:, 1].round().astype(np.int32), 0, intensity_np.shape[0] - 1)
    raw_score = peaks / (alpha * stat_np[cy, cx] + 1e-9)

    out: dict[int, list[dict]] = {}
    for m in min_areas:
        keep = areas >= m
        if not keep.any():
            out[m] = []
            continue
        scores = raw_score[keep]
        s_max = scores.max()
        if s_max > 0:
            scores = np.clip(scores / s_max, 0.0, 1.0)
        else:
            scores = np.clip(scores, 0.0, 1.0)
        kept_x = xs[keep]
        kept_y = ys[keep]
        kept_w = ws[keep]
        kept_h = hs[keep]
        out[m] = [
            {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(kept_x[i]), float(kept_y[i]), float(kept_w[i]), float(kept_h[i])],
                "score": float(scores[i]),
            }
            for i in range(int(keep.sum()))
        ]
    return out


def benchmark(device: str, ca_batch: int, os_batch: int, guard: int, ref: int) -> None:
    corpus = load_corpus(device=device)
    ca_imgs = corpus.images[:ca_batch]
    os_imgs = corpus.images[:os_batch]
    ca_alphas = torch.tensor([ca_alpha(pfa, guard, ref) for pfa in PFAS], device=ca_imgs.device)
    os_alphas = torch.tensor([os_alpha(pfa, guard, ref) for pfa in PFAS], device=os_imgs.device)

    def ca_baseline():
        bg = ca_background(ca_imgs, guard, ref)
        return torch.stack(
            [((ca_imgs > (alpha * bg)) & (ca_imgs > 0)).to(torch.uint8) for alpha in ca_alphas],
            dim=2,
        )

    def ca_optimized():
        bg = ca_background(ca_imgs, guard, ref)
        return build_threshold_masks(ca_imgs, bg, ca_alphas)

    def os_baseline():
        ranks: list[torch.Tensor] = []
        for i in range(os_imgs.shape[0]):
            rank = os_rank_maps(os_imgs[i:i + 1], guard, ref, k_fracs=OS_KS)
            ranks.append(rank.squeeze(1).unsqueeze(0))
        rank_maps = torch.cat(ranks, dim=0)
        image = os_imgs[:, 0]
        return torch.stack(
            [((image[:, None] > (alpha * rank_maps)) & (image[:, None] > 0)).to(torch.uint8)
             for alpha in os_alphas],
            dim=2,
        )

    def os_optimized():
        rank_maps = os_rank_maps_hist(
            os_imgs,
            guard,
            ref,
            k_fracs=OS_KS,
            bins=64,
            bin_chunk=32,
            pool_dtype=torch.float16,
        )
        return build_threshold_masks(os_imgs, rank_maps, os_alphas)

    ca_base_s, ca_base_masks = _timeit(ca_baseline)
    ca_opt_s, ca_opt_masks = _timeit(ca_optimized)
    ca_mismatch = int((ca_base_masks != ca_opt_masks).sum().item())

    os_base_s, os_base_masks = _timeit(os_baseline)
    os_opt_s, os_opt_masks = _timeit(os_optimized)
    os_mismatch = int((os_base_masks != os_opt_masks).sum().item())

    exact_rank = os_rank_maps(os_imgs[:1], guard, ref, k_fracs=OS_KS)
    hist_rank = os_rank_maps_hist(
        os_imgs[:1],
        guard,
        ref,
        k_fracs=OS_KS,
        bins=64,
        bin_chunk=32,
        pool_dtype=torch.float16,
    )
    os_rank_max_abs = float((exact_rank.squeeze(1).unsqueeze(0) - hist_rank).abs().max().item())

    image_id = int(corpus.image_ids[0])
    intensity_np = corpus.images[0, 0].cpu().numpy()
    ca_masks_np = ca_opt_masks[0, 0].cpu().numpy()
    ca_stat_np = ca_background(corpus.images[:1], guard, ref)[0, 0].cpu().numpy()
    ca_alpha_np = np.array(ca_alphas.cpu().tolist(), dtype=np.float32)

    def ca_postproc_old():
        return [
            _old_mask_to_preds(
                ca_masks_np[i],
                intensity_np,
                ca_stat_np,
                float(ca_alpha_np[i]),
                image_id,
            )
            for i in range(ca_masks_np.shape[0])
        ]

    def ca_postproc_new():
        return [
            mask_to_preds(
                ca_masks_np[i],
                intensity_np,
                None,
                image_id,
                stat_np=ca_stat_np,
                alpha=float(ca_alpha_np[i]),
            )
            for i in range(ca_masks_np.shape[0])
        ]

    def ca_postproc_packed():
        return [
            mask_to_pred_arrays(
                ca_masks_np[i],
                intensity_np,
                None,
                image_id,
                stat_np=ca_stat_np,
                alpha=float(ca_alpha_np[i]),
            )
            for i in range(ca_masks_np.shape[0])
        ]

    ca_post_old_s, _ = _timeit(ca_postproc_old, warmup=1, repeat=5)
    ca_post_new_s, _ = _timeit(ca_postproc_new, warmup=1, repeat=5)
    ca_post_packed_s, _ = _timeit(ca_postproc_packed, warmup=1, repeat=5)

    os_masks_np = os_opt_masks[0].cpu().numpy()
    os_stat_np = hist_rank[0].cpu().numpy()
    os_alpha_np = np.array(os_alphas.cpu().tolist(), dtype=np.float32)

    def os_postproc_old():
        out = []
        for stat_idx in range(os_masks_np.shape[0]):
            for alpha_idx in range(os_masks_np.shape[1]):
                out.append(
                    _old_mask_to_preds(
                        os_masks_np[stat_idx, alpha_idx],
                        intensity_np,
                        os_stat_np[stat_idx],
                        float(os_alpha_np[alpha_idx]),
                        image_id,
                    )
                )
        return out

    def os_postproc_new():
        out = []
        for stat_idx in range(os_masks_np.shape[0]):
            for alpha_idx in range(os_masks_np.shape[1]):
                out.append(
                    mask_to_preds(
                        os_masks_np[stat_idx, alpha_idx],
                        intensity_np,
                        None,
                        image_id,
                        stat_np=os_stat_np[stat_idx],
                        alpha=float(os_alpha_np[alpha_idx]),
                    )
                )
        return out

    def os_postproc_packed():
        out = []
        for stat_idx in range(os_masks_np.shape[0]):
            for alpha_idx in range(os_masks_np.shape[1]):
                out.append(
                    mask_to_pred_arrays(
                        os_masks_np[stat_idx, alpha_idx],
                        intensity_np,
                        None,
                        image_id,
                        stat_np=os_stat_np[stat_idx],
                        alpha=float(os_alpha_np[alpha_idx]),
                    )
                )
        return out

    os_post_old_s, _ = _timeit(os_postproc_old, warmup=1, repeat=5)
    os_post_new_s, _ = _timeit(os_postproc_new, warmup=1, repeat=5)
    os_post_packed_s, _ = _timeit(os_postproc_packed, warmup=1, repeat=5)

    subset_ca = np.arange(min(CA_PIPELINE_IMAGES, corpus.images.shape[0]), dtype=np.int64)
    subset_os = np.arange(min(OS_PIPELINE_IMAGES, corpus.images.shape[0]), dtype=np.int64)

    def ca_pipeline_old():
        out = {(p, m): [] for p in PFAS for m in (20, 50, 100)}
        for i in subset_ca:
            img = corpus.images[i:i + 1]
            image_id_local = int(corpus.image_ids[i])
            intensity_local = img.squeeze().cpu().numpy()
            bg_local = ca_background(img, guard, ref)
            for pfa in PFAS:
                thr = ca_alpha(pfa, guard, ref) * bg_local
                mask_np = (((img > thr) & (img > 0)).squeeze().cpu().numpy().astype(np.uint8))
                thr_np = thr.squeeze().cpu().numpy()
                preds = mask_to_preds(mask_np, intensity_local, thr_np, image_id_local, (20, 50, 100))
                for ma in (20, 50, 100):
                    out[(pfa, ma)].extend(preds[ma])
        return sum(len(v) for v in out.values())

    def ca_pipeline_new():
        out = {(p, m): [] for p in PFAS for m in (20, 50, 100)}
        alpha_local = torch.tensor([ca_alpha(p, guard, ref) for p in PFAS], device=corpus.images.device)
        for i in subset_ca:
            img = corpus.images[i:i + 1]
            image_id_local = int(corpus.image_ids[i])
            intensity_local = img.squeeze().cpu().numpy()
            bg_local = ca_background(img, guard, ref)
            masks = (
                (img[:, :, None] > (bg_local[:, :, None] * alpha_local.view(1, 1, -1, 1, 1)))
                & (img[:, :, None] > 0)
            ).to(torch.uint8)
            masks_np = masks[0, 0].cpu().numpy()
            stat_np = bg_local[0, 0].cpu().numpy()
            for p_idx, pfa in enumerate(PFAS):
                preds = mask_to_pred_arrays(
                    masks_np[p_idx],
                    intensity_local,
                    None,
                    image_id_local,
                    (20, 50, 100),
                    stat_np=stat_np,
                    alpha=float(alpha_local[p_idx]),
                )
                for ma in (20, 50, 100):
                    out[(pfa, ma)].append(preds[ma])
        return sum(concat_packed_preds(v).image_ids.size for v in out.values())

    def os_pipeline_old():
        out = {(p, m, k): [] for p in PFAS for m in (20, 50, 100) for k in OS_KS}
        for i in subset_os:
            img = corpus.images[i:i + 1]
            image_id_local = int(corpus.image_ids[i])
            intensity_local = img.squeeze().cpu().numpy()
            ranks_local = os_rank_maps(img, guard, ref, k_fracs=OS_KS)
            for k_idx, k in enumerate(OS_KS):
                rank_local = ranks_local[k_idx:k_idx + 1]
                for pfa in PFAS:
                    thr = os_alpha(pfa, guard, ref) * rank_local
                    mask_np = (((img > thr) & (img > 0)).squeeze().cpu().numpy().astype(np.uint8))
                    thr_np = thr.squeeze().cpu().numpy()
                    preds = mask_to_preds(mask_np, intensity_local, thr_np, image_id_local, (20, 50, 100))
                    for ma in (20, 50, 100):
                        out[(pfa, ma, k)].extend(preds[ma])
        return sum(len(v) for v in out.values())

    def os_pipeline_new():
        out = {(p, m, k): [] for p in PFAS for m in (20, 50, 100) for k in OS_KS}
        imgs_local = corpus.images[subset_os.tolist()]
        image_ids_local = corpus.image_ids[subset_os]
        ranks_local = os_rank_maps_hist(
            imgs_local,
            guard,
            ref,
            k_fracs=OS_KS,
            bins=64,
            bin_chunk=32,
            pool_dtype=torch.float16,
        )
        masks = build_threshold_masks(imgs_local, ranks_local, os_alphas).cpu().numpy()
        intensities_local = imgs_local[:, 0].cpu().numpy()
        stats_local = ranks_local.cpu().numpy()
        for b in range(imgs_local.shape[0]):
            for k_idx, k in enumerate(OS_KS):
                for p_idx, pfa in enumerate(PFAS):
                    preds = mask_to_pred_arrays(
                        masks[b, k_idx, p_idx],
                        intensities_local[b],
                        None,
                        int(image_ids_local[b]),
                        (20, 50, 100),
                        stat_np=stats_local[b, k_idx],
                        alpha=float(os_alphas[p_idx]),
                    )
                    for ma in (20, 50, 100):
                        out[(pfa, ma, k)].append(preds[ma])
        return sum(concat_packed_preds(v).image_ids.size for v in out.values())

    ca_pipe_old_s, _ = _timeit(ca_pipeline_old, warmup=1, repeat=3)
    ca_pipe_new_s, _ = _timeit(ca_pipeline_new, warmup=1, repeat=3)
    os_pipe_old_s, _ = _timeit(os_pipeline_old, warmup=1, repeat=3)
    os_pipe_new_s, _ = _timeit(os_pipeline_new, warmup=1, repeat=3)

    print(f"Benchmark device: {device}")
    print(f"CA batch: {ca_batch}  OS batch: {os_batch}  guard={guard} ref={ref}")
    print()
    print(f"CA baseline torch loop:      {ca_base_s:.4f}s")
    print(f"CA optimized Triton masks:   {ca_opt_s:.4f}s")
    print(f"CA speedup:                  {ca_base_s / max(ca_opt_s, 1e-9):.2f}x")
    print(f"CA mask mismatches:          {ca_mismatch}")
    print()
    print(f"OS baseline exact+loop:      {os_base_s:.4f}s")
    print(f"OS optimized hist+Triton:    {os_opt_s:.4f}s")
    print(f"OS speedup:                  {os_base_s / max(os_opt_s, 1e-9):.2f}x")
    print(f"OS mask mismatches:          {os_mismatch}")
    print(f"OS rank max abs diff:        {os_rank_max_abs:.6f}")
    print()
    print(f"CA postproc old CPU path:    {ca_post_old_s:.4f}s")
    print(f"CA postproc new CPU path:    {ca_post_new_s:.4f}s")
    print(f"CA postproc speedup:         {ca_post_old_s / max(ca_post_new_s, 1e-9):.2f}x")
    print(f"CA postproc packed output:   {ca_post_packed_s:.4f}s")
    print(f"CA packed vs dict:           {ca_post_new_s / max(ca_post_packed_s, 1e-9):.2f}x")
    print()
    print(f"OS postproc old CPU path:    {os_post_old_s:.4f}s")
    print(f"OS postproc new CPU path:    {os_post_new_s:.4f}s")
    print(f"OS postproc speedup:         {os_post_old_s / max(os_post_new_s, 1e-9):.2f}x")
    print(f"OS postproc packed output:   {os_post_packed_s:.4f}s")
    print(f"OS packed vs dict:           {os_post_new_s / max(os_post_packed_s, 1e-9):.2f}x")
    print()
    print(f"CA subset pipeline old:      {ca_pipe_old_s:.4f}s ({len(subset_ca)} images)")
    print(f"CA subset pipeline new:      {ca_pipe_new_s:.4f}s ({len(subset_ca)} images)")
    print(f"CA pipeline speedup:         {ca_pipe_old_s / max(ca_pipe_new_s, 1e-9):.2f}x")
    print()
    print(f"OS subset pipeline old:      {os_pipe_old_s:.4f}s ({len(subset_os)} images)")
    print(f"OS subset pipeline new:      {os_pipe_new_s:.4f}s ({len(subset_os)} images)")
    print(f"OS pipeline speedup:         {os_pipe_old_s / max(os_pipe_new_s, 1e-9):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ca-batch", type=int, default=32)
    ap.add_argument("--os-batch", type=int, default=4)
    ap.add_argument("--guard", type=int, default=15)
    ap.add_argument("--ref", type=int, default=30)
    args = ap.parse_args()
    benchmark(args.device, args.ca_batch, args.os_batch, args.guard, args.ref)


if __name__ == "__main__":
    main()
