"""Grid search that caches per-(method, window) background maps globally.

For each window, the background/order-stat map for every corpus image is
computed once and held on CPU RAM as fp16.  Each experiment then cheaply
combines those cached maps with the free axes (Pfa, min_area, OS-k).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..gpu.ca import ca_background, ca_alpha
from ..gpu.data import Corpus
from ..gpu.eval_coco import build_gt_index, evaluate, fast_ap50
from ..gpu.os_cfar import os_alpha, os_rank_maps_hist
from ..gpu.postproc import PackedPreds, concat_packed_preds, mask_to_pred_arrays
try:
    from ..gpu.triton_mask import build_threshold_masks
except ImportError:
    def build_threshold_masks(images, stat_maps, alphas):
        return ((images[:, :, None] > stat_maps[:, :, None] * alphas.view(1, 1, -1, 1, 1))
                & (images[:, :, None] > 0)).to(torch.uint8)

WINDOWS: list[tuple[int, int]] = [(10, 25), (15, 30), (20, 40)]
PFAS: list[float] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
MIN_AREAS: list[int] = [20, 50, 100]
OS_KS: list[float] = [0.5, 0.75, 0.9]
OS_HIST_BINS = 64
OS_HIST_BIN_CHUNK = 32
OS_CACHE_CHUNK = 8
PREDICT_CHUNK = 16


@dataclass
class Config:
    method: str
    guard: int
    ref: int
    pfa: float
    min_area: int
    os_k: float | None


def enumerate_grid(method: str) -> list[Config]:
    out: list[Config] = []
    for g, r in WINDOWS:
        for pfa in PFAS:
            for ma in MIN_AREAS:
                if method == "CA":
                    out.append(Config("CA", g, r, pfa, ma, None))
                else:
                    for k in OS_KS:
                        out.append(Config("OS", g, r, pfa, ma, k))
    return out


class BackgroundCache:
    """Lazy, global, per-(method, window) cache of CFAR background maps.

    CA entries are [N, 1, H, W] fp16 on CPU RAM.
    OS entries are [N, K, H, W] fp16 on CPU RAM (K = len(OS_KS)).
    """
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.ca: dict[tuple[int, int], torch.Tensor] = {}
        self.os: dict[tuple[int, int], torch.Tensor] = {}

    def ca_for(self, guard: int, ref: int) -> torch.Tensor:
        key = (guard, ref)
        if key in self.ca:
            return self.ca[key]
        t0 = time.time()
        imgs = self.corpus.images
        N = imgs.shape[0]
        out = torch.empty(N, 1, imgs.shape[2], imgs.shape[3], dtype=torch.float16, device="cpu")
        chunk = 32
        for i in range(0, N, chunk):
            bg = ca_background(imgs[i:i + chunk], guard, ref)
            out[i:i + chunk] = bg.to(dtype=torch.float16, device="cpu")
        self.ca[key] = out
        print(f"  [cache] CA g={guard} r={ref} built in {time.time() - t0:.1f}s")
        return out

    def os_for(self, guard: int, ref: int) -> torch.Tensor:
        key = (guard, ref)
        if key in self.os:
            return self.os[key]
        t0 = time.time()
        imgs = self.corpus.images
        N, _, H, W = imgs.shape
        out = torch.empty(N, len(OS_KS), H, W, dtype=torch.float16, device="cpu")
        for i in range(0, N, OS_CACHE_CHUNK):
            rank = os_rank_maps_hist(
                imgs[i:i + OS_CACHE_CHUNK],
                guard,
                ref,
                k_fracs=tuple(OS_KS),
                bins=OS_HIST_BINS,
                bin_chunk=OS_HIST_BIN_CHUNK,
            )
            out[i:i + rank.shape[0]] = rank.to(dtype=torch.float16, device="cpu")
            done = i + rank.shape[0]
            if done % 100 == 0 or done == N:
                elapsed = time.time() - t0
                eta = elapsed / max(done, 1) * max(N - done, 0)
                print(f"    OS g={guard} r={ref}  {done}/{N}  elapsed {elapsed:.0f}s  eta {eta:.0f}s")
        self.os[key] = out
        print(f"  [cache] OS g={guard} r={ref} built in {time.time() - t0:.1f}s")
        return out


# ── per-image threshold → predictions for a list of configs (free axes) ──

def _predict_batch_multi(
    images: torch.Tensor,           # [B, 1, H, W] on GPU
    image_ids: np.ndarray,          # [B]
    stat_maps: torch.Tensor,        # [B, S, H, W] on GPU
    alphas: torch.Tensor,           # [P] on GPU
    min_areas: tuple[int, ...],
) -> list[list[list[dict[int, PackedPreds]]]]:
    """Return packed predictions indexed as [batch_idx][stat_idx][alpha_idx][min_area]."""
    if images.is_cuda:
        masks_gpu = build_threshold_masks(images, stat_maps, alphas)
    else:
        masks_gpu = ((images[:, :, None] > stat_maps[:, :, None] * alphas.view(1, 1, -1, 1, 1))
                     & (images[:, :, None] > 0)).to(torch.uint8)
    masks_np = masks_gpu.cpu().numpy()
    intensities_np = images[:, 0].cpu().numpy()
    stats_np = stat_maps.cpu().numpy()

    batch_results: list[list[list[dict[int, PackedPreds]]]] = []
    for batch_idx in range(images.shape[0]):
        results: list[list[dict[int, PackedPreds]]] = []
        for stat_idx in range(stats_np.shape[1]):
            stat_results: list[dict[int, PackedPreds]] = []
            for alpha_idx in range(alphas.numel()):
                stat_results.append(
                    mask_to_pred_arrays(
                        masks_np[batch_idx, stat_idx, alpha_idx],
                        intensities_np[batch_idx],
                        None,
                        int(image_ids[batch_idx]),
                        min_areas,
                        stat_np=stats_np[batch_idx, stat_idx],
                        alpha=float(alphas[alpha_idx].item()),
                    )
                )
            results.append(stat_results)
        batch_results.append(results)
    return batch_results


def predict_subset(
    corpus: Corpus,
    cache: BackgroundCache,
    subset_idx: np.ndarray,
    method: str,
    guard: int,
    ref: int,
    pfas: list[float] = PFAS,
    min_areas: list[int] = MIN_AREAS,
    os_ks: list[float] = OS_KS,
) -> dict[tuple, PackedPreds]:
    """Produce predictions on ``subset_idx`` for every (pfa, min_area [, os_k]) combo.

    Returns dict keyed by (pfa, min_area [, os_k]) → list[dict].
    """
    if method == "CA":
        bg_all = cache.ca_for(guard, ref)
        builders: dict[tuple, list[PackedPreds]] = {
            (pfa, ma): [] for pfa in pfas for ma in min_areas
        }
        alpha = {pfa: ca_alpha(pfa, guard, ref) for pfa in pfas}
    else:
        bg_all = cache.os_for(guard, ref)
        builders = {(pfa, ma, k): [] for pfa in pfas for ma in min_areas for k in os_ks}
        alpha = {pfa: os_alpha(pfa, guard, ref) for pfa in pfas}
        k_idx_map = {k: OS_KS.index(k) for k in os_ks}

    device = corpus.images.device
    alpha_values = torch.tensor([alpha[pfa] for pfa in pfas], device=device, dtype=torch.float32)
    n_sub = len(subset_idx)
    t0 = time.time()
    if method == "CA":
        for loc, i in enumerate(subset_idx):
            if loc and loc % 200 == 0:
                dt = time.time() - t0
                eta = dt / loc * (n_sub - loc)
                print(f"    predict {method} g={guard} r={ref}  {loc}/{n_sub}  "
                      f"elapsed {dt:.0f}s eta {eta:.0f}s", flush=True)
            img = corpus.images[int(i):int(i) + 1].to(device=device, dtype=torch.float32)
            image_id = int(corpus.image_ids[int(i)])
            bg = bg_all[int(i):int(i) + 1].to(device=device, dtype=torch.float32)
            masks = (
                (img[:, :, None] > (bg[:, :, None] * alpha_values.view(1, 1, -1, 1, 1)))
                & (img[:, :, None] > 0)
            ).to(torch.uint8)
            masks_np = masks[0, 0].cpu().numpy()
            intensity_np = img[0, 0].cpu().numpy()
            stat_np = bg[0, 0].cpu().numpy()

            for pfa_idx, pfa in enumerate(pfas):
                preds_by_min = mask_to_pred_arrays(
                    masks_np[pfa_idx],
                    intensity_np,
                    None,
                    image_id,
                    tuple(min_areas),
                    stat_np=stat_np,
                    alpha=float(alpha_values[pfa_idx].item()),
                )
                for ma in min_areas:
                    builders[(pfa, ma)].append(preds_by_min[ma])
    else:
        for loc in range(0, n_sub, PREDICT_CHUNK):
            chunk_idx = subset_idx[loc:loc + PREDICT_CHUNK]
            chunk_list = [int(i) for i in chunk_idx]
            if loc and loc % 200 == 0:
                dt = time.time() - t0
                eta = dt / loc * (n_sub - loc)
                print(f"    predict {method} g={guard} r={ref}  {loc}/{n_sub}  "
                      f"elapsed {dt:.0f}s eta {eta:.0f}s", flush=True)
            imgs = corpus.images[chunk_list].to(device=device, dtype=torch.float32)      # [B, 1, H, W]
            image_ids = corpus.image_ids[chunk_idx]
            bg = bg_all[chunk_list].to(device=device, dtype=torch.float32)               # [B, K, H, W]
            stat_maps = bg[:, [k_idx_map[k] for k in os_ks]]
            batch_preds = _predict_batch_multi(
                imgs,
                image_ids,
                stat_maps,
                alpha_values,
                tuple(min_areas),
            )
            for batch_idx, preds_nested in enumerate(batch_preds):
                for k_idx, k in enumerate(os_ks):
                    for pfa_idx, pfa in enumerate(pfas):
                        preds_by_min = preds_nested[k_idx][pfa_idx]
                        for ma in min_areas:
                            builders[(pfa, ma, k)].append(preds_by_min[ma])
    return {key: concat_packed_preds(chunks) for key, chunks in builders.items()}


# ── outer grid-search driver ──────────────────────────────────────────────

def run_experiment(
    exp, corpus: Corpus, cache: BackgroundCache,
    gt_index: dict | None = None,
) -> dict:
    tune_idx, val_idx, test_idx = exp.split_fn(corpus)
    method = exp.method
    t0 = time.time()
    if gt_index is None:
        gt_index = build_gt_index(corpus.coco)
    tune_ids = corpus.image_ids[tune_idx]
    val_ids = corpus.image_ids[val_idx]
    test_ids = corpus.image_ids[test_idx]

    # 1. Grid search on tune — fast_ap50 (numpy) in the inner loop
    scored: list[tuple[float, Config]] = []
    for guard, ref in WINDOWS:
        tune_preds = predict_subset(corpus, cache, tune_idx, method, guard, ref)
        for key, preds in tune_preds.items():
            if method == "CA":
                pfa, ma = key
                cfg = Config("CA", guard, ref, pfa, ma, None)
            else:
                pfa, ma, k = key
                cfg = Config("OS", guard, ref, pfa, ma, k)
            ap50 = fast_ap50(preds, gt_index, tune_ids)
            scored.append((ap50, cfg))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_k = scored[:5]

    # 2. Val: pick winner among top-k by fast_ap50 on val
    best_val_ap50 = -1.0
    best_cfg: Config | None = None
    best_tune_ap50 = 0.0
    for tune_ap50, cfg in top_k:
        val_preds = predict_subset(
            corpus, cache, val_idx, cfg.method, cfg.guard, cfg.ref,
            pfas=[cfg.pfa], min_areas=[cfg.min_area],
            os_ks=[cfg.os_k] if cfg.method == "OS" else OS_KS,
        )
        key = (cfg.pfa, cfg.min_area) if cfg.method == "CA" else (cfg.pfa, cfg.min_area, cfg.os_k)
        ap50 = fast_ap50(val_preds[key], gt_index, val_ids)
        if ap50 > best_val_ap50:
            best_val_ap50 = ap50
            best_cfg = cfg
            best_tune_ap50 = tune_ap50

    assert best_cfg is not None

    # 3. Test: evaluate winner
    test_preds = predict_subset(
        corpus, cache, test_idx, best_cfg.method, best_cfg.guard, best_cfg.ref,
        pfas=[best_cfg.pfa], min_areas=[best_cfg.min_area],
        os_ks=[best_cfg.os_k] if best_cfg.method == "OS" else OS_KS,
    )
    key = (best_cfg.pfa, best_cfg.min_area) if best_cfg.method == "CA" \
        else (best_cfg.pfa, best_cfg.min_area, best_cfg.os_k)
    test_m = evaluate(test_preds[key], corpus.coco, corpus.image_ids[test_idx])

    runtime = time.time() - t0
    return {
        "exp_id": exp.exp_id,
        "condition": exp.condition,
        "method": exp.method,
        "tune_set": exp.tune_label,
        "val_set": exp.val_label,
        "test_set": exp.test_label,
        "seed": exp.seed,
        "win_window_guard": best_cfg.guard,
        "win_window_ref": best_cfg.ref,
        "win_pfa": best_cfg.pfa,
        "win_min_area": best_cfg.min_area,
        "win_os_k": best_cfg.os_k if best_cfg.os_k is not None else "",
        "tune_AP50": round(best_tune_ap50, 4),
        "val_AP50": round(best_val_ap50, 4),
        "test_AP": round(test_m["AP"], 4),
        "test_AP50": round(test_m["AP50"], 4),
        "test_AP75": round(test_m["AP75"], 4),
        "test_recall_at_p90": round(test_m["recall_at_p90"], 4),
        "test_fa_per_image": round(test_m["fa_per_image"], 2),
        "total_runtime_sec": round(runtime, 1),
    }
