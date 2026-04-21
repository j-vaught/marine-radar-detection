"""OS-CFAR order-statistic estimator on GPU.

Two implementations:

* ``os_rank_maps``       — sort-based; exact, O(N log N) per CUT.  Kept as
                           reference oracle.
* ``os_rank_maps_hist``  — histogram-based; exploits the fact that input
                           pixels are quantized to 256 levels (PNGs /255),
                           so the ring's order statistic can be recovered
                           from a cumulative 256-bin histogram built by two
                           CA-style box pools on a one-hot tensor.
                           O(256) per pixel; 20–30× faster than sort.

Both produce identical threshold maps within quantization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _guard_kept_rows(ref: int, guard: int) -> torch.Tensor:
    """Boolean mask over (2*ref+1)^2 unfold rows: True = reference cell kept."""
    k = 2 * ref + 1
    cy = cx = ref
    rows = torch.arange(k).view(k, 1).expand(k, k)
    cols = torch.arange(k).view(1, k).expand(k, k)
    guard_mask = (rows >= cy - guard) & (rows <= cy + guard) & \
                 (cols >= cx - guard) & (cols <= cx + guard)
    return (~guard_mask).reshape(-1)


def os_rank_maps(
    image: torch.Tensor,
    guard: int,
    ref: int,
    k_fracs: tuple[float, ...] = (0.5, 0.75, 0.9),
    stripe: int = 256,
) -> torch.Tensor:
    """Compute OS-CFAR background maps for one image, at multiple k fractions.

    Args:
        image: [1, 1, H, W] float tensor.
        guard, ref: half-sizes.
        k_fracs: fractions in (0, 1).  Returns one map per fraction.
        stripe: row stripe height to bound peak VRAM.

    Returns:
        [K, 1, H, W] float tensor (K = len(k_fracs)).
    """
    assert image.dim() == 4 and image.shape[0] == 1 and image.shape[1] == 1
    _, _, H, W = image.shape
    ref_k = 2 * ref + 1

    kept = _guard_kept_rows(ref, guard).to(image.device)   # [ref_k^2]
    n_ref = int(kept.sum().item())
    k_indices = [min(int(f * n_ref), n_ref - 1) for f in k_fracs]

    pad = ref
    padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")

    out = torch.empty(len(k_fracs), 1, H, W, device=image.device, dtype=image.dtype)

    for y0 in range(0, H, stripe):
        y1 = min(H, y0 + stripe)
        tile = padded[:, :, y0:y1 + 2 * pad, :]
        # unfold rows only (vertical extent covers the stripe + padding)
        cols = F.unfold(tile, kernel_size=(ref_k, ref_k), stride=1)
        # cols: [1, ref_k*ref_k, (y1-y0) * W]
        valid = cols[:, kept, :]
        # Sort along the ring dimension; pick k-th via indexing.
        sorted_ring, _ = valid.sort(dim=1)
        for ki, kidx in enumerate(k_indices):
            out[ki, 0, y0:y1, :] = sorted_ring[0, kidx, :].view(y1 - y0, W)

    return out


def os_alpha(pfa: float, guard: int, ref: int) -> float:
    """Exponential-clutter CFAR scaling, identical to CA formula."""
    ref_k = 2 * ref + 1
    guard_k = 2 * guard + 1
    n_ref = ref_k * ref_k - guard_k * guard_k
    return float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))


# ── histogram-based OS (fast path) ───────────────────────────────────────

def _box_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Reflect-padded box SUM with odd kernel size k.  x: [B, C, H, W]."""
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    return F.avg_pool2d(xp, kernel_size=k, stride=1) * (k * k)


def os_rank_maps_hist(
    images: torch.Tensor,            # [B, 1, H, W] float in [0, 1]
    guard: int,
    ref: int,
    k_fracs: tuple[float, ...] = (0.5, 0.75, 0.9),
    bins: int = 256,
    bin_chunk: int = 64,
    pool_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Histogram-based order-statistic maps for a batch of images.

    Bin chunking limits peak VRAM: we build ``one_hot`` one chunk of bins
    at a time and accumulate the cumulative histogram as we go, then index
    it for each requested k fraction.

    Returns:
        [B, K, H, W] float (K = len(k_fracs)).
    """
    assert images.dim() == 4 and images.shape[1] == 1
    B, _, H, W = images.shape
    device = images.device
    dtype = images.dtype
    if pool_dtype is None:
        pool_dtype = torch.float16 if images.is_cuda else torch.float32
    ref_k = 2 * ref + 1
    guard_k = 2 * guard + 1
    n_ref = ref_k * ref_k - guard_k * guard_k

    # Quantize to int bin indices — images are 8-bit PNGs / 255, so this
    # perfectly recovers the original uint8 values.
    q = (images.squeeze(1) * (bins - 1)).round().clamp(0, bins - 1).to(torch.int64)  # [B, H, W]

    k_indices = [int(f * n_ref) for f in k_fracs]
    targets = torch.tensor([k + 1 for k in k_indices], device=device)   # need cum >= target
    K = len(k_fracs)

    # Running cumulative hist is folded into per-k "bin found" tracking.
    # For each k: bin_idx = smallest b such that cum[b] >= target. We sweep
    # bins in order; the first time cum crosses the target we latch bin b.
    bin_idx = torch.full((B, K, H, W), bins - 1, device=device, dtype=torch.int32)
    latched = torch.zeros((B, K, H, W), device=device, dtype=torch.bool)
    cum = torch.zeros((B, H, W), device=device, dtype=torch.float32)

    for b0 in range(0, bins, bin_chunk):
        b1 = min(bins, b0 + bin_chunk)
        # Build one-hot chunk [B, b1-b0, H, W]
        bin_ids = torch.arange(b0, b1, device=device).view(1, -1, 1, 1)
        chunk_oh = (q.unsqueeze(1) == bin_ids).to(dtype=pool_dtype)             # [B, C, H, W]
        # Ring count = ref_count - guard_count
        ring = _box_sum(chunk_oh, ref_k) - _box_sum(chunk_oh, guard_k)          # [B, C, H, W]
        # Cumulative histogram up to each bin inside the chunk
        ring_cs = ring.cumsum(dim=1, dtype=torch.float32) + cum.unsqueeze(1)    # [B, C, H, W]
        # For each k, latch first bin where cum >= target
        for ki, t in enumerate(targets):
            not_yet = ~latched[:, ki]                                            # [B, H, W]
            if not not_yet.any():
                continue
            crossed = ring_cs >= t                                               # [B, C, H, W]
            # first crossing bin within this chunk: argmax over the chunk dim
            first_hit_local = crossed.int().argmax(dim=1)                        # [B, H, W]
            any_hit_in_chunk = crossed.any(dim=1)                                # [B, H, W]
            update = not_yet & any_hit_in_chunk
            bin_idx[:, ki] = torch.where(update, (b0 + first_hit_local).to(torch.int32), bin_idx[:, ki])
            latched[:, ki] = latched[:, ki] | update
        cum = ring_cs[:, -1]                                                     # [B, H, W]

    # Convert bin index to the original float value (bin / (bins - 1))
    out = bin_idx.to(dtype=dtype) / float(bins - 1)
    return out   # [B, K, H, W]
