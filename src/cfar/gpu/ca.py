"""CA-CFAR background estimator on GPU (one pass per window).

Replicates `src/cfar/detector.py` semantics: reflect padding, uniform-filter
box means, exponential-clutter alpha.  Returns the background map so any
Pfa (and later min_area) can be applied analytically downstream.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _box_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Reflect-padded box mean with odd kernel size k. x: [B, 1, H, W]."""
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    return F.avg_pool2d(xp, kernel_size=k, stride=1)


def ca_background(images: torch.Tensor, guard: int, ref: int) -> torch.Tensor:
    """Compute the CA-CFAR background map for a batch of images.

    Args:
        images: [B, 1, H, W] float tensor in [0, 1].
        guard:  guard half-size (integer, same for both axes).
        ref:    reference half-size (integer, same for both axes).

    Returns:
        background: [B, 1, H, W] float tensor.
    """
    ref_k = 2 * ref + 1
    guard_k = 2 * guard + 1
    ref_sum = _box_mean(images, ref_k) * (ref_k * ref_k)
    guard_sum = _box_mean(images, guard_k) * (guard_k * guard_k)
    n_ref = ref_k * ref_k - guard_k * guard_k
    return (ref_sum - guard_sum) / max(n_ref, 1)


def ca_alpha(pfa: float, guard: int, ref: int) -> float:
    ref_k = 2 * ref + 1
    guard_k = 2 * guard + 1
    n_ref = ref_k * ref_k - guard_k * guard_k
    return float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))
