"""Unified Triton threshold/mask kernel for CA- and OS-CFAR stat maps.

The kernel consumes a normalized image tensor plus one or more per-pixel
reference statistic maps and emits detection masks for every requested alpha
value in a single GPU pass.  CA uses one stat map (local ring mean), while OS
uses one map per requested rank fraction.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["hw"],
)
@triton.jit
def _threshold_masks_kernel(
    image_ptr,
    stat_ptr,
    alpha_ptr,
    out_ptr,
    hw,
    stats_per_batch,
    num_alpha,
    image_stride_b,
    stat_stride_b,
    stat_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_a,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_mask = tl.cdiv(hw, BLOCK)
    mask_idx = pid // blocks_per_mask
    pixel_block = pid % blocks_per_mask

    stats_alpha = stats_per_batch * num_alpha
    batch_idx = mask_idx // stats_alpha
    rem = mask_idx - batch_idx * stats_alpha
    stat_idx = rem // num_alpha
    alpha_idx = rem - stat_idx * num_alpha

    offs = pixel_block * BLOCK + tl.arange(0, BLOCK)
    valid = offs < hw

    image = tl.load(image_ptr + batch_idx * image_stride_b + offs, mask=valid, other=0.0)
    stat = tl.load(
        stat_ptr + batch_idx * stat_stride_b + stat_idx * stat_stride_s + offs,
        mask=valid,
        other=0.0,
    )
    alpha = tl.load(alpha_ptr + alpha_idx)
    mask = (image > stat * alpha) & (image > 0.0)

    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + stat_idx * out_stride_s
        + alpha_idx * out_stride_a
        + offs,
        mask.to(tl.uint8),
        mask=valid,
    )


def build_threshold_masks(
    image: torch.Tensor,
    stat_maps: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Generate CFAR detection masks for all stat-map / alpha combinations.

    Args:
        image: ``[B, 1, H, W]`` or ``[B, H, W]`` float tensor in ``[0, 1]``.
        stat_maps: ``[B, S, H, W]`` float tensor, where ``S`` is the number of
            CA/OS statistic maps per image.
        alphas: ``[P]`` float tensor.

    Returns:
        ``[B, S, P, H, W]`` uint8 tensor containing binary masks.
    """
    if image.dim() == 4:
        if image.shape[1] != 1:
            raise ValueError(f"Expected a single-channel image tensor, got {tuple(image.shape)}")
        image = image[:, 0]
    elif image.dim() != 3:
        raise ValueError(f"Expected image to have 3 or 4 dims, got {image.dim()}")

    if stat_maps.dim() != 4:
        raise ValueError(f"Expected stat_maps to have shape [B, S, H, W], got {tuple(stat_maps.shape)}")
    if image.shape[0] != stat_maps.shape[0] or image.shape[-2:] != stat_maps.shape[-2:]:
        raise ValueError(
            "Image and stat map shapes are incompatible: "
            f"{tuple(image.shape)} vs {tuple(stat_maps.shape)}"
        )
    if alphas.dim() != 1:
        raise ValueError(f"Expected alphas to be 1D, got {tuple(alphas.shape)}")

    image = image.contiguous()
    stat_maps = stat_maps.contiguous()
    alphas = alphas.to(device=image.device, dtype=torch.float32).contiguous()

    bsz, stats_per_batch, height, width = stat_maps.shape
    hw = height * width

    image_flat = image.view(bsz, hw)
    stat_flat = stat_maps.view(bsz, stats_per_batch, hw)
    out = torch.empty(
        (bsz, stats_per_batch, alphas.numel(), hw),
        device=image.device,
        dtype=torch.uint8,
    )

    total_masks = bsz * stats_per_batch * alphas.numel()
    grid = lambda meta: (total_masks * triton.cdiv(hw, meta["BLOCK"]),)
    device_index = image.device.index if image.device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(device_index):
        _threshold_masks_kernel[grid](
            image_flat,
            stat_flat,
            alphas,
            out,
            hw,
            stats_per_batch,
            alphas.numel(),
            image_flat.stride(0),
            stat_flat.stride(0),
            stat_flat.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
    return out.view(bsz, stats_per_batch, alphas.numel(), height, width)
