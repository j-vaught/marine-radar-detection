"""Inference benchmarking: latency, FLOPs, and parameter counting."""

import time
from pathlib import Path

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    predict_fn,
    sample_input,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Measure inference latency.

    Args:
        predict_fn: Callable that takes sample_input and returns predictions.
        sample_input: Input to the model.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.

    Returns:
        Dict with mean, std, min, max latency in ms.
    """
    # Warmup
    for _ in range(n_warmup):
        predict_fn(sample_input)

    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        predict_fn(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def estimate_flops(
    model: torch.nn.Module,
    input_shape: tuple = (1, 1, 640, 640),
    device: str = "cpu",
) -> int | None:
    """Estimate FLOPs using fvcore.

    Returns:
        Number of FLOPs, or None if fvcore is not available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        dummy = torch.randn(*input_shape).to(device)
        model = model.to(device)
        model.eval()

        flops = FlopCountAnalysis(model, dummy)
        return flops.total()
    except ImportError:
        print("fvcore not installed, skipping FLOPs estimation")
        return None


def benchmark_report(
    method_name: str,
    model: torch.nn.Module | None,
    predict_fn,
    sample_input,
    input_shape: tuple = (1, 1, 640, 640),
) -> dict:
    """Generate a full benchmark report for a detection method."""
    report = {"method": method_name}

    if model is not None:
        report["parameters"] = count_parameters(model)
        report["flops"] = estimate_flops(model, input_shape)

    timing = measure_inference_time(predict_fn, sample_input)
    report.update(timing)

    return report
