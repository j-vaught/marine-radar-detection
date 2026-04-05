"""Statistical testing for model comparison."""

import numpy as np
from itertools import combinations


def bootstrap_ap(
    gt_matched: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap AP distribution.

    Args:
        gt_matched: Boolean array indicating whether each detection matched a GT.
        scores: Confidence scores for each detection.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Array of bootstrapped AP values.
    """
    rng = np.random.RandomState(seed)
    n = len(gt_matched)
    aps = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        ap = _compute_ap(gt_matched[idx], scores[idx])
        aps.append(ap)

    return np.array(aps)


def _compute_ap(gt_matched: np.ndarray, scores: np.ndarray) -> float:
    """Compute Average Precision from matched/unmatched detections."""
    sorted_idx = np.argsort(-scores)
    gt_matched = gt_matched[sorted_idx]

    tp = np.cumsum(gt_matched)
    fp = np.cumsum(~gt_matched)
    recall = tp / (tp[-1] + 1e-6) if len(tp) > 0 else np.array([0])
    precision = tp / (tp + fp + 1e-6)

    # VOC-style 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_rec = precision[recall >= t]
        ap += prec_at_rec.max() / 11 if len(prec_at_rec) > 0 else 0

    return ap


def pairwise_bootstrap_test(
    ap_distributions: dict[str, np.ndarray],
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> list[dict]:
    """Pairwise bootstrap significance testing between methods.

    Args:
        ap_distributions: Dict mapping method name -> bootstrap AP array.
        alpha: Significance level.
        correction: Multiple testing correction ('bonferroni' or 'none').

    Returns:
        List of pairwise test result dicts.
    """
    methods = list(ap_distributions.keys())
    n_comparisons = len(list(combinations(methods, 2)))

    if correction == "bonferroni":
        adjusted_alpha = alpha / max(n_comparisons, 1)
    else:
        adjusted_alpha = alpha

    results = []
    for m1, m2 in combinations(methods, 2):
        dist1 = ap_distributions[m1]
        dist2 = ap_distributions[m2]
        diff = dist1 - dist2

        p_value = min(
            np.mean(diff <= 0) * 2,
            np.mean(diff >= 0) * 2,
            1.0,
        )

        ci_lower = np.percentile(diff, 100 * adjusted_alpha / 2)
        ci_upper = np.percentile(diff, 100 * (1 - adjusted_alpha / 2))

        results.append({
            "method_1": m1,
            "method_2": m2,
            "mean_diff": float(np.mean(diff)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "significant": p_value < adjusted_alpha,
            "adjusted_alpha": adjusted_alpha,
        })

    return results
