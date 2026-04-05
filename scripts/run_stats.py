"""Statistical comparison of detection methods.

Runs pairwise bootstrap significance tests across all methods,
computing bootstrap AP distributions and applying Bonferroni correction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_coco
from src.evaluation.statistical import pairwise_bootstrap_test


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical analysis of detection results")
    parser.add_argument("--gt-path", type=str, required=True,
                        help="Path to ground truth COCO JSON")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory containing all prediction JSONs")
    parser.add_argument("--output-dir", type=str, default="results/tables",
                        help="Output directory for statistical results")
    parser.add_argument("--n-resamples", type=int, default=10000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    return parser.parse_args()


def compute_per_image_ap(gt_path, predictions, gt_ann):
    """Compute per-image AP@0.5 for bootstrap resampling.

    For each image, computes a local AP by evaluating only that image's
    predictions against its ground truth.

    Returns:
        Array of per-image AP values.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    # Write GT and predictions to temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_ann, f)
        gt_tmp = f.name

    coco_gt = COCO(gt_tmp)

    if not predictions:
        return np.zeros(len(gt_ann["images"]))

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    per_image_aps = []
    for img_info in gt_ann["images"]:
        img_id = img_info["id"]
        coco_eval.params.imgIds = [img_id]
        coco_eval.evaluate()
        coco_eval.accumulate()

        # AP@0.5 (index 1 in the stats)
        if coco_eval.stats is not None and len(coco_eval.stats) > 1:
            per_image_aps.append(coco_eval.stats[1])
        else:
            per_image_aps.append(0.0)

    return np.array(per_image_aps)


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gt_path) as f:
        gt_ann = json.load(f)

    # Find all prediction files
    pred_files = sorted(pred_dir.rglob("*_predictions.json"))
    if not pred_files:
        print("No prediction files found")
        return

    # Compute per-image AP for each method
    print("Computing per-image AP distributions...")
    method_aps = {}

    for pred_path in pred_files:
        method_name = pred_path.stem.replace("_predictions", "")

        with open(pred_path) as f:
            predictions = json.load(f)

        if not predictions:
            continue

        print(f"  {method_name}: {len(predictions)} predictions")
        per_image_ap = compute_per_image_ap(args.gt_path, predictions, gt_ann)
        method_aps[method_name] = per_image_ap

    if len(method_aps) < 2:
        print("Need at least 2 methods for pairwise comparison")
        return

    # Bootstrap resampling for each method
    print(f"\nBootstrap resampling ({args.n_resamples} resamples)...")
    rng = np.random.RandomState(42)
    n_images = len(gt_ann["images"])
    bootstrap_distributions = {}

    for method_name, per_image_ap in method_aps.items():
        boot_aps = []
        for _ in range(args.n_resamples):
            idx = rng.choice(n_images, size=n_images, replace=True)
            boot_aps.append(per_image_ap[idx].mean())
        bootstrap_distributions[method_name] = np.array(boot_aps)

    # Pairwise tests
    print("\nRunning pairwise bootstrap tests...")
    test_results = pairwise_bootstrap_test(
        bootstrap_distributions,
        alpha=args.alpha,
        correction="bonferroni",
    )

    # Save results
    df = pd.DataFrame(test_results)
    df.to_csv(output_dir / "statistical_tests.csv", index=False)

    # Print summary
    print(f"\nPairwise comparisons (alpha={args.alpha}, Bonferroni-corrected):")
    print("-" * 80)
    for result in test_results:
        sig = "***" if result["significant"] else "   "
        print(f"  {sig} {result['method_1']:20s} vs {result['method_2']:20s}  "
              f"diff={result['mean_diff']:+.4f}  "
              f"CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  "
              f"p={result['p_value']:.4f}")

    # Also save bootstrap distributions for plotting
    boot_df = pd.DataFrame(bootstrap_distributions)
    boot_df.to_csv(output_dir / "bootstrap_distributions.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
