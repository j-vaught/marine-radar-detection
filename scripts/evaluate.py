"""Evaluate all methods: COCO metrics, TIDE, and multi-seed aggregation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_coco
from src.evaluation.tide_analysis import run_tide_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection methods")
    parser.add_argument("--gt-path", type=str, required=True,
                        help="Path to ground truth COCO JSON")
    parser.add_argument("--classical-dir", type=str, default="results/classical",
                        help="Directory containing classical prediction JSONs")
    parser.add_argument("--dl-dir", type=str, default="results/deep_learning",
                        help="Directory containing DL prediction JSONs")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for evaluation results")
    parser.add_argument("--run-tide", action="store_true",
                        help="Run TIDE error analysis")
    return parser.parse_args()


def evaluate_predictions(gt_path, pred_path, method_name, tide_dir=None):
    """Evaluate a single prediction file."""
    with open(pred_path) as f:
        predictions = json.load(f)

    if not predictions:
        print(f"  No predictions for {method_name}, skipping")
        return None

    results = evaluate_coco(gt_path, predictions)
    results["method"] = method_name

    print(f"  AP@0.50: {results['AP@0.50']:.4f}  "
          f"AP@0.50:0.95: {results['AP@0.50:0.95']:.4f}  "
          f"AR@100: {results['AR@100']:.4f}")

    if tide_dir:
        try:
            run_tide_analysis(gt_path, str(pred_path), tide_dir, method_name)
        except Exception as e:
            print(f"  TIDE failed for {method_name}: {e}")

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tide_dir = output_dir / "tide" if args.run_tide else None

    all_results = []

    # ── Classical methods (deterministic, single result each) ──
    classical_dir = Path(args.classical_dir)
    if classical_dir.exists():
        classical_preds = sorted(classical_dir.glob("*_predictions.json"))
        if classical_preds:
            print("\n=== Classical Methods ===")
        for pred_path in classical_preds:
            method_name = pred_path.stem.replace("_predictions", "")
            print(f"\n{method_name}:")
            result = evaluate_predictions(args.gt_path, pred_path, method_name, tide_dir)
            if result:
                result["type"] = "classical"
                result["seed"] = "N/A"
                all_results.append(result)

    # ── Deep learning methods (multi-seed) ──
    dl_dir = Path(args.dl_dir)
    if dl_dir.exists():
        dl_preds = sorted(dl_dir.rglob("*_predictions.json"))
        if dl_preds:
            print("\n=== Deep Learning Methods ===")

        # Group by base method name (strip seed)
        from collections import defaultdict
        method_seeds = defaultdict(list)
        for pred_path in dl_preds:
            fname = pred_path.stem.replace("_predictions", "")
            # Parse: "modelname_seed42" -> ("modelname", "42")
            parts = fname.rsplit("_seed", 1)
            if len(parts) == 2:
                base_method, seed = parts
            else:
                base_method, seed = fname, "0"
            method_seeds[base_method].append((seed, pred_path))

        for base_method, seed_paths in method_seeds.items():
            print(f"\n{base_method}:")
            seed_results = []

            for seed, pred_path in seed_paths:
                method_label = f"{base_method}_seed{seed}"
                print(f"  seed={seed}:")
                result = evaluate_predictions(
                    args.gt_path, pred_path, method_label, tide_dir
                )
                if result:
                    result["type"] = "dl"
                    result["seed"] = seed
                    result["base_method"] = base_method
                    all_results.append(result)
                    seed_results.append(result)

            # Compute mean/std across seeds
            if len(seed_results) > 1:
                metric_cols = [k for k in seed_results[0] if k not in
                               ("method", "type", "seed", "base_method")]
                mean_result = {"method": base_method, "type": "dl_mean", "seed": "mean"}
                std_result = {"method": f"{base_method}_std", "type": "dl_std", "seed": "std"}
                for col in metric_cols:
                    vals = [r[col] for r in seed_results if isinstance(r.get(col), (int, float))]
                    if vals:
                        mean_result[col] = float(np.mean(vals))
                        std_result[col] = float(np.std(vals))
                all_results.append(mean_result)
                all_results.append(std_result)
                print(f"  mean AP@0.50: {mean_result.get('AP@0.50', 0):.4f} "
                      f"+/- {std_result.get('AP@0.50', 0):.4f}")

    # ── Save summary ──
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = output_dir / "evaluation_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Print clean comparison table (only classical + DL means)
        display_df = df[df["type"].isin(["classical", "dl_mean"])].copy()
        if not display_df.empty:
            cols = ["method", "type", "AP@0.50", "AP@0.75", "AP@0.50:0.95", "AR@100"]
            available_cols = [c for c in cols if c in display_df.columns]
            print("\n" + display_df[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
