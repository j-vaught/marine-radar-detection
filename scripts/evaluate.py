"""Evaluate all methods: COCO metrics, TIDE, benchmarking, statistical tests."""

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_coco
from src.evaluation.tide_analysis import run_tide_analysis
from src.evaluation.benchmarking import measure_inference_time


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection methods")
    parser.add_argument("--gt-path", type=str, required=True,
                        help="Path to ground truth COCO JSON")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory containing prediction JSONs")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for evaluation results")
    parser.add_argument("--run-tide", action="store_true",
                        help="Run TIDE error analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*_predictions.json"))

    all_results = []

    for pred_path in pred_files:
        method_name = pred_path.stem.replace("_predictions", "")
        print(f"\nEvaluating {method_name}...")

        with open(pred_path) as f:
            predictions = json.load(f)

        if not predictions:
            print(f"  No predictions for {method_name}, skipping")
            continue

        # COCO evaluation
        results = evaluate_coco(args.gt_path, predictions)
        results["method"] = method_name
        all_results.append(results)

        print(f"  AP@0.50: {results['AP@0.50']:.4f}")
        print(f"  AP@0.50:0.95: {results['AP@0.50:0.95']:.4f}")
        print(f"  AR@100: {results['AR@100']:.4f}")

        # TIDE analysis
        if args.run_tide:
            tide_dir = output_dir / "tide"
            run_tide_analysis(args.gt_path, str(pred_path), tide_dir, method_name)

    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = output_dir / "evaluation_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
