"""Generate all figures for the final report."""

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import plot_ap_comparison, plot_pareto, plot_tide_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--results-csv", type=str, required=True,
                        help="Path to evaluation_summary.csv")
    parser.add_argument("--tide-results", type=str, default=None,
                        help="Path to TIDE results JSON")
    parser.add_argument("--benchmark-csv", type=str, default=None,
                        help="Path to benchmarking CSV")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Output directory for figures")
    return parser.parse_args()


CLASSICAL_METHODS = {"ca_cfar", "go_cfar", "so_cfar", "os_cfar",
                     "morphological", "wavelet", "otsu", "ssotsu"}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    df = pd.read_csv(args.results_csv)
    df["type"] = df["method"].apply(lambda m: "classical" if m in CLASSICAL_METHODS else "dl")

    # AP comparison bar charts
    for metric in ["AP@0.50", "AP@0.75", "AP@0.50:0.95"]:
        safe_name = metric.replace("@", "_at_").replace(":", "_").replace(".", "")
        plot_ap_comparison(df, metric=metric, output_path=output_dir / f"ap_{safe_name}.png")
        print(f"Saved {metric} comparison plot")

    # AR comparison
    if "AR@100" in df.columns:
        plot_ap_comparison(df, metric="AR@100", output_path=output_dir / "ar_100.png")

    # Pareto plot (needs benchmark data)
    if args.benchmark_csv:
        bench_df = pd.read_csv(args.benchmark_csv)
        merged = df.merge(bench_df, on="method", how="inner")
        merged["type"] = merged["method"].apply(
            lambda m: "classical" if m in CLASSICAL_METHODS else "dl"
        )
        plot_pareto(merged, output_path=output_dir / "pareto.png")
        print("Saved Pareto plot")

    # TIDE plot
    if args.tide_results:
        import json
        with open(args.tide_results) as f:
            tide_data = json.load(f)
        plot_tide_comparison(tide_data, output_path=output_dir / "tide_errors.png")
        print("Saved TIDE error plot")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
