"""Generate all figures for the final report."""

import argparse
import json
from pathlib import Path

import numpy as np
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
    parser.add_argument("--bootstrap-csv", type=str, default=None,
                        help="Path to bootstrap_distributions.csv")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Output directory for figures")
    return parser.parse_args()


CLASSICAL_METHODS = {"ca_cfar", "go_cfar", "so_cfar", "os_cfar",
                     "morphological", "wavelet", "otsu", "ssotsu"}
DL_METHODS = {"yolov11", "faster_rcnn", "unet", "rt_detr"}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)

    # Filter to only show classical + DL mean rows for main comparison
    display = df[df["type"].isin(["classical", "dl_mean"])].copy()
    if display.empty:
        display = df[~df["type"].isin(["dl_std"])].copy()
    display["type"] = display["method"].apply(
        lambda m: "classical" if m in CLASSICAL_METHODS else "dl"
    )

    # AP comparison bar charts
    for metric in ["AP@0.50", "AP@0.75", "AP@0.50:0.95"]:
        if metric not in display.columns:
            continue
        safe_name = metric.replace("@", "_at_").replace(":", "_").replace(".", "")
        plot_ap_comparison(display, metric=metric, output_path=output_dir / f"ap_{safe_name}.png")
        print(f"Saved {metric} comparison plot")

    # AR comparison
    if "AR@100" in display.columns:
        plot_ap_comparison(display, metric="AR@100", output_path=output_dir / "ar_100.png")
        print("Saved AR@100 comparison plot")

    # Size-stratified AP
    for size_metric in ["AP_small", "AP_medium", "AP_large"]:
        if size_metric in display.columns:
            plot_ap_comparison(
                display, metric=size_metric,
                output_path=output_dir / f"{size_metric.lower()}.png"
            )
            print(f"Saved {size_metric} comparison plot")

    # Pareto plot
    if args.benchmark_csv:
        bench_df = pd.read_csv(args.benchmark_csv)
        merged = display.merge(bench_df, on="method", how="inner")
        merged["type"] = merged["method"].apply(
            lambda m: "classical" if m in CLASSICAL_METHODS else "dl"
        )
        plot_pareto(merged, output_path=output_dir / "pareto.png")
        print("Saved Pareto plot")

    # TIDE plot
    if args.tide_results:
        with open(args.tide_results) as f:
            tide_data = json.load(f)
        plot_tide_comparison(tide_data, output_path=output_dir / "tide_errors.png")
        print("Saved TIDE error plot")

    # Bootstrap confidence interval plot
    if args.bootstrap_csv and Path(args.bootstrap_csv).exists():
        _plot_bootstrap_ci(args.bootstrap_csv, output_dir)

    # LaTeX summary table
    _generate_latex_table(display, output_dir)

    print(f"\nAll figures saved to {output_dir}")


def _plot_bootstrap_ci(bootstrap_csv, output_dir):
    """Plot bootstrap confidence intervals for DL methods."""
    import matplotlib.pyplot as plt
    from src.visualization.plots import GARNET, ATLANTIC, CONGAREE, HORSESHOE, BLACK, WHITE

    boot_df = pd.read_csv(bootstrap_csv)
    methods = boot_df.columns.tolist()
    colors = [GARNET, ATLANTIC, CONGAREE, HORSESHOE]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        vals = boot_df[method].values
        mean = np.mean(vals)
        ci_low = np.percentile(vals, 2.5)
        ci_high = np.percentile(vals, 97.5)

        color = colors[i % len(colors)]
        ax.barh(i, mean, color=color, edgecolor=BLACK, linewidth=0.5, height=0.6)
        ax.errorbar(mean, i, xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt="none", ecolor=BLACK, capsize=4, linewidth=1.5)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("AP@0.50 (Bootstrap Mean)")
    ax.set_title("Bootstrap 95% Confidence Intervals")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_dir / "bootstrap_ci.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved bootstrap CI plot")


def _generate_latex_table(df, output_dir):
    """Generate a LaTeX-formatted results table."""
    metric_cols = ["AP@0.50", "AP@0.75", "AP@0.50:0.95", "AR@100"]
    available = [c for c in metric_cols if c in df.columns]

    if not available:
        return

    table_df = df[["method", "type"] + available].copy()
    for col in available:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    # Find best values per column
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection performance across all methods.}",
        r"\label{tab:results}",
        r"\begin{tabular}{ll" + "c" * len(available) + "}",
        r"\toprule",
        "Method & Type & " + " & ".join(available) + r" \\",
        r"\midrule",
    ]

    for _, row in table_df.iterrows():
        line = f"{row['method']} & {row['type']}"
        for col in available:
            line += f" & {row[col]}"
        line += r" \\"
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = output_dir / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print("Saved LaTeX results table")


if __name__ == "__main__":
    main()
