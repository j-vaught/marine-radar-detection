"""Visualization utilities for results comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Brand colors
GARNET = "#73000A"
BLACK = "#000000"
WHITE = "#FFFFFF"
ROSE = "#CC2E40"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
HORSESHOE = "#65780B"
GRASS = "#CED318"
HONEYCOMB = "#A49137"
BLACK_90 = "#363636"
BLACK_70 = "#5C5C5C"
BLACK_50 = "#A2A2A2"
WARM_GREY = "#676156"

# Palette for methods
CLASSICAL_COLORS = [GARNET, ROSE, CONGAREE, HORSESHOE, HONEYCOMB, WARM_GREY, BLACK_70]
DL_COLORS = [ATLANTIC, BLACK, GRASS, BLACK_90]

plt.rcParams.update({
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": BLACK,
    "text.color": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "font.family": "sans-serif",
})


def plot_ap_comparison(
    results: pd.DataFrame,
    metric: str = "AP@0.50",
    output_path: str | Path | None = None,
):
    """Bar chart comparing AP across all methods.

    Args:
        results: DataFrame with 'method', 'type' (classical/dl), and metric columns.
        metric: Which AP metric to plot.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    classical = results[results["type"] == "classical"]
    dl = results[results["type"] == "dl"]

    x_pos = np.arange(len(results))

    colors = []
    for _, row in results.iterrows():
        if row["type"] == "classical":
            idx = list(classical["method"]).index(row["method"])
            colors.append(CLASSICAL_COLORS[idx % len(CLASSICAL_COLORS)])
        else:
            idx = list(dl["method"]).index(row["method"])
            colors.append(DL_COLORS[idx % len(DL_COLORS)])

    bars = ax.bar(x_pos, results[metric], color=colors, edgecolor=BLACK, linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results["method"], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison: Classical vs. Deep Learning")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    results: pd.DataFrame,
    x_metric: str = "mean_ms",
    y_metric: str = "AP@0.50:0.95",
    output_path: str | Path | None = None,
):
    """Accuracy vs. latency Pareto plot.

    Args:
        results: DataFrame with method names, latency, and AP metrics.
        x_metric: Latency column name.
        y_metric: Accuracy column name.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (_, row) in enumerate(results.iterrows()):
        if row["type"] == "classical":
            color = GARNET
            marker = "s"
        else:
            color = ATLANTIC
            marker = "o"

        ax.scatter(
            row[x_metric], row[y_metric],
            c=color, marker=marker, s=100, edgecolors=BLACK, linewidth=0.5, zorder=3,
        )
        ax.annotate(
            row["method"], (row[x_metric], row[y_metric]),
            textcoords="offset points", xytext=(8, 4), fontsize=8, color=BLACK_70,
        )

    # Compute and draw Pareto front
    points = results[[x_metric, y_metric]].values
    pareto_idx = _pareto_front(points[:, 0], points[:, 1], minimize_x=True, maximize_y=True)
    pareto_points = points[pareto_idx]
    pareto_points = pareto_points[pareto_points[:, 0].argsort()]
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], "--", color=ROSE, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel(y_metric)
    ax.set_title("Accuracy vs. Latency Pareto Analysis")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GARNET, markersize=10, label="Classical"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ATLANTIC, markersize=10, label="Deep Learning"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pareto_front(
    x: np.ndarray, y: np.ndarray,
    minimize_x: bool = True, maximize_y: bool = True,
) -> np.ndarray:
    """Find Pareto-optimal indices."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] <= x[i]) if minimize_x else (x[j] >= x[i])
            y_better = (y[j] >= y[i]) if maximize_y else (y[j] <= y[i])
            x_strict = (x[j] < x[i]) if minimize_x else (x[j] > x[i])
            y_strict = (y[j] > y[i]) if maximize_y else (y[j] < y[i])
            if x_better and y_better and (x_strict or y_strict):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def plot_tide_comparison(
    tide_results: list[dict],
    output_path: str | Path | None = None,
):
    """Stacked bar chart of TIDE error components across methods."""
    df = pd.DataFrame(tide_results)
    error_cols = ["classification", "localization", "cls_and_loc", "duplicate", "background", "missed"]
    colors = [GARNET, ATLANTIC, CONGAREE, HORSESHOE, HONEYCOMB, BLACK_50]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for col, color in zip(error_cols, colors):
        ax.bar(x, df[col], bottom=bottom, label=col.replace("_", " ").title(),
               color=color, edgecolor=BLACK, linewidth=0.5)
        bottom += df[col].values

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=45, ha="right")
    ax.set_ylabel("Error Contribution")
    ax.set_title("TIDE Error Decomposition")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
