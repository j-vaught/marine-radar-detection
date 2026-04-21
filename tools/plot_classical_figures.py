#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "figures" / "classical"

CFAR_MATRIX = RESULTS / "cfar" / "matrix.csv"
OTSU_MATRIX = RESULTS / "otsu" / "matrix.csv"

# Official University of South Carolina brand colors:
# https://sc.edu/about/offices_and_divisions/communications/toolbox/colors/
BRAND = {
    "garnet": "#73000a",
    "black": "#000000",
    "white": "#ffffff",
    "black_90": "#363636",
    "black_70": "#5C5C5C",
    "black_50": "#A2A2A2",
    "black_30": "#C7C7C7",
    "black_10": "#ECECEC",
    "warm_grey": "#676156",
    "sandstorm": "#FFF2E3",
    "rose": "#CC2E40",
    "atlantic": "#466A9F",
    "congaree": "#1F414D",
    "horseshoe": "#65780B",
    "honeycomb": "#A49137",
}

METHOD_ORDER = ["CA", "OS", "OTSU", "SSOTSU"]
METHOD_COLORS = {
    "CA": BRAND["garnet"],
    "OS": BRAND["black_90"],
    "OTSU": BRAND["atlantic"],
    "SSOTSU": BRAND["honeycomb"],
}
CONDITION_ORDER = ["P", "S", "T"]
CONDITION_LABELS = {
    "P": "Pooled",
    "S": "Site",
    "T": "Transfer",
}
SITE_ORDER = ["MANV", "DARC", "DAAN"]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BRAND["white"],
            "axes.facecolor": BRAND["white"],
            "axes.edgecolor": BRAND["black_70"],
            "axes.labelcolor": BRAND["black_90"],
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": BRAND["black_10"],
            "grid.linewidth": 0.8,
            "font.family": "serif",
            "font.serif": ["Georgia", "DejaVu Serif"],
            "font.size": 12,
            "text.color": BRAND["black_90"],
            "xtick.color": BRAND["black_90"],
            "ytick.color": BRAND["black_90"],
            "legend.frameon": False,
            "legend.labelcolor": BRAND["black_90"],
            "savefig.facecolor": BRAND["white"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def combined_rows() -> list[dict[str, str]]:
    return load_rows(CFAR_MATRIX) + load_rows(OTSU_MATRIX)


def percent_formatter(_: float, __: int) -> str:
    return f"{_:,.0f}%"


def parse_site(label: str) -> str:
    for site in SITE_ORDER:
        if site in label:
            return site
    raise ValueError(f"could not parse site from {label!r}")


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.svg")
    plt.close(fig)


def metric_value(row: dict[str, str], key: str) -> float:
    aliases = {
        "test_ap": ("test_ap", "test_AP"),
        "test_ap50": ("test_ap50", "test_AP50"),
        "test_ap75": ("test_ap75", "test_AP75"),
        "test_recall_at_p90": ("test_recall_at_p90", "test_recall_at_p90"),
        "test_fa_per_image": ("test_fa_per_image", "test_fa_per_image"),
        "total_runtime_sec": ("total_runtime_sec",),
    }
    for alias in aliases[key]:
        if alias in row and row[alias] != "":
            return float(row[alias])
    raise KeyError(f"missing metric {key!r} in row")


def mean_metric(rows: list[dict[str, str]], key: str) -> float:
    return mean(metric_value(row, key) for row in rows)


def plot_condition_metric(rows: list[dict[str, str]], metric_key: str, y_label: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for condition in CONDITION_ORDER:
            subset = [
                row
                for row in rows
                if row["method"] == method and row["condition"] == condition
            ]
            values.append(mean_metric(subset, metric_key))
        if "ap" in metric_key:
            values = [value * 100.0 for value in values]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
    ax.set_xlabel("Evaluation Setting")
    ax.set_ylabel(y_label)
    if "ap" in metric_key:
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4)
    save(fig, stem)


def plot_transfer_target_ap50(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(SITE_ORDER))
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for site in SITE_ORDER:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and row["condition"] == "T"
                and parse_site(row["test_set"]) == site
            ]
            values.append(mean_metric(subset, "test_ap50") * 100.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SITE_ORDER)
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Mean AP50 (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4)
    save(fig, "classical_transfer_target_ap50")


def plot_runtime(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    values = [
        mean_metric([row for row in rows if row["method"] == method], "total_runtime_sec")
        for method in METHOD_ORDER
    ]
    y = np.arange(len(METHOD_ORDER))
    ax.barh(
        y,
        values,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        height=0.65,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_xlabel("Mean Runtime / Experiment (s)")
    ax.set_ylabel("Method")
    save(fig, "classical_runtime_by_method")


def plot_otsu_winner_params(rows: list[dict[str, str]]) -> None:
    otsu_rows = [row for row in rows if row["method"] == "OTSU"]
    specs = [
        ("win_min_area", "Min Area"),
        ("win_dilation", "Dilation"),
        ("win_box_padding", "Box Padding"),
    ]

    fig, axes = plt.subplots(1, len(specs), figsize=(12.2, 3.8), sharey=True)
    for axis, (key, xlabel) in zip(axes, specs):
        counts = Counter(row[key] for row in otsu_rows)
        labels = sorted(counts, key=lambda value: float(value))
        values = [counts[label] for label in labels]
        axis.bar(
            labels,
            values,
            color=BRAND["garnet"],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Selection Count")
        axis.set_facecolor(BRAND["sandstorm"])
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    save(fig, "otsu_winner_parameter_counts")


def plot_ssotsu_winner_params(rows: list[dict[str, str]]) -> None:
    ssotsu_rows = [row for row in rows if row["method"] == "SSOTSU"]
    specs = [
        ("win_block_size", "Block Size"),
        ("win_seed_threshold_factor", "Seed Threshold Factor"),
        ("win_region_grow_threshold", "Region Grow Threshold"),
        ("win_min_area", "Min Area"),
    ]

    fig, axes = plt.subplots(1, len(specs), figsize=(15.4, 3.8), sharey=True)
    for axis, (key, xlabel) in zip(axes, specs):
        counts = Counter(row[key] for row in ssotsu_rows)
        labels = sorted(counts, key=lambda value: float(value))
        values = [counts[label] for label in labels]
        axis.bar(
            labels,
            values,
            color=BRAND["atlantic"],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Selection Count")
        axis.set_facecolor(BRAND["sandstorm"])
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    save(fig, "ssotsu_winner_parameter_counts")


def main() -> None:
    configure_matplotlib()
    rows = combined_rows()
    plot_condition_metric(
        rows,
        metric_key="test_ap50",
        y_label="Mean AP50 (%)",
        stem="classical_ap50_by_condition",
    )
    plot_condition_metric(
        rows,
        metric_key="test_ap",
        y_label="Mean AP (%)",
        stem="classical_ap_by_condition",
    )
    plot_condition_metric(
        rows,
        metric_key="test_fa_per_image",
        y_label="Mean False Alarms / Image",
        stem="classical_false_alarms_by_condition",
    )
    plot_transfer_target_ap50(rows)
    plot_runtime(rows)
    plot_otsu_winner_params(rows)
    plot_ssotsu_winner_params(rows)


if __name__ == "__main__":
    main()
