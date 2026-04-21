#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "figures" / "learned"

UNET_MATRIX = RESULTS / "unet_matrix" / "matrix.csv"
FRCNN_MATRIX = RESULTS / "faster_rcnn_matrix" / "matrix.csv"
RTDETR_MATRIX = RESULTS / "rtdetr_matrix" / "matrix.csv"
YOLO26_MATRIX = RESULTS / "yolo26_matrix" / "matrix.csv"

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

METHOD_ORDER = ["UNET", "FRCNN", "RTDETR", "Y26"]
METHOD_LABELS = {
    "UNET": "U-Net",
    "FRCNN": "Faster R-CNN",
    "RTDETR": "RT-DETR",
    "Y26": "YOLO26",
}
METHOD_COLORS = {
    "UNET": BRAND["atlantic"],
    "FRCNN": BRAND["garnet"],
    "RTDETR": BRAND["congaree"],
    "Y26": BRAND["honeycomb"],
}
CONDITION_ORDER = ["P", "S", "T"]
CONDITION_LABELS = {
    "P": "Pooled",
    "S": "Site",
    "T": "Transfer",
}
SITE_ORDER = ["MANV", "DARC", "DAAN"]
SITE_CODES = {
    "MANV": "Mv",
    "DARC": "Dc",
    "DAAN": "Da",
}
SITE_PERMS = [
    ("MANV", "DARC", "DAAN"),
    ("MANV", "DAAN", "DARC"),
    ("DARC", "MANV", "DAAN"),
    ("DARC", "DAAN", "MANV"),
    ("DAAN", "MANV", "DARC"),
    ("DAAN", "DARC", "MANV"),
]
TRANSFER_CMAP = LinearSegmentedColormap.from_list(
    "usc_transfer",
    [BRAND["sandstorm"], BRAND["honeycomb"], BRAND["garnet"]],
)


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


def load_rows(path: Path, method: str) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["method"] = method
    return rows


def combined_rows() -> list[dict[str, str]]:
    return (
        load_rows(UNET_MATRIX, "UNET")
        + load_rows(FRCNN_MATRIX, "FRCNN")
        + load_rows(RTDETR_MATRIX, "RTDETR")
        + load_rows(YOLO26_MATRIX, "Y26")
    )


def metric_value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean_metric(rows: list[dict[str, str]], key: str) -> float:
    return mean(metric_value(row, key) for row in rows)


def percent_formatter(value: float, _: int) -> str:
    return f"{value:,.0f}%"


def minute_formatter(value: float, _: int) -> str:
    return f"{value:,.0f}m"


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


def plot_condition_metric(rows: list[dict[str, str]], metric_key: str, y_label: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for condition in CONDITION_ORDER:
            subset = [row for row in rows if row["method"] == method and row["condition"] == condition]
            values.append(mean_metric(subset, metric_key))
        if metric_key.startswith("test_ap"):
            values = [value * 100.0 for value in values]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
    ax.set_xlabel("Evaluation Setting")
    ax.set_ylabel(y_label)
    if metric_key.startswith("test_ap"):
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    save(fig, stem)


def plot_pooled_seed_ap50(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    seeds = list(range(5))
    x = np.arange(len(seeds))
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for seed in seeds:
            subset = [
                row
                for row in rows
                if row["method"] == method and row["condition"] == "P" and int(row["seed"]) == seed
            ]
            values.append(mean_metric(subset, "test_ap50") * 100.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {seed}" for seed in seeds])
    ax.set_xlabel("Pooled Split Seed")
    ax.set_ylabel("AP50 (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    save(fig, "learned_pooled_seed_ap50")


def plot_site_permutation_ap50(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(SITE_PERMS))
    width = 0.18
    labels = [f"{SITE_CODES[t]}-{SITE_CODES[v]}-{SITE_CODES[s]}" for t, v, s in SITE_PERMS]

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for tune, val, test in SITE_PERMS:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and row["condition"] == "S"
                and row["tune_label"] == tune
                and row["val_label"] == val
                and row["test_label"] == test
            ]
            values.append(mean_metric(subset, "test_ap50") * 100.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Tune-Val-Test Site Ordering")
    ax.set_ylabel("AP50 (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4)
    save(fig, "learned_site_permutation_ap50")


def plot_transfer_target_ap50(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
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
                and parse_site(row["test_label"]) == site
            ]
            values.append(mean_metric(subset, "test_ap50") * 100.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SITE_ORDER)
    ax.set_xlabel("Transfer Test Dataset")
    ax.set_ylabel("Mean AP50 (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    save(fig, "learned_transfer_target_ap50")


def plot_transfer_heatmaps(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for axis, method in zip(axes_flat, METHOD_ORDER, strict=True):
        matrix = np.zeros((len(SITE_ORDER), len(SITE_ORDER)))
        for row_idx, tune_site in enumerate(SITE_ORDER):
            for col_idx, test_site in enumerate(SITE_ORDER):
                subset = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["condition"] == "T"
                    and parse_site(row["tune_label"]) == tune_site
                    and parse_site(row["test_label"]) == test_site
                ]
                matrix[row_idx, col_idx] = mean_metric(subset, "test_ap50") * 100.0

        image = axis.imshow(
            matrix,
            cmap=TRANSFER_CMAP,
            vmin=0.0,
            vmax=100.0,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(METHOD_LABELS[method], color=METHOD_COLORS[method], fontweight="bold")
        axis.set_facecolor(BRAND["sandstorm"])
        axis.grid(False)
        axis.set_xticks(np.arange(len(SITE_ORDER)))
        axis.set_yticks(np.arange(len(SITE_ORDER)))
        axis.set_xticklabels(SITE_ORDER)
        axis.set_yticklabels(SITE_ORDER)
        axis.set_xlabel("Test Site")
        axis.set_ylabel("Tune Site")

        for row_idx in range(len(SITE_ORDER)):
            for col_idx in range(len(SITE_ORDER)):
                value = matrix[row_idx, col_idx]
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=BRAND["black_90"] if value < 65.0 else BRAND["white"],
                    fontsize=11,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=axes_flat, shrink=0.84, pad=0.02)
    cbar.set_label("AP50 (%)")
    save(fig, "learned_transfer_ap50_heatmaps")


def plot_runtime(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    values = [
        mean_metric([row for row in rows if row["method"] == method], "train_seconds") / 60.0
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
    ax.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_xlabel("Mean Train Time / Experiment (min)")
    ax.set_ylabel("Method")
    ax.xaxis.set_major_formatter(FuncFormatter(minute_formatter))
    save(fig, "learned_train_runtime_by_method")


def plot_predictions_by_condition(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for condition in CONDITION_ORDER:
            subset = [row for row in rows if row["method"] == method and row["condition"] == condition]
            values.append(mean_metric(subset, "preds_per_image"))
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
    ax.set_xlabel("Evaluation Setting")
    ax.set_ylabel("Predictions / Image")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    save(fig, "learned_predictions_by_condition")


def plot_unet_winner_params(rows: list[dict[str, str]]) -> None:
    unet_rows = [row for row in rows if row["method"] == "UNET"]
    specs = [
        ("threshold", "Threshold"),
        ("min_area", "Min Area"),
        ("max_area", "Max Area"),
        ("dilation", "Dilation"),
        ("box_padding", "Box Padding"),
    ]

    fig, axes = plt.subplots(1, len(specs), figsize=(16.4, 3.8), sharey=True)
    for axis, (key, xlabel) in zip(axes, specs, strict=True):
        counts = Counter(row[key] for row in unet_rows)
        labels = sorted(counts, key=lambda value: float(value))
        values = [counts[label] for label in labels]
        axis.bar(
            labels,
            values,
            color=METHOD_COLORS["UNET"],
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Selection Count")
        axis.set_facecolor(BRAND["sandstorm"])
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    save(fig, "learned_unet_winner_parameter_counts")


def main() -> None:
    configure_matplotlib()
    rows = combined_rows()
    plot_condition_metric(
        rows,
        metric_key="test_ap50",
        y_label="Mean AP50 (%)",
        stem="learned_ap50_by_condition",
    )
    plot_condition_metric(
        rows,
        metric_key="test_ap",
        y_label="Mean AP (%)",
        stem="learned_ap_by_condition",
    )
    plot_condition_metric(
        rows,
        metric_key="test_ap75",
        y_label="Mean AP75 (%)",
        stem="learned_ap75_by_condition",
    )
    plot_pooled_seed_ap50(rows)
    plot_site_permutation_ap50(rows)
    plot_transfer_target_ap50(rows)
    plot_transfer_heatmaps(rows)
    plot_runtime(rows)
    plot_predictions_by_condition(rows)
    plot_unet_winner_params(rows)


if __name__ == "__main__":
    main()
