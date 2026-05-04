#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_IMAGES = ROOT / "data" / "processed" / "images"
OUT_DIR = RESULTS / "figures" / "paper"
ZOOM_DIR = OUT_DIR / "failure_zoomed"
ZOOM_CACHE_DIR = ZOOM_DIR / "_classical_cache"

MATRIX_PATHS = {
    "CFAR": RESULTS / "cfar" / "matrix.csv",
    "OTSU": RESULTS / "otsu" / "matrix.csv",
    "WAVELET": RESULTS / "wavelet" / "matrix.csv",
    "UNET": RESULTS / "unet_matrix" / "matrix.csv",
    "FRCNN": RESULTS / "faster_rcnn_matrix" / "matrix.csv",
    "RTDETR": RESULTS / "rtdetr_matrix" / "matrix.csv",
    "YOLO26": RESULTS / "yolo26_matrix" / "matrix.csv",
}

CLASSICAL_METHOD_ORDER = ["CA", "OS", "OTSU", "SSOTSU", "WAV"]
LEARNED_METHOD_ORDER = ["UNET", "FRCNN", "RTDETR", "Y26"]
METHOD_ORDER = CLASSICAL_METHOD_ORDER + LEARNED_METHOD_ORDER

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

METHOD_LABELS = {
    "CA": "CA-CFAR",
    "OS": "OS-CFAR",
    "OTSU": "Otsu",
    "SSOTSU": "SS-Otsu",
    "WAV": "Wavelet",
    "UNET": "U-Net",
    "FRCNN": "Faster R-CNN",
    "RTDETR": "RT-DETR",
    "Y26": "YOLO26",
}
METHOD_FAMILY = {
    "CA": "Classical",
    "OS": "Classical",
    "OTSU": "Classical",
    "SSOTSU": "Classical",
    "WAV": "Classical",
    "UNET": "Learned",
    "FRCNN": "Learned",
    "RTDETR": "Learned",
    "Y26": "Learned",
}
METHOD_COLORS = {
    "CA": BRAND["garnet"],
    "OS": BRAND["black_90"],
    "OTSU": BRAND["atlantic"],
    "SSOTSU": BRAND["honeycomb"],
    "WAV": BRAND["warm_grey"],
    "UNET": "#5B7DB1",
    "FRCNN": "#980018",
    "RTDETR": BRAND["congaree"],
    "Y26": "#AD9B3A",
}
METHOD_PRED_COLORS = {
    "CA": "#A10F1B",
    "OS": "#B36B00",
    "OTSU": "#2E5FA7",
    "SSOTSU": "#A07E00",
    "WAV": "#6D7788",
    "UNET": "#2E5FA7",
    "FRCNN": "#8E0013",
    "RTDETR": "#17404B",
    "Y26": "#9E8C2A",
}
CONDITION_ORDER = ["P", "S", "T"]
CONDITION_LABELS = {
    "P": "Pooled",
    "S": "Site",
    "T": "Transfer",
}
SITE_ORDER = ["MANV", "DARC", "DAAN"]
SITE_CODES = {"MANV": "Mv", "DARC": "Dc", "DAAN": "Da"}
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


@dataclass
class NormalizedRow:
    method: str
    family: str
    condition: str
    exp_id: str
    seed: int
    tune_label: str
    val_label: str
    test_label: str
    test_ap: float
    test_ap50: float
    test_ap75: float
    runtime_sec: float
    detection_volume: float | None
    raw: dict[str, str]


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


def save(fig: plt.Figure, stem: str, *, svg: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    if svg:
        fig.savefig(OUT_DIR / f"{stem}.svg")
    plt.close(fig)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_site(label: str) -> str:
    for site in SITE_ORDER:
        if site in label:
            return site
    raise ValueError(f"could not parse site from {label!r}")


def normalize_rows() -> list[NormalizedRow]:
    rows: list[NormalizedRow] = []

    for row in load_csv(MATRIX_PATHS["CFAR"]):
        rows.append(
            NormalizedRow(
                method=row["method"],
                family=METHOD_FAMILY[row["method"]],
                condition=row["condition"],
                exp_id=row["exp_id"],
                seed=int(row["seed"]),
                tune_label=row["tune_set"],
                val_label=row["val_set"],
                test_label=row["test_set"],
                test_ap=float(row["test_AP"]),
                test_ap50=float(row["test_AP50"]),
                test_ap75=float(row["test_AP75"]),
                runtime_sec=float(row["total_runtime_sec"]),
                detection_volume=float(row["test_fa_per_image"]),
                raw=row,
            )
        )

    for row in load_csv(MATRIX_PATHS["OTSU"]):
        rows.append(
            NormalizedRow(
                method=row["method"],
                family=METHOD_FAMILY[row["method"]],
                condition=row["condition"],
                exp_id=row["exp_id"],
                seed=int(row["seed"]),
                tune_label=row["tune_set"],
                val_label=row["val_set"],
                test_label=row["test_set"],
                test_ap=float(row["test_ap"]),
                test_ap50=float(row["test_ap50"]),
                test_ap75=float(row["test_ap75"]),
                runtime_sec=float(row["total_runtime_sec"]),
                detection_volume=float(row["test_fa_per_image"]),
                raw=row,
            )
        )

    for row in load_csv(MATRIX_PATHS["WAVELET"]):
        rows.append(
            NormalizedRow(
                method=row["method"],
                family=METHOD_FAMILY[row["method"]],
                condition=row["condition"],
                exp_id=row["exp_id"],
                seed=int(row["seed"]),
                tune_label=row["tune_set"],
                val_label=row["val_set"],
                test_label=row["test_set"],
                test_ap=float(row["test_ap"]),
                test_ap50=float(row["test_ap50"]),
                test_ap75=float(row["test_ap75"]),
                runtime_sec=float(row["total_runtime_sec"]),
                detection_volume=float(row["test_fa_per_image"]),
                raw=row,
            )
        )

    learned_specs = [
        ("UNET", MATRIX_PATHS["UNET"]),
        ("FRCNN", MATRIX_PATHS["FRCNN"]),
        ("RTDETR", MATRIX_PATHS["RTDETR"]),
        ("Y26", MATRIX_PATHS["YOLO26"]),
    ]
    for method, path in learned_specs:
        for row in load_csv(path):
            rows.append(
                NormalizedRow(
                    method=method,
                    family=METHOD_FAMILY[method],
                    condition=row["condition"],
                    exp_id=row["exp_id"],
                    seed=int(row["seed"]),
                    tune_label=row["tune_label"],
                    val_label=row["val_label"],
                    test_label=row["test_label"],
                    test_ap=float(row["test_ap"]),
                    test_ap50=float(row["test_ap50"]),
                    test_ap75=float(row["test_ap75"]),
                    runtime_sec=float(row["train_seconds"]),
                    detection_volume=float(row["preds_per_image"]),
                    raw=row,
                )
            )
    return rows


def grouped(rows: list[NormalizedRow], *, methods: list[str] | None = None, condition: str | None = None) -> list[NormalizedRow]:
    subset = rows
    if methods is not None:
        allowed = set(methods)
        subset = [row for row in subset if row.method in allowed]
    if condition is not None:
        subset = [row for row in subset if row.condition == condition]
    return subset


def mean_metric(rows: list[NormalizedRow], attr: str) -> float:
    return mean(getattr(row, attr) for row in rows)


def percent_formatter(value: float, _: int) -> str:
    return f"{value:,.0f}%"


def minute_formatter(value: float, _: int) -> str:
    return f"{value:,.0f}m"


def log_second_formatter(value: float, _: int) -> str:
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.0f}m"
    return f"{value:.0f}s"


def scenario_key(row: NormalizedRow) -> tuple[str, str, str, str, int]:
    return (row.condition, row.tune_label, row.val_label, row.test_label, row.seed)


def plot_family_condition_metric(rows: list[NormalizedRow], metric: str, stem: str, y_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    families = [("Classical", CLASSICAL_METHOD_ORDER), ("Learned", LEARNED_METHOD_ORDER)]

    for axis, (family_label, method_order) in zip(axes, families, strict=True):
        x = np.arange(len(CONDITION_ORDER))
        width = 0.8 / len(method_order)
        for idx, method in enumerate(method_order):
            values = []
            for condition in CONDITION_ORDER:
                subset = [row for row in rows if row.method == method and row.condition == condition]
                value = mean_metric(subset, metric)
                if metric.startswith("test_ap"):
                    value *= 100.0
                values.append(value)
            axis.bar(
                x + (idx - (len(method_order) - 1) / 2.0) * width,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor=BRAND["black_70"],
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
        axis.set_title(family_label)
        axis.set_xticks(x)
        axis.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
        axis.set_xlabel("Evaluation Setting")
        axis.grid(axis="x", visible=False)
        if metric.startswith("test_ap"):
            axis.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=min(len(method_order), 3))
    axes[0].set_ylabel(y_label)
    save(fig, stem)


def plot_method_mean_ap50(rows: list[NormalizedRow]) -> None:
    values = [
        (method, mean_metric([row for row in rows if row.method == method], "test_ap50") * 100.0)
        for method in METHOD_ORDER
    ]
    values.sort(key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    y = np.arange(len(values))
    ax.barh(
        y,
        [value for _, value in values],
        color=[METHOD_COLORS[method] for method, _ in values],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        height=0.68,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[method] for method, _ in values])
    ax.set_xlabel("Mean AP50 Across Matrix (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="y", visible=False)
    save(fig, "paper_method_mean_ap50")


def plot_method_mean_ap50_bottom_legend(rows: list[NormalizedRow]) -> None:
    values = [
        (method, mean_metric([row for row in rows if row.method == method], "test_ap50") * 100.0)
        for method in METHOD_ORDER
    ]
    values.sort(key=lambda item: item[1], reverse=True)

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        [value for _, value in values],
        color=[METHOD_COLORS[method] for method, _ in values],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method, _ in values], rotation=25, ha="right")
    ax.set_ylabel("Mean AP50 Across Matrix (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="x", visible=False)
    ax.legend(
        bars,
        [METHOD_LABELS[method] for method, _ in values],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
    )
    save(fig, "paper_method_mean_ap50_bottom_legend")


def plot_method_test_site_ap50_bottom_legend(rows: list[NormalizedRow], test_site: str, stem: str) -> None:
    values = []
    for method in METHOD_ORDER:
        subset = [
            row
            for row in rows
            if row.method == method and test_site in row.test_label
        ]
        if not subset:
            continue
        values.append((method, mean_metric(subset, "test_ap50") * 100.0))
    values.sort(key=lambda item: item[1], reverse=True)

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        [value for _, value in values],
        color=[METHOD_COLORS[method] for method, _ in values],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method, _ in values], rotation=25, ha="right")
    ax.set_ylabel(f"Mean AP50 On {test_site} Test Site (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="x", visible=False)
    ax.legend(
        bars,
        [METHOD_LABELS[method] for method, _ in values],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
    )
    save(fig, stem)


def plot_localization_retention(rows: list[NormalizedRow]) -> None:
    values = []
    for method in METHOD_ORDER:
        subset = [row for row in rows if row.method == method]
        ap50 = mean_metric(subset, "test_ap50")
        ap75 = mean_metric(subset, "test_ap75")
        retention = 0.0 if ap50 <= 0.0 else (ap75 / ap50) * 100.0
        values.append((method, retention))
    values.sort(key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    y = np.arange(len(values))
    ax.barh(
        y,
        [value for _, value in values],
        color=[METHOD_COLORS[method] for method, _ in values],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        height=0.68,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[method] for method, _ in values])
    ax.set_xlabel("AP75 / AP50 Retention (%)")
    ax.set_xlim(0.0, 105.0)
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="y", visible=False)
    save(fig, "paper_localization_retention")


def compute_ranks(rows: list[NormalizedRow]) -> dict[tuple[str, tuple[str, str, str, str, int]], float]:
    grouped_rows: dict[tuple[str, str, str, str, int], list[NormalizedRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[scenario_key(row)].append(row)

    ranks: dict[tuple[str, tuple[str, str, str, str, int]], float] = {}
    for key, scenario_rows in grouped_rows.items():
        sorted_rows = sorted(scenario_rows, key=lambda row: row.test_ap50, reverse=True)
        for rank, row in enumerate(sorted_rows, start=1):
            ranks[(row.method, key)] = float(rank)
    return ranks


def plot_method_mean_rank(rows: list[NormalizedRow]) -> None:
    ranks = compute_ranks(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.08

    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for condition in CONDITION_ORDER:
            condition_ranks = [
                rank
                for (rank_method, key), rank in ranks.items()
                if rank_method == method and key[0] == condition
            ]
            values.append(mean(condition_ranks))
        ax.bar(
            x + (idx - (len(METHOD_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.5,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
    ax.set_xlabel("Evaluation Setting")
    ax.set_ylabel("Mean AP50 Rank (Lower Is Better)")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5)
    save(fig, "paper_method_mean_rank_by_condition")


def plot_pareto_runtime(rows: list[NormalizedRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for method in METHOD_ORDER:
        subset = [row for row in rows if row.method == method]
        runtime = mean_metric(subset, "runtime_sec")
        ap50 = mean_metric(subset, "test_ap50") * 100.0
        ax.scatter(
            runtime,
            ap50,
            s=140,
            color=METHOD_COLORS[method],
            edgecolor=BRAND["black_70"],
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            METHOD_LABELS[method],
            (runtime, ap50),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=11,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean Runtime / Experiment (log scale)")
    ax.set_ylabel("Mean AP50 (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(log_second_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    save(fig, "paper_pareto_ap50_runtime")


def plot_transfer_heatmaps(rows: list[NormalizedRow]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.6), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for axis, method in zip(axes_flat, METHOD_ORDER, strict=True):
        matrix = np.zeros((len(SITE_ORDER), len(SITE_ORDER)))
        for row_idx, tune_site in enumerate(SITE_ORDER):
            for col_idx, test_site in enumerate(SITE_ORDER):
                subset = [
                    row
                    for row in rows
                    if row.method == method
                    and row.condition == "T"
                    and parse_site(row.tune_label) == tune_site
                    and parse_site(row.test_label) == test_site
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
                    fontsize=10,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=axes_flat, shrink=0.84, pad=0.02)
    cbar.set_label("AP50 (%)")
    save(fig, "paper_transfer_heatmaps_all_methods")


def plot_transfer_gap(rows: list[NormalizedRow]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    x = np.arange(len(METHOD_ORDER))
    width = 0.33

    diagonal_values = []
    offdiag_values = []
    for method in METHOD_ORDER:
        diagonal_rows = [
            row
            for row in rows
            if row.method == method and row.condition == "T" and parse_site(row.tune_label) == parse_site(row.test_label)
        ]
        offdiag_rows = [
            row
            for row in rows
            if row.method == method and row.condition == "T" and parse_site(row.tune_label) != parse_site(row.test_label)
        ]
        diagonal_values.append(mean_metric(diagonal_rows, "test_ap50") * 100.0)
        offdiag_values.append(mean_metric(offdiag_rows, "test_ap50") * 100.0)

    ax.bar(
        x - width / 2,
        diagonal_values,
        width=width,
        color=BRAND["garnet"],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        label="Same-Site Transfer",
    )
    ax.bar(
        x + width / 2,
        offdiag_values,
        width=width,
        color=BRAND["black_50"],
        edgecolor=BRAND["black_70"],
        linewidth=0.6,
        label="Cross-Site Transfer",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], rotation=25, ha="right")
    ax.set_ylabel("AP50 (%)")
    ax.set_xlabel("Method")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    save(fig, "paper_transfer_generalization_gap")


def plot_transfer_target_site_breakdown(rows: list[NormalizedRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True)
    families = [("Classical", CLASSICAL_METHOD_ORDER), ("Learned", LEARNED_METHOD_ORDER)]

    for axis, (family_label, method_order) in zip(axes, families, strict=True):
        x = np.arange(len(SITE_ORDER))
        width = 0.8 / len(method_order)
        for idx, method in enumerate(method_order):
            values = []
            for test_site in SITE_ORDER:
                subset = [
                    row
                    for row in rows
                    if row.method == method
                    and row.condition == "T"
                    and parse_site(row.test_label) == test_site
                ]
                values.append(mean_metric(subset, "test_ap50") * 100.0)
            axis.bar(
                x + (idx - (len(method_order) - 1) / 2.0) * width,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor=BRAND["black_70"],
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
        axis.text(
            0.02,
            0.90,
            family_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(SITE_ORDER)
        axis.set_xlabel("Transfer Test Site")
        axis.grid(axis="x", visible=False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(len(method_order), 3))
    axes[0].set_ylabel("AP50 (%)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    save(fig, "paper_transfer_target_site_breakdown")


def plot_pooled_seed_variability(rows: list[NormalizedRow]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    data = []
    for method in METHOD_ORDER:
        subset = [row.test_ap50 * 100.0 for row in rows if row.method == method and row.condition == "P"]
        data.append(subset)

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": BRAND["black_90"], "linewidth": 1.2},
        whiskerprops={"color": BRAND["black_70"], "linewidth": 1.0},
        capprops={"color": BRAND["black_70"], "linewidth": 1.0},
        boxprops={"edgecolor": BRAND["black_70"], "linewidth": 1.0},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER, strict=True):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.75)

    for idx, method in enumerate(METHOD_ORDER, start=1):
        values = [row.test_ap50 * 100.0 for row in rows if row.method == method and row.condition == "P"]
        offsets = np.linspace(-0.12, 0.12, num=len(values))
        ax.scatter(
            np.full(len(values), idx) + offsets,
            values,
            color=BRAND["white"],
            edgecolor=BRAND["black_70"],
            linewidth=0.8,
            s=32,
            zorder=3,
        )

    ax.set_xticks(np.arange(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], rotation=25, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Pooled AP50 Across Seeds (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    save(fig, "paper_pooled_seed_variability")


def plot_site_permutations(rows: list[NormalizedRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True)
    families = [("Classical", CLASSICAL_METHOD_ORDER), ("Learned", LEARNED_METHOD_ORDER)]
    labels = [f"{SITE_CODES[t]}-{SITE_CODES[v]}-{SITE_CODES[s]}" for t, v, s in SITE_PERMS]

    for axis, (family_label, method_order) in zip(axes, families, strict=True):
        x = np.arange(len(SITE_PERMS))
        width = 0.8 / len(method_order)
        for idx, method in enumerate(method_order):
            values = []
            for tune, val, test in SITE_PERMS:
                subset = [
                    row
                    for row in rows
                    if row.method == method
                    and row.condition == "S"
                    and row.tune_label == tune
                    and row.val_label == val
                    and row.test_label == test
                ]
                values.append(mean_metric(subset, "test_ap50") * 100.0)
            axis.bar(
                x + (idx - (len(method_order) - 1) / 2.0) * width,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor=BRAND["black_70"],
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
        axis.set_title(family_label)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=22, ha="right")
        axis.set_xlabel("Tune-Val-Test Site Ordering")
        axis.grid(axis="x", visible=False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(len(method_order), 3))
    axes[0].set_ylabel("AP50 (%)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    save(fig, "paper_site_permutation_ap50_families")


def plot_detection_volume(rows: list[NormalizedRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=False)
    specs = [
        ("Classical", CLASSICAL_METHOD_ORDER, "False Alarms / Image"),
        ("Learned", LEARNED_METHOD_ORDER, "Predictions / Image"),
    ]

    for axis, (family_label, method_order, y_label) in zip(axes, specs, strict=True):
        x = np.arange(len(CONDITION_ORDER))
        width = 0.8 / len(method_order)
        for idx, method in enumerate(method_order):
            values = []
            for condition in CONDITION_ORDER:
                subset = [row for row in rows if row.method == method and row.condition == condition]
                values.append(mean_metric(subset, "detection_volume"))
            axis.bar(
                x + (idx - (len(method_order) - 1) / 2.0) * width,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor=BRAND["black_70"],
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
        axis.set_title(family_label)
        axis.set_xticks(x)
        axis.set_xticklabels([CONDITION_LABELS[key] for key in CONDITION_ORDER])
        axis.set_xlabel("Evaluation Setting")
        axis.set_ylabel(y_label)
        axis.grid(axis="x", visible=False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(len(method_order), 3))
    save(fig, "paper_detection_volume_by_condition")


def _sorted_counter_labels(counter: Counter[str]) -> list[str]:
    def _key(value: str) -> tuple[int, float | str]:
        if value == "":
            return (0, "")
        try:
            return (1, float(value))
        except ValueError:
            return (2, value)

    return sorted(counter, key=_key)


def plot_parameter_counts(rows: list[NormalizedRow], method_filter: str, specs: list[tuple[str, str]], stem: str, color: str) -> None:
    subset = [row.raw for row in rows if row.method == method_filter]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.2 * len(specs), 3.8), sharey=True)
    if len(specs) == 1:
        axes = [axes]

    for axis, (key, xlabel) in zip(axes, specs, strict=True):
        values = [row[key] for row in subset if row[key] != ""]
        counts = Counter(values)
        labels = _sorted_counter_labels(counts)
        axis.bar(
            labels,
            [counts[label] for label in labels],
            color=color,
            edgecolor=BRAND["black_70"],
            linewidth=0.6,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Selection Count")
        axis.set_facecolor(BRAND["sandstorm"])
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    save(fig, stem)


def write_summary_tables(rows: list[NormalizedRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "method_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "family", "condition", "mean_ap", "mean_ap50", "mean_ap75", "mean_runtime_sec", "mean_detection_volume"],
        )
        writer.writeheader()
        for method in METHOD_ORDER:
            method_rows = [row for row in rows if row.method == method]
            writer.writerow(
                {
                    "method": method,
                    "family": METHOD_FAMILY[method],
                    "condition": "ALL",
                    "mean_ap": mean_metric(method_rows, "test_ap"),
                    "mean_ap50": mean_metric(method_rows, "test_ap50"),
                    "mean_ap75": mean_metric(method_rows, "test_ap75"),
                    "mean_runtime_sec": mean_metric(method_rows, "runtime_sec"),
                    "mean_detection_volume": mean_metric(method_rows, "detection_volume"),
                }
            )
            for condition in CONDITION_ORDER:
                condition_rows = [row for row in method_rows if row.condition == condition]
                writer.writerow(
                    {
                        "method": method,
                        "family": METHOD_FAMILY[method],
                        "condition": condition,
                        "mean_ap": mean_metric(condition_rows, "test_ap"),
                        "mean_ap50": mean_metric(condition_rows, "test_ap50"),
                        "mean_ap75": mean_metric(condition_rows, "test_ap75"),
                        "mean_runtime_sec": mean_metric(condition_rows, "runtime_sec"),
                        "mean_detection_volume": mean_metric(condition_rows, "detection_volume"),
                    }
                )

    gap_path = OUT_DIR / "transfer_gap_summary.csv"
    with gap_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "diag_ap50", "offdiag_ap50", "gap_ap50"])
        writer.writeheader()
        for method in METHOD_ORDER:
            diag_rows = [
                row
                for row in rows
                if row.method == method and row.condition == "T" and parse_site(row.tune_label) == parse_site(row.test_label)
            ]
            offdiag_rows = [
                row
                for row in rows
                if row.method == method and row.condition == "T" and parse_site(row.tune_label) != parse_site(row.test_label)
            ]
            diag = mean_metric(diag_rows, "test_ap50")
            offdiag = mean_metric(offdiag_rows, "test_ap50")
            writer.writerow({"method": method, "diag_ap50": diag, "offdiag_ap50": offdiag, "gap_ap50": diag - offdiag})


def load_experiment_payload(exp_dir: Path) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, list[dict]]]:
    gt = json.load((exp_dir / "test_gt.json").open())
    preds = json.load((exp_dir / "test_predictions.json").open())
    image_by_id = {int(image["id"]): image for image in gt["images"]}
    gt_by_id: dict[int, list[dict]] = defaultdict(list)
    pred_by_id: dict[int, list[dict]] = defaultdict(list)

    for ann in gt["annotations"]:
        gt_by_id[int(ann["image_id"])].append(ann)
    for pred in preds:
        pred_by_id[int(pred["image_id"])].append(pred)
    return image_by_id, gt_by_id, pred_by_id


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def match_stats(gt_boxes: list[list[float]], pred_boxes: list[list[float]], threshold: float = 0.5) -> tuple[int, int]:
    matched_gts = sum(1 for gt in gt_boxes if any(bbox_iou(gt, pred) >= threshold for pred in pred_boxes))
    matched_preds = sum(1 for pred in pred_boxes if any(bbox_iou(pred, gt) >= threshold for gt in gt_boxes))
    return matched_gts, matched_preds


def overlay_image(file_name: str, gt_boxes: list[list[float]], pred_boxes: list[list[float]], pred_color: str | tuple[int, int, int], *, gt_color: tuple[int, int, int] = (40, 180, 99), max_preds: int = 40) -> Image.Image:
    image = Image.open(DATA_IMAGES / file_name).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in gt_boxes:
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline=gt_color, width=3)

    pred_boxes = pred_boxes[:max_preds]
    if isinstance(pred_color, str):
        pred_rgb = tuple(int(pred_color.strip("#")[idx : idx + 2], 16) for idx in (0, 2, 4))
    else:
        pred_rgb = pred_color
    for box in pred_boxes:
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline=pred_rgb, width=2)
    return image


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_bounds_from_box_sets(
    image_size: tuple[int, int],
    box_sets: list[list[list[float]]],
    *,
    pad: int = 72,
    min_size: int = 320,
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    coords: list[tuple[float, float, float, float]] = []
    for box_set in box_sets:
        for x, y, w, h in box_set:
            coords.append((x, y, x + w, y + h))

    if not coords:
        return 0, 0, image_w, image_h

    x1 = min(coord[0] for coord in coords)
    y1 = min(coord[1] for coord in coords)
    x2 = max(coord[2] for coord in coords)
    y2 = max(coord[3] for coord in coords)

    box_w = x2 - x1
    box_h = y2 - y1
    crop_size = max(min_size, int(max(box_w, box_h) + 2 * pad))
    crop_size = min(crop_size, max(image_w, image_h))

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    left = int(round(cx - crop_size / 2))
    top = int(round(cy - crop_size / 2))
    left = max(0, min(left, image_w - crop_size))
    top = max(0, min(top, image_h - crop_size))
    right = min(image_w, left + crop_size)
    bottom = min(image_h, top + crop_size)
    return left, top, right, bottom


def box_center(box: list[float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + 0.5 * w, y + 0.5 * h


def focus_bounds(boxes: list[list[float]], pad: int = 0) -> tuple[float, float, float, float]:
    x1 = min(box[0] for box in boxes) - pad
    y1 = min(box[1] for box in boxes) - pad
    x2 = max(box[0] + box[2] for box in boxes) + pad
    y2 = max(box[1] + box[3] for box in boxes) + pad
    return x1, y1, x2, y2


def boxes_with_centers_in_bounds(boxes: list[list[float]], bounds: tuple[float, float, float, float]) -> list[list[float]]:
    x1, y1, x2, y2 = bounds
    kept = []
    for box in boxes:
        cx, cy = box_center(box)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            kept.append(box)
    return kept


def select_focus_gt_cluster(gt_boxes: list[list[float]], pred_groups: list[list[list[float]]], radius: float = 180.0) -> list[list[float]]:
    if not gt_boxes:
        return []

    support = []
    for gt_box in gt_boxes:
        support.append(sum(1 for preds in pred_groups if any(bbox_iou(gt_box, pred_box) >= 0.5 for pred_box in preds)))
    min_support = min(support)
    candidate_indices = [idx for idx, value in enumerate(support) if value == min_support]
    if len(candidate_indices) == 1:
        return [gt_boxes[candidate_indices[0]]]

    candidate_boxes = [gt_boxes[idx] for idx in candidate_indices]
    centers = [box_center(box) for box in candidate_boxes]
    best_seed = 0
    best_count = -1
    for idx, (cx, cy) in enumerate(centers):
        count = sum(1 for ox, oy in centers if (cx - ox) ** 2 + (cy - oy) ** 2 <= radius ** 2)
        if count > best_count:
            best_count = count
            best_seed = idx
    sx, sy = centers[best_seed]
    cluster = [
        box
        for box, (cx, cy) in zip(candidate_boxes, centers, strict=True)
        if (sx - cx) ** 2 + (sy - cy) ** 2 <= radius ** 2
    ]
    return cluster or [candidate_boxes[best_seed]]


def local_failure_crop(
    image_size: tuple[int, int],
    gt_boxes: list[list[float]],
    pred_groups: list[list[list[float]]],
    *,
    focus_pad: int = 140,
    crop_pad: int = 60,
    min_size: int = 320,
) -> tuple[tuple[int, int, int, int], list[list[float]], list[list[list[float]]]]:
    focus_gt = select_focus_gt_cluster(gt_boxes, pred_groups)
    seed_boxes = focus_gt or gt_boxes
    if not seed_boxes:
        crop_box = (0, 0, image_size[0], image_size[1])
        return crop_box, [], [[] for _ in pred_groups]

    focus_box = focus_bounds(seed_boxes, pad=focus_pad)
    local_gt = boxes_with_centers_in_bounds(gt_boxes, focus_box) or seed_boxes
    local_preds = [boxes_with_centers_in_bounds(pred_boxes, focus_box) for pred_boxes in pred_groups]
    crop_box = crop_bounds_from_box_sets(image_size, [local_gt, *local_preds], pad=crop_pad, min_size=min_size)

    crop_bounds = (float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3]))
    cropped_gt = boxes_with_centers_in_bounds(gt_boxes, crop_bounds) or local_gt
    cropped_preds = [boxes_with_centers_in_bounds(pred_boxes, crop_bounds) for pred_boxes in pred_groups]
    return crop_box, cropped_gt, cropped_preds


def save_zoom_panels(
    file_name: str,
    panels: list[dict],
    crop_box: tuple[int, int, int, int],
    out_dir: Path,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        label_slug = panel["label"].lower().replace(" ", "_").replace("-", "")
        overlay = overlay_image(
            file_name,
            panel["gt_boxes"],
            panel.get("pred_boxes", []),
            panel["pred_color"],
            max_preds=80,
        )
        overlay.crop(crop_box).save(out_dir / f"{stem}_{label_slug}.png")


def write_subset_coco(source_gt_path: Path, image_ids: list[int], out_path: Path) -> dict:
    coco = json.load(source_gt_path.open())
    keep = set(image_ids)
    subset = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    subset["images"] = [image for image in coco["images"] if int(image["id"]) in keep]
    subset["annotations"] = [ann for ann in coco["annotations"] if int(ann["image_id"]) in keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset))
    return subset


def run_checked(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def ensure_classical_subset_payloads(
    scenario_tag: str,
    source_gt_path: Path,
    image_ids: list[int],
    direction_tag: str,
) -> dict[str, Path]:
    cache_root = ZOOM_CACHE_DIR / scenario_tag
    subset_gt_path = cache_root / "subset_gt.json"
    subset = write_subset_coco(source_gt_path, image_ids, subset_gt_path)
    subset_images = subset["images"]

    payload_dirs: dict[str, Path] = {}
    for method in ("CA", "OS", "OTSU", "SSOTSU", "WAV"):
        method_dir = cache_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "test_gt.json").write_text(subset_gt_path.read_text())
        payload_dirs[method] = method_dir

    ensure_cfar_subset_predictions(payload_dirs["CA"], subset_images, direction_tag, method="CA")
    ensure_cfar_subset_predictions(payload_dirs["OS"], subset_images, direction_tag, method="OS")
    ensure_otsu_subset_predictions(payload_dirs["OTSU"], subset_gt_path, direction_tag, method="OTSU")
    ensure_otsu_subset_predictions(payload_dirs["SSOTSU"], subset_gt_path, direction_tag, method="SSOTSU")
    ensure_wavelet_subset_predictions(payload_dirs["WAV"], subset_gt_path, direction_tag)
    return payload_dirs


def ensure_cfar_subset_predictions(
    out_dir: Path,
    subset_images: list[dict],
    direction_tag: str,
    *,
    method: str,
) -> None:
    pred_path = out_dir / "test_predictions.json"
    if pred_path.exists():
        return

    import cv2
    import numpy as np

    from src.cfar.detector import detect_cfar

    params = json.load((RESULTS / "cfar" / "per_experiment" / f"T-{method}-{direction_tag}.json").open())
    center = (531, 515)
    radius = 466
    predictions: list[dict] = []
    for info in subset_images:
        image = cv2.imread(str(DATA_IMAGES / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        image = cv2.bitwise_and(image, mask)
        image_f = image.astype(np.float32) / 255.0
        boxes, _, scores = detect_cfar(
            image_f,
            guard=(int(params["win_window_guard"]), int(params["win_window_guard"])),
            ref=(int(params["win_window_ref"]), int(params["win_window_ref"])),
            pfa=float(params["win_pfa"]),
            mode=method,
            k_os=float(params["win_os_k"] or 0.75),
            min_area=int(params["win_min_area"]),
        )
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            keep = np.hypot(cx - center[0], cy - center[1]) <= radius
            boxes = boxes[keep]
            scores = scores[keep]
            xywh = boxes.astype(float).copy()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]
            predictions.extend(
                {
                    "image_id": int(info["id"]),
                    "category_id": 1,
                    "bbox": xywh[idx].tolist(),
                    "score": float(scores[idx]),
                }
                for idx in range(len(xywh))
            )

    pred_path.write_text(json.dumps(predictions))


def ensure_otsu_subset_predictions(
    out_dir: Path,
    subset_gt_path: Path,
    direction_tag: str,
    *,
    method: str,
) -> None:
    pred_path = out_dir / "test_predictions.json"
    if pred_path.exists():
        return

    params = json.load((RESULTS / "otsu" / "per_experiment" / f"T-{method}-{direction_tag}.json").open())
    binary = ROOT / "src" / "otsu" / "target" / "release" / "otsu"
    cmd = [str(binary)] if binary.exists() else [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(ROOT / "src" / "otsu" / "Cargo.toml"),
        "--",
    ]
    cmd.extend(
        [
            "run",
            "--ann",
            str(subset_gt_path),
            "--images",
            str(DATA_IMAGES),
            "--out",
            str(pred_path),
            "--backend",
            "cpu",
            "--method",
            "otsu" if method == "OTSU" else "ssotsu",
            "--min-area",
            str(int(params["win_min_area"])),
            "--max-area",
            str(int(params["win_max_area"])),
            "--min-eccentricity",
            str(float(params["win_min_eccentricity"])),
            "--max-eccentricity",
            str(float(params["win_max_eccentricity"])),
            "--dilation",
            str(int(params["win_dilation"])),
            "--box-padding",
            str(int(params["win_box_padding"])),
        ]
    )
    if method == "SSOTSU":
        cmd.extend(
            [
                "--block-size",
                str(int(params["win_block_size"])),
                "--seed-threshold-factor",
                str(float(params["win_seed_threshold_factor"])),
                "--region-grow-threshold",
                str(float(params["win_region_grow_threshold"])),
            ]
        )
    run_checked(cmd)


def ensure_wavelet_subset_predictions(out_dir: Path, subset_gt_path: Path, direction_tag: str) -> None:
    pred_path = out_dir / "test_predictions.json"
    if pred_path.exists():
        return

    params = json.load((RESULTS / "wavelet" / "per_experiment" / f"T-WAV-{direction_tag}.json").open())
    binary = ROOT / "target" / "release" / "wavelet"
    cmd = [str(binary)] if binary.exists() else [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(ROOT / "src" / "wavelet" / "Cargo.toml"),
        "--",
    ]
    cmd.extend(
        [
            "run",
            "--ann",
            str(subset_gt_path),
            "--images",
            str(DATA_IMAGES),
            "--out",
            str(pred_path),
            "--backend",
            "cpu",
            "--level",
            str(int(params["win_level"])),
            "--threshold-sigma",
            str(float(params["win_threshold_sigma"])),
            "--min-area",
            str(int(params["win_min_area"])),
            "--max-area",
            str(int(params["win_max_area"])),
            "--dilation",
            str(int(params["win_dilation"])),
            "--box-padding",
            str(int(params["win_box_padding"])),
        ]
    )
    run_checked(cmd)


def compose_zoom_strip(
    file_name: str,
    panels: list[dict],
    crop_box: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    scale = max(1.0, 520.0 / max(crop_w, crop_h))
    panel_w = int(round(crop_w * scale))
    panel_h = int(round(crop_h * scale))
    header_h = 48
    footer_h = 36
    gap = 18
    canvas_w = len(panels) * panel_w + (len(panels) - 1) * gap
    canvas_h = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), BRAND["white"])

    title_font = load_font(28)
    meta_font = load_font(22)

    for idx, panel in enumerate(panels):
        label = panel["label"]
        gt_boxes = panel["gt_boxes"]
        pred_boxes = panel.get("pred_boxes", [])
        pred_color = panel["pred_color"]
        footer = panel["footer"]
        overlay = overlay_image(file_name, gt_boxes, pred_boxes, pred_color, max_preds=80)
        crop = overlay.crop(crop_box)
        if scale != 1.0:
            crop = crop.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = idx * (panel_w + gap)
        canvas.paste(crop, (x, header_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((x + 8, 8), label, fill=BRAND["black_90"], font=title_font)
        draw.text((x + 8, header_h + panel_h + 6), footer, fill=BRAND["black_70"], font=meta_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def select_hard_transfer_images() -> list[int]:
    base_dir = RESULTS / "yolo26_matrix" / "T-Y26-MvDc"
    image_by_id, gt_by_id, _ = load_experiment_payload(base_dir)

    method_dirs = {
        "UNET": RESULTS / "unet_matrix" / "T-UNET-MvDc",
        "FRCNN": RESULTS / "faster_rcnn_matrix" / "T-FRCNN-MvDc",
        "Y26": RESULTS / "yolo26_matrix" / "T-Y26-MvDc",
    }
    preds = {}
    for method, exp_dir in method_dirs.items():
        _, _, pred_by_id = load_experiment_payload(exp_dir)
        preds[method] = pred_by_id

    scored: list[tuple[float, int]] = []
    for image_id, image in image_by_id.items():
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
        if not gt_boxes:
            continue
        method_scores = []
        for method, pred_by_id in preds.items():
            pred_boxes = [pred["bbox"] for pred in pred_by_id.get(image_id, [])]
            matched_gts, _ = match_stats(gt_boxes, pred_boxes)
            miss_rate = (len(gt_boxes) - matched_gts) / max(1, len(gt_boxes))
            method_scores.append(miss_rate)
        score = float(mean(method_scores)) + 0.05 * len(gt_boxes)
        scored.append((score, image_id))
    scored.sort(reverse=True)
    return [image_id for _, image_id in scored[:4]]


def select_rtdetr_collapse_images() -> list[int]:
    rtdetr_dir = RESULTS / "rtdetr_matrix" / "T-RTDETR-DcMv"
    yolo_dir = RESULTS / "yolo26_matrix" / "T-Y26-DcMv"
    image_by_id, gt_by_id, rtdetr_pred_by_id = load_experiment_payload(rtdetr_dir)
    _, _, yolo_pred_by_id = load_experiment_payload(yolo_dir)

    scored: list[tuple[float, int]] = []
    for image_id, image in image_by_id.items():
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
        rtdetr_boxes = [pred["bbox"] for pred in rtdetr_pred_by_id.get(image_id, [])]
        yolo_boxes = [pred["bbox"] for pred in yolo_pred_by_id.get(image_id, [])]
        _, rtdetr_matched_preds = match_stats(gt_boxes, rtdetr_boxes)
        yolo_matched_gts, yolo_matched_preds = match_stats(gt_boxes, yolo_boxes)
        rtdetr_fp = max(0, len(rtdetr_boxes) - rtdetr_matched_preds)
        yolo_fp = max(0, len(yolo_boxes) - yolo_matched_preds)
        score = rtdetr_fp - 2.0 * yolo_fp + 3.0 * yolo_matched_gts
        scored.append((score, image_id))
    scored.sort(reverse=True)
    return [image_id for _, image_id in scored[:4]]


def plot_failure_hard_transfer() -> None:
    scenario_dirs = {
        "U-Net": RESULTS / "unet_matrix" / "T-UNET-MvDc",
        "Faster R-CNN": RESULTS / "faster_rcnn_matrix" / "T-FRCNN-MvDc",
        "YOLO26": RESULTS / "yolo26_matrix" / "T-Y26-MvDc",
    }
    image_by_id, gt_by_id, _ = load_experiment_payload(scenario_dirs["YOLO26"])
    pred_payloads = {label: load_experiment_payload(path)[2] for label, path in scenario_dirs.items()}
    chosen_ids = select_hard_transfer_images()

    fig, axes = plt.subplots(len(chosen_ids), 4, figsize=(14.4, 3.3 * len(chosen_ids)))
    if len(chosen_ids) == 1:
        axes = np.array([axes])

    column_labels = ["Ground Truth", "U-Net", "Faster R-CNN", "YOLO26"]
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].set_title(label)

    for row_idx, image_id in enumerate(chosen_ids):
        info = image_by_id[image_id]
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]

        reference = overlay_image(info["file_name"], gt_boxes, [], BRAND["black_50"])
        axes[row_idx, 0].imshow(reference)
        axes[row_idx, 0].set_ylabel(f"{info['file_name']}\nGT={len(gt_boxes)}", rotation=90)

        for col_idx, label in enumerate(["U-Net", "Faster R-CNN", "YOLO26"], start=1):
            pred_by_id = pred_payloads[label]
            pred_boxes = [pred["bbox"] for pred in pred_by_id.get(image_id, [])]
            matched_gts, matched_preds = match_stats(gt_boxes, pred_boxes)
            image = overlay_image(info["file_name"], gt_boxes, pred_boxes, METHOD_PRED_COLORS[{"U-Net": "UNET", "Faster R-CNN": "FRCNN", "YOLO26": "Y26"}[label]])
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_xlabel(f"match={matched_gts}/{len(gt_boxes)} preds={len(pred_boxes)}")

        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].grid(False)

    save(fig, "paper_failure_hard_transfer_mvdc", svg=False)


def plot_failure_rtdetr_collapse() -> None:
    scenario_dirs = {
        "RT-DETR": RESULTS / "rtdetr_matrix" / "T-RTDETR-DcMv",
        "YOLO26": RESULTS / "yolo26_matrix" / "T-Y26-DcMv",
    }
    image_by_id, gt_by_id, rtdetr_pred_by_id = load_experiment_payload(scenario_dirs["RT-DETR"])
    _, _, yolo_pred_by_id = load_experiment_payload(scenario_dirs["YOLO26"])
    chosen_ids = select_rtdetr_collapse_images()

    fig, axes = plt.subplots(len(chosen_ids), 3, figsize=(11.2, 3.3 * len(chosen_ids)))
    if len(chosen_ids) == 1:
        axes = np.array([axes])

    column_labels = ["Ground Truth", "RT-DETR", "YOLO26"]
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].set_title(label)

    for row_idx, image_id in enumerate(chosen_ids):
        info = image_by_id[image_id]
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
        rtdetr_boxes = [pred["bbox"] for pred in rtdetr_pred_by_id.get(image_id, [])]
        yolo_boxes = [pred["bbox"] for pred in yolo_pred_by_id.get(image_id, [])]

        reference = overlay_image(info["file_name"], gt_boxes, [], BRAND["black_50"])
        axes[row_idx, 0].imshow(reference)
        axes[row_idx, 0].set_ylabel(f"{info['file_name']}\nGT={len(gt_boxes)}", rotation=90)

        rtdetr_match, rtdetr_matched_preds = match_stats(gt_boxes, rtdetr_boxes)
        yolo_match, _ = match_stats(gt_boxes, yolo_boxes)

        rtdetr_image = overlay_image(info["file_name"], gt_boxes, rtdetr_boxes, METHOD_PRED_COLORS["RTDETR"], max_preds=50)
        yolo_image = overlay_image(info["file_name"], gt_boxes, yolo_boxes, METHOD_PRED_COLORS["Y26"])
        axes[row_idx, 1].imshow(rtdetr_image)
        axes[row_idx, 1].set_xlabel(f"match={rtdetr_match}/{len(gt_boxes)} preds={len(rtdetr_boxes)}")
        axes[row_idx, 2].imshow(yolo_image)
        axes[row_idx, 2].set_xlabel(f"match={yolo_match}/{len(gt_boxes)} preds={len(yolo_boxes)}")

        for col_idx in range(3):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].grid(False)

    save(fig, "paper_failure_rtdetr_collapse_dcmv", svg=False)


def top_pred_boxes(pred_by_id: dict[int, list[dict]], image_id: int, limit: int) -> list[list[float]]:
    preds = sorted(pred_by_id.get(image_id, []), key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
    return [pred["bbox"] for pred in preds[:limit]]


def render_zoomed_hard_transfer_examples() -> None:
    scenario_dirs = {
        "U-Net": RESULTS / "unet_matrix" / "T-UNET-MvDc",
        "Faster R-CNN": RESULTS / "faster_rcnn_matrix" / "T-FRCNN-MvDc",
        "YOLO26": RESULTS / "yolo26_matrix" / "T-Y26-MvDc",
    }
    image_by_id, gt_by_id, _ = load_experiment_payload(scenario_dirs["YOLO26"])
    pred_payloads = {label: load_experiment_payload(path)[2] for label, path in scenario_dirs.items()}
    chosen_ids = select_hard_transfer_images()
    out_dir = ZOOM_DIR / "hard_transfer_mvdc"
    out_dir.mkdir(parents=True, exist_ok=True)

    color_key = {
        "U-Net": "UNET",
        "Faster R-CNN": "FRCNN",
        "YOLO26": "Y26",
    }

    for image_id in chosen_ids:
        info = image_by_id[image_id]
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
        unet_boxes = top_pred_boxes(pred_payloads["U-Net"], image_id, 18)
        frcnn_boxes = top_pred_boxes(pred_payloads["Faster R-CNN"], image_id, 20)
        yolo_boxes = top_pred_boxes(pred_payloads["YOLO26"], image_id, 20)
        crop_box, local_gt, local_pred_groups = local_failure_crop(
            Image.open(DATA_IMAGES / info["file_name"]).size,
            gt_boxes,
            [unet_boxes, frcnn_boxes, yolo_boxes],
            focus_pad=160,
            crop_pad=56,
            min_size=300,
        )
        local_unet, local_frcnn, local_yolo = local_pred_groups

        panels = [
            {
                "label": "Ground Truth",
                "gt_boxes": local_gt,
                "pred_boxes": [],
                "pred_color": BRAND["black_50"],
                "footer": f"{info['file_name']}  GT={len(local_gt)} local / {len(gt_boxes)} total",
            }
        ]
        for label, pred_boxes in (
            ("U-Net", local_unet),
            ("Faster R-CNN", local_frcnn),
            ("YOLO26", local_yolo),
        ):
            matched_gts, _ = match_stats(local_gt, pred_boxes)
            panels.append(
                {
                    "label": label,
                    "gt_boxes": local_gt,
                    "pred_boxes": pred_boxes,
                    "pred_color": METHOD_PRED_COLORS[color_key[label]],
                    "footer": f"match={matched_gts}/{len(local_gt)} preds={len(pred_boxes)}",
                }
            )

        stem = Path(info["file_name"]).stem.lower()
        save_zoom_panels(info["file_name"], panels, crop_box, out_dir, stem)
        compose_zoom_strip(info["file_name"], panels, crop_box, out_dir / f"{stem}_zoom.png")


def render_zoomed_rtdetr_collapse_examples() -> None:
    scenario_dirs = {
        "RT-DETR": RESULTS / "rtdetr_matrix" / "T-RTDETR-DcMv",
        "YOLO26": RESULTS / "yolo26_matrix" / "T-Y26-DcMv",
    }
    image_by_id, gt_by_id, rtdetr_pred_by_id = load_experiment_payload(scenario_dirs["RT-DETR"])
    _, _, yolo_pred_by_id = load_experiment_payload(scenario_dirs["YOLO26"])
    chosen_ids = select_rtdetr_collapse_images()
    out_dir = ZOOM_DIR / "rtdetr_collapse_dcmv"
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_id in chosen_ids:
        info = image_by_id[image_id]
        gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
        rtdetr_boxes = top_pred_boxes(rtdetr_pred_by_id, image_id, 50)
        yolo_boxes = top_pred_boxes(yolo_pred_by_id, image_id, 20)
        crop_box, local_gt, local_pred_groups = local_failure_crop(
            Image.open(DATA_IMAGES / info["file_name"]).size,
            gt_boxes,
            [rtdetr_boxes, yolo_boxes],
            focus_pad=150,
            crop_pad=60,
            min_size=300,
        )
        local_rtdetr, local_yolo = local_pred_groups

        rtdetr_match, _ = match_stats(local_gt, local_rtdetr)
        yolo_match, _ = match_stats(local_gt, local_yolo)
        panels = [
            {
                "label": "Ground Truth",
                "gt_boxes": local_gt,
                "pred_boxes": [],
                "pred_color": BRAND["black_50"],
                "footer": f"{info['file_name']}  GT={len(local_gt)} local / {len(gt_boxes)} total",
            },
            {
                "label": "RT-DETR",
                "gt_boxes": local_gt,
                "pred_boxes": local_rtdetr,
                "pred_color": METHOD_PRED_COLORS["RTDETR"],
                "footer": f"match={rtdetr_match}/{len(local_gt)} preds={len(local_rtdetr)}",
            },
            {
                "label": "YOLO26",
                "gt_boxes": local_gt,
                "pred_boxes": local_yolo,
                "pred_color": METHOD_PRED_COLORS["Y26"],
                "footer": f"match={yolo_match}/{len(local_gt)} preds={len(local_yolo)}",
            },
        ]

        stem = Path(info["file_name"]).stem.lower()
        save_zoom_panels(info["file_name"], panels, crop_box, out_dir, stem)
        compose_zoom_strip(info["file_name"], panels, crop_box, out_dir / f"{stem}_zoom.png")


def render_zoomed_classical_examples() -> None:
    scenarios = [
        (
            "hard_transfer_mvdc_classical",
            RESULTS / "yolo26_matrix" / "T-Y26-MvDc" / "test_gt.json",
            select_hard_transfer_images(),
            "MvDc",
        ),
        (
            "rtdetr_collapse_dcmv_classical",
            RESULTS / "yolo26_matrix" / "T-Y26-DcMv" / "test_gt.json",
            select_rtdetr_collapse_images(),
            "DcMv",
        ),
    ]

    labels = ["CA-CFAR", "OS-CFAR", "Otsu", "SS-Otsu", "Wavelet"]
    method_keys = ["CA", "OS", "OTSU", "SSOTSU", "WAV"]

    for scenario_tag, source_gt_path, chosen_ids, direction_tag in scenarios:
        payload_dirs = ensure_classical_subset_payloads(scenario_tag, source_gt_path, chosen_ids, direction_tag)
        out_dir = ZOOM_DIR / scenario_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        image_by_id, gt_by_id, _ = load_experiment_payload(payload_dirs["CA"])
        pred_payloads = {method: load_experiment_payload(path)[2] for method, path in payload_dirs.items()}

        for image_id in chosen_ids:
            info = image_by_id[image_id]
            gt_boxes = [ann["bbox"] for ann in gt_by_id.get(image_id, [])]
            pred_groups = [top_pred_boxes(pred_payloads[method], image_id, 24) for method in method_keys]
            crop_box, local_gt, local_pred_groups = local_failure_crop(
                Image.open(DATA_IMAGES / info["file_name"]).size,
                gt_boxes,
                pred_groups,
                focus_pad=170,
                crop_pad=56,
                min_size=300,
            )

            panels = [
                {
                    "label": "Ground Truth",
                    "gt_boxes": local_gt,
                    "pred_boxes": [],
                    "pred_color": BRAND["black_50"],
                    "footer": f"{info['file_name']}  GT={len(local_gt)} local / {len(gt_boxes)} total",
                }
            ]
            for label, method_key, pred_boxes in zip(labels, method_keys, local_pred_groups, strict=True):
                matched_gts, _ = match_stats(local_gt, pred_boxes)
                panels.append(
                    {
                        "label": label,
                        "gt_boxes": local_gt,
                        "pred_boxes": pred_boxes,
                        "pred_color": METHOD_PRED_COLORS[method_key],
                        "footer": f"match={matched_gts}/{len(local_gt)} preds={len(pred_boxes)}",
                    }
                )

            stem = Path(info["file_name"]).stem.lower()
            save_zoom_panels(info["file_name"], panels, crop_box, out_dir, stem)
            compose_zoom_strip(info["file_name"], panels, crop_box, out_dir / f"{stem}_zoom.png")


def main() -> None:
    configure_matplotlib()
    rows = normalize_rows()
    write_summary_tables(rows)

    plot_family_condition_metric(rows, "test_ap50", "paper_ap50_by_condition_families", "AP50 (%)")
    plot_family_condition_metric(rows, "test_ap", "paper_ap_by_condition_families", "AP (%)")
    plot_family_condition_metric(rows, "test_ap75", "paper_ap75_by_condition_families", "AP75 (%)")
    plot_method_mean_ap50(rows)
    plot_method_mean_ap50_bottom_legend(rows)
    plot_method_test_site_ap50_bottom_legend(rows, "DAAN", "paper_method_mean_ap50_daan_bottom_legend")
    plot_localization_retention(rows)
    plot_method_mean_rank(rows)
    plot_pareto_runtime(rows)
    plot_transfer_heatmaps(rows)
    plot_transfer_gap(rows)
    plot_transfer_target_site_breakdown(rows)
    plot_pooled_seed_variability(rows)
    plot_site_permutations(rows)
    plot_detection_volume(rows)

    plot_parameter_counts(
        rows,
        "CA",
        [
            ("win_window_guard", "Guard"),
            ("win_window_ref", "Reference"),
            ("win_pfa", "PFA"),
            ("win_min_area", "Min Area"),
        ],
        "paper_cfar_ca_winner_parameter_counts",
        METHOD_COLORS["CA"],
    )
    plot_parameter_counts(
        rows,
        "OS",
        [
            ("win_window_guard", "Guard"),
            ("win_window_ref", "Reference"),
            ("win_pfa", "PFA"),
            ("win_min_area", "Min Area"),
            ("win_os_k", "OS-k"),
        ],
        "paper_cfar_os_winner_parameter_counts",
        METHOD_COLORS["OS"],
    )
    plot_parameter_counts(
        rows,
        "OTSU",
        [
            ("win_min_area", "Min Area"),
            ("win_dilation", "Dilation"),
            ("win_box_padding", "Box Padding"),
        ],
        "paper_otsu_winner_parameter_counts",
        METHOD_COLORS["OTSU"],
    )
    plot_parameter_counts(
        rows,
        "SSOTSU",
        [
            ("win_block_size", "Block Size"),
            ("win_seed_threshold_factor", "Seed Threshold"),
            ("win_region_grow_threshold", "Grow Threshold"),
            ("win_min_area", "Min Area"),
        ],
        "paper_ssotsu_winner_parameter_counts",
        METHOD_COLORS["SSOTSU"],
    )
    plot_parameter_counts(
        rows,
        "WAV",
        [
            ("win_level", "Level"),
            ("win_threshold_sigma", "Sigma"),
        ],
        "paper_wavelet_winner_parameter_counts",
        METHOD_COLORS["WAV"],
    )
    plot_parameter_counts(
        rows,
        "UNET",
        [
            ("threshold", "Threshold"),
            ("min_area", "Min Area"),
            ("max_area", "Max Area"),
            ("dilation", "Dilation"),
            ("box_padding", "Box Padding"),
        ],
        "paper_unet_winner_parameter_counts",
        METHOD_COLORS["UNET"],
    )

    plot_failure_hard_transfer()
    plot_failure_rtdetr_collapse()
    render_zoomed_hard_transfer_examples()
    render_zoomed_rtdetr_collapse_examples()
    render_zoomed_classical_examples()


if __name__ == "__main__":
    main()
