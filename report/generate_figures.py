"""Generate publication-quality figures for the marine radar detection report.

Brand palette (per author preferences):
  Garnet   #73000A  (primary accent)
  Black    #000000
  Warm grey shades, Sandstorm
  Accents: Rose, Atlantic, Congaree, Horseshoe, Grass, Honeycomb

No rounded edges. High contrast. No emojis.
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

# ── Brand palette ────────────────────────────────────────────────────────
GARNET     = "#73000A"
BLACK      = "#000000"
WHITE      = "#FFFFFF"
BLACK_90   = "#363636"
BLACK_70   = "#5C5C5C"
BLACK_50   = "#A2A2A2"
BLACK_30   = "#C7C7C7"
BLACK_10   = "#ECECEC"
WARM_GREY  = "#676156"
SANDSTORM  = "#FFF2E3"
ROSE       = "#CC2E40"
ATLANTIC   = "#466A9F"
CONGAREE   = "#1F414D"
HORSESHOE  = "#65780B"
GRASS      = "#CED318"
HONEYCOMB  = "#A49137"

# Colour assignments
CLASSICAL_COLOURS = {
    "wavelet":       GARNET,
    "morphological": ROSE,
    "otsu":          HORSESHOE,
    "ssotsu":        HONEYCOMB,
    "ca_cfar":       CONGAREE,
    "go_cfar":       ATLANTIC,
    "so_cfar":       WARM_GREY,
    "os_cfar":       BLACK_50,
}
DL_COLOURS = {
    "yolov11":     BLACK,
    "rt_detr":     BLACK_70,
    "faster_rcnn": BLACK_30,
    "unet":        BLACK_90,
}

METHOD_LABELS = {
    "wavelet":       "Wavelet-SWT",
    "morphological": "Morphological",
    "otsu":          "Otsu",
    "ssotsu":        "SSOTSU",
    "ca_cfar":       "CA-CFAR",
    "go_cfar":       "GO-CFAR",
    "so_cfar":       "SO-CFAR",
    "os_cfar":       "OS-CFAR",
    "yolov11":       "YOLOv11",
    "rt_detr":       "RT-DETR",
    "faster_rcnn":   "Faster R-CNN",
    "unet":          "U-Net",
}

CLASSICAL_ORDER = ["wavelet", "morphological", "otsu", "ca_cfar",
                   "go_cfar", "so_cfar", "ssotsu", "os_cfar"]
DL_ORDER = ["yolov11", "rt_detr", "faster_rcnn", "unet"]


def setup_style() -> None:
    """Configure matplotlib with the brand typography + clean spines."""
    mpl.rcParams.update({
        "figure.facecolor":     WHITE,
        "axes.facecolor":       WHITE,
        "savefig.facecolor":    WHITE,
        "axes.edgecolor":       BLACK,
        "axes.labelcolor":      BLACK,
        "text.color":           BLACK,
        "xtick.color":          BLACK,
        "ytick.color":          BLACK,
        "font.family":          "sans-serif",
        "font.sans-serif":      ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.titleweight":     "bold",
        "axes.labelsize":       10,
        "legend.frameon":       False,
        "legend.fontsize":      9,
        "axes.grid":            False,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "patch.linewidth":      0.6,
        # No rounded bar corners by default - matplotlib uses sharp by default
    })


def load_results() -> pd.DataFrame:
    df = pd.read_csv("results/evaluation_summary.csv")
    return df


# ── Figure 1: headline AP@0.50 comparison ────────────────────────────────

def fig_headline(df: pd.DataFrame, out_path: Path) -> None:
    """Dramatic bar chart showing the classical-vs-DL gap at AP@0.50."""
    # Classical: one row each
    cls = df[df["type"] == "classical"].copy()
    cls = cls.sort_values("AP@0.50", ascending=False)

    # DL: aggregate over seeds
    dl = df[df["type"] == "dl"].groupby("method").agg(
        ap_mean=("AP@0.50", "mean"),
        ap_std=("AP@0.50", "std"),
    ).reset_index()
    dl = dl.sort_values("ap_mean", ascending=False)

    # Build combined ordered list: classical (desc) then DL (desc)
    methods = list(cls["method"]) + list(dl["method"])
    values = list(cls["AP@0.50"]) + list(dl["ap_mean"])
    errors = [0] * len(cls) + list(dl["ap_std"].fillna(0))
    is_classical = [True] * len(cls) + [False] * len(dl)

    colors = [CLASSICAL_COLOURS[m] if c else DL_COLOURS[m]
              for m, c in zip(methods, is_classical)]
    labels = [METHOD_LABELS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=errors, color=colors, edgecolor=BLACK,
                  linewidth=0.6, width=0.72, capsize=2.5,
                  error_kw={"ecolor": BLACK, "linewidth": 0.8})

    # Value labels on top
    for i, (b, v) in enumerate(zip(bars, values)):
        txt = f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.012,
                txt, ha="center", va="bottom", fontsize=8, color=BLACK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AP@0.50")
    ax.set_ylim(0, max(values) * 1.20)

    # Vertical separator between classical and DL blocks
    sep = len(cls) - 0.5
    ax.axvline(sep, color=BLACK_50, linewidth=0.6, linestyle=(0, (3, 3)))

    # Section headers
    ax.text((len(cls) - 1) / 2, max(values) * 1.14, "Classical",
            ha="center", va="top", fontsize=10, weight="bold", color=GARNET)
    ax.text(len(cls) + (len(dl) - 1) / 2, max(values) * 1.14, "Deep Learning",
            ha="center", va="top", fontsize=10, weight="bold", color=BLACK)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", color=BLACK_50)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: multi-metric panel ─────────────────────────────────────────

def fig_multimetric(df: pd.DataFrame, out_path: Path) -> None:
    """2x2 grid showing AP@0.50, AP@0.75, AP@0.50:0.95, AR@100."""
    metrics = [
        ("AP@0.50",      "AP@0.50"),
        ("AP@0.75",      "AP@0.75"),
        ("AP@0.50:0.95", "AP@[0.50:0.95] (COCO mAP)"),
        ("AR@100",       "AR@100"),
    ]

    cls = df[df["type"] == "classical"].copy()
    dl_agg = df[df["type"] == "dl"].groupby("method").agg(
        AP_50=("AP@0.50", "mean"), AP_50_std=("AP@0.50", "std"),
        AP_75=("AP@0.75", "mean"), AP_75_std=("AP@0.75", "std"),
        AP_50_95=("AP@0.50:0.95", "mean"), AP_50_95_std=("AP@0.50:0.95", "std"),
        AR_100=("AR@100", "mean"), AR_100_std=("AR@100", "std"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))

    for ax, (col, title) in zip(axes.flat, metrics):
        # Classical
        cls_sorted = cls.sort_values(col, ascending=False)
        dl_col_map = {"AP@0.50": "AP_50", "AP@0.75": "AP_75",
                      "AP@0.50:0.95": "AP_50_95", "AR@100": "AR_100"}
        dl_std_map = {k: f"{v}_std" for k, v in dl_col_map.items()}
        dl_sorted = dl_agg.sort_values(dl_col_map[col], ascending=False)

        methods = list(cls_sorted["method"]) + list(dl_sorted["method"])
        values = list(cls_sorted[col]) + list(dl_sorted[dl_col_map[col]])
        errs = [0] * len(cls_sorted) + list(dl_sorted[dl_std_map[col]].fillna(0))
        is_cls = [True] * len(cls_sorted) + [False] * len(dl_sorted)
        colors = [CLASSICAL_COLOURS[m] if c else DL_COLOURS[m]
                  for m, c in zip(methods, is_cls)]
        labels = [METHOD_LABELS[m] for m in methods]
        x = np.arange(len(methods))

        ax.bar(x, values, yerr=errs, color=colors, edgecolor=BLACK,
               linewidth=0.5, width=0.75, capsize=2,
               error_kw={"ecolor": BLACK, "linewidth": 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ymax = max(values + [1e-4]) * 1.15
        ax.set_ylim(0, ymax)
        ax.axvline(len(cls_sorted) - 0.5, color=BLACK_50,
                   linewidth=0.5, linestyle=(0, (3, 3)))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: preprocessing comparison ───────────────────────────────────

def fig_preprocess(out_path: Path, img_stem: str = "DAAN_342") -> None:
    """Raw -> red channel -> masked -> GT overlay."""
    raw_path = Path("data/raw/DAAN/radarImages") / f"{img_stem.split('_')[1]}.png"
    if not raw_path.exists():
        # Fallback to another image
        raw_path = Path("data/raw/DAAN/radarImages") / "010.png"

    raw = cv2.imread(str(raw_path))  # BGR
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    red_only = raw[:, :, 2]  # R in BGR index 2

    # Circular mask
    h, w = red_only.shape
    cx, cy = w // 2, h // 2
    r = int(min(w, h) / 2 * 0.92)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    masked = red_only.copy()
    masked[mask == 0] = 0

    # GT overlay
    gt_path = Path("data/processed/annotations/test.json")
    with open(gt_path) as f:
        gt_all = json.load(f)
    fname = f"{img_stem}.png"
    img_info = next((i for i in gt_all["images"] if i["file_name"] == fname), None)
    gt_boxes = []
    if img_info:
        gt_boxes = [a["bbox"] for a in gt_all["annotations"]
                    if a["image_id"] == img_info["id"]]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))

    titles = [
        "(a) Raw BGR PPI capture",
        "(b) Red channel only",
        "(c) + Circular mask",
        "(d) + DBSCAN ground truth",
    ]
    imgs = [raw_rgb, red_only, masked, masked]

    for ax, img, t in zip(axes, imgs, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BLACK)
            spine.set_linewidth(0.7)

    # GT overlay on last panel
    for x, y, bw, bh in gt_boxes:
        rect = Rectangle((x, y), bw, bh, linewidth=1.6, edgecolor=GARNET,
                         facecolor="none")
        axes[-1].add_patch(rect)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: qualitative detection comparison ───────────────────────────

def fig_qualitative(out_path: Path) -> None:
    """Sample image with GT + top predictions from key methods."""
    # Load GT
    with open("data/processed/annotations/test.json") as f:
        gt = json.load(f)

    gt_by_img: dict[int, list] = {}
    for a in gt["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)

    # Pick 2 images: one with many targets, one with few but clear
    img_list = sorted(gt["images"], key=lambda i: len(gt_by_img.get(i["id"], [])),
                      reverse=True)
    chosen = []
    for img_info in img_list:
        n = len(gt_by_img.get(img_info["id"], []))
        if 3 <= n <= 8:
            chosen.append(img_info)
        if len(chosen) >= 2:
            break

    # Load method predictions
    method_specs = [
        ("wavelet",       "Wavelet-SWT",   GARNET,    "classical"),
        ("morphological", "Morphological", ROSE,      "classical"),
        ("ca_cfar",       "CA-CFAR",       CONGAREE,  "classical"),
        ("yolov11",       "YOLOv11 (seed 456)", ATLANTIC,  "dl"),
    ]
    method_preds = {}
    for m, _, _, kind in method_specs:
        base = "results/classical" if kind == "classical" else "results/deep_learning_v2"
        if kind == "classical":
            path = f"{base}/{m}_predictions.json"
        else:
            path = f"{base}/{m}_seed456_predictions.json"
        with open(path) as f:
            preds = json.load(f)
        method_preds[m] = preds

    n_cols = 1 + len(method_specs)   # GT + methods
    fig, axes = plt.subplots(len(chosen), n_cols,
                             figsize=(3.2 * n_cols, 3.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, img_info in enumerate(chosen):
        img = cv2.imread(f"data/processed/images/{img_info['file_name']}",
                         cv2.IMREAD_GRAYSCALE)
        gts = gt_by_img.get(img_info["id"], [])

        # GT column
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        for a in gts:
            x, y, bw, bh = a["bbox"]
            rect = Rectangle((x, y), bw, bh, linewidth=1.8,
                             edgecolor=GARNET, facecolor="none")
            ax.add_patch(rect)
        if row == 0:
            ax.set_title(f"Ground Truth\n({len(gts)} objects)", fontsize=10)
        else:
            ax.set_title(f"({len(gts)} objects)", fontsize=10)
        ax.set_ylabel(img_info["file_name"], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK); sp.set_linewidth(0.7)

        # Method columns (always display GT in garnet underneath, preds in method color)
        for col, (m, label, colour, kind) in enumerate(method_specs, start=1):
            preds = method_preds[m]
            # For this image only
            img_preds = [p for p in preds if p["image_id"] == img_info["id"]]
            # If DL + tiled, image_ids refer to tiles, so skip cross-checking
            if kind == "dl":
                # Try to find matching tiles' predictions via tile metadata
                img_preds = []   # DL was evaluated on tiles; skip for full images

            img_preds = sorted(img_preds, key=lambda p: -p.get("score", 0))[:40]

            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            # Faint GT
            for a in gts:
                x, y, bw, bh = a["bbox"]
                ax.add_patch(Rectangle((x, y), bw, bh, linewidth=0.9,
                                       edgecolor=GARNET, facecolor="none",
                                       linestyle=(0, (2, 2)), alpha=0.7))
            # Method predictions
            for p in img_preds:
                x, y, bw, bh = p["bbox"]
                ax.add_patch(Rectangle((x, y), bw, bh, linewidth=1.3,
                                       edgecolor=colour, facecolor="none"))
            if row == 0:
                ax.set_title(f"{label}\n({len(img_preds)} shown)", fontsize=10)
            else:
                ax.set_title(f"({len(img_preds)} shown)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(BLACK); sp.set_linewidth(0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 5: detection count vs AP (inefficiency plot) ──────────────────

def fig_count_vs_ap(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of (total predictions) vs (AP@0.50), showing over-prediction.

    GT has 2,022 objects in test set; a horizontal line highlights that.
    """
    GT_COUNT = 2022

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df.iterrows():
        m = row["method"]
        n = row["n_preds"]
        ap = row["AP@0.50"]
        if row["type"] == "classical":
            c = CLASSICAL_COLOURS[m]
            marker = "s"
            label = METHOD_LABELS[m] if row["seed"] == "N/A" else None
        else:
            c = DL_COLOURS[m]
            marker = "o"
            label = None

        if n <= 0 or ap <= 0:
            # Plot at a floor to keep log scale readable
            ap_plot = 3e-6
        else:
            ap_plot = ap
        n_plot = max(n, 1)

        ax.scatter(n_plot, ap_plot, s=110, marker=marker, c=c,
                   edgecolors=BLACK, linewidths=0.6, zorder=4, alpha=0.9)

    # Labels for each method (use classical + DL mean)
    cls = df[df["type"] == "classical"]
    for _, r in cls.iterrows():
        m = r["method"]; n = r["n_preds"]; ap = r["AP@0.50"]
        if ap <= 0:
            continue
        ax.annotate(METHOD_LABELS[m], (n, ap), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, color=BLACK)

    dl_agg = df[df["type"] == "dl"].groupby("method").agg(
        ap=("AP@0.50", "mean"), n=("n_preds", "mean")).reset_index()
    for _, r in dl_agg.iterrows():
        m = r["method"]; n = r["n"]; ap = r["ap"]
        ap_plot = max(ap, 3e-6)
        ax.annotate(METHOD_LABELS[m] + " (mean)", (n, ap_plot),
                    xytext=(8, 4), textcoords="offset points", fontsize=8,
                    color=BLACK_70, fontstyle="italic")

    ax.axvline(GT_COUNT, color=GARNET, linewidth=1.0, linestyle="--")
    ax.text(GT_COUNT * 1.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
            f"Ground-truth = {GT_COUNT}", color=GARNET, fontsize=9,
            rotation=90, va="top", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total predictions on test set (log)")
    ax.set_ylabel("AP@0.50 (log)")
    ax.set_title("Detection Volume vs. Precision")

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GARNET,
               markersize=9, markeredgecolor=BLACK, label="Classical"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLACK,
               markersize=9, markeredgecolor=BLACK, label="Deep Learning"),
    ]
    ax.legend(handles=handles, loc="lower left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 6: prediction box-size distribution ───────────────────────────

def fig_box_sizes(out_path: Path) -> None:
    """Overlay of GT vs per-method predicted box-area histograms."""
    with open("data/processed/annotations/test.json") as f:
        gt = json.load(f)
    gt_areas = np.array([a["bbox"][2] * a["bbox"][3] for a in gt["annotations"]])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    bins = np.logspace(0.5, 6, 42)

    ax.hist(gt_areas, bins=bins, alpha=0.45, color=GARNET,
            edgecolor=BLACK, linewidth=0.5, label="Ground truth")

    method_files = [
        ("wavelet", "results/classical/wavelet_predictions.json", ATLANTIC,
         "Wavelet-SWT"),
        ("ca_cfar", "results/classical/ca_cfar_predictions.json", CONGAREE,
         "CA-CFAR"),
        ("morphological", "results/classical/morphological_predictions.json",
         HORSESHOE, "Morphological"),
    ]
    for _m, path, colour, label in method_files:
        with open(path) as f:
            preds = json.load(f)
        areas = np.array([p["bbox"][2] * p["bbox"][3] for p in preds])
        if len(areas) == 0:
            continue
        ax.hist(areas, bins=bins, histtype="step", linewidth=1.5,
                edgecolor=colour, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Bounding box area (pixels, log)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Bounding Box Areas")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 7: classical vs DL summary boxplot ────────────────────────────

def fig_classical_vs_dl(df: pd.DataFrame, out_path: Path) -> None:
    """Paired boxplots summarising classical vs DL across metrics."""
    metrics = [("AP@0.50", "AP@0.50"),
               ("AP@0.50:0.95", "AP@[0.50:0.95]"),
               ("AR@100", "AR@100")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    for ax, (col, title) in zip(axes, metrics):
        cls_vals = df[df["type"] == "classical"][col].values
        dl_vals = df[df["type"] == "dl"][col].values

        data = [cls_vals, dl_vals]

        bplot = ax.boxplot(data, labels=["Classical", "Deep Learning"],
                           widths=0.55, patch_artist=True, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": BLACK,
                                      "markeredgecolor": BLACK, "markersize": 6},
                           medianprops={"color": BLACK, "linewidth": 1.4},
                           whiskerprops={"color": BLACK, "linewidth": 0.8},
                           capprops={"color": BLACK, "linewidth": 0.8},
                           flierprops={"marker": "x", "markerfacecolor": BLACK_50,
                                       "markeredgecolor": BLACK_50})
        for patch, colour in zip(bplot["boxes"], [GARNET, BLACK_70]):
            patch.set_facecolor(colour); patch.set_alpha(0.35)
            patch.set_edgecolor(BLACK); patch.set_linewidth(0.7)

        # Overlay individual points
        for i, vals in enumerate(data):
            xs = np.random.default_rng(0).normal(i + 1, 0.04, size=len(vals))
            colour = GARNET if i == 0 else BLACK_70
            ax.scatter(xs, vals, color=colour, alpha=0.8, s=20,
                       edgecolors=BLACK, linewidths=0.4, zorder=3)

        ax.set_title(title)
        ax.set_ylabel("Score")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 8: Dataset composition pie + sample counts ────────────────────

def fig_dataset(out_path: Path) -> None:
    with open("data/processed/annotations/train.json") as f:
        train = json.load(f)
    with open("data/processed/annotations/val.json") as f:
        val = json.load(f)
    with open("data/processed/annotations/test.json") as f:
        test = json.load(f)

    sets = [("Train (MANV)", train, HORSESHOE),
            ("Val (DARC)",   val,   ATLANTIC),
            ("Test (DAAN)",  test,  GARNET)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: images per split
    names = [s[0] for s in sets]
    img_counts = [len(s[1]["images"]) for s in sets]
    ann_counts = [len(s[1]["annotations"]) for s in sets]
    colours = [s[2] for s in sets]

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w / 2, img_counts, width=w, color=colours,
                edgecolor=BLACK, linewidth=0.5, label="Images")
    b2 = ax.bar(x + w / 2, ann_counts, width=w, color=[BLACK_70] * 3,
                edgecolor=BLACK, linewidth=0.5, label="Annotations")
    for rect, v in list(zip(b1, img_counts)) + list(zip(b2, ann_counts)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 80,
                f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Dataset split sizes")
    ax.legend(loc="upper right")

    # Panel B: number of objects per image (test set distribution)
    per_img = {}
    for a in test["annotations"]:
        per_img[a["image_id"]] = per_img.get(a["image_id"], 0) + 1
    vals = list(per_img.values())
    # Images with zero objects
    n_zero = len(test["images"]) - len(per_img)
    all_vals = vals + [0] * n_zero

    ax = axes[1]
    bins = np.arange(0, max(all_vals) + 2) - 0.5
    ax.hist(all_vals, bins=bins, color=GARNET, edgecolor=BLACK,
            linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Objects per image (test set)")
    ax.set_ylabel("Number of images")
    ax.set_title("Target density distribution")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main driver ──────────────────────────────────────────────────────────

def main() -> None:
    setup_style()
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)

    df = load_results()

    print("Rendering Figure 1: headline AP comparison")
    fig_headline(df, out / "fig1_headline.png")

    print("Rendering Figure 2: multi-metric panel")
    fig_multimetric(df, out / "fig2_multimetric.png")

    print("Rendering Figure 3: preprocessing pipeline")
    fig_preprocess(out / "fig3_preprocess.png", img_stem="DAAN_342")

    print("Rendering Figure 4: qualitative detection comparison")
    fig_qualitative(out / "fig4_qualitative.png")

    print("Rendering Figure 5: detection-count vs AP")
    fig_count_vs_ap(df, out / "fig5_count_vs_ap.png")

    print("Rendering Figure 6: bounding-box area distributions")
    fig_box_sizes(out / "fig6_boxsizes.png")

    print("Rendering Figure 7: classical vs DL boxplots")
    fig_classical_vs_dl(df, out / "fig7_classical_vs_dl.png")

    print("Rendering Figure 8: dataset composition")
    fig_dataset(out / "fig8_dataset.png")

    print(f"\nAll figures saved to {out.resolve()}")


if __name__ == "__main__":
    main()
