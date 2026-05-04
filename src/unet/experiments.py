"""Run the U-Net P/S/T experiment matrix."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from .dataset import (
    RadarSegmentationItemsDataset,
    SegmentationItem,
    load_segmentation_items,
    subset_coco,
    subset_items,
)
from .matrix import ExperimentSpec, build_all
from .model import UNet, predict_prob_map, train_unet
from .utils import binary_mask_to_bboxes_with_scores


def make_loader(
    image_dir: str | Path,
    items: list[SegmentationItem],
    cfg: dict,
    shuffle: bool,
) -> DataLoader:
    input_size = tuple(cfg["training"].get("input_size", ())) or None
    ds = RadarSegmentationItemsDataset(image_dir, items, input_size=input_size)
    workers = int(cfg["training"]["workers"])
    return DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def load_prob_cache(
    model: UNet,
    items: list[SegmentationItem],
    image_dir: str | Path,
    device: str,
    input_size: tuple[int, int] | None,
    desc: str,
) -> list[dict]:
    cache: list[dict] = []
    for item in items:
        image = cv2.imread(str(Path(image_dir) / item.file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(Path(image_dir) / item.file_name)
        prob_map = predict_prob_map(model, image, device=device, input_size=input_size)
        cache.append({"item": item, "prob_map": prob_map.astype("float16")})
    print(f"[unet] cached {len(cache)} probability maps for {desc}")
    return cache


def preds_from_cache(
    cache: list[dict],
    threshold: float,
    min_area: int,
    max_area: int,
    dilation: int,
    box_padding: int,
) -> list[dict]:
    preds: list[dict] = []
    for entry in cache:
        item = entry["item"]
        prob_map = entry["prob_map"].astype("float32")
        mask = (prob_map >= threshold).astype("uint8")
        bboxes, scores = binary_mask_to_bboxes_with_scores(
            mask,
            prob_map,
            min_area=min_area,
            max_area=max_area,
            dilation=dilation,
            box_padding=box_padding,
        )
        for box, score in zip(bboxes, scores, strict=True):
            x1, y1, x2, y2 = box.tolist()
            preds.append(
                {
                    "image_id": item.image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return preds


def eval_predictions(gt_json_path: Path, preds_path: Path, verbose: bool = True) -> dict:
    gt = COCO(str(gt_json_path))
    preds = json.loads(preds_path.read_text())
    if preds:
        dt = gt.loadRes(str(preds_path))
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        if verbose:
            ev.summarize()
        else:
            with redirect_stdout(io.StringIO()):
                ev.summarize()
        stats = ev.stats.tolist()
    else:
        stats = [0.0] * 12
    n_images = len(gt.dataset["images"])
    return {
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "predictions": len(preds),
        "preds_per_image": float(len(preds) / max(1, n_images)),
    }


def tune_postprocessing(
    cache: list[dict],
    gt_json_path: Path,
    out_dir: Path,
    cfg: dict,
) -> tuple[dict, dict]:
    post = cfg["postprocessing"]
    thresholds = [float(v) for v in post.get("threshold_grid", [post.get("threshold", 0.5)])]
    min_areas = [int(v) for v in post.get("min_area_grid", [post.get("min_area", 15)])]
    max_area = int(post.get("max_area", 50000))
    dilation = int(post.get("dilation", 0))
    box_padding = int(post.get("box_padding", 0))

    best_choice: dict | None = None
    best_metrics: dict | None = None
    for threshold in thresholds:
        for min_area in min_areas:
            preds = preds_from_cache(
                cache,
                threshold=threshold,
                min_area=min_area,
                max_area=max_area,
                dilation=dilation,
                box_padding=box_padding,
            )
            pred_path = out_dir / f"val_thr{threshold:.2f}_area{min_area}.json"
            pred_path.write_text(json.dumps(preds))
            metrics = eval_predictions(gt_json_path, pred_path, verbose=False)
            if best_metrics is None or (metrics["ap50"], metrics["ap"]) > (best_metrics["ap50"], best_metrics["ap"]):
                best_choice = {
                    "threshold": threshold,
                    "min_area": min_area,
                    "max_area": max_area,
                    "dilation": dilation,
                    "box_padding": box_padding,
                }
                best_metrics = metrics

    if best_choice is None or best_metrics is None:
        raise RuntimeError("postprocessing sweep failed")
    return best_choice, best_metrics


def write_subset_json(path: Path, ann_path: str | Path, items: list[SegmentationItem]) -> None:
    image_ids = {item.image_id for item in items}
    path.write_text(json.dumps(subset_coco(ann_path, image_ids)))


def run_experiment(
    exp: ExperimentSpec,
    ann_path: str | Path,
    image_dir: str | Path,
    items: list[SegmentationItem],
    cfg: dict,
    out_root: Path,
) -> dict:
    exp_dir = out_root / exp.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = exp.split(items)
    train_items = subset_items(items, train_idx)
    val_items = subset_items(items, val_idx)
    test_items = subset_items(items, test_idx)

    train_json = exp_dir / "train_gt.json"
    val_json = exp_dir / "val_gt.json"
    test_json = exp_dir / "test_gt.json"
    write_subset_json(train_json, ann_path, train_items)
    write_subset_json(val_json, ann_path, val_items)
    write_subset_json(test_json, ann_path, test_items)

    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"].get("base_channels", 32),
    )
    train_loader = make_loader(image_dir, train_items, cfg, shuffle=True)
    val_loader = make_loader(image_dir, val_items, cfg, shuffle=False)
    train_seed = exp.seed

    started = time.time()
    train_stats = train_unet(model, train_loader, val_loader, cfg, exp_dir, seed=train_seed)
    train_seconds = time.time() - started

    ckpt_path = exp_dir / f"best_seed{train_seed}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    device = cfg["training"]["device"]
    input_size = tuple(cfg["training"].get("input_size", ())) or None
    val_cache = load_prob_cache(model, val_items, image_dir, device, input_size, f"{exp.exp_id} val")
    best_post, val_metrics = tune_postprocessing(val_cache, val_json, exp_dir, cfg)
    test_cache = load_prob_cache(model, test_items, image_dir, device, input_size, f"{exp.exp_id} test")
    test_preds = preds_from_cache(test_cache, **best_post)
    test_pred_path = exp_dir / "test_predictions.json"
    test_pred_path.write_text(json.dumps(test_preds))
    test_metrics = eval_predictions(test_json, test_pred_path, verbose=True)

    result = {
        "exp_id": exp.exp_id,
        "condition": exp.condition,
        "tune_label": exp.tune_label,
        "val_label": exp.val_label,
        "test_label": exp.test_label,
        "seed": exp.seed,
        "train_images": len(train_items),
        "val_images": len(val_items),
        "test_images": len(test_items),
        "best_epoch": train_stats["best_epoch"],
        "best_val_loss": train_stats["best_val_loss"],
        "val_ap": val_metrics["ap"],
        "val_ap50": val_metrics["ap50"],
        "test_ap": test_metrics["ap"],
        "test_ap50": test_metrics["ap50"],
        "test_ap75": test_metrics["ap75"],
        "predictions": test_metrics["predictions"],
        "preds_per_image": test_metrics["preds_per_image"],
        "threshold": best_post["threshold"],
        "min_area": best_post["min_area"],
        "max_area": best_post["max_area"],
        "dilation": best_post["dilation"],
        "box_padding": best_post["box_padding"],
        "train_seconds": train_seconds,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    per_exp_dir = out_root / "per_experiment"
    per_exp_dir.mkdir(parents=True, exist_ok=True)
    (per_exp_dir / f"{exp.exp_id}.json").write_text(json.dumps(result, indent=2))
    return result


def filter_experiments(exps: list[ExperimentSpec], only: list[str] | None) -> list[ExperimentSpec]:
    if not only:
        return exps
    wanted = {raw.upper() for raw in only}
    return [exp for exp in exps if exp.exp_id.upper() in wanted or exp.condition.upper() in wanted]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/processed/annotations/all.json")
    ap.add_argument("--images", default="data/processed/images")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--output-dir", default="results/unet_matrix")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--t-max", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["training"]["workers"] = args.workers
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.t_max is not None:
        cfg["training"]["T_max"] = args.t_max
    elif args.epochs is not None:
        cfg["training"]["T_max"] = args.epochs

    items = load_segmentation_items(args.ann)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    exps = filter_experiments(build_all(), args.only)
    print(f"[unet] running {len(exps)} experiments")

    rows: list[dict] = []
    for exp in exps:
        metrics_path = out_root / exp.exp_id / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"[unet] skip {exp.exp_id} (resume)")
            rows.append(json.loads(metrics_path.read_text()))
            continue
        print(f"[unet] start {exp.exp_id}")
        row = run_experiment(exp, args.ann, args.images, items, cfg, out_root)
        rows.append(row)

    csv_path = out_root / "matrix.csv"
    fieldnames = [
        "exp_id",
        "condition",
        "tune_label",
        "val_label",
        "test_label",
        "seed",
        "train_images",
        "val_images",
        "test_images",
        "best_epoch",
        "best_val_loss",
        "val_ap",
        "val_ap50",
        "test_ap",
        "test_ap50",
        "test_ap75",
        "predictions",
        "preds_per_image",
        "threshold",
        "min_area",
        "max_area",
        "dilation",
        "box_padding",
        "train_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[unet] wrote {csv_path}")


if __name__ == "__main__":
    main()
