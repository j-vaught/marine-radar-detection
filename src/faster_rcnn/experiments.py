"""Run the Faster R-CNN P/S/T experiment matrix."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..unet.dataset import load_segmentation_items, subset_coco, subset_items
from .dataset import RadarDetectionDataset, coco_to_items, collate_fn
from .matrix import ExperimentSpec, build_all
from .model import build_model, predict_faster_rcnn, resolve_device, train_faster_rcnn


def make_loader(
    image_dir: str | Path,
    coco_ann: dict,
    cfg: dict,
    shuffle: bool,
) -> DataLoader:
    items = coco_to_items(coco_ann)
    workers = int(cfg["training"]["workers"])
    ds = RadarDetectionDataset(image_dir, items)
    return DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_floor: float,
    desc: str,
) -> list[dict]:
    model.eval()
    preds: list[dict] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            batch_images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(batch_images)
            for target, output in zip(targets, outputs, strict=True):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                for box, score in zip(boxes, scores, strict=True):
                    if float(score) < score_floor:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    preds.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
    return preds


def eval_predictions(gt_json_path: Path, pred_path: Path, verbose: bool = True) -> dict:
    gt = COCO(str(gt_json_path))
    preds = json.loads(pred_path.read_text())
    if preds:
        dt = gt.loadRes(str(pred_path))
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


def tune_thresholds(
    raw_preds: list[dict],
    gt_json_path: Path,
    out_dir: Path,
    cfg: dict,
) -> tuple[dict, dict]:
    post = cfg["postprocessing"]
    thresholds = [float(v) for v in post.get("score_threshold_grid", [post["score_threshold"]])]
    best_choice: dict | None = None
    best_metrics: dict | None = None

    for threshold in thresholds:
        filtered = [pred for pred in raw_preds if float(pred["score"]) >= threshold]
        pred_path = out_dir / f"val_thr{threshold:.2f}.json"
        pred_path.write_text(json.dumps(filtered))
        metrics = eval_predictions(gt_json_path, pred_path, verbose=False)
        if best_metrics is None or (metrics["ap50"], metrics["ap"]) > (best_metrics["ap50"], best_metrics["ap"]):
            best_choice = {"score_threshold": threshold}
            best_metrics = metrics

    if best_choice is None or best_metrics is None:
        raise RuntimeError("threshold sweep failed")
    return best_choice, best_metrics


def write_subset_json(path: Path, ann_path: str | Path, image_ids: set[int]) -> dict:
    subset = subset_coco(ann_path, image_ids)
    path.write_text(json.dumps(subset))
    return subset


def run_experiment(
    exp: ExperimentSpec,
    ann_path: str | Path,
    image_dir: str | Path,
    items: list,
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
    train_coco = write_subset_json(train_json, ann_path, {item.image_id for item in train_items})
    val_coco = write_subset_json(val_json, ann_path, {item.image_id for item in val_items})
    test_coco = write_subset_json(test_json, ann_path, {item.image_id for item in test_items})

    train_loader = make_loader(image_dir, train_coco, cfg, shuffle=True)
    val_loader = make_loader(image_dir, val_coco, cfg, shuffle=False)

    model = build_model(cfg, pretrained=True)
    started = time.time()
    train_stats = train_faster_rcnn(model, train_loader, val_loader, cfg, exp_dir, seed=exp.seed)
    train_seconds = time.time() - started

    torch.cuda.empty_cache()

    eval_model = build_model(
        cfg,
        pretrained=False,
        box_score_thresh=float(cfg["model"].get("eval_box_score_thresh", 0.0)),
        box_detections_per_img=int(cfg["model"].get("eval_box_detections_per_img", 300)),
    )
    ckpt_path = exp_dir / f"best_seed{exp.seed}.pth"
    eval_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    device = resolve_device(cfg["training"]["device"])
    eval_model.to(device)

    val_raw = collect_predictions(
        eval_model,
        make_loader(image_dir, val_coco, cfg, shuffle=False),
        device,
        score_floor=0.0,
        desc=f"{exp.exp_id} val",
    )
    (exp_dir / "val_raw_predictions.json").write_text(json.dumps(val_raw))
    best_post, val_metrics = tune_thresholds(val_raw, val_json, exp_dir, cfg)

    test_raw = collect_predictions(
        eval_model,
        make_loader(image_dir, test_coco, cfg, shuffle=False),
        device,
        score_floor=0.0,
        desc=f"{exp.exp_id} test",
    )
    (exp_dir / "test_raw_predictions.json").write_text(json.dumps(test_raw))
    test_filtered = [pred for pred in test_raw if float(pred["score"]) >= best_post["score_threshold"]]
    test_pred_path = exp_dir / "test_predictions.json"
    test_pred_path.write_text(json.dumps(test_filtered))
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
        "score_threshold": best_post["score_threshold"],
        "train_seconds": train_seconds,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    per_exp_dir = out_root / "per_experiment"
    per_exp_dir.mkdir(parents=True, exist_ok=True)
    (per_exp_dir / f"{exp.exp_id}.json").write_text(json.dumps(result, indent=2))

    del model
    del eval_model
    torch.cuda.empty_cache()
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
    ap.add_argument("--output-dir", default="results/faster_rcnn_matrix")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr-step-size", type=int, default=None)
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
    if args.lr_step_size is not None:
        cfg["training"]["lr_step_size"] = args.lr_step_size

    items = load_segmentation_items(args.ann)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    exps = filter_experiments(build_all(), args.only)
    print(f"[faster_rcnn] running {len(exps)} experiments")

    rows: list[dict] = []
    for exp in exps:
        metrics_path = out_root / exp.exp_id / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"[faster_rcnn] skip {exp.exp_id} (resume)")
            rows.append(json.loads(metrics_path.read_text()))
            continue
        print(f"[faster_rcnn] start {exp.exp_id}")
        rows.append(run_experiment(exp, args.ann, args.images, items, cfg, out_root))

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
        "score_threshold",
        "train_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[faster_rcnn] wrote {csv_path}")


if __name__ == "__main__":
    main()
