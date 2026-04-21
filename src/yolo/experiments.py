"""Run the YOLO26 P/S/T experiment matrix."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..unet.dataset import load_segmentation_items, subset_coco, subset_items
from .coco_to_yolo import write_labels
from .matrix import ExperimentSpec, build_all
from .model import load_config, predict_sahi, resolve_device, train_yolo


def write_subset_json(path: Path, ann_path: str | Path, image_ids: set[int]) -> dict:
    subset = subset_coco(ann_path, image_ids)
    path.write_text(json.dumps(subset))
    return subset


def link_image(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def stage_split(split_root: Path, image_dir: str | Path, subset: dict, subset_json: Path) -> Path:
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    wanted_names = {Path(info["file_name"]).name for info in subset["images"]}
    for existing in images_dir.glob("*"):
        if existing.name not in wanted_names:
            existing.unlink()
    for existing in labels_dir.glob("*.txt"):
        if f"{existing.stem}.png" not in wanted_names:
            existing.unlink()

    for info in subset["images"]:
        src = Path(image_dir) / info["file_name"]
        if not src.exists():
            raise FileNotFoundError(src)
        link_image(src, images_dir / Path(info["file_name"]).name)

    write_labels(subset_json, images_dir, labels_dir)
    return images_dir


def write_data_yaml(root: Path) -> Path:
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: {root}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: target\n".format(root=root.resolve())
    )
    return data_yaml


def collect_predictions_direct(
    weights_path: Path,
    images_dir: Path,
    subset: dict,
    cfg: dict,
    score_floor: float,
    desc: str,
) -> list[dict]:
    from ultralytics import YOLO
    from tqdm import tqdm

    model = YOLO(str(weights_path))
    image_lookup = {Path(info["file_name"]).name: int(info["id"]) for info in subset["images"]}
    preds: list[dict] = []
    results = model.predict(
        source=str(images_dir),
        conf=score_floor,
        device=resolve_device(cfg["training"].get("device")),
        imgsz=int(cfg["training"]["imgsz"]),
        batch=int(cfg["training"]["batch_size"]),
        stream=True,
        verbose=False,
    )

    for result in tqdm(results, total=len(image_lookup), desc=desc):
        image_name = Path(result.path).name
        image_id = image_lookup[image_name]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confs, strict=True):
            x1, y1, x2, y2 = box.tolist()
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    del model
    return preds


def collect_predictions_sahi(
    weights_path: Path,
    images_dir: Path,
    subset: dict,
    cfg: dict,
    score_floor: float,
    desc: str,
    config_path: Path,
) -> list[dict]:
    from tqdm import tqdm

    image_lookup = {Path(info["file_name"]).name: int(info["id"]) for info in subset["images"]}
    preds: list[dict] = []

    for image_name in tqdm(sorted(image_lookup), desc=desc):
        image_id = image_lookup[image_name]
        dets = predict_sahi(
            weights_path,
            images_dir / image_name,
            config_path,
            device=cfg["training"].get("device"),
            cfg=cfg,
        )
        for det in dets:
            score = float(det["score"])
            if score < score_floor:
                continue
            x1, y1, x2, y2 = det["bbox"]
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                }
            )
    return preds


def collect_predictions(
    weights_path: Path,
    images_dir: Path,
    subset: dict,
    cfg: dict,
    score_floor: float,
    desc: str,
    config_path: Path,
) -> list[dict]:
    if bool(cfg.get("sahi", {}).get("enabled", False)):
        return collect_predictions_sahi(weights_path, images_dir, subset, cfg, score_floor, desc, config_path)
    return collect_predictions_direct(weights_path, images_dir, subset, cfg, score_floor, desc)


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


def tune_thresholds(raw_preds: list[dict], gt_json_path: Path, out_dir: Path, cfg: dict) -> tuple[dict, dict]:
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


def read_training_summary(save_dir: Path) -> dict:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        return {"best_epoch": -1, "best_map": 0.0, "best_map50": 0.0}

    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"best_epoch": -1, "best_map": 0.0, "best_map50": 0.0}

    metric_col = None
    for candidate in ("metrics/mAP50-95(B)", "metrics/mAP50(B)"):
        if candidate in rows[0]:
            metric_col = candidate
            break
    if metric_col is None:
        best_idx = len(rows) - 1
    else:
        best_idx = max(range(len(rows)), key=lambda idx: float(rows[idx].get(metric_col, 0.0) or 0.0))

    best_row = rows[best_idx]
    best_epoch = int(float(best_row.get("epoch", best_idx + 1)))
    return {
        "best_epoch": best_epoch,
        "best_map": float(best_row.get("metrics/mAP50-95(B)", 0.0) or 0.0),
        "best_map50": float(best_row.get("metrics/mAP50(B)", 0.0) or 0.0),
    }


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

    yolo_root = exp_dir / "yolo"
    stage_split(yolo_root / "train", image_dir, train_coco, train_json)
    val_images_dir = stage_split(yolo_root / "val", image_dir, val_coco, val_json)
    test_images_dir = stage_split(yolo_root / "test", image_dir, test_coco, test_json)
    data_yaml = write_data_yaml(yolo_root)

    started = time.time()
    train_info = train_yolo(data_yaml, exp_dir / "train_config.yaml", exp_dir, seed=exp.seed, cfg=cfg)
    train_seconds = time.time() - started
    save_dir = Path(train_info["save_dir"])
    weights_path = Path(train_info["weights"])
    if not weights_path.exists():
        fallback = save_dir / "weights" / "last.pt"
        if not fallback.exists():
            raise FileNotFoundError(weights_path)
        weights_path = fallback
    train_summary = read_training_summary(save_dir)

    score_floor = float(cfg["postprocessing"].get("score_floor", 0.001))
    config_path = exp_dir / "train_config.yaml"
    val_raw = collect_predictions(weights_path, val_images_dir, val_coco, cfg, score_floor, f"{exp.exp_id} val", config_path)
    (exp_dir / "val_raw_predictions.json").write_text(json.dumps(val_raw))
    best_post, val_metrics = tune_thresholds(val_raw, val_json, exp_dir, cfg)

    test_raw = collect_predictions(weights_path, test_images_dir, test_coco, cfg, score_floor, f"{exp.exp_id} test", config_path)
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
        "best_epoch": train_summary["best_epoch"],
        "best_map": train_summary["best_map"],
        "best_map50": train_summary["best_map50"],
        "val_ap": val_metrics["ap"],
        "val_ap50": val_metrics["ap50"],
        "test_ap": test_metrics["ap"],
        "test_ap50": test_metrics["ap50"],
        "test_ap75": test_metrics["ap75"],
        "predictions": test_metrics["predictions"],
        "preds_per_image": test_metrics["preds_per_image"],
        "batch_size": int(train_info["batch_size"]),
        "score_threshold": best_post["score_threshold"],
        "sahi_enabled": bool(cfg.get("sahi", {}).get("enabled", False)),
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
    ap.add_argument("--output-dir", default="results/yolo26_matrix")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["training"]["workers"] = args.workers

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    items = load_segmentation_items(args.ann)
    exps = filter_experiments(build_all(), args.only)
    print(f"[yolo] running {len(exps)} experiments")

    rows: list[dict] = []
    for exp in exps:
        metrics_path = out_root / exp.exp_id / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"[yolo] skip {exp.exp_id} (resume)")
            rows.append(json.loads(metrics_path.read_text()))
            continue

        print(f"[yolo] start {exp.exp_id}")
        exp_dir = out_root / exp.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
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
        "best_map",
        "best_map50",
        "val_ap",
        "val_ap50",
        "test_ap",
        "test_ap50",
        "test_ap75",
        "predictions",
        "preds_per_image",
        "batch_size",
        "score_threshold",
        "sahi_enabled",
        "train_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[yolo] wrote {csv_path}")


if __name__ == "__main__":
    main()
