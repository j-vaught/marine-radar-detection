"""Run trained DL models on the test set and produce COCO-format predictions."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import format_predictions_coco


def parse_args():
    parser = argparse.ArgumentParser(description="Run DL inference on test set")
    parser.add_argument("--model", type=str, required=True,
                        choices=["yolov11", "faster_rcnn", "unet", "rt_detr"],
                        help="Model to run")
    parser.add_argument("--weights-dir", type=str, required=True,
                        help="Directory containing trained model weights")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--test-ann", type=str, required=True,
                        help="Path to test annotation JSON (tiled or full-image)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing test images/tiles")
    parser.add_argument("--output-dir", type=str, default="results/deep_learning",
                        help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Confidence threshold (low for full AP evaluation)")
    return parser.parse_args()


def infer_yolov11(args, test_ann):
    """YOLOv11-OBB with optional SAHI tiled inference."""
    from src.deep_learning.yolov11_obb import predict_yolov11, predict_sahi

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    # Find all seed checkpoints
    best_weights = sorted(weights_dir.rglob("best.pt"))
    if not best_weights:
        best_weights = sorted(weights_dir.rglob("*.pt"))

    seeds = cfg["training"]["seeds"]
    use_sahi = cfg.get("sahi", {}).get("enabled", False)

    for i, seed in enumerate(seeds):
        if i >= len(best_weights):
            print(f"  No checkpoint found for seed {seed}, skipping")
            continue

        weight_path = best_weights[i]
        print(f"\n--- YOLOv11 seed={seed} weights={weight_path} ---")

        all_preds = []
        for img_info in tqdm(test_ann["images"], desc=f"YOLOv11 seed={seed}"):
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            if use_sahi:
                predictions = predict_sahi(str(weight_path), str(img_path), args.config)
                for pred in predictions:
                    bbox = pred["bbox"]
                    all_preds.append({
                        "image_id": img_info["id"],
                        "category_id": 1,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "score": pred["score"],
                    })
            else:
                image = cv2.imread(str(img_path))
                detections = predict_yolov11(str(weight_path), image, conf=args.conf, device=args.device)
                for det in detections:
                    x1, y1, x2, y2, score = det
                    all_preds.append({
                        "image_id": img_info["id"],
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    })

        pred_path = output_dir / f"yolov11_seed{seed}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(all_preds, f)
        print(f"  Saved {len(all_preds)} predictions to {pred_path}")


def infer_faster_rcnn(args, test_ann):
    """Faster R-CNN inference on test tiles."""
    from src.deep_learning.faster_rcnn import build_model, predict_faster_rcnn

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    device = f"cuda:{args.device}"

    seeds = cfg["training"]["seeds"]

    for seed in seeds:
        weight_path = weights_dir / f"best_seed{seed}.pth"
        if not weight_path.exists():
            weight_path = weights_dir / f"last_seed{seed}.pth"
        if not weight_path.exists():
            print(f"  No checkpoint for seed {seed}, skipping")
            continue

        print(f"\n--- Faster R-CNN seed={seed} ---")
        model = build_model(cfg)
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))

        all_preds = []
        for img_info in tqdm(test_ann["images"], desc=f"FRCNN seed={seed}"):
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            image_3ch = np.stack([image] * 3, axis=-1)
            tensor = torch.from_numpy(image_3ch).permute(2, 0, 1).float() / 255.0

            detections = predict_faster_rcnn(model, tensor, device=device, score_thresh=args.conf)
            preds = format_predictions_coco(
                img_info["id"],
                detections[:, :4],
                detections[:, 4] if len(detections) > 0 else None,
            )
            all_preds.extend(preds)

        pred_path = output_dir / f"faster_rcnn_seed{seed}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(all_preds, f)
        print(f"  Saved {len(all_preds)} predictions to {pred_path}")


def infer_unet(args, test_ann):
    """U-Net segmentation inference, extracting bboxes from predicted masks."""
    from src.deep_learning.unet import UNet, predict_unet

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    device = f"cuda:{args.device}"

    seeds = cfg["training"]["seeds"]
    post_cfg = cfg.get("postprocessing", {})
    threshold = post_cfg.get("threshold", 0.5)
    min_area = post_cfg.get("min_area", 15)

    for seed in seeds:
        weight_path = weights_dir / f"best_seed{seed}.pth"
        if not weight_path.exists():
            print(f"  No checkpoint for seed {seed}, skipping")
            continue

        print(f"\n--- U-Net seed={seed} ---")
        model = UNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
        )
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))

        all_preds = []
        for img_info in tqdm(test_ann["images"], desc=f"U-Net seed={seed}"):
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0

            bboxes, pred_mask = predict_unet(
                model, tensor, device=device, threshold=threshold, min_area=min_area
            )

            # Use mask intensity as confidence
            scores = np.ones(len(bboxes), dtype=np.float32)
            for j, box in enumerate(bboxes):
                x1, y1, x2, y2 = box.astype(int)
                region = pred_mask[max(0, y1):y2, max(0, x1):x2]
                if region.size > 0:
                    scores[j] = float(region.mean())

            preds = format_predictions_coco(img_info["id"], bboxes, scores)
            all_preds.extend(preds)

        pred_path = output_dir / f"unet_seed{seed}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(all_preds, f)
        print(f"  Saved {len(all_preds)} predictions to {pred_path}")


def infer_rt_detr(args, test_ann):
    """RT-DETR inference."""
    from src.deep_learning.rt_detr import predict_rt_detr

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    seeds = cfg["training"]["seeds"]
    best_weights = sorted(weights_dir.rglob("best.pt"))

    for i, seed in enumerate(seeds):
        if i >= len(best_weights):
            print(f"  No checkpoint for seed {seed}, skipping")
            continue

        weight_path = best_weights[i]
        print(f"\n--- RT-DETR seed={seed} ---")

        all_preds = []
        for img_info in tqdm(test_ann["images"], desc=f"RT-DETR seed={seed}"):
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            detections = predict_rt_detr(str(weight_path), image, conf=args.conf, device=args.device)
            preds = format_predictions_coco(
                img_info["id"],
                detections[:, :4],
                detections[:, 4] if len(detections) > 0 else None,
            )
            all_preds.extend(preds)

        pred_path = output_dir / f"rt_detr_seed{seed}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(all_preds, f)
        print(f"  Saved {len(all_preds)} predictions to {pred_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_ann) as f:
        test_ann = json.load(f)

    dispatch = {
        "yolov11": infer_yolov11,
        "faster_rcnn": infer_faster_rcnn,
        "unet": infer_unet,
        "rt_detr": infer_rt_detr,
    }

    dispatch[args.model](args, test_ann)


if __name__ == "__main__":
    main()
