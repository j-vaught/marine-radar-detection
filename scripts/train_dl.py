"""Train all deep learning models with multi-seed runs."""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["yolov11", "faster_rcnn", "unet", "rt_detr"],
                        help="Model to train")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data config (YOLO yaml or annotation dir)")
    parser.add_argument("--output-dir", type=str, default="results/deep_learning",
                        help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override GPU device")
    return parser.parse_args()


def train_yolov11(config_path, data_path, output_dir, seeds):
    from src.deep_learning.yolov11_obb import train_yolov11 as _train
    for seed in seeds:
        print(f"\n--- YOLOv11-OBB seed={seed} ---")
        _train(data_path, config_path, output_dir, seed=seed)


def train_faster_rcnn(config_path, data_path, output_dir, seeds):
    import torch
    from torch.utils.data import DataLoader
    from src.deep_learning.faster_rcnn import build_model, train_faster_rcnn as _train
    from src.data.dataset import RadarDetectionDataset
    import json

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(data_path)

    # Load annotations
    with open(data_dir / "annotations" / "DAAN_train.json") as f:
        train_ann = json.load(f)
    with open(data_dir / "annotations" / "DAAN_val.json") as f:
        val_ann = json.load(f)

    # Build annotation lists for dataset
    def ann_to_list(ann_dict):
        img_id_to_anns = {}
        for a in ann_dict["annotations"]:
            img_id_to_anns.setdefault(a["image_id"], []).append(a)

        items = []
        for img in ann_dict["images"]:
            anns = img_id_to_anns.get(img["id"], [])
            boxes = []
            labels = []
            for a in anns:
                x, y, w, h = a["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(a["category_id"])
            items.append({
                "file_name": img["file_name"],
                "boxes": boxes,
                "labels": labels,
            })
        return items

    train_items = ann_to_list(train_ann)
    val_items = ann_to_list(val_ann)

    def collate_fn(batch):
        return tuple(zip(*batch))

    for seed in seeds:
        print(f"\n--- Faster R-CNN seed={seed} ---")
        model = build_model(cfg)

        train_ds = RadarDetectionDataset(data_dir / "tiles", train_items)
        val_ds = RadarDetectionDataset(data_dir / "tiles", val_items)

        train_loader = DataLoader(
            train_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=True, num_workers=cfg["training"]["workers"],
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=False, num_workers=cfg["training"]["workers"],
            collate_fn=collate_fn,
        )

        _train(model, train_loader, val_loader, cfg, output_dir, seed=seed)


def train_unet(config_path, data_path, output_dir, seeds):
    import torch
    from torch.utils.data import DataLoader
    from src.deep_learning.unet import UNet, train_unet as _train
    from src.data.dataset import RadarSegmentationDataset

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(data_path)

    for seed in seeds:
        print(f"\n--- U-Net seed={seed} ---")
        model = UNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
        )

        train_ds = RadarSegmentationDataset(
            data_dir / "tiles" / "train",
            data_dir / "masks" / "train",
        )
        val_ds = RadarSegmentationDataset(
            data_dir / "tiles" / "val",
            data_dir / "masks" / "val",
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=True, num_workers=cfg["training"]["workers"],
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=False, num_workers=cfg["training"]["workers"],
        )

        _train(model, train_loader, val_loader, cfg, output_dir, seed=seed)


def train_rtdetr(config_path, data_path, output_dir, seeds):
    from src.deep_learning.rt_detr import train_rt_detr as _train
    for seed in seeds:
        print(f"\n--- RT-DETR seed={seed} ---")
        _train(data_path, config_path, output_dir, seed=seed)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["training"]["seeds"]
    if args.device:
        cfg["training"]["device"] = args.device

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "yolov11": train_yolov11,
        "faster_rcnn": train_faster_rcnn,
        "unet": train_unet,
        "rt_detr": train_rtdetr,
    }

    dispatch[args.model](args.config, args.data, output_dir, seeds)


if __name__ == "__main__":
    main()
