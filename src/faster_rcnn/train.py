"""Train Faster R-CNN with 3-seed multi-run.

Usage:
    python -m src.faster_rcnn.train --data data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from .dataset import RadarDetectionDataset, coco_to_items
from .model import build_model, train_faster_rcnn


def _collate(batch):
    return tuple(zip(*batch))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed data root (contains annotations/ and tiles/)")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--output-dir", default="results/faster_rcnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.device is not None:
        cfg["training"]["device"] = args.device
    seeds = args.seeds or cfg["training"]["seeds"]

    data_dir = Path(args.data)
    with open(data_dir / "annotations" / "train_tiled.json") as f:
        train_ann = json.load(f)
    with open(data_dir / "annotations" / "val_tiled.json") as f:
        val_ann = json.load(f)

    train_items = coco_to_items(train_ann)
    val_items = coco_to_items(val_ann)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n--- Faster R-CNN seed={seed} device={cfg['training']['device']} ---")
        model = build_model(cfg)

        train_ds = RadarDetectionDataset(data_dir / "tiles" / "train", train_items)
        val_ds = RadarDetectionDataset(data_dir / "tiles" / "val", val_items)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"]["workers"],
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["workers"],
            collate_fn=_collate,
        )

        train_faster_rcnn(model, train_loader, val_loader, cfg, str(out), seed=seed)


if __name__ == "__main__":
    main()
