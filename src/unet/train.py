"""Train U-Net with 3-seed multi-run.

Usage:
    python -m src.unet.train --data data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from .dataset import RadarSegmentationDataset
from .model import UNet, train_unet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed data root (contains tiles/ and masks/)")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--output-dir", default="results/unet")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.device is not None:
        cfg["training"]["device"] = args.device
    seeds = args.seeds or cfg["training"]["seeds"]

    data_dir = Path(args.data)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n--- U-Net seed={seed} device={cfg['training']['device']} ---")
        model = UNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
        )
        train_ds = RadarSegmentationDataset(data_dir / "tiles" / "train", data_dir / "masks" / "train")
        val_ds = RadarSegmentationDataset(data_dir / "tiles" / "val", data_dir / "masks" / "val")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"]["workers"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["workers"],
        )

        train_unet(model, train_loader, val_loader, cfg, str(out), seed=seed)


if __name__ == "__main__":
    main()
