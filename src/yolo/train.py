"""Train a YOLO model on the radar dataset.

Usage:
    python -m src.yolo.train --data data/processed/yolo/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .model import load_config, train_yolo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--output-dir", default="results/yolo")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    seeds = args.seeds or cfg["training"]["seeds"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n--- {cfg['model']['name']} seed={seed} device={cfg['training']['device']} ---")
        train_yolo(args.data, args.config, str(out), seed=seed, cfg=cfg)


if __name__ == "__main__":
    main()
