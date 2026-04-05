"""Preprocess DLR DAAN/HAXR datasets: tile, split, and convert annotations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import load_ppi_image, apply_clahe
from src.data.tiling import tile_image, clip_boxes_to_tile, save_tiles
from src.data.splits import temporal_split, save_split, print_split_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess marine radar dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["DAAN", "HAXR"],
                        help="Dataset name")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Tile overlap fraction")
    parser.add_argument("--apply-clahe", action="store_true",
                        help="Apply CLAHE enhancement")
    parser.add_argument("--train-sessions", nargs="+", default=None,
                        help="Session IDs for training split")
    parser.add_argument("--val-sessions", nargs="+", default=None,
                        help="Session IDs for validation split")
    parser.add_argument("--test-sessions", nargs="+", default=None,
                        help="Session IDs for test split")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Load annotations
    ann_files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("**/*.json"))
    if not ann_files:
        print(f"No annotation JSON found in {raw_dir}")
        return

    ann_path = ann_files[0]
    print(f"Loading annotations from {ann_path}")
    with open(ann_path) as f:
        annotations = json.load(f)

    # Build image_id -> annotations mapping
    img_id_to_anns = {}
    for ann in annotations.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # Process each image
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    all_tile_anns = {
        "images": [],
        "annotations": [],
        "categories": annotations.get("categories", [{"id": 1, "name": "target"}]),
    }
    tile_id = 0
    ann_id = 0

    for img_info in tqdm(annotations.get("images", []), desc="Processing images"):
        img_path = raw_dir / img_info["file_name"]
        if not img_path.exists():
            # Try subdirectory
            candidates = list(raw_dir.rglob(img_info["file_name"]))
            if not candidates:
                continue
            img_path = candidates[0]

        image = load_ppi_image(str(img_path), normalize=False)
        if args.apply_clahe:
            image = apply_clahe(image)

        # Get GT boxes for this image
        gt_anns = img_id_to_anns.get(img_info["id"], [])
        gt_boxes = np.array([a["bbox"] for a in gt_anns], dtype=np.float32)  # COCO: x,y,w,h

        # Convert xywh -> xyxy
        if len(gt_boxes) > 0:
            gt_boxes_xyxy = gt_boxes.copy()
            gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]
            gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]
        else:
            gt_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)

        # Tile
        tiles = tile_image(image, tile_size=args.tile_size, overlap=args.overlap)
        frame_id = Path(img_info["file_name"]).stem

        for i, t in enumerate(tiles):
            tile_name = f"{frame_id}_tile{i:04d}.png"
            cv2.imwrite(str(tiles_dir / tile_name), t["tile"])

            # Clip boxes to this tile
            tile_boxes = clip_boxes_to_tile(
                gt_boxes_xyxy, t["x_offset"], t["y_offset"], args.tile_size
            )

            all_tile_anns["images"].append({
                "id": tile_id,
                "file_name": tile_name,
                "width": args.tile_size,
                "height": args.tile_size,
                "source_frame": img_info["file_name"],
                "x_offset": t["x_offset"],
                "y_offset": t["y_offset"],
            })

            for box in tile_boxes:
                # Convert xyxy -> xywh for COCO
                x1, y1, x2, y2 = box
                all_tile_anns["annotations"].append({
                    "id": ann_id,
                    "image_id": tile_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })
                ann_id += 1

            tile_id += 1

    # Save annotations
    ann_output = output_dir / "annotations"
    ann_output.mkdir(parents=True, exist_ok=True)
    save_split(all_tile_anns, ann_output / f"{args.dataset}_all.json")

    # Split if sessions provided
    if args.train_sessions and args.val_sessions and args.test_sessions:
        train, val, test = temporal_split(
            all_tile_anns, args.train_sessions, args.val_sessions, args.test_sessions
        )
        save_split(train, ann_output / f"{args.dataset}_train.json")
        save_split(val, ann_output / f"{args.dataset}_val.json")
        save_split(test, ann_output / f"{args.dataset}_test.json")
        print_split_stats(train, val, test)

    print(f"Done. {tile_id} tiles, {ann_id} annotations saved to {output_dir}")


if __name__ == "__main__":
    main()
