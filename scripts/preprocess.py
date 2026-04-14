"""Unified preprocessing pipeline for DLR (DAAN/DARC/MANV) and HAXR datasets.

Workflow:
    1. Convert DLR polar annotations to COCO bounding boxes
    2. Render HAXR HDF5 radar cycles to PPI images + COCO annotations
    3. Copy/rename images with dataset prefix
    4. Merge all sub-datasets into a single COCO annotation file
    5. Temporal split by dataset / station
    6. Tile images to 640x640 with overlap
    7. Export YOLO labels and U-Net masks
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ais_anchored_gt import convert_all_datasets
from src.data.haxr_renderer import render_dataset as render_haxr
from src.data.preprocessing import apply_clahe
from src.data.radar_mask import apply_radar_mask, create_radar_mask
from src.data.tiling import tile_image, clip_boxes_to_tile
from src.data.splits import save_split, print_split_stats
from src.data.format_converters import (
    coco_to_yolo,
    write_yolo_data_yaml,
    generate_segmentation_masks,
    reorganize_tiles_by_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing for marine radar detection benchmark"
    )
    parser.add_argument(
        "--config", type=str, default="configs/data_split.yaml",
        help="Path to data split config YAML",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Root directory containing raw datasets (DAAN/, DARC/, MANV/, HAXR/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for all processed data",
    )
    parser.add_argument(
        "--skip-haxr", action="store_true",
        help="Skip HAXR rendering (DLR datasets only)",
    )
    parser.add_argument(
        "--skip-dlr", action="store_true",
        help="Skip DLR datasets (HAXR only)",
    )
    parser.add_argument(
        "--apply-clahe", action="store_true",
        help="Apply CLAHE contrast enhancement to images",
    )
    parser.add_argument(
        "--skip-tiling", action="store_true",
        help="Skip tiling (only generate full-image annotations)",
    )
    parser.add_argument(
        "--skip-yolo", action="store_true",
        help="Skip YOLO format export",
    )
    parser.add_argument(
        "--skip-masks", action="store_true",
        help="Skip U-Net mask generation",
    )
    return parser.parse_args()


# ── Step 1: Convert DLR polar annotations to COCO ────────────────────────

def process_dlr(raw_dir: Path, output_dir: Path, cfg: dict, apply_clahe_flag: bool) -> dict:
    """Convert all DLR sub-datasets to COCO format and copy images."""
    dlr_cfg = cfg["dlr"]
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dlr_datasets = ["DAAN", "DARC", "MANV"]
    available = [d for d in dlr_datasets if (raw_dir / d / "polarMeas.json").exists()]

    if not available:
        print("No DLR datasets found, skipping")
        return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "vessel"}]}

    print(f"Processing DLR datasets: {available}")
    merged_coco = convert_all_datasets(
        str(raw_dir),
        datasets=available,
        polar_eps_m=dlr_cfg.get("dbscan_eps", 200.0),
        polar_min_samples=dlr_cfg.get("dbscan_min_samples", 5),
        threshold=dlr_cfg.get("blob_threshold", 25),
        min_area=dlr_cfg.get("blob_min_area", 10),
        max_area=dlr_cfg.get("blob_max_area", 8000),
        match_radius=dlr_cfg.get("match_radius_px", 30),
        include_unclaimed=dlr_cfg.get("include_unclaimed", False),
    )

    # Copy images with dataset-prefixed names
    for img_info in tqdm(merged_coco["images"], desc="Copying DLR images"):
        fname = img_info["file_name"]  # e.g., "DAAN_010.png"
        dataset_name = fname.split("_")[0]
        original_fname = fname[len(dataset_name) + 1:]  # e.g., "010.png"
        src = raw_dir / dataset_name / "radarImages" / original_fname
        dst = images_dir / fname

        if src.exists() and not dst.exists():
            img = cv2.imread(str(src))
            if img is not None:
                # Extract red channel (BGR index 2) -- contains radar returns
                # without blue background or green UI marks (bearing ticks, compass)
                if len(img.shape) == 3:
                    img = img[:, :, 2]  # Red channel in BGR
                # Mask out everything outside the PPI circle (set to true zero)
                radar_msk = create_radar_mask(img.shape[0], img.shape[1])
                img[radar_msk == 0] = 0
                if apply_clahe_flag:
                    img = apply_clahe(img)
                cv2.imwrite(str(dst), img)

    return merged_coco


# ── Step 2: Render HAXR HDF5 to PPI images ───────────────────────────────

def process_haxr(raw_dir: Path, output_dir: Path, cfg: dict) -> dict:
    """Render HAXR data and produce COCO annotations."""
    haxr_dir = raw_dir / "HAXR"
    if not haxr_dir.exists():
        print("HAXR directory not found, skipping")
        return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "vessel"}]}

    haxr_cfg = cfg["haxr"]
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering HAXR radar data...")
    haxr_coco = render_haxr(
        hdf5_dir=str(haxr_dir),
        output_dir=str(images_dir),
        sample_rate=haxr_cfg["sample_rate"],
        img_size=haxr_cfg["img_size"],
        box_size=haxr_cfg["box_size"],
        time_window=haxr_cfg.get("time_window", 1.0),
    )

    return haxr_coco


# ── Step 3: Merge and split ──────────────────────────────────────────────

def merge_coco_dicts(*coco_dicts: dict) -> dict:
    """Merge multiple COCO dicts, re-indexing IDs to be globally unique."""
    merged = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "vessel"}],
    }

    img_id_offset = 0
    ann_id_offset = 0

    for coco_dict in coco_dicts:
        if not coco_dict.get("images"):
            continue

        # Find max IDs in this dict
        old_to_new_img = {}
        for img in coco_dict["images"]:
            new_id = img["id"] + img_id_offset
            old_to_new_img[img["id"]] = new_id
            new_img = {**img, "id": new_id}
            merged["images"].append(new_img)

        for ann in coco_dict.get("annotations", []):
            new_ann = {
                **ann,
                "id": ann["id"] + ann_id_offset,
                "image_id": old_to_new_img[ann["image_id"]],
            }
            merged["annotations"].append(new_ann)

        if coco_dict["images"]:
            img_id_offset = max(img["id"] for img in merged["images"]) + 1
        if coco_dict.get("annotations"):
            ann_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

    return merged


def split_by_dataset(coco_dict: dict, cfg: dict) -> dict[str, dict]:
    """Split COCO annotations by dataset/station prefix in filenames."""
    splits_cfg = cfg["splits"]
    haxr_splits_cfg = cfg.get("haxr_splits", {})

    # Build prefix -> split_name mapping
    prefix_to_split = {}
    for split_name, prefixes in splits_cfg.items():
        for prefix in prefixes:
            prefix_to_split[prefix] = split_name
    for split_name, stations in haxr_splits_cfg.items():
        for station in stations:
            prefix_to_split[station] = split_name

    splits = {name: {"images": [], "annotations": [], "categories": coco_dict["categories"]}
              for name in ["train", "val", "test"]}

    # Map images to splits
    img_id_to_split = {}
    for img in coco_dict["images"]:
        fname = img["file_name"]
        prefix = fname.split("_")[0]
        split_name = prefix_to_split.get(prefix, "train")  # default to train
        splits[split_name]["images"].append(img)
        img_id_to_split[img["id"]] = split_name

    # Map annotations to splits
    for ann in coco_dict["annotations"]:
        split_name = img_id_to_split.get(ann["image_id"])
        if split_name:
            splits[split_name]["annotations"].append(ann)

    return splits


# ── Step 4: Tiling ───────────────────────────────────────────────────────

def tile_split(
    split_ann: dict,
    images_dir: Path,
    tiles_dir: Path,
    tile_size: int = 640,
    overlap: float = 0.2,
    min_visibility: float = 0.3,
) -> dict:
    """Tile images in a split and clip annotations to tiles."""
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Build image_id -> annotations
    img_id_to_anns = {}
    for ann in split_ann.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    tiled_ann = {
        "images": [],
        "annotations": [],
        "categories": split_ann["categories"],
    }
    tile_id = 0
    ann_id = 0

    for img_info in tqdm(split_ann["images"], desc="Tiling"):
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Get GT boxes in xyxy format
        anns = img_id_to_anns.get(img_info["id"], [])
        gt_boxes = np.array([a["bbox"] for a in anns], dtype=np.float32) if anns else np.zeros((0, 4))
        if len(gt_boxes) > 0:
            gt_xyxy = gt_boxes.copy()
            gt_xyxy[:, 2] += gt_xyxy[:, 0]
            gt_xyxy[:, 3] += gt_xyxy[:, 1]
        else:
            gt_xyxy = np.zeros((0, 4), dtype=np.float32)

        tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
        frame_id = Path(img_info["file_name"]).stem

        for i, t in enumerate(tiles):
            tile_name = f"{frame_id}_tile{i:04d}.png"
            cv2.imwrite(str(tiles_dir / tile_name), t["tile"])

            tile_boxes = clip_boxes_to_tile(
                gt_xyxy, t["x_offset"], t["y_offset"], tile_size, min_visibility
            )

            tiled_ann["images"].append({
                "id": tile_id,
                "file_name": tile_name,
                "width": tile_size,
                "height": tile_size,
                "source_frame": img_info["file_name"],
                "x_offset": t["x_offset"],
                "y_offset": t["y_offset"],
            })

            for box in tile_boxes:
                x1, y1, x2, y2 = box
                tiled_ann["annotations"].append({
                    "id": ann_id,
                    "image_id": tile_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })
                ann_id += 1
            tile_id += 1

    return tiled_ann


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tile_cfg = cfg["tiling"]
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1 & 2: Build COCO annotations from raw data ──
    coco_dicts = []

    if not args.skip_dlr:
        dlr_coco = process_dlr(raw_dir, output_dir, cfg, args.apply_clahe)
        coco_dicts.append(dlr_coco)
        print(f"DLR: {len(dlr_coco['images'])} images, {len(dlr_coco['annotations'])} annotations")

    if not args.skip_haxr:
        haxr_coco = process_haxr(raw_dir, output_dir, cfg)
        coco_dicts.append(haxr_coco)
        print(f"HAXR: {len(haxr_coco['images'])} images, {len(haxr_coco['annotations'])} annotations")

    if not coco_dicts:
        print("No datasets processed")
        return

    # ── Step 3: Merge and save full-image annotations ──
    merged = merge_coco_dicts(*coco_dicts)
    save_split(merged, ann_dir / "all.json")
    print(f"\nMerged: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # ── Step 4: Split by dataset ──
    splits = split_by_dataset(merged, cfg)
    for name, split_ann in splits.items():
        save_split(split_ann, ann_dir / f"{name}.json")
    print_split_stats(splits["train"], splits["val"], splits["test"])

    # ── Step 5: Tile each split ──
    if not args.skip_tiling:
        images_dir = output_dir / "images"
        tiles_dir = output_dir / "tiles"

        tiled_splits = {}
        for split_name, split_ann in splits.items():
            print(f"\nTiling {split_name} split...")
            split_tiles_dir = tiles_dir / split_name
            tiled = tile_split(
                split_ann, images_dir, split_tiles_dir,
                tile_size=tile_cfg["tile_size"],
                overlap=tile_cfg["overlap"],
                min_visibility=tile_cfg["min_visibility"],
            )
            save_split(tiled, ann_dir / f"{split_name}_tiled.json")
            tiled_splits[split_name] = tiled
            n_with_anns = sum(1 for img in tiled["images"]
                              if any(a["image_id"] == img["id"] for a in tiled["annotations"]))
            print(f"  {split_name}: {len(tiled['images'])} tiles, "
                  f"{len(tiled['annotations'])} annotations, "
                  f"{n_with_anns} tiles with objects")

        # ── Step 6: YOLO format export ──
        if not args.skip_yolo:
            print("\nExporting YOLO format...")
            yolo_dir = output_dir / "yolo"
            for split_name, tiled in tiled_splits.items():
                coco_to_yolo(
                    tiled,
                    tiles_dir / split_name,
                    yolo_dir,
                    split_name=split_name,
                    symlink=True,
                )
            write_yolo_data_yaml(yolo_dir)
            print(f"  YOLO data exported to {yolo_dir}")

        # ── Step 7: U-Net mask generation ──
        if not args.skip_masks:
            print("\nGenerating U-Net segmentation masks...")
            for split_name in ["train", "val"]:
                tiled = tiled_splits[split_name]
                mask_dir = output_dir / "masks" / split_name
                generate_segmentation_masks(
                    tiled,
                    tiles_dir / split_name,
                    mask_dir,
                    tile_size=tile_cfg["tile_size"],
                )
            print(f"  Masks saved to {output_dir / 'masks'}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
