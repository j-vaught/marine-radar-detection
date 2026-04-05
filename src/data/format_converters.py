"""Format converters: COCO-to-YOLO labels and bounding-box-to-mask generation."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def coco_to_yolo(
    coco_ann: dict,
    image_dir: str | Path,
    output_dir: str | Path,
    split_name: str = "train",
    symlink: bool = True,
):
    """Convert COCO annotations to YOLO format directory structure.

    Produces:
        output_dir/
          {split_name}/
            images/    (symlinks or copies of tile images)
            labels/    (per-image .txt files with normalized cx cy w h)

    Args:
        coco_ann: COCO annotation dict.
        image_dir: Directory containing the source images.
        output_dir: Root output directory.
        split_name: Split name (train, val, test).
        symlink: If True, symlink images instead of copying.
    """
    image_dir = Path(image_dir)
    out_images = Path(output_dir) / split_name / "images"
    out_labels = Path(output_dir) / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Build image_id -> annotations mapping
    img_id_to_anns = {}
    for ann in coco_ann.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco_ann["images"]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Link/copy image
        src = image_dir / fname
        dst = out_images / fname
        if src.exists() and not dst.exists():
            if symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        # Write YOLO label file
        label_path = out_labels / (Path(fname).stem + ".txt")
        anns = img_id_to_anns.get(img_id, [])

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: x, y, w, h (top-left)
            # Convert to YOLO: class cx cy w h (all normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            cat_id = ann.get("category_id", 1) - 1  # YOLO is 0-indexed
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))


def write_yolo_data_yaml(
    output_dir: str | Path,
    class_names: list[str] | None = None,
):
    """Write the YOLO data.yaml configuration file.

    Args:
        output_dir: Root YOLO output directory.
        class_names: List of class names.
    """
    output_dir = Path(output_dir).resolve()
    if class_names is None:
        class_names = ["vessel"]

    yaml_content = (
        f"path: {output_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


def generate_segmentation_masks(
    coco_ann: dict,
    image_dir: str | Path,
    mask_dir: str | Path,
    tile_size: int = 640,
):
    """Generate binary segmentation masks from COCO bounding box annotations.

    For each image, creates a binary mask PNG where annotated bounding boxes
    are filled with 255 (foreground) and everything else is 0 (background).

    Args:
        coco_ann: COCO annotation dict.
        image_dir: Directory containing source images (to verify existence).
        mask_dir: Output directory for mask PNGs.
        tile_size: Default mask size if image dimensions not in annotation.
    """
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_dir)

    img_id_to_anns = {}
    for ann in coco_ann.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco_ann["images"]:
        fname = img_info["file_name"]
        img_w = img_info.get("width", tile_size)
        img_h = img_info.get("height", tile_size)

        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        anns = img_id_to_anns.get(img_info["id"], [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))
            mask[y1:y2, x1:x2] = 255

        cv2.imwrite(str(mask_dir / fname), mask)


def reorganize_tiles_by_split(
    coco_splits: dict[str, dict],
    source_tile_dir: str | Path,
    output_base: str | Path,
):
    """Reorganize tiles into split-specific subdirectories.

    Creates:
        output_base/tiles/{train,val,test}/
        output_base/masks/{train,val,test}/  (if masks exist)

    Args:
        coco_splits: Dict mapping split name -> COCO annotation dict.
        source_tile_dir: Directory containing all tiles.
        output_base: Base output directory.
    """
    source_tile_dir = Path(source_tile_dir)
    output_base = Path(output_base)

    for split_name, ann_dict in coco_splits.items():
        split_dir = output_base / "tiles" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for img_info in ann_dict["images"]:
            src = source_tile_dir / img_info["file_name"]
            dst = split_dir / img_info["file_name"]
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
