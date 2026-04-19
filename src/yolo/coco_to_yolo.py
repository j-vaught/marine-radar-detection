"""COCO → YOLO format converter.

Writes one <image_stem>.txt per image into <out>/labels/ and a data.yaml
pointing at train/val/test image directories. Self-contained — no other
src imports so each DL folder can fork freely.

Usage:
    python -m src.yolo.coco_to_yolo \\
        --train-ann data/processed/annotations/train_tiled.json \\
        --val-ann   data/processed/annotations/val_tiled.json \\
        --test-ann  data/processed/annotations/test_tiled.json \\
        --train-images data/processed/tiles/train \\
        --val-images   data/processed/tiles/val \\
        --test-images  data/processed/tiles/test \\
        --out data/processed/yolo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def extract_names(coco_path: str | Path) -> list[str]:
    with open(coco_path) as f:
        coco = json.load(f)
    categories = sorted(coco["categories"], key=lambda cat: cat["id"])
    return [category["name"] for category in categories]


def write_labels(coco_path: str | Path, image_dir: str | Path, label_dir: str | Path) -> int:
    with open(coco_path) as f:
        coco = json.load(f)

    imgs = {img["id"]: (img["width"], img["height"], Path(img["file_name"]).stem)
            for img in coco["images"]}

    by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)

    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    for img_id, (w, h, stem) in imgs.items():
        lines = []
        for ann in by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines))
    return len(imgs)


def ensure_split_image_link(image_dir: str | Path, out_root: Path, split: str) -> None:
    split_image_dir = out_root / "images" / split
    resolved_images = Path(image_dir).resolve()
    if split_image_dir.exists() or split_image_dir.is_symlink():
        if split_image_dir.is_dir() and not split_image_dir.is_symlink():
            shutil.rmtree(split_image_dir)
        else:
            split_image_dir.unlink()

    split_image_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(resolved_images.iterdir()):
        if not image_path.is_file():
            continue
        (split_image_dir / image_path.name).symlink_to(image_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build YOLO labels + data.yaml from COCO")
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--test-ann", required=True)
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    for split, ann, images in [
        ("train", args.train_ann, args.train_images),
        ("val", args.val_ann, args.val_images),
        ("test", args.test_ann, args.test_images),
    ]:
        ensure_split_image_link(images, out, split)
        label_dir = out / "labels" / split
        n = write_labels(ann, images, label_dir)
        print(f"{split}: {n} labels in {label_dir}")

    names = extract_names(args.train_ann)
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_yaml}\n"
    )
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
