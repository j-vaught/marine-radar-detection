"""Temporal splitting for train/val/test to prevent data leakage."""

import json
from pathlib import Path

import numpy as np


def temporal_split(
    annotations: dict,
    train_sessions: list[str],
    val_sessions: list[str],
    test_sessions: list[str],
) -> tuple[dict, dict, dict]:
    """Split COCO annotations by recording session to prevent temporal leakage.

    Args:
        annotations: Full COCO annotation dict.
        train_sessions: Session IDs for training.
        val_sessions: Session IDs for validation.
        test_sessions: Session IDs for testing.

    Returns:
        Tuple of (train_ann, val_ann, test_ann) COCO annotation dicts.
    """
    def filter_by_sessions(ann_dict, sessions):
        session_set = set(sessions)
        # Assumes image file_name contains session identifier
        image_ids = set()
        images = []
        for img in ann_dict["images"]:
            session = _extract_session(img["file_name"])
            if session in session_set:
                images.append(img)
                image_ids.add(img["id"])

        anns = [a for a in ann_dict["annotations"] if a["image_id"] in image_ids]

        return {
            "images": images,
            "annotations": anns,
            "categories": ann_dict["categories"],
        }

    train = filter_by_sessions(annotations, train_sessions)
    val = filter_by_sessions(annotations, val_sessions)
    test = filter_by_sessions(annotations, test_sessions)

    return train, val, test


def _extract_session(file_name: str) -> str:
    """Extract session identifier from filename.

    Override this based on the actual DLR DAAN naming convention.
    """
    # Default: use directory or prefix before first underscore
    return file_name.split("_")[0]


def save_split(ann_dict: dict, output_path: str | Path):
    """Save a COCO annotation split to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ann_dict, f)


def print_split_stats(train: dict, val: dict, test: dict):
    """Print dataset split statistics."""
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_images = len(split["images"])
        n_anns = len(split["annotations"])
        print(f"{name}: {n_images} images, {n_anns} annotations")
