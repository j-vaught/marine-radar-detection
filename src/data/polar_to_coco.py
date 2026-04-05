"""Convert DLR radar dataset polar coordinate annotations to COCO-format bounding boxes.

Each dataset (DAAN, DARC, MANV) contains a ``polarMeas.json`` where keys are
frame indices (as strings) and values are flat lists structured as:

    [obj_id, heading, [range, azimuth], [range, azimuth], ...,
     obj_id, heading, [range, azimuth], ...]

This module parses those lists, clusters the boundary points with DBSCAN,
projects clusters to pixel space, and produces a COCO annotation dictionary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

# ---- image constants --------------------------------------------------------

IMG_W = 1050
IMG_H = 1024


# ---- parsing ----------------------------------------------------------------


def parse_polar_annotations(polar_data: list) -> list[dict]:
    """Parse the flat annotation list for a single frame.

    The list begins with an integer object ID and a heading value, followed by
    zero or more ``[range, azimuth]`` coordinate pairs.  When a frame contains
    multiple objects the sequence repeats: after the last coordinate pair of
    the first object the next integer ID and heading start the second object.

    Args:
        polar_data: The raw flat list from ``polarMeas.json`` for one frame.

    Returns:
        A list of dicts, each with keys:
            ``id``      -- integer object identifier
            ``heading`` -- heading in degrees (float)
            ``points``  -- list of ``[range_m, azimuth_rad]`` pairs
    """
    if not polar_data:
        return []

    objects: list[dict] = []
    current: dict | None = None
    i = 0

    while i < len(polar_data):
        element = polar_data[i]

        # An integer signals the start of a new object record.
        if isinstance(element, int):
            if current is not None:
                objects.append(current)
            obj_id = element
            heading = polar_data[i + 1] if i + 1 < len(polar_data) else None
            current = {"id": obj_id, "heading": heading, "points": []}
            i += 2
        elif isinstance(element, list):
            if current is None:
                # Malformed entry -- create a placeholder object.
                current = {"id": -1, "heading": None, "points": []}
            current["points"].append(element)
            i += 1
        else:
            # Unexpected scalar (e.g., a stray float); skip it.
            i += 1

    if current is not None:
        objects.append(current)

    return objects


# ---- coordinate conversion --------------------------------------------------


def polar_to_pixel(
    r: np.ndarray,
    theta: np.ndarray,
    img_w: int,
    img_h: int,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar (range, azimuth) coordinates to pixel coordinates.

    The radar PPI display is centered in the image.  Azimuth zero points north
    (up) and increases clockwise, matching standard radar convention.

    Args:
        r:         Range values in metres (array-like).
        theta:     Azimuth values in radians (array-like).
        img_w:     Image width in pixels.
        img_h:     Image height in pixels.
        max_range: Maximum range represented by the display radius, in metres.

    Returns:
        ``(px, py)`` -- pixel x and y coordinate arrays.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    cx = img_w / 2.0
    cy = img_h / 2.0
    radius_px = min(img_w, img_h) / 2.0 * 0.92

    r_px = (r / max_range) * radius_px

    px = cx + r_px * np.sin(theta)
    py = cy - r_px * np.cos(theta)

    return px, py


# ---- clustering -------------------------------------------------------------


def cluster_points(
    points: list,
    eps: float = 200,
    min_samples: int = 5,
) -> list[np.ndarray]:
    """Cluster boundary points using DBSCAN in Cartesian space.

    Points are projected from polar (range, azimuth) to Cartesian before
    clustering so that Euclidean distance is meaningful.

    Args:
        points:      List of ``[range_m, azimuth_rad]`` pairs.
        eps:         DBSCAN neighbourhood radius in metres.
        min_samples: DBSCAN minimum cluster size.

    Returns:
        A list of arrays, each shaped ``(N, 2)`` and containing
        ``[range_m, azimuth_rad]`` pairs belonging to one cluster.
        Noise points (label ``-1``) are excluded.
    """
    if len(points) < min_samples:
        return []

    pts = np.array(points, dtype=float)          # (N, 2): [range, azimuth]
    r = pts[:, 0]
    theta = pts[:, 1]

    x = r * np.sin(theta)
    y = r * np.cos(theta)
    coords = np.column_stack([x, y])

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_

    clusters: list[np.ndarray] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        clusters.append(pts[mask])               # (M, 2): [range, azimuth]

    return clusters


# ---- bounding box construction ----------------------------------------------


def cluster_to_bbox(
    cluster_r: np.ndarray,
    cluster_theta: np.ndarray,
    img_w: int,
    img_h: int,
    max_range: float,
    padding: int = 12,
) -> list[float]:
    """Convert a cluster of polar points to a COCO bounding box.

    Args:
        cluster_r:     Range values for the cluster in metres.
        cluster_theta: Azimuth values for the cluster in radians.
        img_w:         Image width in pixels.
        img_h:         Image height in pixels.
        max_range:     Maximum range represented by the display radius.
        padding:       Extra pixels added to each side of the tight box.

    Returns:
        ``[x, y, w, h]`` in pixel coordinates, clipped to the image bounds.
        All values are Python floats.
    """
    px, py = polar_to_pixel(cluster_r, cluster_theta, img_w, img_h, max_range)

    x_min = float(np.clip(px.min() - padding, 0, img_w - 1))
    y_min = float(np.clip(py.min() - padding, 0, img_h - 1))
    x_max = float(np.clip(px.max() + padding, 0, img_w - 1))
    y_max = float(np.clip(py.max() + padding, 0, img_h - 1))

    w = x_max - x_min
    h = y_max - y_min

    return [x_min, y_min, w, h]


# ---- dataset conversion -----------------------------------------------------


def convert_dataset(
    dataset_dir: str | Path,
    dataset_name: str,
    max_range: float = 6000,
    eps: float = 200,
    min_samples: int = 5,
    padding: int = 12,
) -> dict[str, Any]:
    """Convert one DLR radar sub-dataset to a COCO annotation dictionary.

    Args:
        dataset_dir:  Path to the dataset root (e.g., ``data/raw/DAAN/``).
                      Must contain ``polarMeas.json`` and a ``radarImages/``
                      subdirectory.
        dataset_name: Short name used to prefix image filenames in the output
                      (e.g., ``"DAAN"``).
        max_range:    Maximum range in metres for pixel conversion.
        eps:          DBSCAN neighbourhood radius in metres.
        min_samples:  DBSCAN minimum cluster size.
        padding:      Bounding box padding in pixels.

    Returns:
        A COCO-format dict with keys ``"info"``, ``"categories"``,
        ``"images"``, and ``"annotations"``.
    """
    dataset_dir = Path(dataset_dir)
    polar_path = dataset_dir / "polarMeas.json"
    images_dir = dataset_dir / "radarImages"

    with open(polar_path, "r") as fh:
        polar_data: dict[str, Any] = json.load(fh)

    # Collect available image files keyed by integer frame number.
    available_images: dict[int, str] = {}
    if images_dir.is_dir():
        for fname in os.listdir(images_dir):
            if fname.lower().endswith(".png"):
                frame_num = int(os.path.splitext(fname)[0])
                available_images[frame_num] = fname

    coco: dict[str, Any] = {
        "info": {
            "description": f"DLR radar dataset {dataset_name} -- COCO format",
            "dataset": dataset_name,
        },
        "categories": [{"id": 1, "name": "vessel", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }

    image_id = 1
    annotation_id = 1

    for raw_key in sorted(polar_data.keys(), key=lambda k: int(k)):
        frame_num = int(raw_key)
        frame_entry = polar_data[raw_key]

        # Skip frames with no annotation.
        if frame_entry is None:
            continue

        # Skip frames whose image file is absent.
        if frame_num not in available_images:
            continue

        orig_filename = available_images[frame_num]
        coco_filename = f"{dataset_name}_{orig_filename}"

        # Parse all objects in this frame, then pool all their boundary points.
        objects = parse_polar_annotations(frame_entry)
        all_points: list[list[float]] = []
        for obj in objects:
            all_points.extend(obj["points"])

        # Skip frames that do not have enough points for at least one cluster.
        if len(all_points) < min_samples:
            continue

        # Determine image dimensions from the first available image (fast path:
        # use the known constants; avoids opening every file).
        img_w = IMG_W
        img_h = IMG_H

        # Cluster the pooled points.
        clusters = cluster_points(all_points, eps=eps, min_samples=min_samples)

        if not clusters:
            continue

        # Register image record.
        coco["images"].append(
            {
                "id": image_id,
                "file_name": coco_filename,
                "width": img_w,
                "height": img_h,
                "frame": frame_num,
                "dataset": dataset_name,
            }
        )

        # Create one annotation per cluster.
        for cluster_pts in clusters:
            cluster_r = cluster_pts[:, 0]
            cluster_theta = cluster_pts[:, 1]

            bbox = cluster_to_bbox(
                cluster_r, cluster_theta, img_w, img_h, max_range, padding
            )
            area = bbox[2] * bbox[3]

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

        image_id += 1

    return coco


# ---- multi-dataset convenience ----------------------------------------------


def convert_all_datasets(
    raw_dir: str | Path,
    datasets: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convert DAAN, DARC, and MANV and merge into one COCO dictionary.

    Image and annotation IDs are reassigned to be globally unique across the
    combined output.

    Args:
        raw_dir:  Path to the directory that contains the dataset sub-folders
                  (e.g., ``data/raw/``).
        datasets: Sub-dataset names to process.  Defaults to
                  ``["DAAN", "DARC", "MANV"]``.
        **kwargs: Forwarded to :func:`convert_dataset` (``max_range``,
                  ``eps``, ``min_samples``, ``padding``).

    Returns:
        A merged COCO-format dict covering all requested sub-datasets.
    """
    if datasets is None:
        datasets = ["DAAN", "DARC", "MANV"]

    raw_dir = Path(raw_dir)

    merged: dict[str, Any] = {
        "info": {
            "description": "DLR radar datasets (DAAN + DARC + MANV) -- COCO format",
            "datasets": datasets,
        },
        "categories": [{"id": 1, "name": "vessel", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }

    global_image_id = 1
    global_annotation_id = 1

    for name in datasets:
        dataset_dir = raw_dir / name
        if not dataset_dir.is_dir():
            continue

        subset = convert_dataset(dataset_dir, name, **kwargs)

        # Build remapping from local IDs to global IDs.
        local_to_global_image: dict[int, int] = {}
        for img in subset["images"]:
            local_to_global_image[img["id"]] = global_image_id
            img["id"] = global_image_id
            merged["images"].append(img)
            global_image_id += 1

        for ann in subset["annotations"]:
            ann["id"] = global_annotation_id
            ann["image_id"] = local_to_global_image[ann["image_id"]]
            merged["annotations"].append(ann)
            global_annotation_id += 1

    return merged
