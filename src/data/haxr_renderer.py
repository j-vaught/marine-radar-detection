"""Render HAXR (Hamburg X-band Radar) HDF5 files into PPI images.

Generates Cartesian PPI images from raw pulse data and produces COCO-format
annotations from AIS ground truth. Does not depend on the haxr library.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-file helpers
# ---------------------------------------------------------------------------


def get_max_range(hdf5_path: str | Path) -> float:
    """Return the maximum range value (meters) from the r2 dataset.

    Args:
        hdf5_path: Path to an HAXR HDF5 file.

    Returns:
        Maximum range in meters.
    """
    with h5py.File(hdf5_path, "r") as f:
        max_range = float(f["r2"][:].max())
    return max_range


def render_ppi(
    hdf5_path: str | Path,
    cycle_idx: int,
    img_size: int = 1024,
    max_range: float | None = None,
) -> np.ndarray:
    """Render a single radar cycle into a Cartesian PPI image.

    The pulse cloud is first binned into a polar histogram (azimuth x range),
    then mapped pixel-by-pixel to a square Cartesian image centred on the
    radar.  Amplitude values are accumulated in each bin and the result is
    normalised to uint8.

    Azimuth convention: 0 deg is North (up), increasing clockwise, matching
    the standard radar PPI convention.  The HDF5 azimuth values are already
    in this convention.

    Args:
        hdf5_path: Path to an HAXR HDF5 file.
        cycle_idx: Zero-based cycle index within the file.
        img_size: Side length (pixels) of the output square image.
        max_range: Maximum range in meters.  Inferred from r2 if None.

    Returns:
        Grayscale PPI image as a (img_size, img_size) uint8 array.
    """
    with h5py.File(hdf5_path, "r") as f:
        first = int(f["cycle/first"][cycle_idx])
        last = int(f["cycle/last"][cycle_idx])

        if last <= first:
            # Degenerate cycle - return blank image
            return np.zeros((img_size, img_size), dtype=np.uint8)

        amp = f["amp"][first:last].astype(np.float32)
        az1 = f["az1"][first:last].astype(np.float32)
        az2 = f["az2"][first:last].astype(np.float32)
        r1 = f["r1"][first:last].astype(np.float32)
        r2 = f["r2"][first:last].astype(np.float32)

        if max_range is None:
            max_range = float(f["r2"][:].max())

    # Mid-point polar coordinates
    az_mid = (az1 + az2) / 2.0  # degrees, 0=North, clockwise
    r_mid = (r1 + r2) / 2.0     # meters

    # Discard pulses beyond max_range
    valid = r_mid <= max_range
    az_mid = az_mid[valid]
    r_mid = r_mid[valid]
    amp = amp[valid]

    # Build polar histogram: rows=az_bins, cols=range_bins
    n_az = 720
    n_r = img_size // 2

    az_bin = np.floor(az_mid / 360.0 * n_az).astype(np.int32)
    az_bin = np.clip(az_bin, 0, n_az - 1)

    r_bin = np.floor(r_mid / max_range * n_r).astype(np.int32)
    r_bin = np.clip(r_bin, 0, n_r - 1)

    # Accumulate amplitude into histogram; use float64 to avoid overflow
    polar = np.zeros((n_az, n_r), dtype=np.float64)
    np.add.at(polar, (az_bin, r_bin), amp)

    # Normalise histogram to [0, 255]
    p_max = polar.max()
    if p_max > 0:
        polar = polar / p_max * 255.0
    polar = polar.astype(np.float32)

    # Polar-to-Cartesian mapping
    # For each output pixel compute its (range, azimuth) and sample the histogram.
    half = img_size / 2.0
    pixels_per_meter = half / max_range

    # Coordinate grids in pixel space (origin at centre)
    ys, xs = np.mgrid[0:img_size, 0:img_size]
    dx = xs.astype(np.float32) - half
    dy = ys.astype(np.float32) - half  # positive-y is downward in image

    # Range in pixels from centre
    r_px = np.sqrt(dx * dx + dy * dy)

    # Azimuth: 0=North (up=negative-dy), clockwise
    # atan2(dx, -dy) gives angle from North, clockwise, in [-pi, pi]
    az_rad = np.arctan2(dx, -dy)                  # radians from North
    az_deg = np.degrees(az_rad) % 360.0           # [0, 360)

    # Map to histogram bin indices
    az_idx = (az_deg / 360.0 * n_az).astype(np.int32)
    az_idx = np.clip(az_idx, 0, n_az - 1)

    r_norm = r_px / half  # fraction of max_range
    r_idx = (r_norm * n_r).astype(np.int32)

    # Pixels outside max_range radius get zero
    outside = r_px >= half
    r_idx = np.clip(r_idx, 0, n_r - 1)

    cart = polar[az_idx, r_idx]
    cart[outside] = 0.0

    return cart.astype(np.uint8)


# ---------------------------------------------------------------------------
# AIS helpers
# ---------------------------------------------------------------------------


def get_ais_for_cycle(
    csv_path: str | Path,
    cycle_tod: float,
    time_window: float = 1.0,
) -> list[dict]:
    """Return AIS vessel records whose time of day is within time_window of cycle_tod.

    Args:
        csv_path: Path to an HAXR AIS CSV file.
        cycle_tod: Cycle time-of-day in seconds since midnight.
        time_window: Maximum allowed time difference in seconds.

    Returns:
        List of dicts with keys ``uid``, ``range_m``, ``azimuth_deg``.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Rename to short internal names regardless of header whitespace
    col_map = {
        "tod": "tod",
        "uid": "uid",
        "range (meters)": "range_m",
        "azimuth (degrees)": "azimuth_deg",
    }
    df = df.rename(columns=col_map)

    mask = np.abs(df["tod"].values - cycle_tod) <= time_window
    rows = df[mask]

    return [
        {
            "uid": str(row.uid),
            "range_m": float(row.range_m),
            "azimuth_deg": float(row.azimuth_deg),
        }
        for _, row in rows.iterrows()
    ]


def ais_to_bbox(
    range_m: float,
    azimuth_deg: float,
    img_size: int,
    max_range: float,
    box_size: int = 20,
) -> list[float]:
    """Convert a polar AIS position to a COCO [x, y, w, h] bounding box.

    The box is centred on the vessel pixel position and clipped to the image
    boundary.  Azimuth is measured clockwise from North.

    Args:
        range_m: Slant range to vessel in meters.
        azimuth_deg: Azimuth to vessel in degrees (0=North, clockwise).
        img_size: Side length of the PPI image in pixels.
        max_range: Maximum range represented in the image in meters.
        box_size: Side length of the bounding box in pixels.

    Returns:
        COCO bbox as ``[x, y, width, height]`` floats, clipped to image bounds.
    """
    half = img_size / 2.0
    pixels_per_meter = half / max_range

    az_rad = np.radians(azimuth_deg)
    # North is up (-y direction), East is right (+x direction)
    px = half + range_m * pixels_per_meter * np.sin(az_rad)
    py = half - range_m * pixels_per_meter * np.cos(az_rad)

    half_box = box_size / 2.0
    x1 = float(np.clip(px - half_box, 0, img_size - 1))
    y1 = float(np.clip(py - half_box, 0, img_size - 1))
    x2 = float(np.clip(px + half_box, 0, img_size - 1))
    y2 = float(np.clip(py + half_box, 0, img_size - 1))

    w = x2 - x1
    h = y2 - y1

    return [x1, y1, w, h]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def render_dataset(
    hdf5_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int = 10,
    img_size: int = 1024,
    box_size: int = 20,
) -> dict:
    """Process all HAXR HDF5 files and build a COCO annotation dictionary.

    Every ``sample_rate``-th cycle is rendered to a PNG image.  AIS ground
    truth is matched by time of day and converted to bounding boxes.

    Output images are named ``{station}_{hour}_{cycle:06d}.png`` and saved
    under ``output_dir/images/``.

    Args:
        hdf5_dir: Directory containing ``.hdf5`` and ``.csv`` pairs plus
            ``stations.csv``.
        output_dir: Root directory for rendered images and annotations.
        sample_rate: Render every N-th cycle (1 = all cycles).
        img_size: Side length of each output PPI image in pixels.
        box_size: Bounding-box side length in pixels for each AIS vessel.

    Returns:
        COCO-format annotation dict with keys ``info``, ``licenses``,
        ``images``, ``annotations``, and ``categories``.
    """
    hdf5_dir = Path(hdf5_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {
            "description": "HAXR Hamburg X-band Radar vessel detection dataset",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "vessel", "supercategory": "vehicle"}],
        "images": [],
        "annotations": [],
    }

    image_id = 0
    ann_id = 0

    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
    if not hdf5_files:
        logger.warning("No HDF5 files found in %s", hdf5_dir)
        return coco

    for hdf5_path in hdf5_files:
        stem = hdf5_path.stem                 # e.g. "altona_08-UTC"
        csv_path = hdf5_dir / f"{stem}.csv"

        # Parse station and hour from stem, e.g. "altona_08-UTC" -> "altona", "08"
        # Stem format is "{station}_{HH}-UTC"
        parts = stem.rsplit("_", 1)           # split on last underscore
        if len(parts) == 2:
            station = parts[0]
            hour_str = parts[1].replace("-UTC", "")
        else:
            station = stem
            hour_str = "00"

        logger.info("Processing %s", hdf5_path.name)

        try:
            file_max_range = get_max_range(hdf5_path)
        except Exception as exc:
            logger.error("Failed to read max range from %s: %s", hdf5_path.name, exc)
            continue

        with h5py.File(hdf5_path, "r") as f:
            n_cycles = int(f["cycle/first"].shape[0])
            cycle_tods = f["cycle/tod"][:]

        has_ais = csv_path.exists()
        if not has_ais:
            logger.warning("No AIS CSV found for %s; skipping annotations", stem)

        for cycle_idx in range(0, n_cycles, sample_rate):
            cycle_tod = float(cycle_tods[cycle_idx])

            # Render PPI
            try:
                ppi = render_ppi(
                    hdf5_path,
                    cycle_idx,
                    img_size=img_size,
                    max_range=file_max_range,
                )
            except Exception as exc:
                logger.warning(
                    "render_ppi failed for %s cycle %d: %s",
                    hdf5_path.name,
                    cycle_idx,
                    exc,
                )
                continue

            # Skip blank (degenerate) cycles
            if ppi.max() == 0:
                logger.debug("Skipping blank cycle %d in %s", cycle_idx, hdf5_path.name)
                continue

            # Save image
            file_name = f"{station}_{hour_str}_{cycle_idx:06d}.png"
            img_save_path = images_dir / file_name
            cv2.imwrite(str(img_save_path), ppi)

            image_id += 1
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": img_size,
                    "height": img_size,
                    "station": station,
                    "hour_utc": hour_str,
                    "cycle_idx": cycle_idx,
                    "cycle_tod": cycle_tod,
                }
            )

            # Match AIS ground truth
            if has_ais:
                try:
                    vessels = get_ais_for_cycle(csv_path, cycle_tod)
                except Exception as exc:
                    logger.warning(
                        "AIS lookup failed for %s cycle %d: %s",
                        hdf5_path.name,
                        cycle_idx,
                        exc,
                    )
                    vessels = []

                for vessel in vessels:
                    bbox = ais_to_bbox(
                        vessel["range_m"],
                        vessel["azimuth_deg"],
                        img_size=img_size,
                        max_range=file_max_range,
                        box_size=box_size,
                    )
                    # Skip degenerate boxes
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    ann_id += 1
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "uid": vessel["uid"],
                        }
                    )

    logger.info(
        "Rendered %d images with %d annotations",
        len(coco["images"]),
        len(coco["annotations"]),
    )
    return coco
