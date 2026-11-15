"""Frozen, frame-index-based partitions shared by all revision experiments."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from pathlib import Path


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    """Seed available random number generators before model construction."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _safe_image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe image path: {name}")
    return path


def build_manifest(ann_path, image_dir):
    """Build partitions with verified image hashes and explicit index provenance."""
    ann_path, image_dir = Path(ann_path), Path(image_dir)
    data = json.loads(ann_path.read_text())
    images = data["images"]
    ids = [im["id"] for im in images]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs")
    if len({im["file_name"] for im in images}) != len(images):
        raise ValueError("Duplicate image filenames")
    grouped = {}
    records = []
    for im in images:
        site, frame = im["dataset"], im["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError(f"Non-integer frame index for image {im['id']}")
        grouped.setdefault(site, []).append(im)
        records.append(
            {
                "id": im["id"],
                "site": site,
                "frame": frame,
                "file_name": im["file_name"],
                "sha256": sha256_file(_safe_image_path(image_dir, im["file_name"])),
            }
        )
    sites = {}
    for site, group in sorted(grouped.items()):
        group.sort(key=lambda im: im["frame"])
        if len({im["frame"] for im in group}) != len(group):
            raise ValueError(f"Duplicate frame index in {site}")
        n = len(group)
        boundaries = [0, *(math.floor(p * n) for p in (0.15, 0.20, 0.80, 0.85)), n]
        names = ("validation", "validation_buffer", "train", "test_buffer", "test")
        parts = {
            name: [im["id"] for im in group[a:b]]
            for name, a, b in zip(names, boundaries, boundaries[1:])
        }
        adjacent = {}
        for seed in range(5):
            # SHA-256 ordering is independent of interpreter RNG implementation.
            ranked = sorted(
                parts["train"],
                key=lambda image_id: (
                    hashlib.sha256(f"{site}:{seed}:{image_id}".encode()).hexdigest(),
                    image_id,
                ),
            )
            removed = set(ranked[: len(parts["test_buffer"])])
            if len(removed) != len(parts["test_buffer"]):
                raise ValueError(f"Insufficient train frames for adjacency in {site}")
            adjacent[str(seed)] = [i for i in parts["train"] if i not in removed] + parts[
                "test_buffer"
            ]
        frames = {im["id"]: im["frame"] for im in group}

        def distances(train_ids, reference_ids, frames=frames):
            reference = sorted(frames[i] for i in reference_ids)
            if not reference:
                return {str(i): None for i in train_ids}
            result = {}
            for i in train_ids:
                frame = frames[i]
                pos = bisect.bisect_left(reference, frame)
                neighbors = reference[max(0, pos - 1) : pos + 1]
                result[str(i)] = min(abs(frame - other) for other in neighbors)
            return result

        distance_audit = {}
        for condition, train_ids in [
            ("buffered", parts["train"]),
            *[(f"adjacency_{seed}", ids) for seed, ids in adjacent.items()],
        ]:
            distance_audit[condition] = {
                target: distances(train_ids, parts[target]) for target in ("validation", "test")
            }
        sites[site] = {
            "count": n,
            "boundaries": boundaries,
            "partitions": parts,
            "counts": {k: len(v) for k, v in parts.items()},
            "adjacency": adjacent,
            "frame_index_distances": distance_audit,
        }
    all_ids = set(ids)
    if any(a["image_id"] not in all_ids for a in data["annotations"]):
        raise ValueError("Annotation references an unknown image")
    return {
        "schema_version": 1,
        "annotation_sha256": sha256_file(ann_path),
        "image_count": len(images),
        "annotation_count": len(data["annotations"]),
        "provenance": {
            "ordering": "numeric frame field within dataset",
            "meaning": "frame-index blocking; chronology and persistent tracks unverified",
            "adjacency_selection": "SHA-256 of site:seed:image_id, ascending",
        },
        "images": sorted(records, key=lambda x: (x["site"], x["frame"])),
        "sites": sites,
    }


def _subset(data, ids):
    ids = set(ids)
    return {
        **data,
        "images": [im for im in data["images"] if im["id"] in ids],
        "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
    }


def prepare_job(job):
    """Load the frozen manifest and return full and source-specific COCO datasets."""
    ann_path, image_dir = Path(job["ann"]), Path(job["images"])
    manifest = json.loads(Path(job["manifest"]).read_text())
    if sha256_file(ann_path) != manifest["annotation_sha256"]:
        raise ValueError("Annotation hash does not match manifest")
    data = json.loads(ann_path.read_text())
    site = manifest["sites"][job["source"]]
    condition = job["condition"]
    if condition == "buffered":
        train_ids = site["partitions"]["train"]
    elif condition == "adjacency":
        train_ids = site["adjacency"][str(job["seed"])]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    limit = job.get("train_limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("train_limit must be a positive integer")
        train_ids = train_ids[:limit]
    overfit = job.get("overfit", False)
    if overfit and job["stage"] != "diagnostic":
        raise ValueError("Overfit is restricted to diagnostics")
    train = _subset(data, train_ids)
    validation = _subset(data, train_ids if overfit else site["partitions"]["validation"])
    targets = job.get("eval_targets", [])
    if job["stage"] == "diagnostic" and targets:
        raise ValueError("Diagnostics cannot access test holdouts")
    tests = {
        target: _subset(data, manifest["sites"][target]["partitions"]["test"]) for target in targets
    }
    return {
        "annotations": data,
        "train": train,
        "validation": validation,
        "tests": tests,
        "image_dir": image_dir,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.ann, args.images)
    save_json(args.output, manifest)
    print(
        f"Wrote {args.output}: {manifest['image_count']} images, "
        f"{manifest['annotation_count']} annotations, sha256={sha256_file(args.output)}"
    )
    for site, info in manifest["sites"].items():
        print(f"{site}: {info['counts']}")


if __name__ == "__main__":
    main()
