"""Stratify one-to-one matches from the completed DARC factorial archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from bisect import bisect_right
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import darc_pixel_audit as pixel
from . import protocol
from .darc_summary import _artifact, _preflight, _release_references
from .finalize import _serialized_digest

SPEC_SHA256 = "231c49a946756ba4307ee25be496501967b5b43afe8cd537484a9a1b9203510e"
AREA = "native_box_area_px2"
CONTRAST = "rendered_contrast_dn"
CONTRAST_KEY = "box_p90_minus_annulus_p90"
METRICS = ("iou_0.5", "center_8px")


def read_json(path: Path):
    return json.loads(path.read_bytes())


def checked_hash(path: Path, expected: str) -> str:
    digest = protocol.sha256_file(path)
    if digest != expected:
        raise ValueError(f"Frozen input digest differs: {path}")
    return digest


def load_spec(path: Path) -> dict:
    checked_hash(path, SPEC_SHA256)
    return read_json(path)


def bin_value(value: float | None, family: str, spec: dict) -> str:
    if value is None and family == CONTRAST:
        return "missing"
    if value is None or not math.isfinite(value):
        raise ValueError(f"Nonfinite {family}: {value}")
    if (family == AREA and value <= 0) or (family == CONTRAST and not -255 <= value <= 255):
        raise ValueError(f"Out-of-domain {family}: {value}")
    bins = spec["bins"][family]
    return bins["names"][bisect_right(bins["edges"], value)]


def audit_contrast_summary(
    site: str, values: list[float], archived: dict, csv_stats: dict, n_labels: int
) -> None:
    actual = pixel.summarize(values)
    expected = archived["per_box"][CONTRAST_KEY]
    if n_labels != archived["counts"]["annotations"] or actual != expected:
        raise ValueError(
            f"{site}: reconstructed object contrast differs from the archived PNG audit"
        )
    csv_count, csv_values = csv_stats[site]
    if csv_count != actual["n"] or csv_values != {"mean": actual["mean"], **actual["quantiles"]}:
        raise ValueError(f"{site}: recovered contrast differs from archived CSV")


def make_features(args, spec: dict) -> dict:
    """Recover per-label data using the original audit code and untouched images."""
    audited = spec["pixel_audit"]
    checked_hash(args.pixel_audit_json, audited["json_sha256"])
    checked_hash(args.pixel_audit_csv, audited["csv_sha256"])
    archive = read_json(args.pixel_audit_json)
    split_sha = checked_hash(args.manifest, archive["manifest_sha256"])
    ann_sha = checked_hash(args.annotations, archive["annotation_sha256"])
    checked_hash(Path(pixel.__file__), archive["implementation_sha256"])
    split, coco = map(read_json, (args.manifest, args.annotations))
    if split["annotation_sha256"] != ann_sha:
        raise ValueError("Frozen manifest and label bytes disagree")

    targets = {target for _, target in spec["directions"]}
    by_id = {image["id"]: image for image in coco["images"]}
    archived_groups = {
        group["site"]: group
        for group in archive["groups"]
        if group["site"] in targets and group["split"] == "test"
    }
    csv_stats = {
        site: (group["per_box"][CONTRAST_KEY]["n"], {}) for site, group in archived_groups.items()
    }
    with args.pixel_audit_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row["site"] in targets
                and row["split"] == "test"
                and row["section"] == "per_box"
                and row["metric"] == CONTRAST_KEY
            ):
                n, stats = csv_stats[row["site"]]
                if int(row["n"]) != n or row["statistic"] in stats:
                    raise ValueError("Invalid or duplicate statistic in archived pixel-audit CSV")
                stats[row["statistic"]] = float(row["value"])

    rows, verified = {}, {}
    for site in sorted(targets):
        wanted = set(split["sites"][site]["partitions"]["test"])
        site_refs = protocol._subset(coco, wanted)
        annotations = defaultdict(list)
        for annotation in site_refs["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        present, values, results = set(), [], []
        for record in split["images"]:
            image_id = record["id"]
            if image_id not in wanted:
                continue
            info = by_id[image_id]
            if (
                record["site"] != site
                or info["dataset"] != site
                or record["file_name"] != info["file_name"]
                or record["frame"] != info["frame"]
            ):
                raise ValueError("Frozen frame identity differs between COCO and manifest")
            path = protocol._safe_image_path(args.images, info["file_name"])
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"PNG digest differs: {path}")
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if (
                image is None
                or image.dtype != np.uint8
                or image.shape != (info["height"], info["width"])
            ):
                raise ValueError(f"Unexpected native raster: {path}")
            labels = annotations[image_id]
            samples = pixel.box_samples(image, pixel.disc_mask(image.shape), labels)
            for annotation, sample in zip(labels, samples, strict=True):
                contrast = sample.get(CONTRAST_KEY)
                if contrast is not None:
                    values.append(contrast)
                results.append(
                    {
                        "annotation_id": annotation["id"],
                        "image_id": image_id,
                        AREA: math.prod(annotation["bbox"][2:]),
                        CONTRAST: contrast,
                        "annulus_pixels": sample.get("annulus_pixels", 0),
                    }
                )
            present.add(image_id)
        if present != wanted or len(present) != archived_groups[site]["counts"]["frames"]:
            raise ValueError(f"{site}: incomplete archived target-frame match")
        audit_contrast_summary(site, values, archived_groups[site], csv_stats, len(results))
        rows[site], verified[site] = results, len(present)
    return {
        "schema": "darc-processed-strata-features-1",
        "spec_sha256": SPEC_SHA256,
        "manifest_sha256": split_sha,
        "annotation_sha256": ann_sha,
        "pixel_audit_json_sha256": audited["json_sha256"],
        "pixel_audit_csv_sha256": audited["csv_sha256"],
        "original_pixel_audit_implementation_sha256": archive["implementation_sha256"],
        "contrast_summary_reproduced_exactly": True,
        "verified_reference_frames": verified,
        "runtime": runtime_identity(),
        "references": rows,
    }


def runtime_identity() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "opencv": cv2.__version__,
        "script_sha256": protocol.sha256_file(__file__),
    }


def intersection_over_union(refs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ref_min, det_min = refs[:, None, :2], boxes[None, :, :2]
    ref_max = ref_min + refs[:, None, 2:]
    det_max = det_min + boxes[None, :, 2:]
    lengths = np.maximum(0, np.minimum(ref_max, det_max) - np.maximum(ref_min, det_min))
    intersection = lengths.prod(axis=2)
    return intersection / (
        refs[:, 2:].prod(axis=1)[:, None] + boxes[:, 2:].prod(axis=1) - intersection
    )


def matched_row_ids(ids: list[int], valid: np.ndarray, cost: np.ndarray) -> set[int]:
    # The penalty outweighs the combined costs of all possible feasible edges.
    penalty = len(ids) + valid.shape[1] + 1
    rows, columns = linear_sum_assignment(np.where(valid, cost, penalty))
    return {ids[i] for i, j in zip(rows, columns, strict=True) if valid[i, j]}


def match_predictions(
    references: dict, predictions: list[dict], scoring: dict
) -> dict[str, set[int]]:
    grouped_refs, grouped_preds = defaultdict(list), defaultdict(list)
    images = {image["id"]: image for image in references["images"]}
    for annotation in references["annotations"]:
        if annotation.get("iscrowd", 0) or annotation["category_id"] != 1:
            raise ValueError("Reference contains unsupported category or crowd annotations")
        grouped_refs[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        image_id = prediction["image_id"]
        box, score = prediction["bbox"], prediction["score"]
        if image_id not in images or prediction["category_id"] != 1:
            raise ValueError("Stored prediction refers to different target IDs or category")
        x, y, w, h = box
        if (
            not all(math.isfinite(v) for v in [*box, score])
            or not 0 <= score <= 1
            or not (0 <= x < x + w <= images[image_id]["width"])
            or not (0 <= y < y + h <= images[image_id]["height"])
        ):
            raise ValueError("Stored prediction has invalid native coordinates or score")
        grouped_preds[image_id].append(prediction)

    matched = {metric: set() for metric in METRICS}
    radius = scoring["center_distance_maximum_px"]
    for image_id, image_refs in grouped_refs.items():
        ranked = sorted(grouped_preds[image_id], key=lambda p: -p["score"])
        top = ranked[: scoring["top_predictions_per_image"]]
        if not top:
            continue
        image_refs.sort(key=lambda a: a["id"])
        ids = [a["id"] for a in image_refs]
        refs = np.asarray([a["bbox"] for a in image_refs], dtype=float)
        boxes = np.asarray([p["bbox"] for p in top], dtype=float)
        iou = intersection_over_union(refs, boxes)
        ref_centers = refs[:, :2] + refs[:, 2:] / 2
        det_centers = boxes[:, :2] + boxes[:, 2:] / 2
        distances = np.linalg.norm(ref_centers[:, None] - det_centers[None, :], axis=2)
        matched["iou_0.5"].update(matched_row_ids(ids, iou >= scoring["iou_minimum"], 1 - iou))
        matched["center_8px"].update(matched_row_ids(ids, distances <= radius, distances / radius))
    return matched


def reference_strata(reference: dict, features: list[dict], spec: dict) -> dict:
    annotations = {row["id"]: row for row in reference["annotations"]}
    rows = {row["annotation_id"]: row for row in features}
    if set(rows) != set(annotations) or len(rows) != len(features):
        raise ValueError("Per-label pixel features differ from the fixed reference inventory")
    groups = {("all", "all"): set(rows)}
    for family in (AREA, CONTRAST):
        names = spec["bins"][family]["names"] + (["missing"] if family == CONTRAST else [])
        groups.update({(family, name): set() for name in names})
    for annotation_id, row in rows.items():
        ann = annotations[annotation_id]
        if row["image_id"] != ann["image_id"] or row[AREA] != math.prod(ann["bbox"][2:]):
            raise ValueError("Per-label geometry differs from released reference")
        for family in (AREA, CONTRAST):
            groups[family, bin_value(row[family], family, spec)].add(annotation_id)
    return groups


def seed_rows(matched: dict, strata: dict, identity: dict) -> list[dict]:
    return [
        {
            **identity,
            "criterion": metric,
            "family": family,
            "bin": name,
            "reference_count": len(ids),
            "matched_count": len(matches & ids),
            "recall": len(matches & ids) / len(ids) if ids else None,
        }
        for metric, matches in matched.items()
        for (family, name), ids in strata.items()
    ]


def paired_rows(rows: list[dict], spec: dict) -> list[dict]:
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ("source", "target", "criterion", "family", "bin"))
        cell = (row["seed"], row["arm"])
        if cell in cells[key]:
            raise ValueError(f"Duplicate paired strata cell: {key} {cell}")
        cells[key][cell] = row
    result = []
    expected = {(seed, arm) for seed in spec["seeds"] for arm in spec["arms"]}
    for key, cells_for_stratum in cells.items():
        if set(cells_for_stratum) != expected:
            raise ValueError(f"Incomplete paired strata cell: {key}")
        for name, (plus, minus) in spec["paired_differences"].items():
            pairs = [
                (cells_for_stratum[seed, plus], cells_for_stratum[seed, minus])
                for seed in spec["seeds"]
            ]
            supports = {arm["reference_count"] for pair in pairs for arm in pair}
            if len(supports) != 1:
                raise ValueError("Paired arms or seeds differ in reference support")
            support = supports.pop()
            effects = (
                [float(high["recall"] - low["recall"]) for high, low in pairs] if support else []
            )
            result.append(
                {
                    **dict(
                        zip(("source", "target", "criterion", "family", "bin"), key, strict=True)
                    ),
                    "contrast": name,
                    "reference_count": support,
                    **{
                        f"seed{seed}": effects[index] if support else None
                        for index, seed in enumerate(spec["seeds"])
                    },
                    "effect_mean": fmean(effects) if support else None,
                    "effect_sample_sd": stdev(effects) if support else None,
                }
            )
    return result


def verified_predictions(
    receipt, target: str, reference: dict, ledger: Path
) -> tuple[list[dict], str]:
    metrics = json.loads(receipt.metrics_bytes)
    record = metrics["tests"][target]
    if (
        metrics["job_id"] != receipt.job["job_id"]
        or record["checkpoint_sha256"] != metrics["best_weights_sha256"]
        or record["gt_sha256"] != _serialized_digest(reference)
    ):
        raise ValueError(
            "Archived prediction does not bind the selected checkpoint and target labels"
        )
    _artifact(receipt, record["gt_path"], record["gt_sha256"])
    path = _artifact(receipt, record["raw_predictions_path"], record["raw_predictions_sha256"])
    digest = record["raw_predictions_sha256"]
    proof = read_json(ledger / receipt.job["job_id"] / f"{receipt.completion['attempt']}.json")
    inventory = proof["files"]
    stored = inventory[path.relative_to(receipt.folder).as_posix()]
    if not proof["verified"] or stored["sha256"] != digest or stored["size"] != path.stat().st_size:
        raise ValueError("Verified archive inventory differs from raw target predictions")
    return read_json(path), digest


def score_archive(args, spec: dict) -> dict:
    ledger = args.runs / ".ledger"
    if not ledger.is_dir():
        raise ValueError("The verified 42-fit archive ledger is required")
    protocol_sha = checked_hash(args.protocol, spec["protocol_sha256"])
    study, jobs, receipts = _preflight(args.runs, args.state, args.jobs, args.gate, args.protocol)
    refs = _release_references(study, jobs, args.manifest, args.annotations)
    features = read_json(args.features)
    if (
        features["schema"] != "darc-processed-strata-features-1"
        or features["spec_sha256"] != SPEC_SHA256
        or features["manifest_sha256"] != jobs["split_sha256"]
        or features["annotation_sha256"] != jobs["annotations_sha256"]
        or features["pixel_audit_json_sha256"] != spec["pixel_audit"]["json_sha256"]
        or features["pixel_audit_csv_sha256"] != spec["pixel_audit"]["csv_sha256"]
        or features["contrast_summary_reproduced_exactly"] is not True
    ):
        raise ValueError("Per-object features are not bound to the frozen strata and image audit")

    rows, proofs, support = [], [], {}
    for source, target in spec["directions"]:
        reference = refs[target]["test"]
        values = features["references"][target]
        strata = reference_strata(reference, values, spec)
        support[target] = {
            "frames": len(reference["images"]),
            "negative_contrast_in_lt64": sum(
                row[CONTRAST] is not None and row[CONTRAST] < 0 for row in values
            ),
            "bins": [
                {"family": family, "bin": name, "references": len(ids)}
                for (family, name), ids in strata.items()
            ],
        }
        if support[target]["frames"] != features["verified_reference_frames"][target]:
            raise ValueError("Contrast features lack verified coverage of the whole target holdout")
        for seed in spec["seeds"]:
            for arm in spec["arms"]:
                job_id = f"darc-v2-fit-{source}-s{seed}-{arm}"
                fit = receipts[job_id]
                predictions, digest = verified_predictions(fit, target, reference, ledger)
                matched = match_predictions(reference, predictions, spec["scoring"])
                rows.extend(
                    seed_rows(
                        matched,
                        strata,
                        {"source": source, "target": target, "seed": seed, "arm": arm},
                    )
                )
                proofs.append(
                    {
                        "job_id": job_id,
                        "target": target,
                        "raw_predictions_sha256": digest,
                        "metrics_sha256": fit.completion["metrics_sha256"],
                    }
                )
    return {
        "schema": "darc-processed-strata-results-1",
        "scope": spec["interpretation"],
        "spec_sha256": SPEC_SHA256,
        "protocol_sha256": protocol_sha,
        "jobs_sha256": protocol.sha256_file(args.jobs),
        "scoring_gate_sha256": protocol.sha256_file(args.gate),
        "features_sha256": protocol.sha256_file(args.features),
        "completed_verified_fits": len(receipts),
        "new_fits": 0,
        "runtime": runtime_identity(),
        "target_support": support,
        "reused_prediction_proofs": proofs,
        "recall_by_seed": rows,
        "paired_contrasts": paired_rows(rows, spec),
    }


def write_results(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    protocol.save_json(output / "strata-results.json", report)
    for key, name in (
        ("recall_by_seed", "recall-by-seed.csv"),
        ("paired_contrasts", "paired-contrasts.csv"),
    ):
        rows = report[key]
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def points(value):
        return "NA" if value is None else f"{100 * value:+.2f}"

    lines = [
        "DARC processed-image error stratification",
        "",
        "One-to-one matched-object recall at 30 raw proposals per image on fixed released labels.",
        "Match across the full frame before grouping reference boxes. Positive differences favor the intervention.",
        "The paired columns are percentage points. NA denotes zero reference support.",
        "Negative contrast remains in lt64. Contrast is measured in rendered digital levels.",
        "",
        "Source -> target  criterion   family/bin                            n  contrast             s0      s1      s2    mean      SD",
    ]
    for row in report["paired_contrasts"]:
        name = f"{row['family']}/{row['bin']}"
        columns = [
            points(row[key])
            for key in ("seed0", "seed1", "seed2", "effect_mean", "effect_sample_sd")
        ]
        lines.append(
            f"{row['source']:>4} -> {row['target']:<4}  {row['criterion']:<10}  {name:<33} "
            f"{row['reference_count']:>5}  {row['contrast']:<19} {' '.join(f'{v:>7}' for v in columns)}"
        )
    (output / "strata-report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    features = modes.add_parser("features")
    score = modes.add_parser("score")
    for mode in (features, score):
        for name in ("spec", "manifest", "annotations", "output"):
            mode.add_argument("--" + name, type=Path, required=True)
    for name in ("images", "pixel-audit-json", "pixel-audit-csv"):
        features.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "jobs", "gate", "state", "runs", "features"):
        score.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.mode == "features":
        cv2.setNumThreads(1)
        protocol.save_json(args.output, make_features(args, spec))
    else:
        write_results(score_archive(args, spec), args.output)


if __name__ == "__main__":
    main()
