"""Verify archived source-only DARC diagnostics before enabling target scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from . import archive, evaluation, fleet, protocol
from .darc_fleet import ALLOWED, ARMS, GPU_MODEL, REQUIRED_CHECKS, SOURCES
from .darc_summary import Receipt, _fit_plan, _fit_schedule, _runtime_signature, _successful_state
from .summarize import _digest_json

UPDATES, INTERVAL, BATCH_SIZE = 32, 16, 2
PRESENTATIONS = UPDATES * BATCH_SIZE
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
NATIVE_HW = [1024, 1050]
DATA_FILES = (
    "annotations.json",
    "darc_data.py",
    "manifest.json",
    "source_pixel_smoke.json",
    "source_spatial_audit.json",
    "specification.json",
    "split_manifest.json",
)
SOURCE_PREDICTIONS = {"val_raw_predictions.json", "val_predictions.json"}
THREAD_POLICY = {
    "torch_intraop_threads": 8,
    "torch_interop_threads": 1,
    "opencv_threads": 1,
    "worker_torch_intraop_threads": 1,
    "worker_opencv_threads": 1,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def finite(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class Evidence:
    """Keep the digest of every local file as it was checked."""

    def __init__(self) -> None:
        self.entries: dict[Path, str] = {}

    def file(self, path: Path, expected: str | None = None) -> Path:
        require(not path.is_symlink() and path.is_file(), f"Missing regular evidence file: {path}")
        path = path.resolve(strict=True)
        actual = protocol.sha256_file(path)
        require(expected is None or expected == actual, f"Evidence checksum mismatch: {path}")
        require(self.entries.setdefault(path, actual) == actual, f"Changed evidence file: {path}")
        return path

    def load(self, path: Path, expected: str | None = None):
        checked = self.file(path, expected)
        content = checked.read_bytes()
        require(
            hashlib.sha256(content).hexdigest() == self.entries[checked],
            f"Changed evidence file before parsing: {checked}",
        )
        return json.loads(content)

    def proof(self, paths) -> list[dict[str, str]]:
        return [
            {"path": str(path), "sha256": self.entries[path]}
            for path in sorted({Path(path).resolve() for path in paths})
        ]


def canonical_data(evidence: Evidence, root: Path, plan: dict, summary_path: Path, release: dict):
    summary = evidence.load(summary_path)
    checksums = evidence.load(root / "checksums.json")
    require(
        set(checksums) == set(DATA_FILES) and summary["checksums"] == checksums,
        "Canonical data build inventory does not match its staged summary",
    )
    require(
        checksums["manifest.json"] == release["data_manifest_sha256"]
        and checksums["annotations.json"] == plan["annotations_sha256"]
        and checksums["split_manifest.json"] == plan["split_sha256"]
        and checksums["specification.json"] == plan["protocol_sha256"],
        "Canonical data build is not bound to the training release and job inputs",
    )
    for name, digest in checksums.items():
        evidence.file(root / name, digest)
    manifest = evidence.load(root / "manifest.json")
    parts = evidence.load(root / "split_manifest.json")
    annotations = evidence.load(root / "annotations.json")
    spec = evidence.load(root / "specification.json")
    require(
        manifest["annotation_sha256"] == parts["annotation_sha256"] == plan["annotations_sha256"]
        and manifest["split_sha256"] == plan["split_sha256"] == spec["inputs"]["split_sha256"]
        and manifest["specification_sha256"] == plan["protocol_sha256"]
        and manifest["loader_sha256"] == checksums["darc_data.py"]
        and spec["inputs"]["annotations_sha256"] == plan["annotations_sha256"]
        and manifest["protocol_id"] == spec["protocol_id"] == "darc-processed-v1",
        "Canonical data build metadata differs from frozen inputs",
    )
    require(
        len(manifest["images"]) == parts["image_count"] == summary["manifest_image_count"]
        and len(annotations["images"]) == parts["image_count"]
        and len(annotations["annotations"]) == parts["annotation_count"],
        "Canonical image or annotation inventory count differs",
    )
    refs = {}
    for source in SOURCES:
        source_parts = parts["sites"][source]["partitions"]
        assigned = [image_id for ids in source_parts.values() for image_id in ids]
        require(len(assigned) == len(set(assigned)), f"Overlapping {source} partitions")
        refs[source] = {}
        for split in ("train", "validation"):
            ids = set(source_parts[split])
            selected = protocol._subset(annotations, ids)
            expected = spec["inputs"]["partitions"][source]
            require(
                {image["id"] for image in selected["images"]} == ids
                and len(selected["images"]) == len(ids) == expected[f"{split}_frames"]
                and all(
                    image["dataset"] == source and [image["height"], image["width"]] == NATIVE_HW
                    for image in selected["images"]
                ),
                f"{source} frozen {split} frames differ",
            )
            if f"{split}_boxes" in expected:
                require(
                    len(selected["annotations"]) == expected[f"{split}_boxes"],
                    f"{source} frozen {split} annotation count differs",
                )
            refs[source][split] = selected
    smoke = evidence.load(root / "source_pixel_smoke.json")
    spatial = evidence.load(root / "source_spatial_audit.json")
    require(
        smoke["passed"] is True
        and summary["pixel_smoke_passed"] is True
        and len(smoke["rows"]) == summary["pixel_smoke_row_count"] == 15,
        "Canonical source pixel smoke is incomplete",
    )
    streams = verify_canonical_spatial(spatial, smoke, refs, spec)
    return spec, refs, streams


def verify_canonical_spatial(spatial: dict, smoke: dict, refs: dict, spec: dict) -> dict:
    policy = spec["fit_policy"]
    long_count = policy["optimizer_updates"] * policy["batch_size"]
    recorded = {(row["source"], row["seed"]): row for row in spatial["runs"]}
    require(
        len(spatial["runs"]) == 9
        and set(recorded) == {(source, seed) for source in SOURCES for seed in range(3)},
        "Canonical spatial audit lacks the nine frozen source/seed streams",
    )
    streams = {}
    for source in SOURCES:
        train_ids = {image["id"] for image in refs[source]["train"]["images"]}
        for seed in range(3):
            row = recorded[source, seed]
            stream = row["stream"]
            counts = row["overall"]
            require(
                row["train_ids"] == sorted(train_ids)
                and len(stream) == counts["presentations"] == long_count
                and all(
                    idx == i and image_id in train_ids for i, (idx, image_id) in enumerate(stream)
                )
                and _digest_json(stream) == row["stream_sha256"]
                and counts["accepted_grow"] > 0
                and counts["accepted_shrink"] > 0
                and sum(
                    counts[k]
                    for k in (
                        "accepted_grow",
                        "accepted_shrink",
                        "identity_gate",
                        "identity_rejected",
                    )
                )
                == long_count
                and counts["scale_min_p05_median_p95_max"][0] < 1
                and counts["scale_min_p05_median_p95_max"][-1] > 1,
                f"{source}/s{seed}: canonical scale audit is invalid",
            )
        stream = recorded[source, 0]["stream"]
        streams[source] = stream[:PRESENTATIONS]
        rows = [row for row in smoke["rows"] if row["source"] == source]
        bridges = [row for row in rows if row.get("I0_bridge_exact") is True]
        jitters = [row for row in rows if "direction" in row]
        require(
            len(rows) == 5
            and len(bridges) == 3
            and len({r["image_id"] for r in bridges}) == 3
            and all(r["image_id"] in train_ids for r in bridges)
            and len(jitters) == 2
            and {row["direction"] for row in jitters} == {"grow", "shrink"},
            f"{source}: canonical pixel smoke is missing bridge, shrink or grow evidence",
        )
        for row in jitters:
            index = row["presentation_index"]
            require(
                type(index) is int
                and 0 <= index < len(stream)
                and stream[index] == [index, row["image_id"]]
                and row["object_count"] > 0
                and 0 <= row["inverse_max_absolute_error_px"] <= 1e-4
                and is_sha(row["I0_float32_sha256"])
                and is_sha(row["I1_float32_sha256"])
                and is_scale(row["scale"])
                and (row["scale"] > 1 if row["direction"] == "grow" else row["scale"] < 1),
                f"{source}: canonical pixel warp proof is invalid",
            )
    return streams


def is_scale(value) -> bool:
    return finite(value) and 2**-0.5 <= value < 2**0.5


def verify_thread_parity(parity: dict, refs: dict, plan: dict, loader_sha: str) -> None:
    rows = parity["rows"]
    require(
        parity["passed"] is True
        and parity["protocol_sha256"] == plan["protocol_sha256"]
        and parity["split_sha256"] == plan["split_sha256"]
        and parity["annotation_sha256"] == plan["annotations_sha256"]
        and parity["loader_sha256"] == loader_sha
        and parity["environment"]["tested_opencv_threads"] == [1, 112]
        and len(rows) == parity["comparison_count"] == 90
        and parity["max_absolute_tensor_difference"] == 0.0,
        "OpenCV thread parity is unverified or belongs to different benchmark bytes",
    )
    for source in SOURCES:
        selected = [row for row in rows if row["source"] == source]
        ids = {row["image_id"] for row in selected}
        require(
            len(ids) == 5
            and ids <= {row["id"] for row in refs[source]["train"]["images"]}
            and len(selected) == 30
            and {(row["image_id"], row["intensity"], row["geometry"]) for row in selected}
            == {
                (i, intensity, geometry)
                for i in ids
                for intensity in ("I0", "I1")
                for geometry in ("canonical", "shrink", "grow")
            }
            and all(
                row["passed"] is True
                and row["all_targets_bitwise_equal"] is True
                and row["tensor_bitwise_equal"] is True
                and row["max_absolute_tensor_difference"] == 0.0
                for row in selected
            ),
            f"{source}: incomplete or differing source-only render parity outcomes",
        )


def metadata_file(name: str) -> bool:
    path = Path(name)
    return path.suffix.lower() in {".json", ".csv", ".log"} or path.name == "command.txt"


def read_attempt(
    evidence: Evidence, state_path: Path, metadata: Path, ledger: dict, archive_root: str
) -> tuple[Receipt, dict, dict]:
    job_id = state_path.stem
    evidence.file(state_path)
    state = _successful_state(state_path.parent, job_id)
    if state is None:
        raise ValueError(f"{job_id}: missing successful current diagnostic completion")
    attempt = state["attempt"]
    require(
        type(attempt) is int
        and attempt > 0
        and state["host"] in ALLOWED
        and Path(state["out"]).is_absolute()
        and state["receipt"] == str(Path(state["out"]) / "receipt.json")
        and state["completion"] == str(Path(state["out"]) / "completion.json")
        and re.fullmatch(r"[0-9a-f]{32}", state["nonce"]) is not None,
        f"{job_id}: unexpected state identity",
    )
    identity = archive.archive_identity(job_id, str(attempt), state, "comech-2080", archive_root)
    verified = ledger["verified"].get(identity, {})
    require(
        verified.get("job") == job_id and verified.get("attempt") == str(attempt),
        f"{job_id}: no verified archive identity for current attempt",
    )
    folder = metadata / job_id / str(attempt)
    require(
        not folder.is_symlink() and folder.resolve().is_relative_to(metadata.resolve()),
        f"{job_id}: unsafe archive metadata directory",
    )
    ledger_path = metadata / ".ledger" / job_id / f"{attempt}.json"
    remote = evidence.load(ledger_path)
    require(bool(remote["verified"]), f"{job_id}: unverified central inventory")
    inventory = remote["files"]
    local_names = {
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    }
    require(
        local_names == {name for name in inventory if metadata_file(name)},
        f"{job_id}: incomplete local copy of the verified archive metadata",
    )
    for name in sorted(local_names):
        path = folder / name
        require(path.resolve().is_relative_to(folder.resolve()), f"{job_id}: invalid archive path")
        proof = inventory[name]
        require(
            is_sha(proof["sha256"]) and path.stat().st_size == proof["size"],
            f"{job_id}: invalid archived file size or hash for {name}",
        )
        evidence.file(path, proof["sha256"])
    require(
        not any(
            Path(name).name.startswith(tuple(f"{site}_" for site in SOURCES))
            or ("predictions" in Path(name).name and name not in SOURCE_PREDICTIONS)
            for name in inventory
        ),
        f"{job_id}: diagnostic archive contains unexpected target artifacts",
    )
    job, completion, receipt = [
        evidence.load(folder / name) for name in ("job.json", "completion.json", "receipt.json")
    ]
    require(
        completion == state["result"]
        and all(
            receipt.get(k) is not None and receipt[k] == completion[k]
            for k in ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id")
        )
        and type(receipt["pid"]) is int
        and receipt["pid"] > 0
        and isinstance(receipt["start_ticks"], str)
        and re.fullmatch(r"[1-9][0-9]*", receipt["start_ticks"]) is not None
        and isinstance(receipt["boot_id"], str)
        and bool(receipt["boot_id"])
        and job["attempt_nonce"] == state["nonce"]
        and job["out_dir"] == state["out"]
        and completion["exit_code"] == 0
        and bool(completion["finished"]),
        f"{job_id}: state, receipt, completion, and resolved job do not agree",
    )
    metrics_path = evidence.file(folder / "metrics.json", completion["metrics_sha256"])
    return (
        Receipt(job, completion, folder, metrics_path.read_bytes(), state["host"]),
        state,
        inventory,
    )


def validate_runtime(fit: Receipt, state: dict, runtime: dict, profile: dict, profile_sha: str):
    require(
        state["runtime_profile_sha256"] == profile_sha
        and state["gpu_uuid"] == fit.job["gpu_uuid"]
        and fit.job.get("runtime_profile_sha256", profile_sha) == profile_sha
        and fit.completion["runtime_provenance"]["cuda_visible_devices"] == state["gpu_uuid"]
        and runtime["gpu_capability"] == [8, 9]
        and runtime["torch_version"] == profile["torch"]
        and runtime["torchvision_version"] == profile["torchvision"]
        and runtime["cuda_version"] == profile["cuda_build"]
        and runtime["cudnn_version"] == profile["cudnn"]
        and runtime["torch_threads"] == profile["threads"]
        and runtime["torch_interop_threads"] == THREAD_POLICY["torch_interop_threads"]
        and runtime["thread_policy"] == profile["thread_policy"] == THREAD_POLICY
        and ".".join(runtime["python_version"].split(".")[:2]) == profile["python_minor"]
        and profile["thread_environment"]
        == fit.completion["runtime_provenance"]["thread_environment"]
        and all(
            runtime[field] == "8"
            for field in ("omp_num_threads", "mkl_num_threads", "openblas_num_threads")
        )
        and runtime["training_updates"] == UPDATES
        and runtime["validation_count"] == UPDATES // INTERVAL
        and runtime["training_rng_seed"] == fit.job["seed"]
        and finite(runtime["train_seconds"])
        and runtime["train_seconds"] > 0
        and type(runtime["peak_vram_bytes"]) is int
        and runtime["peak_vram_bytes"] > 0,
        f"{fit.job['job_id']}: runtime receipt does not attest the frozen policy",
    )
    return (fit.host, _runtime_signature(fit, runtime), runtime["opencv_version"])


def valid_box(box, width: int, height: int) -> bool:
    if len(box) != 4 or not all(finite(v) for v in box):
        return False
    x, y, w, h = box
    return 0 <= x < x + w <= width and 0 <= y < y + h <= height


def check_predictions(
    evidence: Evidence, folder: Path, val: dict, metrics: dict
) -> tuple[int, int]:
    raw = evidence.load(folder / "val_raw_predictions.json")
    filtered = evidence.load(folder / "val_predictions.json")
    input_audit = evidence.load(folder / "validation_input_audit.json")
    by_id = {image["id"]: image for image in val["images"]}
    job_id = metrics["job_id"]
    require(
        [row["image_id"] for row in input_audit] == list(by_id)
        and all(
            row["scale"] == 1
            and row["scale_level"] == "S0"
            and row["intensity"] == metrics["intensity"]
            and row["native_size_hw"] == NATIVE_HW
            and row["object_count"] == 0
            and row["presentation_index"] is None
            for row in input_audit
        ),
        f"{job_id}: source validation input audit has foreign IDs, geometry or labels",
    )
    require(
        isinstance(raw, list)
        and bool(raw)
        and all(
            pred["image_id"] in by_id
            and type(pred["image_id"]) is int
            and type(pred["category_id"]) is int
            and pred["category_id"] == 1
            and finite(pred["score"])
            and 0 <= pred["score"] <= 1
            and valid_box(
                pred["bbox"], by_id[pred["image_id"]]["width"], by_id[pred["image_id"]]["height"]
            )
            for pred in raw
        )
        and max(Counter(pred["image_id"] for pred in raw).values()) <= 300,
        f"{job_id}: source validation predictions are empty or not valid native boxes",
    )
    threshold = metrics["score_threshold"]
    selection = evidence.load(folder / "postprocessing_selection.json")
    require(
        threshold in THRESHOLDS
        and selection["selection_split"] == "source_validation"
        and selection["score_threshold"] == threshold
        and selection["metrics"] == metrics["source_validation"]
        and bool(filtered)
        and filtered == [pred for pred in raw if pred["score"] >= threshold],
        f"{job_id}: source validation threshold or operating predictions differ",
    )
    scores = {
        t: evaluation.evaluate_coco(val, [pred for pred in raw if pred["score"] >= t])
        for t in THRESHOLDS
    }
    winner = max(THRESHOLDS, key=lambda t: (scores[t]["ap50"], scores[t]["ap"], -t))
    reported = metrics["source_validation"]
    require(
        winner == threshold
        and scores[winner].keys() == reported.keys()
        and all(
            finite(reported[key]) and math.isclose(value, reported[key], rel_tol=0, abs_tol=1e-9)
            for key, value in scores[winner].items()
        ),
        f"{job_id}: source-only threshold replay differs",
    )
    return len(raw), len(filtered)


def verify_training_geometry(
    training: dict, train: dict, stream: list, job: dict, resize_hw: list[int], mask: dict
) -> dict:
    job_id = job["job_id"]
    rows, summary = training["presentations"], training["summary"]
    require(
        len(rows) == summary["presentations"] == PRESENTATIONS
        and [[row["presentation_index"], row["image_id"]] for row in rows] == stream
        and summary["native_hw"] == NATIVE_HW
        and summary["model_resize_hw"] == resize_hw,
        f"{job_id}: expected the 64 frozen source presentations with measured model geometry",
    )
    boxes = defaultdict(list)
    for annotation in train["annotations"]:
        box = annotation["bbox"]
        require(valid_box(box, NATIVE_HW[1], NATIVE_HW[0]), "Invalid released training box")
        boxes[annotation["image_id"]].append(box)
    scales, before, after, reasons = [], [], [], Counter()
    cx, cy = mask["center_xy"]
    radius2 = mask["radius_px"] ** 2
    for row in rows:
        scale, reason = row["scale"], row["scale_identity_reason"]
        attempted, rejections = row["attempted_scales"], row["scale_rejections"]
        labels = boxes[row["image_id"]]
        original = row["box_areas_before"]
        transformed = row["box_areas_after"]
        require(
            is_scale(scale)
            and row["scale_level"] == job["scale"]
            and row["intensity"] == job["intensity"]
            and row["native_size_hw"] == NATIVE_HW
            and len(labels) == row["object_count"] == len(original) == len(transformed)
            and all(is_scale(x) for x in attempted)
            and type(rejections) is int
            and 0 <= rejections <= 2,
            f"{job_id}: invalid training row identity or label count",
        )
        if scale == 1:
            allowed = (
                reason == "baseline" and not attempted and rejections == 0
                if job["scale"] == "S0"
                else (reason == "gate" and not attempted and rejections == 0)
                or (reason == "rejected" and len(attempted) == rejections == 2)
            )
        else:
            allowed = (
                job["scale"] == "S1"
                and reason is None
                and len(attempted) == rejections + 1
                and rejections < 2
                and attempted[-1] == scale
            )
        require(allowed, f"{job_id}: inconsistent scale draw decisions")
        for (x, y, w, h), area_before, area_after in zip(
            labels, original, transformed, strict=True
        ):
            expected = w * h
            require(
                finite(area_before)
                and area_before == expected
                and finite(area_after)
                and area_after > 0
                and math.isclose(area_after, expected * scale**2, rel_tol=1e-9, abs_tol=1e-8),
                f"{job_id}: sampled labels were altered beyond the recorded transform",
            )
            if scale != 1:
                require(
                    all(
                        ((a - cx) * scale) ** 2 + ((b - cy) * scale) ** 2 <= radius2 + 1e-8
                        for a in (x, x + w)
                        for b in (y, y + h)
                    ),
                    f"{job_id}: accepted zoom moved a label outside the radar disc",
                )
        scales.append(scale)
        before.extend(original)
        after.extend(transformed)
        reasons[reason] += 1

    shrink, grow = sum(scale < 1 for scale in scales), sum(scale > 1 for scale in scales)
    require(
        bool(before)
        and summary["accepted_shrink"] == shrink
        and summary["accepted_grow"] == grow
        and summary["scale_identity_reasons"]
        == {k: reasons[k] for k in ("baseline", "gate", "rejected")}
        and summary["scale_rejections"] == sum(row["scale_rejections"] for row in rows)
        and summary["intensity_fallbacks"] == sum(bool(row["intensity_fallback"]) for row in rows)
        and (shrink > 0 and grow > 0 if job["scale"] == "S1" else shrink == grow == 0),
        f"{job_id}: diagnostic did not exercise declared identity or both scale directions",
    )
    ratio = math.prod(resize_hw) / math.prod(NATIVE_HW)
    distributions = {
        "scale_summary": scales,
        "box_area_before_native_px2": before,
        "box_area_after_native_px2": after,
        "box_area_after_model_resize_px2": [area * ratio for area in after],
    }
    for key, values in distributions.items():
        recorded = summary[key]
        require(
            recorded["count"] == len(values)
            and all(
                finite(recorded[k]) and math.isclose(recorded[k], number, rel_tol=1e-9)
                for k, number in (
                    ("min", min(values)),
                    ("median", median(values)),
                    ("max", max(values)),
                )
            ),
            f"{job_id}: realized {key} disagrees with individual presentations",
        )
    return {
        "boxes_preserved": len(before),
        "accepted_shrink": shrink,
        "accepted_grow": grow,
        "realized_scale_min_max": [min(scales), max(scales)],
        "draw_sha256": _digest_json(
            [
                [
                    row[key]
                    for key in (
                        "image_id",
                        "scale",
                        "scale_identity_reason",
                        "attempted_scales",
                        "scale_rejections",
                        "box_areas_before",
                        "box_areas_after",
                    )
                ]
                for row in rows
            ]
        ),
    }


def validate_attempt(
    evidence: Evidence,
    fit: Receipt,
    state: dict,
    inventory: dict,
    planned: dict,
    inputs: dict,
    profile: dict,
    profile_sha: str,
    refs: dict,
    stream: list,
    spec: dict,
) -> tuple[dict, tuple]:
    job, folder = fit.job, fit.folder
    job_id, release = planned["job_id"], profile["training_release"]
    require(
        all(
            job[k] == value
            for k, value in planned.items()
            if k not in {"ann", "images", "manifest", "frozen_protocol", "initial_state"}
        )
        and planned["diagnostic_updates"] == UPDATES
        and planned["diagnostic_val_interval"] == INTERVAL
        and job["eval_targets"] == []
        and job["split_sha256"] == inputs["split_sha256"]
        and job["annotations_sha256"] == inputs["annotations_sha256"]
        and job["frozen_protocol_sha256"] == inputs["protocol_sha256"]
        and job["code_commit"] == fit.completion["code_commit"] == release
        and Path(job["images"]).parent == Path(state["data_path"])
        and job["ann"] == str(Path(state["data_path"]) / "annotations" / "all.json")
        and not any(job.get(k) for k in ("overfit", "train_limit", "config_overrides", "resume")),
        f"{job_id}: foreign job, protocol, source or target-scoring instruction",
    )
    metrics = json.loads(fit.metrics_bytes)
    require(
        all(
            metrics[key] == job[key]
            for key in (
                "job_id",
                "source",
                "seed",
                "stage",
                "condition",
                "intensity",
                "scale",
                "size_control",
                "initial_state_sha256",
                "initial_tensor_sha256",
            )
        )
        and metrics["protocol_id"] == spec["protocol_id"]
        and metrics["frozen_protocol_sha256"] == inputs["protocol_sha256"]
        and metrics["manifest_sha256"] == inputs["split_sha256"]
        and metrics["annotation_sha256"] == inputs["annotations_sha256"]
        and metrics["tests"] == {}
        and metrics["subset_ids_sha256"] is None,
        f"{job_id}: diagnostic metrics contain a different job, split, or target scores",
    )
    expected_runtime = str(Path(job["out_dir"]) / "runtime.json")
    expected_checkpoint = str(Path(job["out_dir"]) / "best.pt")
    checkpoint = metrics["best_weights_sha256"]
    require(
        metrics["runtime_path"] == expected_runtime
        and metrics["checkpoint"] == expected_checkpoint
        and is_sha(checkpoint)
        and inventory["best.pt"]["sha256"] == checkpoint
        and inventory["best.pt"]["size"] > 0,
        f"{job_id}: selected checkpoint is missing or archive digest differs",
    )
    runtime = evidence.load(folder / "runtime.json", metrics["runtime_sha256"])
    require(
        metrics["thread_policy"] == runtime["thread_policy"] == THREAD_POLICY,
        f"{job_id}: effective main and worker thread policies differ",
    )
    signature = validate_runtime(fit, state, runtime, profile, profile_sha)
    history = evidence.load(folder / "history.json")
    _fit_schedule(
        fit,
        metrics,
        history,
        {
            **spec["fit_policy"],
            "optimizer_updates": UPDATES,
            "updates_per_validation": INTERVAL,
            "batch_size": BATCH_SIZE,
        },
        spec["protocol_id"],
    )
    require(
        all(
            finite(row[key]) and row[key] >= 0
            for row in history
            for key in ("train_loss_last_batch", "train_loss_mean", "source_val_loss")
        ),
        f"{job_id}: nonfinite or invalid diagnostic loss trace",
    )
    resolved = evidence.load(folder / "job_resolved.json")
    resize_hw = runtime["model_resize_hw"]
    require(
        len(resize_hw) == 2
        and all(type(dim) is int and dim > 0 for dim in resize_hw)
        and resolved["job"] == job
        and resolved["protocol_id"] == spec["protocol_id"]
        and resolved["updates"] == UPDATES
        and resolved["validation_interval"] == INTERVAL
        and resolved["batch_size"] == BATCH_SIZE
        and resolved["decay_after_update"] == spec["fit_policy"]["lr_decay_after_update"]
        and resolved["model_resize_hw"] == resize_hw
        and resolved["subset_ids"] is None
        and resolved["subset_ids_sha256"] is None
        and resolved["presentation_stream_sha256"]
        == metrics["presentation_stream_sha256"]
        == _digest_json(stream)
        and resolved["config"]["postprocessing"]["score_threshold_grid"] == list(THRESHOLDS)
        and bool(resolved["initial_architecture"]),
        f"{job_id}: resolved configuration disagrees with the diagnostic stream or runtime",
    )
    val, train = refs["validation"], refs["train"]
    require(
        evidence.load(folder / "train_gt.json", metrics["train_gt_sha256"]) == train
        and evidence.load(folder / "validation_gt.json", metrics["validation_gt_sha256"]) == val,
        f"{job_id}: source annotations or image IDs were changed",
    )
    training = evidence.load(
        folder / "source_training_audit.json", metrics["training_audit_sha256"]
    )
    scale = verify_training_geometry(
        training, train, stream, job, resize_hw, spec["inputs"]["mask"]
    )
    n_raw, n_filtered = check_predictions(evidence, folder, val, metrics)
    return {
        "job_id": job_id,
        "attempt": state["attempt"],
        "host": fit.host,
        "source": job["source"],
        "scale": job["scale"],
        "intensity": job["intensity"],
        "training_updates": metrics["training_updates"],
        "presentations": training["summary"]["presentations"],
        "source_validation_frames": len(val["images"]),
        "source_validation_updates": [row["update"] for row in history],
        "source_validation_raw_predictions": n_raw,
        "source_validation_operating_predictions": n_filtered,
        "initial_state_sha256": job["initial_state_sha256"],
        "initial_tensor_sha256": job["initial_tensor_sha256"],
        "metrics_sha256": fit.completion["metrics_sha256"],
        "best_weights_sha256": checkpoint,
        **scale,
    }, (signature, _digest_json(resolved["initial_architecture"]))


def audit(args: argparse.Namespace) -> dict:
    evidence = Evidence()
    plan = evidence.load(args.jobs)
    _, diagnostics = _fit_plan(plan)
    planned = {job["job_id"]: job for job in plan["jobs"] if job["job_id"] in diagnostics}
    release_evidence = evidence.load(args.release_evidence)
    profile = evidence.load(args.runtime_profile)
    jobs_sha = evidence.entries[args.jobs.resolve()]
    profile_sha = evidence.entries[args.runtime_profile.resolve()]
    release = profile["training_release"]
    require(
        is_sha(release, 40)
        and release_evidence["release_commit"] == release
        and is_sha(profile["package_inventory_sha256"])
        and profile["status"] == "frozen"
        and profile["protocol_sha256"] == plan["protocol_sha256"]
        and profile["gpu_model"] == GPU_MODEL
        and profile["threads"] == 8
        and profile["thread_policy"] == THREAD_POLICY
        and profile["precision"]
        == {
            "dtype": "float32",
            "autocast": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        and set(profile["thread_environment"])
        == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"}
        and set(profile["thread_environment"].values()) == {"8"},
        "Runtime profile is not the frozen Ada/FP32 release",
    )
    catalog = evidence.load(args.initial_catalog, plan["initial_catalog_sha256"])
    ledger = evidence.load(args.client_ledger)
    require(
        ledger["version"] == 1 and Path(args.archive_root).is_absolute(),
        "Invalid archive ledger or root",
    )
    # Detect incomplete execution before opening any archived predictions or large data build.
    incomplete = [
        job_id for job_id in sorted(diagnostics) if _successful_state(args.state, job_id) is None
    ]
    require(not incomplete, f"Missing successful current diagnostics: {', '.join(incomplete)}")
    spec, refs, streams = canonical_data(
        evidence, args.data_build, plan, args.data_build_summary, release_evidence
    )
    verify_thread_parity(
        evidence.load(args.thread_parity),
        refs,
        plan,
        evidence.entries[(args.data_build / "darc_data.py").resolve()],
    )
    rows, signatures, matched_draws = [], {}, {}
    for source in SOURCES:
        init = catalog[source]["0"]
        keys = ("initial_state_sha256", "initial_tensor_sha256")
        require(
            is_sha(init["sha256"]) and is_sha(init["tensor_sha256"]),
            f"{source}: invalid initial digest",
        )
        for scale, intensity in ARMS:
            job_id = f"darc-v2-diag-{source}-s0-{scale}{intensity}"
            declared = planned[job_id]
            require(
                not declared["size_control"]
                and tuple(declared[k] for k in keys) == (init["sha256"], init["tensor_sha256"])
                and declared["initial_state"] == init["path"],
                f"{job_id}: same-source initial file or tensor identity differs",
            )
            fit, state, inventory = read_attempt(
                evidence, args.state / (job_id + ".json"), args.metadata, ledger, args.archive_root
            )
            row, signature = validate_attempt(
                evidence,
                fit,
                state,
                inventory,
                declared,
                plan,
                profile,
                profile_sha,
                refs[source],
                streams[source],
                spec,
            )
            require(
                signatures.setdefault(source, signature) == signature,
                f"{job_id}: matched host/runtime/data/model architecture differs",
            )
            require(
                matched_draws.setdefault((source, scale), row["draw_sha256"]) == row["draw_sha256"],
                f"{job_id}: matched intensity jobs have different scale or box draws",
            )
            rows.append(row)
    require(
        len({row["best_weights_sha256"] for row in rows}) == len(rows),
        "Different diagnostics share a selected checkpoint digest",
    )

    roots = {"metadata": args.metadata.resolve(), "state": args.state.resolve()}
    attempts = [
        p
        for p in evidence.entries
        if p.is_relative_to(roots["metadata"]) or p.is_relative_to(roots["state"])
    ]
    evidence.file(Path(__file__))
    common = [args.jobs, args.runtime_profile, args.release_evidence, Path(__file__)]
    canonical = [
        args.data_build_summary,
        args.thread_parity,
        *(args.data_build / name for name in DATA_FILES),
        args.data_build / "checksums.json",
    ]
    check_files = {
        "diagnostic_artifacts": [*attempts, args.client_ledger],
        "shared_initialization": [
            args.initial_catalog,
            *(p for p in attempts if p.name in {"metrics.json", "job_resolved.json"}),
        ],
        "spatial_and_box_geometry": [
            *canonical,
            *(p for p in attempts if p.name == "source_training_audit.json"),
        ],
        "finite_gradients_and_valid_predictions": [
            p
            for p in attempts
            if p.name
            in {"history.json", "completion.json", "val_raw_predictions.json", "worker.log"}
        ],
        "source_only_selection": [
            p
            for p in attempts
            if p.name
            in {
                "history.json",
                "validation_gt.json",
                "validation_input_audit.json",
                "postprocessing_selection.json",
                "val_raw_predictions.json",
                "val_predictions.json",
            }
        ],
        "runtime_precision": [
            args.thread_parity,
            *(p for p in attempts if p.name in {"runtime.json", "receipt.json", "completion.json"}),
        ],
        "source_scale_area_audit": [
            *canonical,
            *(p for p in attempts if p.name == "source_training_audit.json"),
        ],
    }
    require(set(check_files) == REQUIRED_CHECKS["scoring"], "Scoring gate algorithm has changed")
    return {
        "schema": "darc-v2-scoring-audit-1",
        "jobs_sha256": jobs_sha,
        "protocol_sha256": plan["protocol_sha256"],
        "release_sha": release,
        "runtime_profile_sha256": profile_sha,
        "diagnostic_job_ids": [row["job_id"] for row in rows],
        "diagnostic_metrics_sha256": {row["job_id"]: row["metrics_sha256"] for row in rows},
        "target_scores_read": 0,
        "gradient_provenance": "The immutable trainer fails on any missing or nonfinite gradient before each optimizer call. This audit verifies success, the call budget and the persisted finite loss trace.",
        "diagnostics": rows,
        "checks": {
            name: {"passed": True, "evidence": evidence.proof([*common, *paths])}
            for name, paths in check_files.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "jobs",
        "state",
        "metadata",
        "client-ledger",
        "runtime-profile",
        "release-evidence",
        "initial-catalog",
        "data-build",
        "data-build-summary",
        "thread-parity",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive-root", required=True)
    args = parser.parse_args()
    report = audit(args)
    fleet.atomic_json(args.output, report)
    print(
        f"Verified {len(report['diagnostics'])} successful source-only diagnostics. Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
