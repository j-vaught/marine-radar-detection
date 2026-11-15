"""Verify and summarize the frozen 42-fit processed DARC factorial."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import TemporaryDirectory

from . import evaluation, protocol
from . import summarize as shared
from .annotation_sensitivity import center_agreement
from .finalize import _serialized_digest

SITES = ("DAAN", "DARC", "MANV")
SEEDS = (0, 1, 2)
ARMS = ("S0I0", "S0I1", "S1I0", "S1I1")
REDUCED = "S0I0-n316"
GPU_MODEL = "NVIDIA RTX 6000 Ada Generation"
OUTCOMES = ("ap50", "ap", "ap75", "ar100", "center_recall_8px_top30")
COEFFICIENTS = {
    "intensity_at_S0": {"S0I1": 1, "S0I0": -1},
    "intensity_at_S1": {"S1I1": 1, "S1I0": -1},
    "scale_at_I0": {"S1I0": 1, "S0I0": -1},
    "scale_at_I1": {"S1I1": 1, "S0I1": -1},
    "scale_by_intensity": {"S1I1": 1, "S1I0": -1, "S0I1": -1, "S0I0": 1},
    "reduced_minus_full": {REDUCED: 1, "S0I0": -1},
}
FOCUS = (("MANV", "DARC"), ("DARC", "MANV"), ("DAAN", "DARC"))
FitKey = tuple[str, int, str]


@dataclass
class Receipt:
    job: dict
    completion: dict
    folder: Path
    metrics_bytes: bytes
    host: str


@dataclass
class Fit:
    receipt: Receipt
    metrics: dict
    runtime: dict
    gpu_memory: dict
    training: dict
    prediction_paths: dict


def _load(path: Path):
    return json.loads(path.read_bytes())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _job_key(job: dict) -> FitKey:
    arm = REDUCED if job["size_control"] else job["scale"] + job["intensity"]
    return job["source"], job["seed"], arm


def _fit_plan(jobs: dict) -> tuple[dict[str, dict], set[str]]:
    if jobs["schema"] != "darc-v2-jobs-1":
        raise ValueError("Unexpected DARC job manifest schema")
    planned, diagnostics = {}, set()
    expected = set(itertools.product(SITES, SEEDS, ARMS)) | {
        (source, seed, REDUCED) for source, seed in itertools.product(("DAAN", "MANV"), SEEDS)
    }
    seen = set()
    for job in jobs["jobs"]:
        source, seed, arm = _job_key(job)
        stage = job["stage"]
        if (
            job["method"] != "faster_rcnn"
            or job["condition"] != "buffered"
            or type(seed) is not int
            or type(job["size_control"]) is not bool
            or (job["size_control"] and (job["scale"], job["intensity"]) != ("S0", "I0"))
        ):
            raise ValueError("Invalid frozen DARC job identity")
        prefix = {"core": "fit", "diagnostic": "diag"}.get(stage)
        job_id = f"darc-v2-{prefix}-{source}-s{seed}-{arm}"
        if prefix is None or job["job_id"] != job_id or job_id in planned or job_id in diagnostics:
            raise ValueError("Invalid or duplicate DARC job ID")
        if stage == "diagnostic":
            if source not in SITES or seed != 0 or arm not in ARMS or job["eval_targets"]:
                raise ValueError("Diagnostics must be the 12 source-only factorial pilots")
            diagnostics.add(job_id)
            continue
        key = (source, seed, arm)
        if key not in expected or key in seen or sorted(job["eval_targets"]) != sorted(SITES):
            raise ValueError(f"{job_id}: unexpected fit or evaluation targets")
        seen.add(key)
        planned[job_id] = job
    if seen != expected or len(diagnostics) != 12:
        raise ValueError("The frozen DARC design requires 42 fits and 12 diagnostics")
    return planned, diagnostics


def _successful_state(state_dir: Path, job_id: str) -> dict | None:
    path = state_dir / f"{job_id}.json"
    if not path.is_file():
        return None
    state = _load(path)
    result = state.get("result") or {}
    if (
        state.get("state") != "succeeded"
        or result.get("state") != "succeeded"
        or result.get("exit_code") != 0
        or result.get("job_id") != job_id
        or result.get("attempt") != state.get("attempt")
        or result.get("nonce") != state.get("nonce")
    ):
        return None
    shared._sha(result.get("metrics_sha256"), f"{job_id} completion metrics")
    return state


def _preflight(
    runs_dir: Path, state_dir: Path, jobs_path: Path, gate_path: Path, protocol_path: Path
) -> tuple[dict, dict, dict[str, Receipt]]:
    """Require the approved scoring gate and a complete archive before opening scores."""
    jobs_bytes = jobs_path.read_bytes()
    jobs = json.loads(jobs_bytes)
    spec_digest = protocol.sha256_file(protocol_path)
    spec = _load(protocol_path)
    plan, diag = _fit_plan(jobs)
    gate = _load(gate_path)
    if (
        gate.get("approved") is not True
        or gate.get("kind") != "scoring"
        or gate.get("jobs_sha256") != _digest(jobs_bytes)
        or gate.get("protocol_sha256") != spec_digest
        or jobs.get("protocol_sha256") != spec_digest
        or set(gate.get("diagnostic_job_ids", ())) != diag
        or spec["protocol_id"] != "darc-processed-v1"
    ):
        raise ValueError("An approved scoring gate for this frozen DARC matrix is required")
    for field in ("runtime_profile_sha256", "audit_sha256"):
        shared._sha(gate.get(field), field)
    release = gate["release_sha"]
    if (
        not isinstance(release, str)
        or len(release) != 40
        or any(x not in "0123456789abcdef" for x in release)
    ):
        raise ValueError("Scoring gate lacks the verified worker release SHA")
    incomplete_diagnostics = sorted(
        job for job in diag if _successful_state(state_dir, job) is None
    )
    if incomplete_diagnostics:
        raise ValueError(
            "Source-only diagnostic completions missing " + ", ".join(incomplete_diagnostics)
        )

    states = {job_id: _successful_state(state_dir, job_id) for job_id in plan}
    missing = [job_id for job_id, state in states.items() if state is None]
    if missing:
        raise ValueError(
            f"Expected 42 completed DARC fits, missing {len(missing)}: {', '.join(missing)}"
        )
    receipts = {}
    ledger_dir = runs_dir / ".ledger"
    fields = ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id")
    for job_id, state in states.items():
        assert state is not None
        if type(state["attempt"]) is not int or state["attempt"] < 1:
            raise ValueError(f"{job_id}: invalid final attempt")
        folder = runs_dir / job_id / str(state["attempt"])
        completion, receipt, job = (
            _load(folder / name) for name in ("completion.json", "receipt.json", "job.json")
        )
        metrics_bytes = (folder / "metrics.json").read_bytes()
        planned = plan[job_id]
        if (
            completion != state["result"]
            or any(
                receipt.get(key) is None or receipt[key] != completion.get(key) for key in fields
            )
            or completion["code_commit"] != release
            or job["code_commit"] != release
            or job["attempt_nonce"] != completion["nonce"]
            or any(
                job[key] != planned[key]
                for key in (
                    "job_id",
                    "method",
                    "source",
                    "stage",
                    "condition",
                    "seed",
                    "scale",
                    "intensity",
                    "size_control",
                    "eval_targets",
                    "initial_state_sha256",
                    "initial_tensor_sha256",
                )
            )
            or job["split_sha256"] != jobs["split_sha256"]
            or job["annotations_sha256"] != jobs["annotations_sha256"]
            or state["runtime_profile_sha256"] != gate["runtime_profile_sha256"]
            or Path(job["images"]).parent != Path(state["data_path"])
            or job["gpu_uuid"] != state["gpu_uuid"]
            or any(key in job for key in ("diagnostic_updates", "diagnostic_val_interval"))
            or any(job.get(key) for key in ("config_overrides", "train_limit", "overfit"))
            or _digest(metrics_bytes) != completion["metrics_sha256"]
        ):
            raise ValueError(f"{job_id}: completed artifact identity differs from the frozen plan")
        if (
            "runtime_profile_sha256" in job
            and job["runtime_profile_sha256"] != gate["runtime_profile_sha256"]
        ):
            raise ValueError(f"{job_id}: worker runtime profile differs from the scoring gate")
        if ledger_dir.is_dir():
            ledger = _load(ledger_dir / job_id / f"{state['attempt']}.json")["files"]
            for name in ("job.json", "receipt.json", "completion.json", "metrics.json"):
                if protocol.sha256_file(folder / name) != ledger[name]["sha256"]:
                    raise ValueError(f"{job_id}: archive ledger differs for {name}")
        receipts[job_id] = Receipt(job, completion, folder, metrics_bytes, state["host"])
    return spec, jobs, receipts


def _artifact(fit: Receipt, location: str, expected_hash: str) -> Path:
    path = Path(location)
    if path.is_absolute():
        path = path.relative_to(Path(fit.job["out_dir"]))
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe path in {fit.job['job_id']} artifact")
    local = fit.folder / path
    if not local.resolve().is_relative_to(fit.folder.resolve()):
        raise ValueError(f"Artifact escapes the approved attempt in {fit.job['job_id']}")
    if protocol.sha256_file(local) != shared._sha(expected_hash, path.name):
        raise ValueError(f"{fit.job['job_id']}: artifact digest mismatch for {path.name}")
    return local


def _release_references(spec: dict, jobs: dict, split_path: Path, ann_path: Path) -> dict:
    split_digest, ann_digest = protocol.sha256_file(split_path), protocol.sha256_file(ann_path)
    if (
        split_digest != jobs["split_sha256"]
        or split_digest != spec["inputs"]["split_sha256"]
        or ann_digest != jobs["annotations_sha256"]
        or ann_digest != spec["inputs"]["annotations_sha256"]
    ):
        raise ValueError("The released split or annotation file differs from the frozen inputs")
    manifest, coco = _load(split_path), _load(ann_path)
    if manifest["annotation_sha256"] != ann_digest:
        raise ValueError("Split manifest references different annotation bytes")
    refs = {}
    for site in SITES:
        parts = manifest["sites"][site]["partitions"]
        if any(
            not parts[key] or len(parts[key]) != len(set(parts[key]))
            for key in ("train", "validation", "test")
        ):
            raise ValueError(f"{site}: split contains missing or duplicate IDs")
        if len(set(parts["train"] + parts["validation"] + parts["test"])) != sum(
            len(parts[key]) for key in ("train", "validation", "test")
        ):
            raise ValueError(f"{site}: split overlaps the frozen target")
        refs[site] = {
            key: shared._subset_coco(coco, set(parts[key]))
            for key in ("train", "validation", "test")
        }
        expected = spec["inputs"]["partitions"][site]
        for key, data in refs[site].items():
            if len(data["images"]) != expected[f"{key}_frames"] or any(
                image["dataset"] != site
                or [image["width"], image["height"]] != spec["inputs"]["dimensions_wh"]
                for image in data["images"]
            ):
                raise ValueError(f"{site}: frozen {key} frame support differs")
            if f"{key}_boxes" in expected and len(data["annotations"]) != expected[f"{key}_boxes"]:
                raise ValueError(f"{site}: frozen {key} reference box support differs")
    return refs


def _fit_schedule(
    receipt: Receipt, metrics: dict, history: list[dict], policy: dict, protocol_id: str
) -> None:
    updates, interval, batch = (
        policy["optimizer_updates"],
        policy["updates_per_validation"],
        policy["batch_size"],
    )
    job_id, source, seed = (receipt.job["job_id"], receipt.job["source"], receipt.job["seed"])
    if (
        metrics["training_updates"] != updates
        or metrics["validation_interval"] != interval
        or metrics["batch_size"] != batch
        or [row["update"] for row in history] != list(range(interval, updates + 1, interval))
    ):
        raise ValueError(f"{job_id}: optimizer update, batch, or validation budget differs")
    for index, row in enumerate(history):
        first_lr = policy["lr_first_update"]
        lr = (
            first_lr
            if row["update"] <= policy["lr_decay_after_update"]
            else first_lr * policy["lr_decay_multiplier"]
        )
        rng = f"{protocol_id}:validation:{source}:{seed}:{index}".encode()
        expected_seed = int.from_bytes(hashlib.sha256(rng).digest()[:4], "big")
        if (
            not math.isclose(row["lr"], lr, rel_tol=1e-12, abs_tol=0)
            or row["validation_seed"] != expected_seed
            or not math.isfinite(row["source_val_loss"])
            or not math.isfinite(row["train_loss_last_batch"])
        ):
            raise ValueError(f"{job_id}: invalid fixed learning rate, loss, or validation RNG")
    best = min(history, key=lambda row: (row["source_val_loss"], row["update"]))
    if (
        metrics["best_update"] != best["update"]
        or metrics["best_val_loss"] != best["source_val_loss"]
    ):
        raise ValueError(f"{job_id}: checkpoint selection differs from the source-validation trace")


def _runtime_signature(receipt: Receipt, runtime: dict) -> tuple:
    job, completion = receipt.job, receipt.completion
    provenance = completion["runtime_provenance"]
    gpu, torch = provenance["gpu"], provenance["torch"]
    required_threads = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    if (
        runtime["device"] != job["device"]
        or runtime["device"] != "cuda:0"
        or runtime["gpu_uuid"] != job["gpu_uuid"]
        or gpu["uuid"] != job["gpu_uuid"]
        or runtime["gpu_name"] != gpu["model"]
        or gpu["model"] != GPU_MODEL
        or runtime["precision"] != "fp32"
        or runtime["autocast"] is not False
        or runtime["cuda_matmul_allow_tf32"] is not False
        or runtime["cudnn_allow_tf32"] is not False
        or torch["matmul_allow_tf32"] is not False
        or torch["cudnn_allow_tf32"] is not False
        or torch["version"] != runtime["torch_version"]
        or torch["cuda_version"] != runtime["cuda_version"]
        or runtime["gpu_capability"] != gpu["compute_capability"]
        or runtime["workers"] != job.get("workers", 2)
        or runtime["torch_threads"] != 8
        or any(provenance["thread_environment"].get(name) != "8" for name in required_threads)
    ):
        raise ValueError(f"{job['job_id']}: GPU assignment or FP32 policy differs")
    for name in ("cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms"):
        if type(runtime[name]) is not bool or runtime[name] != torch[name]:
            raise ValueError(f"{job['job_id']}: numerical policy differs on {name}")
    return (
        gpu["model"],
        tuple(gpu["compute_capability"]),
        runtime["torch_version"],
        runtime["torchvision_version"],
        runtime["cuda_version"],
        runtime["cudnn_version"],
        runtime["numpy_version"],
        runtime["python_version"],
        runtime["torch_threads"],
        runtime["torch_interop_threads"],
        runtime["precision"],
        runtime["cudnn_deterministic"],
        runtime["cudnn_benchmark"],
        runtime["deterministic_algorithms"],
        tuple(runtime["model_resize_hw"]),
        tuple(sorted(provenance["thread_environment"].items())),
        job.get("workers", 2),
        job["images"],
    )


def _memory_evidence(receipt: Receipt, runtime: dict) -> dict:
    """Preserve the individual capacity report and enforce plausible Ada headroom."""
    gpu = receipt.completion["runtime_provenance"]["gpu"]
    capacity, peak = gpu["total_memory_bytes"], runtime["peak_vram_bytes"]
    gib = 1024**3
    if type(capacity) is not int or not 45 * gib <= capacity <= 49 * gib:
        raise ValueError(f"{receipt.job['job_id']}: reported GPU capacity outside 45–49 GiB")
    if type(peak) is not int or not 0 < peak < capacity:
        raise ValueError(f"{receipt.job['job_id']}: invalid peak GPU allocation")
    return {"gpu_uuid": gpu["uuid"], "total_memory_bytes": capacity, "peak_vram_bytes": peak}


def _verify_memory_block(block: tuple[str, int], evidence: list[dict]) -> dict:
    """Allow small capacity-report variation while checking headroom on every device."""
    capacities = [row["total_memory_bytes"] for row in evidence]
    low, high = min(capacities), max(capacities)
    spread = high - low
    peak = max(row["peak_vram_bytes"] for row in evidence)
    if spread * 100 > low:
        raise ValueError(f"{block}: GPU capacity spread exceeds 1% within the matched block")
    if peak >= low:
        raise ValueError(f"{block}: peak GPU allocation exceeds matched device capacity")
    return {
        "min_total_memory_bytes": low,
        "max_total_memory_bytes": high,
        "spread_bytes": spread,
        "relative_spread": spread / low,
        "max_peak_vram_bytes": peak,
        "max_peak_fraction_of_min_capacity": peak / low,
    }


def _prepare_fits(
    receipts: dict[str, Receipt], references: dict, spec: dict, spec_digest: str
) -> tuple[dict[FitKey, Fit], list[dict]]:
    from .darc_data import select_count_subset, train_stream

    policy, contract = spec["fit_policy"], spec["inputs"]
    presentations = policy["optimizer_updates"] * policy["batch_size"]
    train_cache = {}
    signatures = {}
    memory = defaultdict(list)
    states = {}
    scale_audits = {}
    checkpoint_digests = set()
    fits = {}
    for fit in receipts.values():
        metrics = json.loads(fit.metrics_bytes)
        job = fit.job
        source, seed, arm = _job_key(job)
        job_id = job["job_id"]
        for key in (
            "job_id",
            "source",
            "seed",
            "stage",
            "condition",
            "scale",
            "intensity",
            "size_control",
            "initial_state_sha256",
            "initial_tensor_sha256",
        ):
            if metrics[key] != job[key]:
                raise ValueError(f"{job_id}: metrics differ from validated job field {key}")
        if (
            metrics["protocol_id"] != spec["protocol_id"]
            or metrics["frozen_protocol_sha256"] != spec_digest
            or metrics["manifest_sha256"] != contract["split_sha256"]
            or metrics["annotation_sha256"] != contract["annotations_sha256"]
            or set(metrics["tests"]) != set(SITES)
        ):
            raise ValueError(f"{job_id}: run data or target identity differs")
        initial = tuple(
            shared._sha(metrics[k], k) for k in ("initial_state_sha256", "initial_tensor_sha256")
        )
        if states.setdefault((source, seed), initial) != initial:
            raise ValueError(f"{job_id}: factorial block has different initialization")
        checkpoint = shared._sha(metrics["best_weights_sha256"], "selected checkpoint")
        if checkpoint in checkpoint_digests:
            raise ValueError(f"{job_id}: selected checkpoint is reused by another fit")
        checkpoint_digests.add(checkpoint)

        history = _load(fit.folder / "history.json")
        _fit_schedule(fit, metrics, history, policy, spec["protocol_id"])
        flagged = arm == REDUCED
        cache_key = (source, seed, flagged)
        if cache_key not in train_cache:
            train = references[source]["train"]
            subset_ids = (
                select_count_subset(
                    train["images"], source, seed, spec["size_control"]["distinct_frames_per_arm"]
                )
                if flagged
                else None
            )
            if subset_ids is not None:
                train = shared._subset_coco(train, set(subset_ids))
            stream = train_stream(train["images"], source, seed, presentations=presentations)
            train_cache[cache_key] = train, subset_ids, stream
        train, subset_ids, stream = train_cache[cache_key]
        if (
            metrics["presentation_stream_sha256"] != shared._digest_json(stream)
            or metrics["subset_ids_sha256"]
            != (shared._digest_json(subset_ids) if subset_ids is not None else None)
            or metrics["train_gt_sha256"] != _serialized_digest(train)
            or metrics["validation_gt_sha256"]
            != _serialized_digest(references[source]["validation"])
        ):
            raise ValueError(
                f"{job_id}: source subset, presentation order, or source references differ"
            )
        resolved = _load(fit.folder / "job_resolved.json")
        if (
            resolved["job"] != job
            or resolved["protocol_id"] != spec["protocol_id"]
            or resolved["decay_after_update"] != policy["lr_decay_after_update"]
            or resolved["subset_ids"] != subset_ids
        ):
            raise ValueError(f"{job_id}: resolved training plan differs")
        _artifact(fit, "train_gt.json", metrics["train_gt_sha256"])
        _artifact(fit, "validation_gt.json", metrics["validation_gt_sha256"])
        audit = _load(
            _artifact(fit, "source_training_audit.json", metrics["training_audit_sha256"])
        )
        audit_rows = audit["presentations"]
        if [(row["presentation_index"], row["image_id"]) for row in audit_rows] != stream:
            raise ValueError(
                f"{job_id}: observed presentation stream differs from the frozen order"
            )
        if audit["summary"]["presentations"] != presentations:
            raise ValueError(f"{job_id}: training audit presentation total differs")
        scale_hash = shared._digest_json(
            [
                (row["scale"], row["scale_identity_reason"], row["scale_rejections"])
                for row in audit_rows
            ]
        )
        draw_key = (source, seed, job["scale"], flagged)
        if scale_audits.setdefault(draw_key, scale_hash) != scale_hash:
            raise ValueError(f"{job_id}: matched intensity fits used different scale draws")

        runtime = _load(_artifact(fit, metrics["runtime_path"], metrics["runtime_sha256"]))
        signature = (
            fit.host,
            _runtime_signature(fit, runtime),
            shared._digest_json(resolved["initial_architecture"]),
        )
        if signatures.setdefault((source, seed), signature) != signature:
            raise ValueError(
                f"{job_id}: hardware, environment, or precision differs within the matched block"
            )
        evidence = _memory_evidence(fit, runtime)
        memory[source, seed].append(evidence)
        if runtime["training_updates"] != policy["optimizer_updates"] or runtime[
            "validation_count"
        ] != len(history):
            raise ValueError(f"{job_id}: observed training update or validation count differs")
        selection = _load(fit.folder / "postprocessing_selection.json")
        threshold = metrics["score_threshold"]
        if (
            selection["selection_split"] != "source_validation"
            or selection["score_threshold"] != threshold
            or not 0 <= threshold <= 1
            or selection["metrics"] != metrics["source_validation"]
        ):
            raise ValueError(f"{job_id}: source-validation operating selection differs")

        paths = {}
        for target in SITES:
            payload = metrics["tests"][target]
            if payload["checkpoint_sha256"] != checkpoint or payload[
                "gt_sha256"
            ] != _serialized_digest(references[target]["test"]):
                raise ValueError(f"{job_id}/{target}: checkpoint or target reference differs")
            _artifact(fit, payload["gt_path"], payload["gt_sha256"])
            operating_path = payload.get(
                "filtered_predictions_path", payload.get("predictions_path")
            )
            paths[target] = (
                _artifact(fit, payload["raw_predictions_path"], payload["raw_predictions_sha256"]),
                _artifact(fit, operating_path, payload["predictions_sha256"]),
            )
        fits[source, seed, arm] = Fit(
            fit,
            metrics,
            runtime,
            evidence,
            {
                "train_frames": len(train["images"]),
                "train_boxes": len(train["annotations"]),
                "realized_training": audit["summary"],
                "validation_history": history,
            },
            paths,
        )
    block_memory = [
        {"source": source, "seed": seed, **_verify_memory_block((source, seed), evidence)}
        for (source, seed), evidence in sorted(memory.items())
    ]
    return fits, block_memory


def _test_rows(fits: dict[FitKey, Fit], references: dict) -> list[dict]:
    rows = []
    for (source, seed, arm), fit in sorted(fits.items()):
        for target in SITES:
            gt = references[target]["test"]
            raw, operating = map(_load, fit.prediction_paths[target])
            threshold = fit.metrics["score_threshold"]
            job_id = fit.receipt.job["job_id"]
            record = fit.metrics["tests"][target]
            if operating != [p for p in raw if p["score"] >= threshold]:
                raise ValueError(
                    f"{job_id}/{target}: predictions differ from source-selected threshold"
                )
            scores = evaluation.evaluate_coco(gt, raw)
            operating_scores = evaluation.evaluate_coco(gt, operating)
            for label, replay in (("raw", scores), ("filtered", operating_scores)):
                if any(
                    not math.isclose(replay[k], record[label][k], rel_tol=0, abs_tol=1e-9)
                    for k in shared.ROW_METRICS
                ):
                    raise ValueError(f"{job_id}/{target}: {label} metric replay differs")
            centers = center_agreement(gt, raw)
            center = next(
                row
                for row in centers["rows"]
                if (row["radius_px"], row["top_proposals_per_image_category"]) == (8, 30)
            )
            recorded_center = record["center_recall_8px_top30"]
            if (
                center["matched_centers"] != recorded_center["matched"]
                or centers["reference_annotations"] != recorded_center["references"]
                or center["center_recall"] != recorded_center["recall"]
                or recorded_center["max_detections"] != 30
                or recorded_center["tolerance_px"] != 8
            ):
                raise ValueError(f"{job_id}/{target}: one-to-one center recall differs")
            rows.append(
                {
                    "job_id": job_id,
                    "attempt": fit.receipt.completion["attempt"],
                    "source": source,
                    "target": target,
                    "arm": arm,
                    "seed": seed,
                    "test_frames": len(gt["images"]),
                    "test_boxes": len(gt["annotations"]),
                    "train_frames": fit.training["train_frames"],
                    "train_boxes": fit.training["train_boxes"],
                    "checkpoint_sha256": fit.metrics["best_weights_sha256"],
                    "target_gt_sha256": record["gt_sha256"],
                    "best_update": fit.metrics["best_update"],
                    "operating_threshold": threshold,
                    **scores,
                    **{f"operating_{k}": value for k, value in operating_scores.items()},
                    "center_recall_8px_top30": center["center_recall"],
                    "center_matched_8px_top30": center["matched_centers"],
                    "center_references": centers["reference_annotations"],
                }
            )
    return rows


def _paired_effects(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    cells = {(r["source"], r["target"], r["seed"], r["arm"]): r for r in rows}
    if len(rows) != 126 or len(cells) != 126:
        raise ValueError("Paired analysis requires 126 unique destination cells")
    gaps = []
    seed_effects = []
    for (source, target, seed, arm), row in sorted(cells.items()):
        if source == target:
            continue
        local_arm = "S0I0" if arm == REDUCED else arm
        local = cells[target, target, seed, local_arm]
        for metric in OUTCOMES:
            gaps.append(
                {
                    "source": source,
                    "target": target,
                    "seed": seed,
                    "arm": arm,
                    "metric": metric,
                    "local_reference_arm": local_arm,
                    "local": local[metric],
                    "incoming": row[metric],
                    "local_minus_incoming": local[metric] - row[metric],
                }
            )
    for source, target, seed in itertools.product(SITES, SITES, SEEDS):
        for name, coeffs in COEFFICIENTS.items():
            size = name == "reduced_minus_full"
            if size and source == "DARC":
                continue
            for metric in OUTCOMES:
                cell_scores = {arm: cells[source, target, seed, arm][metric] for arm in coeffs}
                change = sum(coeffs[arm] * value for arm, value in cell_scores.items())
                row = {
                    "source": source,
                    "target": target,
                    "seed": seed,
                    "contrast": name,
                    "metric": metric,
                    "score_change": change,
                    "incoming_values": cell_scores,
                    "local_values": None,
                    "local_reference_change": None,
                    "gap_change": None,
                }
                if source != target:
                    local_scores = {
                        arm: cells[target, target, seed, ("S0I0" if size else arm)][metric]
                        for arm in coeffs
                    }
                    local_change = sum(coeffs[arm] * value for arm, value in local_scores.items())
                    row.update(
                        local_values=local_scores,
                        local_reference_change=local_change,
                        gap_change=local_change - change,
                    )
                seed_effects.append(row)
    groups = defaultdict(list)
    for row in seed_effects:
        key = tuple(row[field] for field in ("source", "target", "contrast", "metric"))
        groups[key].append(row)
    estimates = []
    for key, group in sorted(groups.items()):
        if [r["seed"] for r in group] != list(SEEDS):
            raise ValueError("Every paired contrast requires all three seed values")
        result = dict(zip(("source", "target", "contrast", "metric"), key, strict=True))
        for name in ("score_change", "local_reference_change", "gap_change"):
            values = [r[name] for r in group]
            result[f"{name}_by_seed"] = values
            result[f"{name}_mean"] = fmean(values) if values[0] is not None else None
            result[f"{name}_sample_sd"] = stdev(values) if values[0] is not None else None
        estimates.append(result)
    return gaps, seed_effects, estimates


def summarize(
    *,
    protocol_path: Path,
    jobs_path: Path,
    score_gate: Path,
    state_dir: Path,
    runs_dir: Path,
    split_path: Path,
    annotations: Path,
) -> dict:
    spec, jobs, receipts = _preflight(runs_dir, state_dir, jobs_path, score_gate, protocol_path)
    references = _release_references(spec, jobs, split_path, annotations)
    fits, block_memory = _prepare_fits(receipts, references, spec, jobs["protocol_sha256"])
    tests = _test_rows(fits, references)
    gaps, effects, estimates = _paired_effects(tests)
    return {
        "schema_version": 1,
        "protocol_id": spec["protocol_id"],
        "input_scope": "Released processed radar images and their fixed image and box inventory.",
        "manual_labels_added": 0,
        "raw_radar_rebuild_executed": False,
        "interpretation": "Post hoc descriptive contrasts on the three frozen processed-image holdouts.",
        "value_scale": "Scores and effects use a 0–1 fraction scale. Text tables convert differences to points.",
        "protocol_sha256": protocol.sha256_file(protocol_path),
        "jobs_sha256": protocol.sha256_file(jobs_path),
        "scoring_gate_sha256": protocol.sha256_file(score_gate),
        "runtime_profile_sha256": _load(score_gate)["runtime_profile_sha256"],
        "initial_catalog_sha256": jobs["initial_catalog_sha256"],
        "split_sha256": protocol.sha256_file(split_path),
        "annotations_sha256": protocol.sha256_file(annotations),
        "evaluator_sha256": protocol.sha256_file(evaluation.__file__),
        "analysis_code_sha256": protocol.sha256_file(__file__),
        "support": {
            site: {f"{part}_frames": len(ref["images"]) for part, ref in groups.items()}
            | {f"{part}_boxes": len(ref["annotations"]) for part, ref in groups.items()}
            for site, groups in references.items()
        },
        "counts": {
            "factorial_fits": 36,
            "size_control_fits": 6,
            "fits": 42,
            "target_evaluations": 126,
        },
        "gpu_memory_pairing": block_memory,
        "fits": [
            {
                "source": key[0],
                "seed": key[1],
                "arm": key[2],
                "job_id": fit.receipt.job["job_id"],
                "attempt": fit.receipt.completion["attempt"],
                "worker_host": fit.receipt.host,
                "worker_commit": fit.receipt.job["code_commit"],
                "metrics_sha256": fit.receipt.completion["metrics_sha256"],
                "initial_state_sha256": fit.metrics["initial_state_sha256"],
                "initial_tensor_sha256": fit.metrics["initial_tensor_sha256"],
                "presentation_stream_sha256": fit.metrics["presentation_stream_sha256"],
                "subset_ids_sha256": fit.metrics["subset_ids_sha256"],
                "checkpoint_sha256": fit.metrics["best_weights_sha256"],
                "runtime": fit.runtime,
                "gpu_memory": fit.gpu_memory,
                **fit.training,
            }
            for key, fit in sorted(fits.items())
        ],
        "test_rows": tests,
        "target_local_minus_incoming": gaps,
        "paired_effects_by_seed": effects,
        "paired_effect_estimates": estimates,
    }


def _plain_report(report: dict) -> str:
    lines = [
        "DARC processed-image factorial",
        "",
        "The analysis uses the frozen DAAN, DARC, and MANV holdouts with their released image and box inventory.",
        "Manual labels added 0. Raw-radar rebuild status unrun.",
        "The intensity factor changes rendered pixel values and the scale factor changes source-training spatial coverage.",
        "The contrasts are post hoc descriptive summaries on the frozen holdouts.",
        "",
        "Completed fits 42. Fixed-target evaluations 126. Three seeds per paired effect.",
        "",
        "Priority directions and own-source sample-count comparisons",
        "Intensity effects are I1 minus I0. Scale effects are S1 minus S0.",
        "The interaction is the intensity effect at S1 minus the intensity effect at S0.",
        "The size effect compares a reduced source with its own full S0I0 fit.",
        "Each value is the mean ± sample standard deviation in points across the three paired seeds.",
        "source → target  contrast  metric  score change  local reference change  gap change",
    ]
    for row in report["paired_effect_estimates"]:
        selected = (row["source"], row["target"]) in FOCUS or row[
            "contrast"
        ] == "reduced_minus_full"
        if not selected or row["metric"] not in ("ap50", "ap", "center_recall_8px_top30"):
            continue
        numbers = []
        for field in ("score_change", "local_reference_change", "gap_change"):
            mean, sd = row[f"{field}_mean"], row[f"{field}_sample_sd"]
            numbers.append("—" if mean is None else f"{100 * mean:+.2f} ± {100 * sd:.2f}")
        lines.append(
            f"{row['source']} → {row['target']}  {row['contrast']}  {row['metric']}  "
            + "  ".join(numbers)
        )
    lines.extend(
        (
            "",
            "Absolute seed results on the three priority directions",
            "AP and center recall are shown on the 0–100 scale.",
            "source → target  arm  seed  AP50  AP  AP75  AR100  center recall 8 px top 30",
        )
    )
    for row in report["test_rows"]:
        if (row["source"], row["target"]) in FOCUS:
            lines.append(
                f"{row['source']} → {row['target']}  {row['arm']}  {row['seed']}  "
                + "  ".join(f"{100 * row[metric]:.2f}" for metric in OUTCOMES)
            )
    return "\n".join(lines) + "\n"


def _write_report(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as stage:
        pending = Path(stage) / "report"
        pending.mkdir()
        protocol.save_json(pending / "summary.json", report)
        for key in (
            "test_rows",
            "target_local_minus_incoming",
            "paired_effects_by_seed",
            "paired_effect_estimates",
        ):
            rows = [
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                    for k, v in row.items()
                }
                for row in report[key]
            ]
            shared._write_csv(pending / f"{key}.csv", rows, tuple(rows[0]))
        (pending / "report.txt").write_text(_plain_report(report), encoding="utf-8")
        if output.exists():
            if all(
                (output / p.name).is_file()
                and protocol.sha256_file(output / p.name) == protocol.sha256_file(p)
                for p in pending.iterdir()
            ):
                return
            raise FileExistsError("Choose a new output directory for changed DARC analysis")
        os.rename(pending, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "jobs",
        "score-gate",
        "state-dir",
        "runs-dir",
        "split",
        "annotations",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        protocol_path=args.protocol,
        jobs_path=args.jobs,
        score_gate=args.score_gate,
        state_dir=args.state_dir,
        runs_dir=args.runs_dir,
        split_path=args.split,
        annotations=args.annotations,
    )
    _write_report(args.output_dir, report)
    print(f"Verified DARC report for 42 fits and 126 evaluations in {args.output_dir}.")


if __name__ == "__main__":
    main()
