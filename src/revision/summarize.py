"""Summarize the fixed-target revision matrix from collected successful jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SITES = ("DAAN", "DARC", "MANV")
LEARNED = ("faster_rcnn", "yolo", "unet", "rtdetr")
CLASSICAL = ("ca_cfar", "os_cfar", "otsu", "ss_otsu", "wavelet")
ADJACENCY = ("faster_rcnn", "yolo")
AP_METRICS = ("ap", "ap50", "ap75")
ROW_METRICS = (*AP_METRICS, "ar100", "predictions", "preds_per_image")
EDGES = tuple(itertools.permutations(SITES, 2))
T_975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105}
IDENTITY = ("job_id", "method", "source", "seed", "condition", "stage")
Signature = tuple[str, str, str, int]


@dataclass
class Inputs:
    sha256: str
    manifest: dict
    annotations: dict
    holdout_hashes: dict[str, str]
    object_counts: dict[str, int]


@dataclass
class Run:
    job: dict
    attempt: Path
    metrics_path: Path
    inputs: Inputs
    metrics: dict
    tests: dict[str, dict]
    validation: dict
    artifact_sha256: str

    def value(self, target: str, metric: str) -> float:
        return self.tests[target][metric]


def _load(path: Path):
    return json.loads(path.read_bytes())


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_json(value) -> str:
    return _digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _read_inputs(manifest_path: Path, ann_path: Path, recorded_sha: str) -> Inputs:
    manifest_raw = manifest_path.read_bytes()
    if _digest(manifest_raw) != recorded_sha:
        raise ValueError(f"split manifest SHA-256 does not match recorded run: {manifest_path}")
    manifest = json.loads(manifest_raw)
    annotations_raw = ann_path.read_bytes()
    annotation_sha = _digest(annotations_raw)
    if annotation_sha != manifest["annotation_sha256"]:
        raise ValueError(f"annotation SHA-256 does not match split manifest: {ann_path}")
    annotations = json.loads(annotations_raw)
    image_records = {row["id"]: row for row in manifest["images"]}
    if len(image_records) != len(manifest["images"]):
        raise ValueError("duplicate image IDs in split manifest")
    coco_ids = {row["id"] for row in annotations["images"]}
    if len(coco_ids) != len(annotations["images"]):
        raise ValueError("duplicate image IDs in COCO annotations")
    object_counts = Counter(row["image_id"] for row in annotations["annotations"])
    hashes, counts = {}, {}
    for site in SITES:
        ids = manifest["sites"][site]["partitions"]["test"]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= coco_ids:
            raise ValueError(f"empty holdout, duplicate, or unknown image IDs for {site}")
        images = [image_records[image_id] for image_id in ids]
        if any(image["site"] != site for image in images):
            raise ValueError(f"holdout contains another site's image: {site}")
        for image in images:
            _sha(image.get("sha256"), f"frozen image {image['id']}")
        counts[site] = sum(object_counts[image_id] for image_id in ids)
        if not counts[site]:
            raise ValueError(f"holdout has no reference objects: {site}")
        hashes[site] = _digest_json([annotation_sha, site, images])
    return Inputs(recorded_sha, manifest, annotations, hashes, counts)


def _input_path(job: dict, key: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    path = str(job[key])
    if "{" in path or "}" in path:
        raise ValueError(f"unresolved worker input {key}; provide the collected input override")
    return Path(path).resolve()


def _sha(value, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid or missing SHA-256 for {name}")
    return value


def _raw_metrics(payload: dict) -> dict:
    return payload.get("raw_metrics", payload.get("raw", payload))


def _evaluation_metrics(payload: dict) -> dict:
    values = {}
    for key in ROW_METRICS:
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid evaluation metric {key}: {value!r}")
        if key in (*AP_METRICS, "ar100") and not 0 <= value <= 1:
            raise ValueError(f"expected a defined [0, 1] evaluation metric: {key}={value!r}")
        if value < 0 or (key == "predictions" and not isinstance(value, int)):
            raise ValueError(f"invalid prediction count or rate: {key}={value!r}")
        values[key] = value
    return values


def _job_identity(job: dict, expected: dict) -> None:
    if any(job[key] != expected[key] for key in IDENTITY):
        raise ValueError("run identity differs from the planned job identity")


def _load_run(
    planned: dict,
    folder: Path,
    split_manifest: Path | None,
    annotations: Path | None,
    verified: dict[tuple[Path, Path, str], Inputs],
) -> Run:
    metrics_path = folder / "metrics.json"
    metrics = _load(metrics_path)
    actual = _load(folder / "job.json")
    _job_identity(actual, planned)
    if set(actual["eval_targets"]) != set(planned["eval_targets"]):
        raise ValueError("worker evaluation targets differ from the job manifest")
    if any(key in metrics and metrics[key] != actual[key] for key in IDENTITY):
        raise ValueError("metrics identity differs from the recorded worker job")
    if "job_id" not in metrics or "method" not in metrics:
        raise ValueError("metrics missing job_id or method")
    if "job" in metrics:
        _job_identity(metrics["job"], actual)

    record_sha = metrics.get("manifest_sha256", metrics.get("paths", {}).get("manifest_sha256"))
    inputs_key = (
        _input_path(actual, "manifest", split_manifest),
        _input_path(actual, "ann", annotations),
        _sha(record_sha, "recorded split manifest"),
    )
    if inputs_key not in verified:
        verified[inputs_key] = _read_inputs(*inputs_key)
    inputs = verified[inputs_key]
    learned = actual["method"] in LEARNED
    if learned:
        artifact = _sha(
            metrics.get(
                "checkpoint_sha256", metrics.get("best_sha256", metrics.get("best_weights_sha256"))
            ),
            "selected checkpoint",
        )
    else:
        selected = metrics["selected_config"]
        artifact = _digest_json(selected.get("config", selected))

    test_artifacts = metrics["tests"]
    if set(test_artifacts) != set(SITES):
        raise ValueError("a primary job must provide exactly one test result per frozen holdout")
    tests = {}
    for site in SITES:
        payload = test_artifacts[site]
        if "checkpoint_sha256" in payload and payload["checkpoint_sha256"] != artifact:
            raise ValueError(f"{site} did not use the run's selected checkpoint")
        tests[site] = _evaluation_metrics(_raw_metrics(payload))
        count = len(inputs.manifest["sites"][site]["partitions"]["test"])
        if payload.get("images", count) != count or not math.isclose(
            tests[site]["predictions"] / count, tests[site]["preds_per_image"], abs_tol=1e-12
        ):
            raise ValueError(f"{site} metric support disagrees with the frozen holdout")
    validation = _evaluation_metrics(_raw_metrics(metrics["validation"]))
    return Run(actual, folder, metrics_path, inputs, metrics, tests, validation, artifact)


def _planned_signatures() -> set[Signature]:
    return {
        (stage, method, source, seed)
        for stage, methods, seeds in (
            ("core", LEARNED, range(5)),
            ("core", CLASSICAL, (0,)),
            ("adjacency", ADJACENCY, range(3)),
        )
        for method, source, seed in itertools.product(methods, SITES, seeds)
    }


def _signature(job: dict) -> Signature:
    return job["stage"], job["method"], job["source"], job["seed"]


def _attempts(root: Path, job_id: str) -> list[dict]:
    attempts = []
    for folder in sorted((root / job_id).glob("*"), key=lambda p: (len(p.name), p.name)):
        if not folder.is_dir() or re.fullmatch(r"[1-9][0-9]*", folder.name) is None:
            continue
        attempt = int(folder.name)
        completion_path = folder / "completion.json"
        if not completion_path.is_file():
            attempts.append({"attempt": attempt, "state": "no_completion_record"})
            continue
        try:
            receipt = _load(completion_path)
            if receipt["job_id"] != job_id or receipt["attempt"] != attempt:
                raise ValueError("completion identity differs from its attempt directory")
            state = receipt["state"]
            if state == "succeeded" and receipt["exit_code"] != 0:
                raise ValueError("successful completion must have exit code zero")
            if state not in {"succeeded", "failed"}:
                raise ValueError(f"unsupported completion state: {state}")
            attempts.append(receipt)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            attempts.append({"attempt": attempt, "state": "invalid_completion", "error": str(exc)})
    return attempts


def _load_attempt_metrics(folder: Path, completion: dict) -> tuple[dict, str]:
    payload = (folder / "metrics.json").read_bytes()
    digest = _digest(payload)
    if completion["metrics_sha256"] != digest:
        raise ValueError("Attempt metrics do not match the completed receipt")
    return json.loads(payload), digest


def _completed_attempt(root: Path, planned: dict, attempts: list[dict]) -> dict | None:
    """Select a unique success or its verified, same-checkpoint YOLO evaluation repair."""
    successes = [attempt for attempt in attempts if attempt["state"] == "succeeded"]
    if not successes:
        return None
    job_id = planned["job_id"]
    candidate = successes[-1]
    folder = root / job_id / str(candidate["attempt"])
    actual = _load(folder / "job.json")
    requested_parent = planned.get("evaluation_from")
    repair_parent = actual.get("evaluation_from")
    if requested_parent and not repair_parent and len(successes) == 1:
        return None
    if repair_parent is None and len(successes) == 1:
        if planned["method"] == "yolo" and planned["stage"] in {"core", "adjacency"}:
            metrics, _ = _load_attempt_metrics(folder, candidate)
            if metrics.get("evaluation_recipe_version") != "yolo-exact-letterbox-v1":
                return None
        return candidate
    if (
        len(successes) != 2
        or planned["method"] != "yolo"
        or not repair_parent
        or (requested_parent is not None and requested_parent != repair_parent)
    ):
        raise ValueError("Multiple successful attempts without a declared YOLO evaluation repair")

    original = successes[0]
    parent = root / job_id / str(original["attempt"])
    parent_job = _load(parent / "job.json")
    receipt_bytes = (parent / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    repair_receipt = _load(folder / "receipt.json")
    receipt_fields = ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id")
    _job_identity(actual, planned)
    _job_identity(parent_job, planned)
    if (
        parent_job.get("evaluation_from")
        or parent_job["out_dir"] != repair_parent
        or parent_job["attempt_nonce"] != original["nonce"]
        or actual["attempt_nonce"] != candidate["nonce"]
        or any(receipt[key] != original[key] for key in receipt_fields)
        or any(repair_receipt[key] != candidate[key] for key in receipt_fields)
        or any(
            actual.get(key) != parent_job.get(key)
            for key in (
                "initialization",
                "epochs",
                "patience",
                "overfit",
                "train_limit",
                "eval_targets",
                "config_overrides",
            )
        )
        or actual["device"] != "cpu"
        or actual["protocol_sha256"] != candidate["manifest_sha256"]
        or parent_job["protocol_sha256"] != original["manifest_sha256"]
        or candidate["manifest_sha256"] != original["manifest_sha256"]
    ):
        raise ValueError("Evaluation repair receipt or parent identity mismatch")

    parent_metrics, metrics_sha = _load_attempt_metrics(parent, original)
    latest, _ = _load_attempt_metrics(folder, candidate)
    recorded = latest["training_parent"]
    checkpoint = _sha(parent_metrics["best_sha256"], "training parent checkpoint")
    if (
        recorded["directory"] != repair_parent
        or recorded["metrics_sha256"] != metrics_sha
        or recorded["receipt_sha256"] != _digest(receipt_bytes)
        or recorded["completion_sha256"] != _digest((parent / "completion.json").read_bytes())
        or recorded["checkpoint_sha256"] != checkpoint
        or latest["best_sha256"] != checkpoint
        or recorded["code_commit"] != original["code_commit"]
        or recorded.get("runtime_provenance") != original.get("runtime_provenance")
        or latest["evaluation_recipe_version"] != "yolo-exact-letterbox-v1"
        or latest["evaluation_device"] != "cpu"
        or any(
            latest.get(field) != parent_metrics.get(field)
            for field in ("best_epoch", "training_seconds", "peak_vram_bytes")
        )
    ):
        raise ValueError("Evaluation repair provenance differs from the archived training parent")
    return candidate


def _collect(
    jobs: list[dict], root: Path, split_manifest: Path | None, annotations: Path | None
) -> tuple[dict[Signature, Run], list[dict], set[Signature]]:
    plan = _planned_signatures()
    declared: set[Signature] = set()
    ids = set()
    reports = []
    runs = {}
    verified: dict[tuple[Path, Path, str], Inputs] = {}
    for job in jobs:
        job_id = job["job_id"]
        if (
            not isinstance(job_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) is None
            or job_id in {".", ".."}
            or job_id in ids
        ):
            raise ValueError(f"invalid or duplicate job ID: {job_id!r}")
        ids.add(job_id)
        stage = job["stage"]
        signature = _signature(job)
        if isinstance(job["seed"], bool) or not isinstance(job["seed"], int):
            raise TypeError(f"invalid seed for {job_id}")
        if stage in {"core", "adjacency"}:
            condition = "buffered" if stage == "core" else "adjacency"
            if signature not in plan or signature in declared or job["condition"] != condition:
                raise ValueError(f"out-of-protocol or duplicate primary job: {job_id}")
            if sorted(job["eval_targets"]) != sorted(SITES):
                raise ValueError(f"primary job {job_id} must declare all three fixed holdouts")
            declared.add(signature)
        elif stage != "diagnostic":
            raise ValueError(f"unsupported stage for {job_id}: {stage!r}")
        attempts = _attempts(root, job_id)
        report = {key: job[key] for key in IDENTITY}
        report["attempts"] = attempts
        report["state"] = attempts[-1]["state"] if attempts else "missing"
        try:
            selected = _completed_attempt(root, job, attempts)
            if selected is None:
                if report["state"] == "succeeded":
                    report["state"] = "waiting_for_evaluation_repair"
            else:
                folder = root / job_id / str(selected["attempt"])
                if stage == "diagnostic":
                    result = _load(folder / "metrics.json")
                    if result["job_id"] != job_id or result.get("tests"):
                        raise ValueError("diagnostic metrics identity or source-only scope failed")
                else:
                    runs[signature] = _load_run(job, folder, split_manifest, annotations, verified)
                report.update(state="complete", metrics_path=str(folder / "metrics.json"))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            report.update(state="invalid", error=str(exc))
        reports.append(report)
    return runs, reports, plan - declared


def _verify_matches(runs: dict[Signature, Run]) -> dict[str, dict]:
    holdouts = {}
    for (stage, method, source, seed), run in runs.items():
        for site in SITES:
            fingerprint = run.inputs.holdout_hashes[site]
            support = {
                "holdout_sha256": fingerprint,
                "frames": len(run.inputs.manifest["sites"][site]["partitions"]["test"]),
                "objects": run.inputs.object_counts[site],
            }
            if holdouts.setdefault(site, support) != support:
                raise ValueError(
                    f"mismatched fixed target holdout for {site} in {run.job['job_id']}"
                )
        if stage == "adjacency" and (core := runs.get(("core", method, source, seed))) is not None:
            core_site = core.inputs.manifest["sites"][source]
            adjacent_site = run.inputs.manifest["sites"][source]
            if core_site["partitions"]["validation"] != adjacent_site["partitions"][
                "validation"
            ] or len(core_site["partitions"]["train"]) != len(
                adjacent_site["adjacency"][str(seed)]
            ):
                raise ValueError(f"unmatched validation or training count in {run.job['job_id']}")
    return holdouts


def _full_rows(runs: dict[Signature, Run], holdouts: dict[str, dict]) -> list[dict]:
    rows = []
    for _, run in sorted(runs.items()):
        for site in SITES:
            learned = run.job["method"] in LEARNED
            operating = run.metrics["tests"][site]
            operating = operating.get("operating_metrics", operating.get("filtered", {}))
            rows.append(
                {
                    **{key: run.job[key] for key in IDENTITY},
                    "target": site,
                    "attempt": int(run.attempt.name),
                    "checkpoint_sha256": run.artifact_sha256 if learned else None,
                    "selection_sha256": run.artifact_sha256 if not learned else None,
                    "split_manifest_sha256": run.inputs.sha256,
                    **holdouts[site],
                    **run.tests[site],
                    **{"validation_" + key: val for key, val in run.validation.items()},
                    **{"operating_" + key: operating.get(key) for key in ROW_METRICS},
                    "operating_threshold": run.metrics.get(
                        "selected_threshold", run.metrics.get("threshold")
                    ),
                    "metrics_path": str(run.metrics_path),
                }
            )
    return rows


def _seed_summary(values: dict, seeds: tuple[int, ...], deterministic: bool) -> dict:
    missing = [seed for seed in seeds if seed not in values]
    ordered_values = [values[seed] for seed in seeds if seed in values]
    mean = statistics.fmean(ordered_values) if not missing else None
    sd = None
    interval = None
    if mean is not None and not deterministic and len(ordered_values) >= 2:
        sd = statistics.stdev(ordered_values)
        radius = T_975[len(ordered_values) - 1] * sd / math.sqrt(len(ordered_values))
        interval = [mean - radius, mean + radius]
    return {
        "expected_seeds": list(seeds),
        "seed_values": [{"seed": seed, "value": values[seed]} for seed in seeds if seed in values],
        "missing_seeds": missing,
        "n_complete": len(ordered_values),
        "deterministic": deterministic,
        "mean": mean,
        "sample_sd": sd,
        "approximate_t95": interval,
    }


def _scores(matrix: dict[tuple[str, str], float]) -> dict[tuple[str, str, str], float]:
    scores = {("cell", source, target): value for (source, target), value in matrix.items()}
    for source, target in EDGES:
        if (target, target) in matrix and (source, target) in matrix:
            scores["target_local_minus_incoming", source, target] = (
                matrix[target, target] - matrix[source, target]
            )
    for source, target in itertools.combinations(SITES, 2):
        if (source, target) in matrix and (target, source) in matrix:
            scores["reciprocal_forward_minus_reverse", source, target] = (
                matrix[source, target] - matrix[target, source]
            )
    for name, terms in (
        ("macro_local", [("cell", s, s) for s in SITES]),
        ("macro_transfer", [("cell", s, t) for s, t in EDGES]),
        (
            "macro_target_local_minus_incoming",
            [("target_local_minus_incoming", s, t) for s, t in EDGES],
        ),
    ):
        if all(term in scores for term in terms):
            scores[name, "", ""] = statistics.fmean(scores[term] for term in terms)
    return scores


SCORE_KEYS = tuple(_scores(dict.fromkeys(itertools.product(SITES, SITES), 0.0)))


def _statistics(runs: dict[Signature, Run], adjacency: bool) -> list[dict]:
    summaries = []
    methods = ADJACENCY if adjacency else (*LEARNED, *CLASSICAL)
    for method, metric in itertools.product(methods, AP_METRICS):
        deterministic = method in CLASSICAL
        seeds = (0,) if deterministic else tuple(range(3 if adjacency else 5))
        per_seed = {}
        for seed in seeds:
            buffered = _scores(_matrix(runs, "core", method, seed, metric))
            if adjacency:
                adjacent = _scores(_matrix(runs, "adjacency", method, seed, metric))
                per_seed[seed] = {
                    key: adjacent[key] - buffered[key] for key in adjacent.keys() & buffered.keys()
                }
            else:
                per_seed[seed] = buffered
        for key in SCORE_KEYS:
            statistic, source, target = key
            values = {seed: results[key] for seed, results in per_seed.items() if key in results}
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "statistic": statistic,
                    "source": source or None,
                    "target": target or None,
                    **_seed_summary(values, seeds, deterministic),
                }
            )
    return summaries


def _matrix(runs: dict[Signature, Run], stage: str, method: str, seed: int, metric: str) -> dict:
    cells = {}
    for source in SITES:
        if (run := runs.get((stage, method, source, seed))) is not None:
            cells.update({(source, target): run.value(target, metric) for target in SITES})
    return cells


def _prediction_path(run: Run, target: str) -> Path:
    payload = run.metrics["tests"][target]
    stored = (
        payload.get("raw_path")
        or payload.get("raw_predictions_path")
        or run.metrics.get("paths", {})
        .get("subsets", {})
        .get(f"test_{target}", {})
        .get("predictions")
    )
    if stored:
        # Worker originals may be on a different host. Collectors retain run filenames.
        return run.attempt / Path(stored).name
    return run.attempt / f"{target}_raw_predictions.json"


def _subset_coco(coco: dict, selected: set[int]) -> dict:
    return {
        **coco,
        "images": [im for im in coco["images"] if im["id"] in selected],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in selected],
    }


def _paired_ranges(base: dict, deletions: list[dict]) -> dict:
    return {
        metric: {
            "full_holdout": base[metric],
            "min": min(row[metric] for row in deletions),
            "max": max(row[metric] for row in deletions),
            "by_deleted_block": [row[metric] for row in deletions],
        }
        for metric in AP_METRICS
    }


def _block_deletion(runs: dict[Signature, Run], blocks_by_site: dict) -> dict:
    from .evaluation import evaluate_coco

    if not runs or not blocks_by_site or set(blocks_by_site) - set(SITES):
        raise ValueError("block deletion requires completed primary runs and declared site blocks")
    results = {}
    baseline_inputs = next(iter(runs.values())).inputs
    for site, blocks in sorted(blocks_by_site.items()):
        reference = baseline_inputs.manifest["sites"][site]["partitions"]["test"]
        all_ids = set(reference)
        flat = [image_id for block in blocks for image_id in block]
        if len(blocks) < 2 or not all(blocks) or set(flat) != all_ids or len(flat) != len(all_ids):
            raise ValueError(f"blocks must be a disjoint partition of the {site} holdout")
        gt = _subset_coco(baseline_inputs.annotations, all_ids)
        subsets = [_subset_coco(gt, all_ids - set(block)) for block in blocks]
        if any(not subset["annotations"] for subset in subsets):
            raise ValueError(f"deleting one {site} block leaves no reference objects")
        rescored = {}
        scores = []
        gaps = []
        for key, run in sorted(runs.items()):
            predictions = _load(_prediction_path(run, site))
            original = evaluate_coco(gt, predictions)
            if any(
                not math.isclose(original[m], run.value(site, m), rel_tol=0, abs_tol=1e-9)
                for m in ROW_METRICS
            ):
                raise ValueError(f"raw prediction replay differs for {run.job['job_id']} on {site}")
            deletions = [
                evaluate_coco(subset, [p for p in predictions if p["image_id"] not in removed])
                for subset, removed in zip(subsets, map(set, blocks))
            ]
            rescored[key] = deletions
            scores.append(
                {
                    "job_id": run.job["job_id"],
                    "stage": run.job["stage"],
                    "method": run.job["method"],
                    "source": run.job["source"],
                    "seed": run.job["seed"],
                    "ap_ranges": _paired_ranges(original, deletions),
                }
            )
        for key, incoming in sorted(runs.items()):
            stage, method, source, seed = key
            local_key = (stage, method, site, seed)
            if source == site or local_key not in runs:
                continue
            local = runs[local_key]
            base = {
                metric: local.value(site, metric) - incoming.value(site, metric)
                for metric in AP_METRICS
            }
            deletion_gaps = [
                {metric: local_score[metric] - incoming_score[metric] for metric in AP_METRICS}
                for local_score, incoming_score in zip(rescored[local_key], rescored[key])
            ]
            gaps.append(
                {
                    "stage": stage,
                    "method": method,
                    "source": source,
                    "seed": seed,
                    "ap_ranges": _paired_ranges(base, deletion_gaps),
                }
            )
        results[site] = {
            "blocks": blocks,
            "test_cells": scores,
            "target_local_minus_incoming": gaps,
        }
    return results


def summarize(
    jobs: list[dict],
    runs_dir: Path,
    *,
    split_manifest: Path | None = None,
    annotations: Path | None = None,
    blocks: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Build a report and full per-test rows without reducing incomplete seed groups."""
    runs, status, undeclared = _collect(jobs, runs_dir, split_manifest, annotations)
    holdouts = _verify_matches(runs)
    rows = _full_rows(runs, holdouts)
    counts = {}
    for name, methods, stage, planned in (
        ("learned_core", LEARNED, "core", 60),
        ("classical_core", CLASSICAL, "core", 15),
        ("learned_adjacency", ADJACENCY, "adjacency", 18),
    ):
        complete = sum(key[0] == stage and key[1] in methods for key in runs)
        counts[name] = {
            "expected_jobs": planned,
            "complete_jobs": complete,
            "expected_test_evaluations": planned * len(SITES),
            "complete_test_evaluations": complete * len(SITES),
        }
    diagnostics = [job for job in status if job["stage"] == "diagnostic"]
    counts["source_only_diagnostics"] = {
        "declared_jobs": len(diagnostics),
        "complete_jobs": sum(job["state"] == "complete" for job in diagnostics),
    }
    checkpoints = {}
    for key, run in sorted(runs.items()):
        if key[1] in LEARNED:
            checkpoints[run.job["job_id"]] = {
                "sha256": run.artifact_sha256,
                "evaluated_targets": list(run.tests),
            }
    missing = [job for job in status if job["state"] != "complete"]
    result = {
        "schema_version": 1,
        "protocol": "three fixed destination holdouts; raw ranked prediction metrics",
        "estimates": {
            "scores": "COCO AP/AP50/AP75 on a 0–1 scale.",
            "macros": "Equal arithmetic weights across three local sites or six directed edges per seed.",
            "reciprocal": "Source→target minus target→source, on their respective fixed destination holdouts.",
            "adjacency": "Adjacent minus buffered, paired by method, source and seed 0–2.",
            "interval": "Approximate two-sided 95% Student t interval conditional on optimization seeds, the three fixed corpora and the declared split.",
            "missing_seeds": "All planned seeds must be available for a group mean, SD or interval; available seed values remain visible.",
            "block_deletion": "Paired deletion ranges describe test-composition sensitivity, with no sampling interval.",
        },
        "counts": counts,
        "holdouts": holdouts,
        "job_status_counts": dict(Counter(job["state"] for job in status)),
        "missing_jobs": missing,
        "undeclared_primary_jobs": [
            dict(zip(("stage", "method", "source", "seed"), signature))
            for signature in sorted(undeclared)
        ],
        "learned_checkpoints": checkpoints,
        "unique_learned_checkpoint_hashes": len({item["sha256"] for item in checkpoints.values()}),
        "core": _statistics(runs, adjacency=False),
        "adjacency": _statistics(runs, adjacency=True),
    }
    if blocks is not None:
        result["block_deletion"] = _block_deletion(runs, blocks)
    return result, rows


ROW_FIELDS = (
    *IDENTITY,
    "target",
    "attempt",
    "checkpoint_sha256",
    "selection_sha256",
    "split_manifest_sha256",
    "holdout_sha256",
    "frames",
    "objects",
    *ROW_METRICS,
    *("validation_" + key for key in ROW_METRICS),
    *("operating_" + key for key in ROW_METRICS),
    "operating_threshold",
    "metrics_path",
)


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(val, ensure_ascii=False) if isinstance(val, list) else val
                for key, val in row.items()
            }
            for row in rows
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", type=Path, required=True, help="JSON job list or object with jobs"
    )
    parser.add_argument(
        "--runs-dir", type=Path, required=True, help="Collected job/attempt directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, help="Relocated worker split manifest")
    parser.add_argument("--annotations", type=Path, help="Relocated worker reference COCO JSON")
    parser.add_argument(
        "--blocks", type=Path, help="JSON mapping of sites to disjoint holdout ID blocks"
    )
    parser.add_argument(
        "--require-complete", action="store_true", help="Exit 1 after writing an incomplete report"
    )
    args = parser.parse_args()

    jobs = _load(args.jobs)
    if isinstance(jobs, dict):
        jobs = jobs["jobs"]
    summary, rows = summarize(
        jobs,
        args.runs_dir,
        split_manifest=args.split_manifest,
        annotations=args.annotations,
        blocks=_load(args.blocks) if args.blocks else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "test_rows.csv", rows, ROW_FIELDS)
    aggregates = [
        {"condition": group, **row} for group in ("core", "adjacency") for row in summary[group]
    ]
    _write_csv(args.output_dir / "estimates.csv", aggregates, aggregates[0])
    missing = len(summary["missing_jobs"]) + len(summary["undeclared_primary_jobs"])
    print(f"Wrote {len(rows)} test rows and {len(aggregates)} estimates to {args.output_dir}.")
    print(f"Incomplete declared or planned jobs: {missing}.")
    if args.require_complete and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
