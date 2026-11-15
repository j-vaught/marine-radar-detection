"""Finalize a complete revision archive using collected JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from . import annotation_sensitivity as geometry
from . import evaluation, protocol
from . import summarize as summary

ENVELOPE = "fixed_source_envelope"
METHODS = (*summary.LEARNED, *summary.CLASSICAL)
RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")
JOB_RECIPE_FIELDS = (
    "eval_targets",
    "initialization",
    "epochs",
    "patience",
    "overfit",
    "train_limit",
    "config_overrides",
)
GEOMETRY_VALUES = (
    "ap_value",
    "ap_change_from_baseline",
    "otsu_minus_method",
    "otsu_gap_change_from_baseline",
)
OUTPUT_FILES = (
    "summary.json",
    "test_rows.csv",
    "estimates.csv",
    "reference_geometry.json",
    "reference_geometry_seed_rows.csv",
    "reference_geometry_estimates.csv",
    "learning_curves.json",
)


@dataclass
class CollectedRun:
    job: dict
    completion: dict
    folder: Path

    @property
    def identifier(self) -> str:
        return self.job["job_id"]

    @cached_property
    def metrics(self) -> dict:
        payload = (self.folder / "metrics.json").read_bytes()
        if summary._digest(payload) != self.completion["metrics_sha256"]:
            raise ValueError(f"{self.identifier}: collected metrics changed after preflight")
        metrics = json.loads(payload)
        if metrics["job_id"] != self.identifier:
            raise ValueError(f"{self.identifier}: collected metrics refer to a different job")
        return metrics


def _atomic_json(path: Path, value: dict) -> None:
    with TemporaryDirectory(dir=path.parent) as staging:
        pending = Path(staging) / path.name
        protocol.save_json(pending, value)
        os.replace(pending, path)


def _preflight(jobs: list[dict], runs_dir: Path) -> tuple[dict, dict[str, CollectedRun]]:
    """Inspect receipts and metric bytes before touching reference or prediction data."""
    ready = {}
    status = []
    ids = set()
    primary = []
    for planned in jobs:
        job_id = planned["job_id"]
        if not RUN_ID.fullmatch(job_id) or job_id in ids or job_id in {".", ".."}:
            raise ValueError(f"Invalid or duplicate job ID: {job_id!r}")
        ids.add(job_id)
        if planned["stage"] != "diagnostic":
            primary.append(summary._signature(planned))
        attempts = summary._attempts(runs_dir, job_id)
        item = {"job_id": job_id, "stage": planned["stage"]}
        try:
            completion = summary._completed_attempt(runs_dir, planned, attempts)
            if completion is None:
                if attempts:
                    state = attempts[-1]["state"]
                    item.update(
                        state="waiting_for_evaluation_repair" if state == "succeeded" else state,
                        attempts=attempts,
                    )
                else:
                    item["state"] = "uncollected"
            else:
                folder = runs_dir / job_id / str(completion["attempt"])
                metrics_file = folder / "metrics.json"
                if protocol.sha256_file(metrics_file) != completion["metrics_sha256"]:
                    raise ValueError("Collected metrics do not match the worker receipt SHA-256")
                actual = summary._load(folder / "job.json")
                summary._job_identity(actual, planned)
                if any(actual.get(key) != planned.get(key) for key in JOB_RECIPE_FIELDS):
                    raise ValueError("Collected job recipe differs from the declared configuration")
                if (
                    actual["attempt_nonce"] != completion["nonce"]
                    or actual["protocol_sha256"] != completion["manifest_sha256"]
                    or actual.get("code_commit") != completion.get("code_commit")
                ):
                    raise ValueError("Worker receipt differs from the collected job metadata")
                ledger_root = runs_dir / ".ledger"
                ledger = ledger_root / job_id / f"{completion['attempt']}.json"
                if ledger_root.is_dir() and not ledger.is_file():
                    raise FileNotFoundError("Completed attempt is waiting for its archive ledger")
                ready[job_id] = CollectedRun(actual, completion, folder)
                item["state"] = "ready"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item.update(state="invalid_or_uncollected", error=str(exc))
        status.append(item)

    signature_counts = Counter(primary)
    expected = summary._planned_signatures()
    missing = sorted(expected - signature_counts.keys())
    unexpected = sorted(signature_counts.keys() - expected)
    duplicates = sorted(key for key, count in signature_counts.items() if count != 1)
    issues = [item for item in status if item["state"] != "ready"]
    return (
        {
            "state": "incomplete" if issues or missing or unexpected or duplicates else "ready",
            "jobs_by_stage": {
                stage: {
                    "declared": sum(item["stage"] == stage for item in status),
                    "ready": sum(
                        item["stage"] == stage and item["state"] == "ready" for item in status
                    ),
                }
                for stage in ("diagnostic", "core", "adjacency")
            },
            "job_state_counts": dict(Counter(item["state"] for item in status)),
            "incomplete_jobs": issues,
            "missing_primary_signatures": missing,
            "unexpected_primary_signatures": unexpected,
            "duplicate_primary_signatures": duplicates,
        },
        ready,
    )


def _artifact_path(run: CollectedRun, remote_path: str) -> Path:
    """Find a collected artifact using its relative worker path or its basename."""
    path = Path(remote_path)
    original_root = Path(run.job["out_dir"])
    if path.is_absolute():
        path = path.relative_to(original_root)
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Artifact path escapes the attempt: {remote_path}")
    original_layout = run.folder / path
    if original_layout.is_file():
        return original_layout
    return run.folder / path.name


def _test_artifacts(run: CollectedRun, target: str) -> tuple[str, str | None]:
    payload = run.metrics["tests"][target]
    if run.job["method"] in summary.CLASSICAL:
        paths = run.metrics["paths"]["subsets"][f"test_{target}"]
        return paths["annotations_sha256"], paths["predictions"]
    return (
        payload.get("gt_sha256", payload.get("ground_truth_sha256")),
        payload.get("raw_predictions_path", payload.get("raw_path")),
    )


def _serialized_digest(value: dict) -> str:
    """Hash the canonical serialization emitted by protocol.save_json."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return summary._digest(payload.encode())


def _checkpoint_digest(run: CollectedRun) -> str:
    metrics = run.metrics
    return summary._sha(
        metrics.get(
            "checkpoint_sha256", metrics.get("best_sha256", metrics.get("best_weights_sha256"))
        ),
        "selected checkpoint",
    )


def _curve_record(run: CollectedRun) -> tuple[dict, Path | None]:
    metrics, job = run.metrics, run.job
    location = metrics.get("training_log") or "history.json"
    local = _artifact_path(run, location)
    present = local.is_file()
    rows = None
    if present:
        if local.suffix == ".csv":
            with local.open(newline="") as stream:
                rows = sum(1 for _ in csv.DictReader(stream))
        else:
            rows = len(summary._load(local))
    parent = metrics.get("training_parent")
    training = parent if parent is not None else run.completion
    training_runtime = training.get("runtime_provenance") or {}
    training_gpu = (training_runtime.get("gpu") or {}).get("uuid")
    if training_gpu is None:
        visible_gpu = training_runtime.get("cuda_visible_devices")
        if isinstance(visible_gpu, str) and visible_gpu.startswith("GPU-"):
            training_gpu = visible_gpu
        elif parent is None:
            training_gpu = run.completion.get("observed_gpu_uuid")
    record = {
        **{key: job[key] for key in summary.IDENTITY},
        "requested_epochs": job.get("epochs"),
        "requested_patience": job.get("patience"),
        "initialization": job.get("initialization"),
        "best_epoch_as_recorded": metrics.get("best_epoch"),
        "best_epoch_base": 0 if job["method"] in {"yolo", "rtdetr"} else 1,
        "best_validation_loss": metrics.get("best_val_loss"),
        "best_validation_fitness": metrics.get("best_fitness"),
        "training_seconds": metrics.get("training_seconds", metrics.get("train_seconds")),
        "peak_vram_bytes": metrics.get("peak_vram_bytes"),
        "validation": metrics["validation"],
        "curve_worker_path": location,
        "curve_collected": present,
        "curve_local_path": str(local) if present else None,
        "epoch_rows": rows,
        "checkpoint_sha256": _checkpoint_digest(run),
        "checkpoint_verified_by_worker": bool(metrics.get("tests")),
        "code_commit": training.get("code_commit"),
        "gpu_uuid": training_gpu,
        "training_runtime_provenance": training.get("runtime_provenance"),
        "evaluation_attempt": run.completion["attempt"],
        "evaluation_code_commit": run.completion.get("code_commit"),
        "evaluation_device": metrics.get("evaluation_device", job.get("device")),
        "evaluation_runtime_provenance": run.completion.get("runtime_provenance"),
        "evaluation_recipe_version": metrics.get("evaluation_recipe_version"),
    }
    return record, local if present else None


def _inventory(
    runs: dict[str, CollectedRun],
    manifest_path: Path,
    annotation_path: Path,
    *,
    include_adjacency_predictions: bool = False,
) -> tuple[dict, dict, dict, list[dict]]:
    manifest_sha = protocol.sha256_file(manifest_path)
    annotation_sha = protocol.sha256_file(annotation_path)
    manifest = summary._load(manifest_path)
    if manifest["annotation_sha256"] != annotation_sha:
        raise ValueError("Released annotation SHA-256 differs from the frozen split")
    annotations = summary._load(annotation_path)
    references = {
        target: summary._subset_coco(
            annotations, set(manifest["sites"][target]["partitions"]["test"])
        )
        for target in summary.SITES
    }
    ref_hashes = {target: _serialized_digest(coco) for target, coco in references.items()}
    file_hashes = {}
    prediction_paths = {}
    curves = []
    for job_id, run in runs.items():
        recorded_manifest = run.metrics.get(
            "manifest_sha256", run.metrics.get("paths", {}).get("manifest_sha256")
        )
        if (
            run.completion["manifest_sha256"] != manifest_sha
            or recorded_manifest != manifest_sha
            or run.metrics["annotation_sha256"] != annotation_sha
        ):
            raise ValueError(f"{job_id}: source protocol or annotation digest mismatch")
        file_hashes[f"runs/{job_id}/job.json"] = protocol.sha256_file(run.folder / "job.json")
        file_hashes[f"runs/{job_id}/completion.json"] = protocol.sha256_file(
            run.folder / "completion.json"
        )
        file_hashes[f"runs/{job_id}/metrics.json"] = run.completion["metrics_sha256"]
        if run.job["method"] in summary.LEARNED:
            curve, curve_path = _curve_record(run)
            if curve_path is not None:
                digest = protocol.sha256_file(curve_path)
                curve["curve_sha256"] = digest
                file_hashes[f"runs/{job_id}/epoch_curve"] = digest
            curves.append(curve)
        used_artifacts = {
            name: file_hashes[f"runs/{job_id}/{name}"]
            for name in ("job.json", "completion.json", "metrics.json")
        }
        for target in run.job["eval_targets"]:
            reference_sha, prediction = _test_artifacts(run, target)
            if reference_sha != ref_hashes[target]:
                raise ValueError(
                    f"{job_id}/{target}: evaluated COCO SHA-256 differs from frozen target"
                )
            if run.job["method"] in summary.LEARNED:
                actual_checkpoint = run.metrics["tests"][target]["checkpoint_sha256"]
                if actual_checkpoint != _checkpoint_digest(run):
                    raise ValueError(
                        f"{job_id}/{target}: checkpoint metadata differs from selected model"
                    )
            if run.job["stage"] == "core" or include_adjacency_predictions:
                if not prediction:
                    raise ValueError(f"{job_id}/{target}: missing raw prediction path")
                path = _artifact_path(run, prediction)
                if not path.is_file():
                    raise FileNotFoundError(f"{job_id}/{target}: collect raw predictions at {path}")
                digest = protocol.sha256_file(path)
                file_hashes[f"runs/{job_id}/{target}/raw_predictions"] = digest
                prediction_paths[job_id, target] = (path, digest)
                used_artifacts[path.relative_to(run.folder).as_posix()] = digest
        ledger_root = run.folder.parents[1] / ".ledger"
        if ledger_root.is_dir():
            ledger = ledger_root / job_id / f"{run.completion['attempt']}.json"
            archived = summary._load(ledger)["files"]
            if any(archived[name]["sha256"] != digest for name, digest in used_artifacts.items()):
                raise ValueError(f"{job_id}: artifact content differs from verified archive ledger")
            file_hashes[f"runs/{job_id}/archive_ledger"] = protocol.sha256_file(ledger)
    file_hashes["split_manifest"] = manifest_sha
    file_hashes["released_annotations"] = annotation_sha
    return manifest, file_hashes, prediction_paths, curves


def _pair_rankings(results: dict) -> dict:
    ranks = {}
    for view, by_method in results.items():
        cohorts = {}
        for seed in range(5):
            cohort = {
                method: by_method[method, 0 if method in summary.CLASSICAL else seed]["coco"]
                for method in METHODS
            }
            cohorts[seed] = {
                metric: {
                    method: 1 + sum(other[metric] > values[metric] for other in cohort.values())
                    for method, values in cohort.items()
                }
                for metric in summary.AP_METRICS
            }
        ranks[view] = cohorts
    return {
        view: {
            seed: {
                metric: {
                    method: {
                        "rank": rank,
                        "change_from_baseline": rank - ranks["baseline"][seed][metric][method],
                    }
                    for method, rank in by_method.items()
                }
                for metric, by_method in by_metric.items()
            }
            for seed, by_metric in cohorts.items()
        }
        for view, cohorts in ranks.items()
    }


def _verify_geometry_replay(saved: dict, replayed: dict, job_id: str, target: str) -> None:
    if any(
        not math.isclose(saved[metric], replayed[metric], rel_tol=0, abs_tol=1e-9)
        for metric in summary.ROW_METRICS
    ):
        raise ValueError(f"{job_id}/{target}: baseline prediction replay differs")


def _geometry_analysis(
    runs: dict[str, CollectedRun],
    manifest: dict,
    ann_path: Path,
    prediction_paths: dict,
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    pairs = []
    seed_rows = []
    for source in summary.SITES:
        source_runs = {
            (run.job["method"], run.job["seed"]): run
            for run in runs.values()
            if run.job["stage"] == "core" and run.job["source"] == source
        }
        for target in summary.SITES:
            refs, _, envelope = geometry._load_references(ann_path, manifest, source, target)
            predictions = {}
            labels = {}
            for key, run in source_runs.items():
                method, seed = key
                label = f"{method}/{seed}"
                path, digest = prediction_paths[run.identifier, target]
                payload = path.read_bytes()
                if summary._digest(payload) != digest:
                    raise ValueError(
                        f"{run.identifier}/{target}: prediction changed during finalization"
                    )
                predictions[label] = json.loads(payload)
                labels[label] = key
            views = geometry.build_reference_views(refs, envelope["width_height_px"])
            scored = geometry.score_reference_views(views, predictions, cache_dir=cache_dir)
            results = {
                view: {labels[label]: values for label, values in result["methods"].items()}
                for view, result in scored.items()
            }
            baseline = results["baseline"]
            for key, run in source_runs.items():
                saved = summary._raw_metrics(run.metrics["tests"][target])
                _verify_geometry_replay(saved, baseline[key]["coco"], run.identifier, target)
            for view, by_method in results.items():
                for (method, seed), values in by_method.items():
                    otsu = by_method["otsu", 0]["coco"]
                    for metric in summary.AP_METRICS:
                        gap = None if method == "otsu" else otsu[metric] - values["coco"][metric]
                        baseline_gap = (
                            baseline["otsu", 0]["coco"][metric]
                            - baseline[method, seed]["coco"][metric]
                        )
                        seed_rows.append(
                            {
                                "source": source,
                                "target": target,
                                "method": method,
                                "seed": seed,
                                "reference": view,
                                "metric": metric,
                                "ap_value": values["coco"][metric],
                                "ap_change_from_baseline": values["ap_change_from_baseline"][
                                    metric
                                ],
                                "otsu_minus_method": gap,
                                "otsu_gap_change_from_baseline": None
                                if gap is None
                                else gap - baseline_gap,
                            }
                        )
            pairs.append(
                {
                    "source": source,
                    "target": target,
                    "target_test_image_ids": [im["id"] for im in refs["images"]],
                    "fixed_source_envelope": envelope,
                    "clipping": {view: result["clipping"] for view, result in scored.items()},
                    "method_ranks_by_learned_seed_cohort": _pair_rankings(results),
                    "center_agreement": {
                        view: {
                            f"{method}/{seed}": result["center_agreement"]
                            for (method, seed), result in module.items()
                        }
                        for view, module in results.items()
                    },
                }
            )
    return pairs, seed_rows


def _geometry_estimates(rows: list[dict]) -> list[dict]:
    # Keep the exact cohort dimensions explicit for both fixed-envelope and common views.
    by_pair: dict[tuple, dict[int, float]] = defaultdict(dict)
    by_scope: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        source, target, method, ref, metric, seed = (
            row[k] for k in ("source", "target", "method", "reference", "metric", "seed")
        )
        for name in GEOMETRY_VALUES:
            value = row[name]
            if value is None:
                continue
            by_pair["pair", source, target, method, ref, metric, name][seed] = value
            if ref != ENVELOPE:
                scope = "macro_local" if source == target else "macro_transfer"
                by_scope[scope, method, ref, metric, name, seed].append(value)
    for (scope, method, ref, metric, name, seed), values in by_scope.items():
        if len(values) != (3 if scope == "macro_local" else 6):
            raise ValueError(f"Incomplete geometry support for {scope}/{method}/{ref}/{seed}")
        by_pair[scope, "", "", method, ref, metric, name][seed] = statistics.fmean(values)
    estimates = []
    for key, values in sorted(by_pair.items()):
        scope, source, target, method, ref, metric, name = key
        classical = method in summary.CLASSICAL
        seeds = (0,) if classical else tuple(range(5))
        record = summary._seed_summary(values, seeds, classical)
        if record["missing_seeds"]:
            raise ValueError(f"Incomplete geometry seed cohort for {key}")
        estimates.append(
            {
                "scope": scope,
                "source": source or None,
                "target": target or None,
                "method": method,
                "reference": ref,
                "metric": metric,
                "quantity": name,
                **record,
            }
        )
    return estimates


def _run_strict_summary(
    directory: Path,
    jobs: Path,
    completed: Path,
    split_manifest: Path,
    annotations: Path,
    blocks: Path | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.revision.summarize",
        "--jobs",
        str(jobs),
        "--runs-dir",
        str(completed),
        "--output-dir",
        str(directory),
        "--split-manifest",
        str(split_manifest),
        "--annotations",
        str(annotations),
        "--require-complete",
    ]
    if blocks is not None:
        command.extend(["--blocks", str(blocks)])
    subprocess.run(command, check=True)


def _analysis_identity(jobs_sha: str, inventory: dict) -> dict:
    return {
        "schema_version": 1,
        "inputs": {**inventory, "job_manifest": jobs_sha},
        "implementations": {
            Path(module.__file__).name: protocol.sha256_file(module.__file__)
            for module in (summary, geometry, evaluation, protocol)
        }
        | {Path(__file__).name: protocol.sha256_file(__file__)},
        "versions": {package: version(package) for package in ("numpy", "scipy", "pycocotools")},
    }


def _cached_report(output: Path, identity: str) -> bool:
    marker = output / "finalized.json"
    try:
        record = summary._load(marker)
        return (
            record["input_content_sha256"] == identity
            and set(record["outputs"]) == set(OUTPUT_FILES)
            and all(
                protocol.sha256_file(output / filename) == record["outputs"][filename]
                for filename in OUTPUT_FILES
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _validate_immutable_inputs(
    runs: dict[str, CollectedRun],
    jobs: Path,
    manifest: Path,
    annotations: Path,
    identity: dict,
    blocks: Path | None,
    prediction_paths: dict,
) -> None:
    sources = identity["inputs"]
    checks = [
        (jobs, sources["job_manifest"]),
        (manifest, sources["split_manifest"]),
        (annotations, sources["released_annotations"]),
    ]
    if blocks is not None:
        checks.append((blocks, sources["index_blocks"]))
    checks.extend(prediction_paths.values())
    for job_id, run in runs.items():
        for name in ("job.json", "completion.json", "metrics.json"):
            checks.append((run.folder / name, sources[f"runs/{job_id}/{name}"]))
        ledger_key = f"runs/{job_id}/archive_ledger"
        if ledger_key in sources:
            ledger = (
                run.folder.parents[1] / ".ledger" / job_id / f"{run.completion['attempt']}.json"
            )
            checks.append((ledger, sources[ledger_key]))
    for path, expected in checks:
        if protocol.sha256_file(path) != expected:
            raise ValueError(f"Finalization input changed during analysis: {path}")


def _write_final(
    output: Path,
    jobs: Path,
    runs_dir: Path,
    split: Path,
    annotations: Path,
    runs: dict[str, CollectedRun],
    manifest: dict,
    paths: dict,
    curves: list[dict],
    identity: dict,
    digest: str,
    cache_dir: Path | None,
    blocks: Path | None,
) -> None:
    with TemporaryDirectory(dir=output) as work:
        staging = Path(work)
        _run_strict_summary(staging, jobs, runs_dir, split, annotations, blocks)
        pairs, seed_rows = _geometry_analysis(runs, manifest, annotations, paths, cache_dir)
        estimates = _geometry_estimates(seed_rows)
        protocol.save_json(
            staging / "reference_geometry.json",
            {
                "schema_version": 1,
                "input_content_sha256": digest,
                "evaluated_source_target_pairs": len(pairs),
                "source_envelope_scope": "The source-training median width and height define "
                "one envelope per source. Envelope comparisons are within source and target.",
                "reference_inventory": "All views inherit the released object identities.",
                "rank_scope": "Classical methods are deterministic comparators in each "
                "matched learned seed cohort.",
                "conditional_interval": "Approximate Student t intervals describe optimization "
                "seed variation conditional on the three sites and fixed holdouts.",
                "pairs": pairs,
                "estimates": estimates,
            },
        )
        summary._write_csv(
            staging / "reference_geometry_seed_rows.csv", seed_rows, tuple(seed_rows[0])
        )
        summary._write_csv(
            staging / "reference_geometry_estimates.csv", estimates, tuple(estimates[0])
        )
        protocol.save_json(
            staging / "learning_curves.json",
            {
                "schema_version": 1,
                "input_content_sha256": digest,
                "scope": "Training and source validation only.",
                "collected_epoch_logs": sum(record["curve_collected"] for record in curves),
                "learned_job_records": curves,
            },
        )
        _validate_immutable_inputs(runs, jobs, split, annotations, identity, blocks, paths)
        output_hashes = {name: protocol.sha256_file(staging / name) for name in OUTPUT_FILES}
        for name in OUTPUT_FILES:
            os.replace(staging / name, output / name)
        _atomic_json(
            output / "finalized.json",
            {"input_content_sha256": digest, **identity, "outputs": output_hashes},
        )


def finalize(
    *,
    jobs: Path,
    runs_dir: Path,
    split_manifest: Path,
    annotations: Path,
    output_dir: Path,
    cache_dir: Path | None = None,
    blocks: Path | None = None,
) -> bool:
    """Return false for an incomplete archive, with progress and no completion marker."""
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    marker = output / "finalized.json"
    with (output / ".finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        progress = {"state": "preflight"}
        try:
            jobs_bytes = jobs.read_bytes()
            planned = json.loads(jobs_bytes)
            if isinstance(planned, dict):
                planned = planned["jobs"]
            progress, runs = _preflight(planned, runs_dir)
            if progress["state"] == "incomplete":
                marker.unlink(missing_ok=True)
                _atomic_json(progress_path, progress)
                return False
            progress["state"] = "checking_collected_inputs"
            _atomic_json(progress_path, progress)
            manifest, inventory, paths, curves = _inventory(
                runs, split_manifest, annotations, include_adjacency_predictions=blocks is not None
            )
            if blocks is not None:
                inventory["index_blocks"] = protocol.sha256_file(blocks)
            identity = _analysis_identity(summary._digest(jobs_bytes), inventory)
            digest = summary._digest_json(identity)
            progress["input_content_sha256"] = digest
            if not _cached_report(output, digest):
                marker.unlink(missing_ok=True)
                progress["state"] = "finalizing"
                _atomic_json(progress_path, progress)
                _write_final(
                    output,
                    jobs,
                    runs_dir,
                    split_manifest,
                    annotations,
                    runs,
                    manifest,
                    paths,
                    curves,
                    identity,
                    digest,
                    cache_dir,
                    blocks,
                )
            progress["state"] = "complete"
            _atomic_json(progress_path, progress)
        except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
            marker.unlink(missing_ok=True)
            _atomic_json(progress_path, {**progress, "state": "failed", "error": str(exc)})
            raise
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, required=True, help="Collected job/attempt directory"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Reuse precomputed reference scores")
    parser.add_argument(
        "--blocks", type=Path, help="Fixed target frame-index blocks for paired deletion"
    )
    args = parser.parse_args()
    if not finalize(**vars(args)):
        print(f"Archive incomplete. Progress is in {args.output_dir / 'progress.json'}.")
        raise SystemExit(1)
    print(f"Verified final report in {args.output_dir}.")


if __name__ == "__main__":
    main()
