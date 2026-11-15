"""Cache independent geometry scores for completed, verified core attempts."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

from . import annotation_sensitivity as geometry
from . import finalize, protocol
from . import summarize as summary


@dataclass
class Cell:
    job_id: str
    source: str
    target: str
    prediction: Path
    prediction_sha256: str
    metrics_sha256: str
    baseline: dict


@lru_cache(maxsize=9)
def _views(
    annotation_path: Path, split_path: Path, split_sha: str, source: str, target: str
) -> dict:
    if protocol.sha256_file(split_path) != split_sha:
        raise ValueError("Frozen split changed during cache preparation")
    manifest = summary._load(split_path)
    refs, _, envelope = geometry._load_references(annotation_path, manifest, source, target)
    return geometry.build_reference_views(refs, envelope["width_height_px"])


def _score_cell(
    cell: Cell, annotation_path: Path, split_path: Path, split_sha: str, cache_dir: Path
) -> tuple[int, int]:
    payload = cell.prediction.read_bytes()
    if summary._digest(payload) != cell.prediction_sha256:
        raise ValueError(f"{cell.job_id}/{cell.target}: archived predictions changed")
    predictions = json.loads(payload)
    views = _views(annotation_path, split_path, split_sha, cell.source, cell.target)
    baseline = {"baseline": views["baseline"]}
    result, baseline_hits = geometry.score_model_views(baseline, predictions, cache_dir=cache_dir)
    finalize._verify_geometry_replay(
        cell.baseline, result["baseline"]["coco"], cell.job_id, cell.target
    )
    rest = {name: reference for name, reference in views.items() if name != "baseline"}
    _, other_hits = geometry.score_model_views(rest, predictions, cache_dir=cache_dir)
    return len(views), baseline_hits + other_hits


def _cache_identity(split_sha: str, cell: Cell) -> tuple[str, ...]:
    return (split_sha, cell.job_id, cell.target, cell.prediction_sha256, cell.metrics_sha256)


def _validate_archive_ledger(
    runs_dir: Path, runs: dict[str, finalize.CollectedRun], paths: dict, inventory: dict
) -> None:
    ledger = runs_dir / ".ledger"
    if not ledger.is_dir():
        return
    for job_id, run in runs.items():
        files = summary._load(ledger / job_id / f"{run.completion['attempt']}.json")["files"]
        artifacts = {
            name: inventory[f"runs/{job_id}/{name}"]
            for name in ("job.json", "completion.json", "metrics.json")
        }
        for target in summary.SITES:
            path, digest = paths[job_id, target]
            artifacts[path.relative_to(run.folder).as_posix()] = digest
        if any(files[name]["sha256"] != digest for name, digest in artifacts.items()):
            raise ValueError(f"{job_id}: artifact content differs from verified archive ledger")


def precompute(
    *,
    jobs: Path,
    runs_dir: Path,
    split_manifest: Path,
    annotations: Path,
    cache_dir: Path,
    workers: int,
    previous: set[tuple[str, ...]] | None = None,
) -> dict:
    """Score a snapshot of completed core jobs without combining model scores."""
    if not 1 <= workers <= 12:
        raise ValueError("Use between 1 and 12 geometry workers")
    planned = summary._load(jobs)
    if isinstance(planned, dict):
        planned = planned["jobs"]
    progress, ready = finalize._preflight(planned, runs_dir)
    if any(
        progress[key]
        for key in (
            "missing_primary_signatures",
            "unexpected_primary_signatures",
            "duplicate_primary_signatures",
        )
    ):
        raise ValueError("Precomputation requires the full, unique revision benchmark matrix")
    if any(
        row["condition"] != "buffered" or sorted(row["eval_targets"]) != sorted(summary.SITES)
        for row in planned
        if row["stage"] == "core"
    ):
        raise ValueError("Each core job must declare the three buffered test targets")

    core = {job_id: run for job_id, run in ready.items() if run.job["stage"] == "core"}
    archived = runs_dir / ".ledger"
    receipt_count = len(core)
    if archived.is_dir():
        # The archive promotes the attempt directory before writing its verified ledger.
        core = {
            job_id: run
            for job_id, run in core.items()
            if (archived / job_id / f"{run.completion['attempt']}.json").is_file()
        }
    record = {
        "state": "scanning",
        "core_jobs_ready": len(core),
        "core_waiting_for_archive_ledger": receipt_count - len(core),
        "core_jobs_declared": progress["jobs_by_stage"]["core"]["declared"],
        "verified_cells": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "already_verified_in_process": 0,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "progress.json"
    finalize._atomic_json(progress_path, record)
    if not core:
        record["state"] = "waiting_for_core_artifacts"
        finalize._atomic_json(progress_path, record)
        return record

    seen = previous if previous is not None else set()
    try:
        _, inventory, paths, _ = finalize._inventory(core, split_manifest, annotations)
        _validate_archive_ledger(runs_dir, core, paths, inventory)
        split_sha = inventory["split_manifest"]
        cells = [
            Cell(
                job_id,
                core[job_id].job["source"],
                target,
                path,
                prediction_sha,
                core[job_id].completion["metrics_sha256"],
                summary._raw_metrics(core[job_id].metrics["tests"][target]),
            )
            for (job_id, target), (path, prediction_sha) in paths.items()
        ]
        pending = [cell for cell in cells if _cache_identity(split_sha, cell) not in seen]
        record["already_verified_in_process"] = len(cells) - len(pending)
        record["verified_cells"] = record["already_verified_in_process"]
        record["state"] = "scoring"
        finalize._atomic_json(progress_path, record)
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                names = {
                    pool.submit(
                        _score_cell, cell, annotations, split_manifest, split_sha, cache_dir
                    ): cell
                    for cell in pending
                }
                for done in as_completed(names):
                    total, hits = done.result()
                    seen.add(_cache_identity(split_sha, names[done]))
                    record["verified_cells"] += 1
                    record["cache_hits"] += hits
                    record["cache_misses"] += total - hits
                    finalize._atomic_json(progress_path, record)
        record["state"] = "snapshot_cached"
        finalize._atomic_json(progress_path, record)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        finalize._atomic_json(progress_path, {**record, "state": "failed", "error": str(exc)})
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, required=True, help="Archive job/attempt directory"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=6)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=float, default=60, help="Seconds between archive scans")
    args = parser.parse_args()
    if args.poll <= 0:
        parser.error("--poll must be positive")
    options = vars(args).copy()
    del options["watch"], options["poll"]
    seen: set[tuple[str, ...]] = set()
    while True:
        result = precompute(**options, previous=seen)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not args.watch or result["core_jobs_ready"] == result["core_jobs_declared"]:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
