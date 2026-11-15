"""Derive the separate matched-GPU YOLO sensitivity from archived predictions."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from . import evaluation, finalize, protocol
from . import summarize as summary

TARGETS = summary.SITES
SEEDS = (0, 1, 2)
STAGES = ("core", "adjacency")
METRICS = summary.AP_METRICS
MACROS = ("macro_local", "macro_transfer", "macro_target_local_minus_incoming")


def _verified_json(path: Path, expected_sha256: str):
    payload = path.read_bytes()
    if summary._digest(payload) != expected_sha256:
        raise ValueError(f"Digest differs for archived study artifact {path.name}")
    return json.loads(payload)


def _verify_input_audit(audit: dict, replay: dict, study_sha: str, frame_counts: dict) -> None:
    observed = audit["predictor_observed"]
    if (
        audit["study_sha256"] != study_sha
        or audit["job_id"] != replay["job_id"]
        or audit["gpu_uuid"] != replay["gpu_uuid"]
        or audit["manifest_sha256"] != replay["manifest_sha256"]
        or audit["images_verified"] != sum(frame_counts.values())
        or set(audit["per_target"]) != set(TARGETS)
        or observed["actual_results"] != 1
        or observed["args_batch"] != replay["batch"]
        or observed["last_batch_size"] != replay["batch"]
        or not observed["args_rect"]
        or observed["args_half"]
        or observed["args_imgsz"] != replay["imgsz"]
        or any(
            observed[name] != replay[name]
            for name in ("torch_threads", "cudnn_allow_tf32", "matmul_allow_tf32")
        )
    ):
        raise ValueError(f"{replay['job_id']}: recorded input or predictor audit differs")
    for target, frames in frame_counts.items():
        result = audit["per_target"][target]
        if result["images"] != frames or any(
            result[name]
            for name in (
                "manifest_hash_mismatches",
                "mask_pixel_mismatches",
                "staged_hash_mismatches",
            )
        ):
            raise ValueError(f"{replay['job_id']}/{target}: staged image audit differs")


def _verified_replays(
    studies: Path,
    runs: dict,
    selected: dict,
    inventory: dict,
    paths: dict,
    blocks: dict,
    archive_hashes: dict[str, str],
):
    """Verify all replay identities and score their full/deleted target subsets."""
    replacement = {}
    reports = []
    originals = {run.job["job_id"]: (key, run) for key, run in runs.items()}
    expected = sorted(job_id for job_id in originals if selected[job_id].job.get("evaluation_from"))
    if len(expected) != 4 or {p.parent.name for p in studies.glob("*/study.json")} != set(expected):
        raise ValueError("This sensitivity requires exactly the four repaired core YOLO jobs")
    for job_id in expected:
        key, run = originals[job_id]
        stage, method, source, seed = key
        if stage != "core" or method != "yolo" or run.metrics["evaluation_device"] != "cpu":
            raise ValueError(f"{job_id}: expected a repaired CPU core evaluation")
        study_sha = archive_hashes[f"{job_id}/study.json"]
        replay = _verified_json(studies / job_id / "study.json", study_sha)
        audit_sha = archive_hashes[f"{job_id}/input_verification.json"]
        audit = _verified_json(studies / job_id / "input_verification.json", audit_sha)
        frame_counts = {
            target: len(run.inputs.manifest["sites"][target]["partitions"]["test"])
            for target in TARGETS
        }
        adjacent_key = ("adjacency", "yolo", source, seed)
        archived = selected[job_id]
        adjacent = selected[runs[adjacent_key].job["job_id"]]
        required_model = adjacent.completion["runtime_provenance"]["gpu"]["model"]
        if (
            replay["job_id"] != job_id
            or replay["checkpoint_sha256"] != finalize._checkpoint_digest(archived)
            or replay["primary_metrics_sha256"] != archived.completion["metrics_sha256"]
            or replay["manifest_sha256"] != inventory["split_manifest"]
            or replay["annotation_sha256"] != inventory["released_annotations"]
            or replay["evaluation_recipe_version"] != run.metrics["evaluation_recipe_version"]
            or replay["selected_threshold"] != run.metrics["selected_threshold"]
            or replay["score_floor"] != run.metrics["config"]["postprocessing"]["score_floor"]
            or replay["imgsz"] != run.metrics["config"]["training"]["imgsz"]
            or not replay["float32"]
            or replay["batch"] != 1
            or replay["gpu_model"] != required_model
            or set(replay["targets"]) != set(TARGETS)
        ):
            raise ValueError(f"{job_id}: replay identity or matched GPU model differs")
        if Path(replay["primary_run"]).parts[-2:] != (
            job_id,
            str(archived.completion["attempt"]),
        ):
            raise ValueError(f"{job_id}: replay references a different CPU attempt")
        _verify_input_audit(audit, replay, study_sha, frame_counts)
        for target in TARGETS:
            row = replay["targets"][target]
            gt_sha, _ = finalize._test_artifacts(archived, target)
            cpu_sha = paths[job_id, target][1]
            if row["ground_truth_sha256"] != gt_sha or row["cpu_raw_sha256"] != cpu_sha:
                raise ValueError(f"{job_id}/{target}: replay reference or CPU prediction differs")
            expected_name = f"{target}_gpu_raw.json"
            if Path(row["gpu_raw_path"]).name != expected_name:
                raise ValueError(f"{job_id}/{target}: unexpected replay artifact name")
            raw_sha = archive_hashes[f"{job_id}/{expected_name}"]
            if raw_sha != row["gpu_raw_sha256"]:
                raise ValueError(f"{job_id}/{target}: replay prediction inventory differs")
            ids = set(run.inputs.manifest["sites"][target]["partitions"]["test"])
            gt = summary._subset_coco(run.inputs.annotations, ids)
            predictions = _verified_json(studies / job_id / expected_name, raw_sha)
            baseline = evaluation.evaluate_coco(gt, predictions)
            finalize._verify_geometry_replay(row["gpu_raw_metrics"], baseline, job_id, target)
            deleted = []
            for block in blocks[target]:
                removed = set(block)
                subset = summary._subset_coco(gt, ids - removed)
                pruned = [row for row in predictions if row["image_id"] not in removed]
                deleted.append(evaluation.evaluate_coco(subset, pruned))
            replacement[key, target] = {
                metric: [baseline[metric], *(row[metric] for row in deleted)] for metric in METRICS
            }
        reports.append(
            {
                "job_id": job_id,
                "source": source,
                "seed": seed,
                "primary_attempt": archived.completion["attempt"],
                "paired_adjacency_job_id": adjacent.job["job_id"],
                "checkpoint_sha256": replay["checkpoint_sha256"],
                "primary_metrics_sha256": replay["primary_metrics_sha256"],
                "replay_record_sha256": study_sha,
                "input_verification_sha256": audit_sha,
                "gpu_model": replay["gpu_model"],
                "replay_gpu_uuid": replay["gpu_uuid"],
                "adjacency_gpu_uuid": adjacent.completion["observed_gpu_uuid"],
                "selected_threshold": replay["selected_threshold"],
                "score_floor": replay["score_floor"],
                "imgsz": replay["imgsz"],
                "batch": replay["batch"],
                "float32": replay["float32"],
                "code_commit": replay["code_commit"],
                "runtime": {
                    field: replay[field]
                    for field in (
                        "torch_version",
                        "cuda_version",
                        "ultralytics_version",
                        "torch_threads",
                        "cudnn_allow_tf32",
                        "matmul_allow_tf32",
                    )
                },
                "targets": {
                    target: {
                        field: replay["targets"][target][field]
                        for field in (
                            "ground_truth_sha256",
                            "cpu_raw_sha256",
                            "gpu_raw_sha256",
                        )
                    }
                    for target in TARGETS
                },
            }
        )
    return replacement, reports


def _adjacency_effects(scores: dict, removed_target: str | None, deleted_block: int | None):
    """Compute three paired seed effects, with one matched target deletion if requested."""
    results = {}
    for seed, metric in itertools.product(SEEDS, METRICS):
        arms = {}
        for stage in STAGES:
            cells = {}
            for source, target in itertools.product(TARGETS, repeat=2):
                slot = (
                    deleted_block + 1
                    if target == removed_target and deleted_block is not None
                    else 0
                )
                cells[source, target] = scores[(stage, "yolo", source, seed), target][metric][slot]
            arms[stage] = summary._scores(cells)
        results[seed, metric] = {
            key: arms["adjacency"][key] - arms["core"][key] for key in arms["core"]
        }
    return results


def _compare(primary: dict, gpu: dict, target: str | None, block: int | None) -> list[dict]:
    original = _adjacency_effects(primary, target, block)
    altered = _adjacency_effects(gpu, target, block)
    rows = []
    for metric, key in itertools.product(METRICS, summary.SCORE_KEYS):
        old = {seed: original[seed, metric][key] for seed in SEEDS}
        new = {seed: altered[seed, metric][key] for seed in SEEDS}
        change = {seed: new[seed] - old[seed] for seed in SEEDS}
        rows.append(
            {
                "deleted_target": target,
                "deleted_block_index_zero_based": block,
                "metric": metric,
                "statistic": key[0],
                "source": key[1] or None,
                "target": key[2] or None,
                "primary_effect": summary._seed_summary(old, SEEDS, False),
                "gpu_matched_effect": summary._seed_summary(new, SEEDS, False),
                "gpu_matched_minus_primary": summary._seed_summary(change, SEEDS, False),
            }
        )
    return rows


def _range(rows: list[dict], key: str) -> list[float]:
    values = [row[key]["mean"] for row in rows]
    return [min(values), max(values)]


def study(
    jobs: Path,
    archive: Path,
    split: Path,
    ann: Path,
    blocks_path: Path,
    blocks_spec: Path,
    studies: Path,
):
    planned = summary._load(jobs)
    if isinstance(planned, dict):
        planned = planned["jobs"]
    _, ready = finalize._preflight(planned, archive)
    runs, _, _ = summary._collect(planned, archive, split, ann)
    keys = set(itertools.product(STAGES, ("yolo",), TARGETS, SEEDS))
    if not keys <= runs.keys():
        raise ValueError("Both YOLO protocol arms must be complete across all three seeds")
    runs = {key: runs[key] for key in keys}
    selected = {run.job["job_id"]: ready[run.job["job_id"]] for run in runs.values()}
    _, inventory, paths, _ = finalize._inventory(
        selected, split, ann, include_adjacency_predictions=True
    )
    blocks = summary._load(blocks_path)
    if set(blocks) != set(TARGETS) or any(len(group) != 5 for group in blocks.values()):
        raise ValueError("Expected the five frozen blocks at all three test sites")
    blocks_sha = protocol.sha256_file(blocks_path)
    spec = summary._load(blocks_spec)
    if (
        spec["index_blocks_sha256"] != blocks_sha
        or spec["split_manifest_sha256"] != inventory["split_manifest"]
    ):
        raise ValueError("Block specification differs from the frozen split or groups")
    # Use the official block deletion function for the two primary archived arms.
    original = summary._block_deletion(runs, blocks)
    by_job = {run.job["job_id"]: key for key, run in runs.items()}
    primary = {}
    for target, result in original.items():
        for row in result["test_cells"]:
            primary[by_job[row["job_id"]], target] = {
                metric: [record["full_holdout"], *record["by_deleted_block"]]
                for metric, record in row["ap_ranges"].items()
            }
    if len(primary) != 54:
        raise ValueError("Expected 54 primary YOLO eval cells across both arms")
    replay_inventory_path = studies / "archive-verification.json"
    replay_inventory = summary._load(replay_inventory_path)
    archive_hashes = replay_inventory["files"]
    if not (
        replay_inventory["mismatches"] == 0
        and replay_inventory["source_files"]
        == replay_inventory["archived_files"]
        == len(archive_hashes)
    ):
        raise ValueError("GPU replay archive lacks a complete source-to-archive verification")
    replacement, replay_summary = _verified_replays(
        studies, runs, selected, inventory, paths, blocks, archive_hashes
    )
    if len(replacement) != 12 or not set(replacement) <= set(primary):
        raise ValueError("Expected exactly 12 substituted core cells")
    gpu = {**primary, **replacement}
    full = _compare(primary, gpu, None, None)
    deletions = [
        row
        for target in TARGETS
        for block in range(5)
        for row in _compare(primary, gpu, target, block)
        if row["statistic"] in MACROS
    ]
    ranges = []
    for metric, statistic in itertools.product(METRICS, MACROS):
        selection = [
            row for row in deletions if row["metric"] == metric and row["statistic"] == statistic
        ]
        assert len(selection) == 15
        ranges.append(
            {
                "metric": metric,
                "statistic": statistic,
                "primary_effect": _range(selection, "primary_effect"),
                "gpu_matched_effect": _range(selection, "gpu_matched_effect"),
                "gpu_matched_minus_primary": _range(selection, "gpu_matched_minus_primary"),
            }
        )
    return {
        "state": "supplemental_yolo_inference_backend_sensitivity",
        "schema_version": 1,
        "primary_jobs_sha256": protocol.sha256_file(jobs),
        "split_manifest_sha256": inventory["split_manifest"],
        "annotations_sha256": inventory["released_annotations"],
        "index_blocks_sha256": blocks_sha,
        "index_blocks_spec_sha256": protocol.sha256_file(blocks_spec),
        "replay_archive_verification_sha256": protocol.sha256_file(replay_inventory_path),
        "analysis_code_sha256": protocol.sha256_file(__file__),
        "effects": "Adjacent minus buffered for the three matched seeds. COCO AP values and changes use the 0–1 scale.",
        "deletion": "Each of 15 scenarios omits one prespecified block in one destination from both arms. The other two destinations retain every image. The resulting ranges describe test-composition sensitivity.",
        "substitution": "Four corrected CPU core evaluations replaced by predictions from the same checkpoints on the paired adjacency GPU model. This supplemental analysis uses the primary archive as its reference.",
        "replays": replay_summary,
        "full_holdout_effects": full,
        "block_deletion_macro_effects": deletions,
        "block_deletion_macro_ranges": ranges,
    }


def _write_study(output: Path, result: dict) -> None:
    rows = []
    for row in (*result["full_holdout_effects"], *result["block_deletion_macro_effects"]):
        units = {
            name + "_ap_points": row[name]["mean"] * 100.0
            for name in ("primary_effect", "gpu_matched_effect", "gpu_matched_minus_primary")
        }
        rows.append(
            {key: value for key, value in row.items() if not isinstance(value, dict)} | units
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as stage:
        pending = Path(stage) / "report"
        pending.mkdir()
        protocol.save_json(pending / "summary.json", result)
        summary._write_csv(pending / "effects.csv", rows, tuple(rows[0]))
        if output.exists():
            if all(
                (output / p.name).is_file() and (output / p.name).read_bytes() == p.read_bytes()
                for p in pending.iterdir()
            ):
                return
            raise FileExistsError("Choose a separate output path for changed supplemental results")
        os.rename(pending, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "jobs",
        "archive",
        "split",
        "ann",
        "blocks",
        "blocks-spec",
        "studies",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    output = a.output_dir.resolve()
    if "final-report" in output.parts or any(
        (path / "finalized.json").exists() for path in (output, *output.parents)
    ):
        parser.error("Use a separate supplemental analysis directory")
    result = study(a.jobs, a.archive, a.split, a.ann, a.blocks, a.blocks_spec, a.studies)
    _write_study(output, result)
    print(
        json.dumps(
            {
                "state": result["state"],
                "verified_replayed_jobs": len(result["replays"]),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
