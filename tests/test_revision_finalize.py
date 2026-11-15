import json
from pathlib import Path

import pytest

from src.revision import finalize, protocol
from src.revision.summarize import CLASSICAL, ROW_METRICS, SITES, _subset_coco

JOB_SPEC = Path(__file__).parents[1] / "experiments/review-2026-11/jobs.json"


def _metrics(ap):
    return dict(zip(ROW_METRICS, (ap, ap, ap, ap, 1, 1.0)))


def _archive(tmp_path):
    images = []
    boxes = []
    partitions = {}
    for index, site in enumerate(SITES):
        ids = list(range(index * 10 + 1, index * 10 + 5))
        partitions[site] = {
            "partitions": {"validation": ids[:1], "train": ids[1:2], "test": ids[-1:]},
            "adjacency": {str(seed): ids[2:3] for seed in range(3)},
        }
        for image_id in ids:
            images.append(
                {
                    "id": image_id,
                    "dataset": site,
                    "site": site,
                    "frame": image_id,
                    "sha256": "0" * 64,
                    "width": 40,
                    "height": 40,
                }
            )
            boxes.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [5, 5, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            )
    ann_path = tmp_path / "annotations.json"
    coco = {"images": images, "annotations": boxes, "categories": [{"id": 1, "name": "vessel"}]}
    protocol.save_json(ann_path, coco)
    ann_hash = protocol.sha256_file(ann_path)
    manifest = tmp_path / "split.json"
    protocol.save_json(
        manifest, {"sites": partitions, "images": images, "annotation_sha256": ann_hash}
    )
    manifest_digest = protocol.sha256_file(manifest)
    hashes = {
        site: finalize._serialized_digest(
            _subset_coco(coco, set(partitions[site]["partitions"]["test"]))
        )
        for site in SITES
    }
    runs = tmp_path / "collected"
    for planned in json.loads(JOB_SPEC.read_text()):
        job_id = planned["job_id"]
        folder = runs / job_id / "1"
        remote = Path("/remote") / job_id / "1"
        actual = {
            **planned,
            "out_dir": str(remote),
            "attempt_nonce": f"nonce-{job_id}",
            "protocol_sha256": manifest_digest,
            "code_commit": "commit",
        }
        results = {}
        paths = {"subsets": {}}
        for target in planned["eval_targets"]:
            ap = 0.2 + 0.05 * planned["seed"] + 0.1 * (target == planned["source"])
            method = planned["method"]
            if method in CLASSICAL:
                path = remote / f"test_{target}_predictions.json"
                paths["subsets"][f"test_{target}"] = {
                    "annotations_sha256": hashes[target],
                    "predictions": str(path),
                }
                results[target] = _metrics(ap)
            elif method in {"yolo", "rtdetr"}:
                path = remote / f"{target}_raw.json"
                results[target] = {
                    "raw_path": str(path),
                    "raw_metrics": _metrics(ap),
                    "ground_truth_sha256": hashes[target],
                    "checkpoint_sha256": "a" * 64,
                }
            else:
                path = remote / f"{target}_raw_predictions.json"
                results[target] = {
                    "raw_predictions_path": str(path),
                    "raw": _metrics(ap),
                    "gt_sha256": hashes[target],
                    "checkpoint_sha256": "a" * 64,
                }
            protocol.save_json(folder / path.name, [{"ap": ap}])
        learning = {}
        if planned["method"] in CLASSICAL:
            learning["selected_config"] = {"config": {"threshold": 0.5}}
        elif planned["method"] in {"yolo", "rtdetr"}:
            learning.update(best_sha256="a" * 64, training_log=str(remote / "run" / "results.csv"))
        else:
            learning["best_weights_sha256"] = "a" * 64
        metrics = {
            **{key: actual[key] for key in finalize.summary.IDENTITY},
            "manifest_sha256": manifest_digest,
            "annotation_sha256": ann_hash,
            "validation": _metrics(0.5),
            "tests": results,
            "paths": paths,
            **learning,
        }
        if planned["method"] == "yolo":
            metrics["evaluation_recipe_version"] = "yolo-exact-letterbox-v1"
        protocol.save_json(folder / "job.json", actual)
        protocol.save_json(folder / "metrics.json", metrics)
        protocol.save_json(
            folder / "completion.json",
            {
                "job_id": job_id,
                "attempt": 1,
                "nonce": actual["attempt_nonce"],
                "state": "succeeded",
                "exit_code": 0,
                "manifest_sha256": manifest_digest,
                "metrics_sha256": protocol.sha256_file(folder / "metrics.json"),
                "code_commit": "commit",
            },
        )
    return {
        "jobs": JOB_SPEC,
        "runs_dir": runs,
        "split_manifest": manifest,
        "annotations": ann_path,
        "output_dir": tmp_path / "output",
    }


def _fake_score(views, predictions, *, cache_dir=None):
    del cache_dir
    return {
        name: {
            "clipping": entry[1],
            "methods": {
                key: {
                    "coco": _metrics(raw[0]["ap"] if name == "baseline" else raw[0]["ap"] / 2),
                    "ap_change_from_baseline": dict.fromkeys(
                        finalize.summary.AP_METRICS, 0 if name == "baseline" else -raw[0]["ap"] / 2
                    ),
                    "center_agreement": {},
                }
                for key, raw in predictions.items()
            },
        }
        for name, entry in views.items()
    }


def test_incomplete_preflight_does_not_open_heldouts(tmp_path):
    out = tmp_path / "out"
    protocol.save_json(out / "finalized.json", {"stale": True})
    assert not finalize.finalize(
        jobs=JOB_SPEC,
        runs_dir=tmp_path / "uncollected",
        split_manifest=tmp_path / "split-is-not-present",
        annotations=tmp_path / "annotations-are-not-present",
        output_dir=out,
    )
    result = json.loads((out / "progress.json").read_text())
    assert not (out / "finalized.json").exists()
    assert result["state"] == "incomplete"
    assert result["jobs_by_stage"]["core"] == {"declared": 75, "ready": 0}
    assert result["job_state_counts"] == {"uncollected": 110}


def test_finalizes_all_schemas_and_skips_identical_verified_outputs(tmp_path, monkeypatch):
    request = _archive(tmp_path)
    out = request["output_dir"]
    torch_attempt = request["runs_dir"] / "core-unet-DAAN-s0" / "1"
    protocol.save_json(torch_attempt / "history.json", [{"epoch": 1}, {"epoch": 2}])
    ultra_attempt = request["runs_dir"] / "core-yolo-DAAN-s0" / "1"
    (ultra_attempt / "results.csv").write_text("epoch,fitness\n1,0.5\n")
    monkeypatch.setattr(finalize.geometry, "score_reference_views", _fake_score)
    assert finalize.finalize(**request)
    marker = json.loads((out / "finalized.json").read_text())
    assert set(marker["outputs"]) == set(finalize.OUTPUT_FILES)
    report = json.loads((out / "reference_geometry.json").read_text())
    assert report["evaluated_source_target_pairs"] == 9
    assert all(len(pair["clipping"]) == 12 for pair in report["pairs"])
    assert all(
        row["scope"] == "pair"
        for row in report["estimates"]
        if row["reference"] == "fixed_source_envelope"
    )
    macro = next(
        row
        for row in report["estimates"]
        if (
            row["scope"],
            row["method"],
            row["reference"],
            row["metric"],
            row["quantity"],
        )
        == ("macro_transfer", "yolo", "scale_0.8", "ap", "ap_change_from_baseline")
    )
    assert macro["missing_seeds"] == []
    assert macro["mean"] == pytest.approx(-0.15)
    assert sum(1 for _ in (out / "reference_geometry_seed_rows.csv").open()) == 225 * 12 * 3 + 1
    curves = json.loads((out / "learning_curves.json").read_text())
    assert curves["collected_epoch_logs"] == 2
    assert {
        row["method"]: row["epoch_rows"]
        for row in curves["learned_job_records"]
        if row["curve_collected"]
    } == {"unet": 2, "yolo": 1}

    def fail_on_cache_miss(*_args):
        pytest.fail("identical inputs and checksummed outputs should skip report generation")

    monkeypatch.setattr(finalize, "_write_final", fail_on_cache_miss)
    assert finalize.finalize(**request)
    metrics = request["runs_dir"] / "core-yolo-DAAN-s0" / "1" / "metrics.json"
    metrics.write_text("{}")
    assert not finalize.finalize(**request)
    assert not (out / "finalized.json").exists()
    assert json.loads((out / "progress.json").read_text())["state"] == "incomplete"


def test_rejects_verified_worker_metrics_on_wrong_reference(tmp_path):
    request = _archive(tmp_path)
    run = request["runs_dir"] / "core-yolo-DAAN-s0" / "1"
    metrics_file = run / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    metrics["tests"]["DARC"]["ground_truth_sha256"] = "b" * 64
    protocol.save_json(metrics_file, metrics)
    receipt_file = run / "completion.json"
    receipt = json.loads(receipt_file.read_text())
    receipt["metrics_sha256"] = protocol.sha256_file(metrics_file)
    protocol.save_json(receipt_file, receipt)
    with pytest.raises(ValueError, match="differs from frozen target"):
        finalize.finalize(**request)
    assert not (request["output_dir"] / "finalized.json").exists()


def test_precompute_scores_only_verified_core_attempts(tmp_path):
    from src.revision import annotation_sensitivity, evaluation, precompute

    req = _archive(tmp_path)
    one_run = tmp_path / "partial"
    one_run.mkdir()
    job_id = "core-unet-DAAN-s0"
    (req["runs_dir"] / job_id).rename(one_run / job_id)
    req["runs_dir"] = one_run
    holding = one_run / job_id / "1"
    coco = json.loads(req["annotations"].read_text())
    manifest = json.loads(req["split_manifest"].read_text())
    metrics = json.loads((holding / "metrics.json").read_text())
    for target in SITES:
        refs = _subset_coco(coco, set(manifest["sites"][target]["partitions"]["test"]))
        raw = [
            {key: ann[key] for key in ("image_id", "category_id", "bbox")} | {"score": 0.9}
            for ann in refs["annotations"]
        ]
        protocol.save_json(holding / f"{target}_raw_predictions.json", raw)
        metrics["tests"][target]["raw"] = evaluation.evaluate_coco(refs, raw)
    protocol.save_json(holding / "metrics.json", metrics)
    completion = json.loads((holding / "completion.json").read_text())
    completion["metrics_sha256"] = protocol.sha256_file(holding / "metrics.json")
    protocol.save_json(holding / "completion.json", completion)

    cache = tmp_path / "cache"
    options = {key: value for key, value in req.items() if key != "output_dir"}
    report = precompute.precompute(**options, cache_dir=cache, workers=2)
    assert report["core_jobs_ready"] == 1
    assert report["core_jobs_declared"] == 75
    assert report["verified_cells"] == 3
    assert report["cache_hits"] + report["cache_misses"] == 36
    assert not (cache / "finalized.json").exists()
    assert not finalize.finalize(**req, cache_dir=cache)

    _, _, env = annotation_sensitivity._load_references(
        req["annotations"], manifest, "DAAN", "DAAN"
    )
    test = _subset_coco(coco, set(manifest["sites"]["DAAN"]["partitions"]["test"]))
    views = annotation_sensitivity.build_reference_views(test, env["width_height_px"])
    preds = json.loads((holding / "DAAN_raw_predictions.json").read_text())
    cached, hits = annotation_sensitivity.score_model_views(views, preds, cache_dir=cache)
    uncached, _ = annotation_sensitivity.score_model_views(views, preds)
    assert cached == uncached
    assert hits == 12

    (holding / "metrics.json").write_text("{}")
    report = precompute.precompute(**options, cache_dir=cache, workers=2)
    assert report["state"] == "waiting_for_core_artifacts"
    assert report["core_jobs_ready"] == 0


def test_blocks_pass_to_summary_cli_and_change_report_input_identity(tmp_path, monkeypatch):
    request = _archive(tmp_path)
    output = request["output_dir"]
    blocks = tmp_path / "blocks.json"
    protocol.save_json(blocks, {"DAAN": [[4]]})
    arguments = []
    original_summarize = finalize._run_strict_summary

    def capture_summary(directory, jobs, runs, split, annotations, block_path):
        arguments.append(block_path)
        # The full archive fixture has a one-image holdout and fake prediction scores.
        original_summarize(directory, jobs, runs, split, annotations)

    monkeypatch.setattr(finalize, "_run_strict_summary", capture_summary)
    monkeypatch.setattr(finalize.geometry, "score_reference_views", _fake_score)
    assert finalize.finalize(**request)
    assert arguments == [None]
    marker_path = output / "finalized.json"
    geometry_only = json.loads(marker_path.read_text())
    assert finalize.finalize(**request, blocks=blocks)
    paired = json.loads(marker_path.read_text())
    assert arguments == [None, blocks]
    assert paired["input_content_sha256"] != geometry_only["input_content_sha256"]
    assert paired["inputs"]["index_blocks"] == protocol.sha256_file(blocks)
    assert "runs/adjacency-yolo-DAAN-s0/DAAN/raw_predictions" in paired["inputs"]
    assert finalize.finalize(**request, blocks=blocks)
    assert arguments == [None, blocks]
    blocks.write_text('{"DAAN": [[4]]}')
    assert finalize.finalize(**request, blocks=blocks)
    assert arguments == [None, blocks, blocks]


def test_strict_summary_command_forwards_blocks(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(finalize.subprocess, "run", lambda command, **_: captured.extend(command))
    blocks = tmp_path / "blocks.json"
    finalize._run_strict_summary(tmp_path, JOB_SPEC, tmp_path, tmp_path, tmp_path, blocks)
    assert "--require-complete" in captured
    assert captured[-2:] == ["--blocks", str(blocks)]


def test_only_linked_corrected_yolo_attempt_can_replace_frozen_training(tmp_path):
    import shutil

    request = _archive(tmp_path)
    runs = request["runs_dir"]
    job_id = "core-yolo-DAAN-s0"
    parent = runs / job_id / "1"
    plan = next(row for row in json.loads(JOB_SPEC.read_text()) if row["job_id"] == job_id)
    parent_job = json.loads((parent / "job.json").read_text())
    plan_without_repair = {**plan}
    plan["evaluation_from"] = parent_job["out_dir"]
    old_metrics = json.loads((parent / "metrics.json").read_text())
    old_metrics.pop("evaluation_recipe_version")
    old_metrics.update(best_epoch=13, training_seconds=7.5, peak_vram_bytes=100)
    protocol.save_json(parent / "metrics.json", old_metrics)
    parent_completion = json.loads((parent / "completion.json").read_text())
    parent_completion.update(
        pid=123,
        start_ticks="4",
        boot_id="parent-host",
        runtime_provenance={"gpu": {"uuid": "GPU-training", "model": "training GPU"}},
        metrics_sha256=protocol.sha256_file(parent / "metrics.json"),
    )
    protocol.save_json(parent / "completion.json", parent_completion)
    keys = ("job_id", "attempt", "nonce", "pid", "start_ticks", "boot_id")
    protocol.save_json(parent / "receipt.json", {key: parent_completion[key] for key in keys})
    report, ready = finalize._preflight([plan], runs)
    assert not ready
    assert report["incomplete_jobs"][0]["state"] == "waiting_for_evaluation_repair"
    report, ready = finalize._preflight([plan_without_repair], runs)
    assert not ready
    assert report["incomplete_jobs"][0]["state"] == "waiting_for_evaluation_repair"

    repair = parent.with_name("2")
    shutil.copytree(parent, repair)
    eval_job = {
        **parent_job,
        "out_dir": str(Path(parent_job["out_dir"]).with_name("2")),
        "attempt_nonce": "nonce-repair",
        "evaluation_from": plan["evaluation_from"],
        "device": "cpu",
        "code_commit": "repair-release",
    }
    protocol.save_json(repair / "job.json", eval_job)
    repaired: dict = {
        **old_metrics,
        "training_parent": {
            "directory": str(plan["evaluation_from"]),
            **{
                name + "_sha256": protocol.sha256_file(parent / f"{name}.json")
                for name in ("receipt", "completion", "metrics")
            },
            "checkpoint_sha256": old_metrics["best_sha256"],
            "code_commit": parent_completion["code_commit"],
            "runtime_provenance": parent_completion["runtime_provenance"],
        },
        "evaluation_recipe_version": "yolo-exact-letterbox-v1",
        "evaluation_device": "cpu",
        "training_log": str(Path(eval_job["out_dir"]) / "results.csv"),
    }
    for result in repaired["tests"].values():
        name = Path(result["raw_path"]).name
        result["raw_path"] = str(Path(eval_job["out_dir"]) / name)
    protocol.save_json(repair / "metrics.json", repaired)
    receipt = {
        **parent_completion,
        "attempt": 2,
        "nonce": eval_job["attempt_nonce"],
        "pid": 456,
        "start_ticks": "5",
        "metrics_sha256": protocol.sha256_file(repair / "metrics.json"),
        "runtime_provenance": {"processor": "inference-only CPU"},
        "code_commit": eval_job["code_commit"],
    }
    protocol.save_json(repair / "receipt.json", {key: receipt[key] for key in keys})
    protocol.save_json(repair / "completion.json", receipt)

    _, chosen = finalize._preflight([plan], runs)
    assert chosen[job_id].folder == repair
    result, rows = finalize.summary.summarize(
        [plan], runs, split_manifest=request["split_manifest"], annotations=request["annotations"]
    )
    assert not result["missing_jobs"]
    assert {row["attempt"] for row in rows} == {2}
    record, _ = finalize._curve_record(chosen[job_id])
    assert record["gpu_uuid"] == "GPU-training"
    assert record["training_seconds"] == 7.5
    assert record["best_epoch_as_recorded"] == 13
    assert record["code_commit"] == parent_completion["code_commit"]
    assert record["evaluation_runtime_provenance"] == receipt["runtime_provenance"]
    assert record["evaluation_device"] == "cpu"
    assert record["evaluation_attempt"] == 2

    ledger = runs / ".ledger" / job_id / "2.json"
    protocol.save_json(
        ledger,
        {"files": {p.name: {"sha256": protocol.sha256_file(p)} for p in repair.glob("*.json")}},
    )
    _, _, paths, _ = finalize._inventory(chosen, request["split_manifest"], request["annotations"])
    assert set(paths) == {(job_id, target) for target in SITES}
    assert all(path.parent == repair for path, _ in paths.values())
    bad = paths[job_id, "DARC"][0]
    bad.write_text("[]")
    with pytest.raises(ValueError, match="differs from verified archive ledger"):
        finalize._inventory(chosen, request["split_manifest"], request["annotations"])

    protocol.save_json(parent / "receipt.json", {"altered": True})
    report, chosen = finalize._preflight([plan], runs)
    assert not chosen
    assert report["incomplete_jobs"][0]["state"] == "invalid_or_uncollected"
