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
            if planned["stage"] == "core":
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


def _fake_score(views, predictions):
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
