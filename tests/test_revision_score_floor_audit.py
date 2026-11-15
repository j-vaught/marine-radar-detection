import pytest

from src.revision import protocol, score_floor_audit


def test_source_guard_excludes_targets_and_checks_manifest_images(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    (run / "fit/weights").mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    image = images / "frame.png"
    image.write_bytes(b"pixel data")
    ann, manifest = tmp_path / "ann.json", tmp_path / "manifest.json"
    ann.write_text("{}")
    record = {
        "id": 1,
        "file_name": "frame.png",
        "site": "DAAN",
        "sha256": protocol.sha256_file(image),
    }
    protocol.save_json(manifest, {"images": [record]})
    gt = {
        "images": [{"id": 1, "file_name": "frame.png"}],
        "annotations": [],
        "categories": [],
    }
    protocol.save_json(run / "validation_gt.json", gt)
    protocol.save_json(run / "validation_raw.json", [])
    checkpoint = run / "fit/weights/best.pt"
    checkpoint.write_bytes(b"weights")
    job = {"job_id": "x", "method": "yolo", "source": "DAAN", "eval_targets": ["DARC"]}
    protocol.save_json(run / "job.json", job)
    metrics = {
        "job_id": "x",
        "evaluation_recipe_version": "yolo-exact-letterbox-v1",
        "manifest_sha256": protocol.sha256_file(manifest),
        "annotation_sha256": protocol.sha256_file(ann),
        "best_sha256": protocol.sha256_file(checkpoint),
    }
    protocol.save_json(run / "metrics.json", metrics)
    completion = {
        "job_id": "x",
        "state": "succeeded",
        "exit_code": 0,
        "metrics_sha256": protocol.sha256_file(run / "metrics.json"),
        "manifest_sha256": protocol.sha256_file(manifest),
    }
    protocol.save_json(run / "completion.json", completion)
    observed = []

    def prepare(source_job):
        observed.append(source_job["eval_targets"])
        return {"validation": gt, "tests": {}}

    monkeypatch.setattr(protocol, "prepare_job", prepare)
    result = score_floor_audit.source_validation_inputs(run, ann, images, manifest)
    assert observed == [[]]
    assert result["gt"] == gt
    image.write_bytes(b"changed pixels")
    with pytest.raises(ValueError, match="image checksum mismatch"):
        score_floor_audit.source_validation_inputs(run, ann, images, manifest)


def test_source_guard_rejects_legacy_yolo_before_accessing_data(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    protocol.save_json(run / "job.json", {"job_id": "x", "method": "yolo"})
    protocol.save_json(run / "metrics.json", {"job_id": "x"})
    protocol.save_json(
        run / "completion.json",
        {
            "job_id": "x",
            "state": "succeeded",
            "exit_code": 0,
            "metrics_sha256": protocol.sha256_file(run / "metrics.json"),
        },
    )
    with pytest.raises(ValueError, match="corrected prediction artifacts"):
        score_floor_audit.source_validation_inputs(
            run, tmp_path / "missing", tmp_path, tmp_path / "missing"
        )
