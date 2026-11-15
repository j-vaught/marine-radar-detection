import json
from statistics import median

import pytest

from src.revision import archive, darc_diagnostic_audit, evaluation, protocol
from src.revision.darc_summary import Receipt


def write(path, obj):
    protocol.save_json(path, obj)
    return path


def inventory(folder):
    return {
        path.relative_to(folder).as_posix(): {
            "size": path.stat().st_size,
            "sha256": protocol.sha256_file(path),
        }
        for path in folder.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def attempt(tmp_path):
    job_id = "darc-v2-diag-DARC-s0-S0I0"
    out = f"/remote/darc-v2/runs/{job_id}/1"
    metadata = tmp_path / "metadata"
    folder = metadata / job_id / "1"
    nonce = "a" * 32
    receipt = {
        "job_id": job_id,
        "attempt": 1,
        "nonce": nonce,
        "pid": 101,
        "start_ticks": "123",
        "boot_id": "boot-id",
    }
    metrics = write(folder / "metrics.json", {"job_id": job_id, "tests": {}})
    completion = {
        **receipt,
        "state": "succeeded",
        "exit_code": 0,
        "finished": "now",
        "metrics_sha256": protocol.sha256_file(metrics),
    }
    write(folder / "completion.json", completion)
    write(folder / "receipt.json", receipt)
    write(folder / "job.json", {"job_id": job_id, "attempt_nonce": nonce, "out_dir": out})
    state = {
        "state": "succeeded",
        "attempt": 1,
        "nonce": nonce,
        "host": "comech-2422",
        "out": out,
        "receipt": out + "/receipt.json",
        "completion": out + "/completion.json",
        "result": completion,
    }
    state_path = write(tmp_path / "state" / f"{job_id}.json", state)
    archive_root = "/remote/darc-v2/archive"
    identity = archive.archive_identity(job_id, "1", state, "comech-2080", archive_root)
    ledger = {"verified": {identity: {"job": job_id, "attempt": "1"}}}
    remote_ledger = write(
        metadata / ".ledger" / job_id / "1.json",
        {"files": inventory(folder), "verified": "now"},
    )
    return state_path, metadata, ledger, archive_root, folder, remote_ledger


def read_attempt(params):
    return darc_diagnostic_audit.read_attempt(darc_diagnostic_audit.Evidence(), *params[:4])


def test_archive_binds_current_receipt_and_completion(attempt):
    fit, state, proof = read_attempt(attempt)
    assert fit.host == state["host"]
    assert fit.completion["metrics_sha256"] == proof["metrics.json"]["sha256"]
    assert json.loads(fit.metrics_bytes)["tests"] == {}


def test_archive_rejects_tampered_or_incomplete_local_metadata(attempt):
    folder = attempt[4]
    (folder / "metrics.json").write_bytes(b"x" * (folder / "metrics.json").stat().st_size)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_attempt(attempt)
    (folder / "metrics.json").unlink()
    with pytest.raises(ValueError, match="incomplete local"):
        read_attempt(attempt)


def test_archive_rejects_failed_current_attempt_even_with_valid_success_archive(attempt):
    state_path = attempt[0]
    state = json.loads(state_path.read_text())
    state.update(state="failed", attempt=2, nonce="f" * 32)
    write(state_path, state)
    with pytest.raises(ValueError, match="missing successful current"):
        read_attempt(attempt)


def test_archive_rejects_target_predictions_even_if_they_match_inventory(attempt):
    folder, ledger = attempt[4:]
    write(folder / "DARC_raw_predictions.json", [])
    write(ledger, {"files": inventory(folder), "verified": "now"})
    with pytest.raises(ValueError, match="target artifacts"):
        read_attempt(attempt)


def test_source_validation_replays_threshold_and_rejects_foreign_or_invalid_predictions(tmp_path):
    native = {"id": 1, "width": 1050, "height": 1024}
    gt = {
        "images": [native],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "vessel"}],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
    metrics = {
        "job_id": "diagnostic",
        "intensity": "I0",
        "score_threshold": 0.05,
        "source_validation": evaluation.evaluate_coco(gt, predictions),
    }
    write(
        tmp_path / "validation_input_audit.json",
        [
            {
                "image_id": 1,
                "scale": 1,
                "scale_level": "S0",
                "presentation_index": None,
                "intensity": "I0",
                "native_size_hw": [1024, 1050],
                "object_count": 0,
            }
        ],
    )
    write(tmp_path / "val_raw_predictions.json", predictions)
    write(tmp_path / "val_predictions.json", predictions)
    write(
        tmp_path / "postprocessing_selection.json",
        {
            "selection_split": "source_validation",
            "score_threshold": 0.05,
            "metrics": metrics["source_validation"],
        },
    )

    def replay():
        return darc_diagnostic_audit.check_predictions(
            darc_diagnostic_audit.Evidence(), tmp_path, gt, metrics
        )

    assert replay() == (1, 1)
    write(tmp_path / "val_raw_predictions.json", [{**predictions[0], "image_id": 2}])
    with pytest.raises(ValueError, match="valid native boxes"):
        replay()
    write(tmp_path / "val_raw_predictions.json", [{**predictions[0], "bbox": [0, 0, 1051, 1]}])
    with pytest.raises(ValueError, match="valid native boxes"):
        replay()
    write(tmp_path / "val_raw_predictions.json", predictions)
    metrics["score_threshold"] = 0.10
    write(
        tmp_path / "postprocessing_selection.json",
        {
            "selection_split": "source_validation",
            "score_threshold": 0.10,
            "metrics": metrics["source_validation"],
        },
    )
    with pytest.raises(ValueError, match="threshold replay differs"):
        replay()
    metrics["score_threshold"] = 0.05
    metrics["source_validation"]["ap"] -= 1e-12
    write(
        tmp_path / "postprocessing_selection.json",
        {
            "selection_split": "source_validation",
            "score_threshold": 0.05,
            "metrics": metrics["source_validation"],
        },
    )
    with pytest.raises(ValueError, match="threshold replay differs"):
        replay()


def test_geometry_detects_label_loss_and_area_tampering():
    job = {"job_id": "diag", "scale": "S1", "intensity": "I1"}
    stream = [[i, 1] for i in range(64)]
    train = {"annotations": [{"image_id": 1, "bbox": [512, 510, 10, 10]}]}
    scales = [0.75, 1.25] * 32
    before = [100] * 64
    after = [100 * scale**2 for scale in scales]
    resize = [800, 821]
    area_ratio = 800 * 821 / (1024 * 1050)

    def stats(values):
        return {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "max": max(values),
        }

    training = {
        "presentations": [
            {
                "presentation_index": index,
                "image_id": 1,
                "scale": scale,
                "scale_identity_reason": None,
                "attempted_scales": [scale],
                "scale_rejections": 0,
                "object_count": 1,
                "box_areas_before": [100],
                "box_areas_after": [after[index]],
                "scale_level": "S1",
                "intensity": "I1",
                "native_size_hw": [1024, 1050],
                "intensity_fallback": False,
            }
            for index, scale in enumerate(scales)
        ],
        "summary": {
            "presentations": 64,
            "native_hw": [1024, 1050],
            "model_resize_hw": resize,
            "accepted_shrink": 32,
            "accepted_grow": 32,
            "scale_rejections": 0,
            "scale_identity_reasons": {"baseline": 0, "gate": 0, "rejected": 0},
            "intensity_fallbacks": 0,
            "scale_summary": stats(scales),
            "box_area_before_native_px2": stats(before),
            "box_area_after_native_px2": stats(after),
            "box_area_after_model_resize_px2": stats([v * area_ratio for v in after]),
        },
    }

    def check():
        return darc_diagnostic_audit.verify_training_geometry(
            training, train, stream, job, resize, {"center_xy": [531, 515], "radius_px": 466}
        )

    assert check()["boxes_preserved"] == 64
    training["presentations"][0]["object_count"] = 0
    with pytest.raises(ValueError, match="label count"):
        check()
    training["presentations"][0]["object_count"] = 1
    training["presentations"][0]["box_areas_after"] = [1.0]
    with pytest.raises(ValueError, match="altered beyond"):
        check()


def test_runtime_rejects_wrong_effective_threads(tmp_path):
    fit = Receipt(
        {"job_id": "diag", "seed": 0, "gpu_uuid": "GPU-0"},
        {"runtime_provenance": {"cuda_visible_devices": "GPU-0"}},
        folder=tmp_path,
        metrics_bytes=b"",
        host="comech-2422",
    )
    state = {"runtime_profile_sha256": "profile", "gpu_uuid": "GPU-0"}
    profile = {
        "torch": "x",
        "torchvision": "y",
        "cuda_build": "12.1",
        "cudnn": 90100,
        "threads": 8,
        "thread_policy": darc_diagnostic_audit.THREAD_POLICY,
    }
    runtime = {
        "gpu_capability": [8, 9],
        "torch_version": "x",
        "torchvision_version": "y",
        "cuda_version": "12.1",
        "cudnn_version": 90100,
        "torch_threads": 8,
        "torch_interop_threads": 56,
    }
    with pytest.raises(ValueError, match="frozen policy"):
        darc_diagnostic_audit.validate_runtime(fit, state, runtime, profile, "profile")
