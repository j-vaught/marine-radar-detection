import json
from unittest.mock import patch

from src.revision import fleet, jobs


def test_matrix_counts_and_targets():
    generated = jobs.standard_jobs()
    assert len(generated) == 97
    assert len({j["job_id"] for j in generated}) == 97
    assert sum(j["stage"] == "adjacency" for j in generated) == 18
    assert sum(j["stage"] == "diagnostic" and j["train_limit"] == 32 for j in generated) == 4
    assert all(len(j["eval_targets"]) == 3 for j in generated if j["stage"] != "diagnostic")


def test_gpu_free_rejects_busy_uuid():
    with patch.object(fleet, "ssh", side_effect=["0, GPU-a\n1, GPU-b\n", "GPU-a, 123\n"]):
        assert fleet.gpu_free("host", {}) == [(1, "GPU-b")]


def test_worker_records_model_failure(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {"job_id": "bad", "method": "unknown", "out_dir": str(tmp_path), "attempt_nonce": "n"}
        )
    )
    dest = tmp_path / "completion.json"
    with patch.object(
        fleet, "process_identity", return_value={"pid": 1, "start_ticks": "1", "boot_id": "b"}
    ):
        assert fleet.worker(str(job), str(dest), 1) == 1
    completion = json.loads(dest.read_text())
    assert completion["state"] == "failed"
    assert completion["failure_kind"] == "model"


def test_worker_requires_metrics(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {"job_id": "x", "method": "unet", "out_dir": str(tmp_path), "attempt_nonce": "n"}
        )
    )
    with (
        patch.object(
            fleet, "process_identity", return_value={"pid": 1, "start_ticks": "1", "boot_id": "b"}
        ),
        patch("src.revision.run.run_job", return_value=None),
    ):
        assert fleet.worker(str(job), str(tmp_path / "completion.json"), 1) == 1


def test_resolve_nested_paths():
    assert fleet.resolve(
        {"ann": "{data}/a", "target": ["{source}"]}, {"data": "/d", "source": "DAAN"}
    ) == {"ann": "/d/a", "target": ["DAAN"]}


def test_receipt_identity_must_match():
    record = {"attempt": 1, "nonce": "secret"}
    assert fleet.matches(record, {"job_id": "a", "attempt": 1, "nonce": "secret"}, "a")
    assert not fleet.matches(record, {"job_id": "a", "attempt": 1, "nonce": "wrong"}, "a")


def test_validate_rejects_wrong_targets(tmp_path):
    ann = tmp_path / "ann"
    ann.write_text("x")
    import hashlib

    digest = hashlib.sha256(b"x").hexdigest()
    job = {
        "job_id": "a",
        "protocol_sha256": "hash",
        "method": "unet",
        "stage": "core",
        "condition": "buffered",
        "ann": str(ann),
        "eval_targets": ["DAAN"],
        "out_dir": str(tmp_path),
    }
    import pytest

    with pytest.raises(ValueError, match="targets"):
        fleet.validate_metrics(
            job,
            {
                "job_id": "a",
                "manifest_sha256": "hash",
                "annotation_sha256": digest,
                "method": "unet",
                "stage": "core",
                "condition": "buffered",
                "tests": {},
            },
        )


def test_prepare_persists_paths_and_uuid():
    cfg = {"result_root": "/r", "code_path": "/c", "venv_python": "/p"}
    rec = fleet.prepare_launch("h", cfg, {"job_id": "a"}, 2, (3, "GPU-abc"))
    assert rec["receipt"] == "/r/runs/a/2/receipt.json"
    assert rec["gpu_uuid"] == "GPU-abc"
    assert rec["device"] == 3 and rec["nonce"]
