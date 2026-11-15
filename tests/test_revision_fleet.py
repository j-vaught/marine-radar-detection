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
    with patch.object(
        fleet, "ssh", return_value="0, GPU-a\n1, GPU-b\n__REVISION_PROCESS_LIST__\nGPU-a, 123\n"
    ):
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
    assert "thread_environment" in completion["runtime_provenance"]


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


def test_runtime_provenance_reports_gpu_and_backend_settings(monkeypatch):
    import sys
    from types import SimpleNamespace as NS

    fake = NS(
        __version__="2.4.1",
        version=NS(cuda="12.1"),
        are_deterministic_algorithms_enabled=lambda: True,
        backends=NS(
            cudnn=NS(deterministic=True, benchmark=False, allow_tf32=True),
            cuda=NS(matmul=NS(allow_tf32=False)),
        ),
        cuda=NS(
            get_device_properties=lambda _: NS(
                name="GPU model", total_memory=1234, major=8, minor=6
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    result = fleet.runtime_provenance({"gpu_uuid": "GPU-abc"})
    assert result["cuda_visible_devices"] == "GPU-abc"
    assert result["gpu"] == {
        "uuid": "GPU-abc",
        "model": "GPU model",
        "total_memory_bytes": 1234,
        "compute_capability": [8, 6],
    }
    assert result["torch"]["version"] == "2.4.1"
    assert result["torch"]["cuda_version"] == "12.1"
    assert result["torch"]["deterministic_algorithms"] is True
    assert result["torch"]["matmul_allow_tf32"] is False
    assert result["thread_environment"]["OMP_NUM_THREADS"] == "4"
    assert "cpu_count" in result and "machine" in result and "processor" in result


def test_runtime_provenance_cpu_without_torch(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    result = fleet.runtime_provenance({})
    assert result["thread_environment"]["MKL_NUM_THREADS"] == "3"
    assert result["thread_environment"]["RAYON_NUM_THREADS"] == "2"
    assert "torch" not in sys.modules
    assert "gpu" not in result


def test_method_gate_approves_only_completed_diagnostics():
    jobs = [
        {"job_id": "u1", "method": "unet", "stage": "diagnostic"},
        {"job_id": "u2", "method": "unet", "stage": "diagnostic"},
        {"job_id": "r1", "method": "rtdetr", "stage": "diagnostic"},
        {"job_id": "u-core", "method": "unet", "stage": "core"},
        {"job_id": "r-core", "method": "rtdetr", "stage": "core"},
    ]
    records = {
        "u1": {"state": "succeeded"},
        "u2": {"state": "succeeded"},
        "r1": {"state": "running"},
    }
    gate = {
        "approved": True,
        "jobs_sha256": "abc",
        "diagnostic_job_ids": ["u1", "u2", "r1"],
        "approved_methods": ["unet", "rtdetr"],
    }
    assert fleet.approved_gate_methods(gate, jobs, records, "abc") == {"unet"}
    assert not fleet.approved_gate_methods(gate, jobs, records, "different")
    assert not fleet.approved_gate_methods(
        {**gate, "diagnostic_job_ids": ["u1", "u2"]}, jobs, records, "abc"
    )
    assert not fleet.approved_gate_methods(
        {k: v for k, v in gate.items() if k != "approved_methods"}, jobs, records, "abc"
    )
    records["r1"]["state"] = "succeeded"
    assert fleet.approved_gate_methods(gate, jobs, records, "abc") == {"unet", "rtdetr"}
    assert fleet.approved_gate_methods(
        {k: v for k, v in gate.items() if k != "approved_methods"}, jobs, records, "abc"
    ) == {"unet", "rtdetr"}


def test_schedule_starts_approved_unet_but_blocks_running_rtdetr(tmp_path):
    import hashlib
    from types import SimpleNamespace

    manifest = tmp_path / "jobs.json"
    job_list = [
        {"job_id": "uc", "method": "unet", "stage": "core"},
        {"job_id": "rc", "method": "rtdetr", "stage": "core"},
        {"job_id": "ud", "method": "unet", "stage": "diagnostic"},
        {"job_id": "rd", "method": "rtdetr", "stage": "diagnostic"},
    ]
    manifest.write_text(json.dumps(job_list))
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps(
            {
                "hosts": {
                    "h": {
                        "result_root": "/r",
                        "code_path": "/c",
                        "venv_python": "/p",
                        "data_path": "/d",
                        "data_verified": True,
                        "protocol_sha256": "split",
                        "code_commit": "release",
                    }
                }
            }
        )
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "approved": True,
                "jobs_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "split_sha256": "split",
                "release_sha": "release",
                "diagnostic_job_ids": ["ud", "rd"],
                "approved_methods": ["unet", "rtdetr"],
            }
        )
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "ud.json").write_text(json.dumps({"state": "succeeded"}))
    (state / "rd.json").write_text(
        json.dumps({"state": "running", "host": "other", "device": 9, "attempt": 1, "nonce": "n"})
    )
    status = {"receipt": {"job_id": "rd", "attempt": 1, "nonce": "n"}, "alive": True}
    args = SimpleNamespace(
        state=state,
        status=False,
        manifest=manifest,
        hosts=hosts,
        approval=approval,
        once=True,
        poll=60,
    )
    with (
        patch.object(fleet, "remote_status", return_value=status),
        patch.object(fleet, "gpu_free", return_value=[(0, "GPU-0"), (1, "GPU-1")]),
        patch.object(fleet, "launch") as launch,
    ):
        fleet.schedule(args)
    assert launch.call_count == 1
    assert launch.call_args.args[2]["job_id"] == "uc"
    assert not (state / "rc.json").exists()


def test_launch_sets_cpu_and_gpu_thread_environment(tmp_path, monkeypatch):
    import shlex
    import subprocess
    import sys
    from types import SimpleNamespace

    captured = []

    def capture(host, command):
        args = shlex.split(command)
        script, script_args = args[2], args[3:]

        def popen(*args, **kwargs):
            captured.append(kwargs["env"])
            return SimpleNamespace(pid=1)

        with (
            patch.object(sys, "argv", ["-c", *script_args]),
            patch.object(subprocess, "Popen", side_effect=popen),
        ):
            exec(script, {})  # noqa: S102 - execute the trusted launcher under a mocked Popen
        return "1"

    monkeypatch.setattr(fleet, "ssh", capture)
    cfg = {
        "result_root": str(tmp_path),
        "code_path": str(tmp_path),
        "venv_python": "/p",
        "data_path": "/d",
        "threads": 4,
        "cpu_threads": 12,
    }
    for index, device in enumerate([None, (0, "GPU-abc")]):
        job = {"job_id": f"job{index}", "source": "DAAN"}
        fleet.launch("host", cfg, job, fleet.prepare_launch("host", cfg, job, 1, device))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        assert captured[0][name] == "12"
        assert captured[1][name] == "4"
    assert captured[1]["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert captured[0]["CUDA_VISIBLE_DEVICES"] == ""


def test_subprocess_error_does_not_log_command():
    import subprocess

    assert (
        fleet.concise_error(subprocess.TimeoutExpired("large secret script", 30))
        == "command timed out after 30s"
    )
    assert (
        fleet.concise_error(subprocess.CalledProcessError(255, "large secret script"))
        == "command exited with status 255"
    )


def test_schedule_repair_uses_parent_host_cpu_and_new_attempt(tmp_path):
    import hashlib
    from types import SimpleNamespace

    jobs = [
        {"job_id": "diag", "method": "yolo", "stage": "diagnostic"},
        {
            "job_id": "ordinary",
            "method": "yolo",
            "stage": "core",
            "resource": "cpu",
            "evaluation_from": "/r/runs/ordinary/1",
            "required_host": "parent",
        },
        {
            "job_id": "repair",
            "method": "yolo",
            "stage": "core",
            "resource": "cpu",
            "evaluation_from": "/r/runs/repair/1",
            "required_host": "parent",
        },
    ]
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(jobs))
    cfg = {
        "result_root": "/r",
        "code_path": "/c",
        "venv_python": "/p",
        "data_path": "/d",
        "data_verified": True,
        "protocol_sha256": "split",
        "code_commit": "release",
        "cpu_workers": 1,
    }
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"hosts": {"other": cfg, "parent": cfg}}))
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "approved": True,
                "jobs_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "split_sha256": "split",
                "release_sha": "release",
                "diagnostic_job_ids": ["diag"],
                "approved_methods": ["yolo"],
            }
        )
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "diag.json").write_text(json.dumps({"state": "succeeded"}))
    (state / "repair.json").write_text(
        json.dumps({"state": "repair_pending", "attempt": 1, "host": "parent"})
    )
    args = SimpleNamespace(
        state=state,
        status=False,
        manifest=manifest,
        hosts=hosts,
        approval=approval,
        once=True,
        poll=60,
    )
    with (
        patch.object(fleet, "gpu_free", return_value=[(0, "GPU-0")]),
        patch.object(fleet, "launch") as launch,
    ):
        fleet.schedule(args)
    assert launch.call_count == 1
    host, _, job, record = launch.call_args.args
    assert host == "parent" and job["job_id"] == "repair"
    assert record["device"] is None and record["attempt"] == 2
    assert record["out"] == "/r/runs/repair/2"


def test_repair_worker_bypasses_checkpoint_scratch(tmp_path):
    import hashlib
    from contextlib import nullcontext

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    job = {
        "job_id": "repair",
        "method": "yolo",
        "out_dir": str(tmp_path),
        "attempt_nonce": "n",
        "manifest": str(manifest),
        "protocol_sha256": hashlib.sha256(b"{}").hexdigest(),
        "evaluation_from": "/r/runs/repair/1",
        "device": "cpu",
    }
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps(job))
    (tmp_path / "metrics.json").write_text("{}")
    with (
        patch.object(
            fleet, "process_identity", return_value={"pid": 1, "start_ticks": "1", "boot_id": "b"}
        ),
        patch("src.revision.run.run_job") as run,
        patch("src.revision.storage.stage_checkpoint_scratch", return_value=nullcontext()) as stage,
        patch.object(fleet, "validate_metrics"),
    ):
        assert fleet.worker(str(job_file), str(tmp_path / "completion.json"), 2) == 0
    run.assert_called_once()
    stage.assert_not_called()


def test_adjacency_matches_training_gpu_and_repaired_history(tmp_path):
    adj = {"method": "yolo", "source": "DAAN", "seed": 0, "stage": "adjacency", "match_core_gpu_model": True}
    core_id = "core-yolo-DAAN-s0"
    threads = {k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}
    result = {"job_id": core_id, "state": "succeeded", "runtime_provenance": {
        "gpu": {"model": "NVIDIA RTX A5000"}, "thread_environment": threads}}
    training = {"state": "succeeded", "host": "parent", "python": "/venv/python", "result": result}
    cfg = {"venv_python": "/venv/python", "threads": 4}
    assert fleet.adjacency_matches_core(adj, {core_id: training}, tmp_path, "parent", cfg, (0, "GPU-a", "NVIDIA RTX A5000"))
    assert not fleet.adjacency_matches_core(adj, {core_id: training}, tmp_path, "parent", cfg, (0, "GPU-a", "RTX 6000"))
    assert not fleet.adjacency_matches_core(adj, {core_id: training}, tmp_path, "other", cfg, (0, "GPU-a", "NVIDIA RTX A5000"))
    assert not fleet.adjacency_matches_core(adj, {core_id: {"state": "running"}}, tmp_path, "parent", cfg, (0, "GPU-a", "NVIDIA RTX A5000"))
    repaired = {"state": "succeeded", "result": {"runtime_provenance": {}}}
    assert not fleet.adjacency_matches_core(adj, {core_id: repaired}, tmp_path, "parent", cfg, (0, "GPU-a", "NVIDIA RTX A5000"))
    (tmp_path / "history").mkdir()
    (tmp_path / "history" / f"{core_id}-attempt1.json").write_text(json.dumps(training))
    assert fleet.adjacency_matches_core(adj, {core_id: repaired}, tmp_path, "parent", cfg, (0, "GPU-a", "NVIDIA RTX A5000"))
    assert not fleet.adjacency_matches_core(adj, {core_id: repaired}, tmp_path, "parent", {**cfg, "threads": 2}, (0, "GPU-a", "NVIDIA RTX A5000"))


def test_gpu_query_includes_model_when_requested():
    with patch.object(fleet, "ssh", return_value="0, GPU-a, NVIDIA RTX A5000\n__REVISION_PROCESS_LIST__\n"):
        assert fleet.gpu_free("host", {}, include_models=True) == [(0, "GPU-a", "NVIDIA RTX A5000")]


def test_adjacency_cross_host_requires_matching_environment_attestations(tmp_path):
    job = {"method": "yolo", "source": "DAAN", "seed": 0, "stage": "adjacency",
           "match_core_gpu_model": True, "paired_environment_sha256": "digest"}
    core_id = "core-yolo-DAAN-s0"
    threads = {key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}
    training = {"state": "succeeded", "host": "original", "python": "/original/python",
        "result": {"job_id": core_id, "state": "succeeded", "runtime_provenance": {
            "gpu": {"model": "NVIDIA RTX 6000 Ada Generation"}, "thread_environment": threads}}}
    records = {core_id: training}
    cfg = {"venv_python": "/candidate/python", "threads": 4, "paired_environment_sha256": "digest"}
    device = (0, "GPU-a", "NVIDIA RTX 6000 Ada Generation")
    hosts = {"original": {"paired_environment_sha256": "digest"}, "candidate": cfg}
    assert fleet.adjacency_matches_core(job, records, tmp_path, "candidate", cfg, device, hosts)
    assert not fleet.adjacency_matches_core(job, records, tmp_path, "candidate", {**cfg, "paired_environment_sha256": "wrong"}, device, hosts)
    assert not fleet.adjacency_matches_core(job, records, tmp_path, "candidate", cfg, device, {"original": {}})
    assert not fleet.adjacency_matches_core({**job, "paired_environment_sha256": ""}, records, tmp_path, "candidate", cfg, device, hosts)
    assert not fleet.adjacency_matches_core(job, records, tmp_path, "candidate", {**cfg, "threads": 2}, device, hosts)
    assert not fleet.adjacency_matches_core(job, records, tmp_path, "candidate", cfg, (0, "GPU-a", "other"), hosts)
