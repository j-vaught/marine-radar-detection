import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.revision import darc_fleet as d


def catalog():
    return {source: {str(seed): {"path": "{root}/initial/" + source + f"/s{seed}.pt",
                                 "sha256": f"file-{source}-{seed}", "tensor_sha256": f"tensor-{source}-{seed}"}
                     for seed in range(3)} for source in d.SOURCES}


def test_factory_exact_matrix_and_shared_initialization():
    spec = d.make_jobs("protocol", catalog())
    jobs = spec["jobs"]
    assert sum(j["stage"] == "core" for j in jobs) == 42
    assert sum(j["stage"] == "diagnostic" for j in jobs) == 12
    assert sum(j["size_control"] for j in jobs) == 6
    assert len({j["job_id"] for j in jobs}) == 54
    for source in d.SOURCES:
        for seed in range(3):
            block = [j for j in jobs if j["source"] == source and j["seed"] == seed]
            assert len({j["initial_state_sha256"] for j in block}) == 1
            assert len({j["initial_tensor_sha256"] for j in block}) == 1
    assert all(j["eval_targets"] == [] and j["diagnostic_updates"] == 32 for j in jobs if j["stage"] == "diagnostic")


def test_scoring_gate_requires_all_diagnostics(tmp_path):
    jobs = d.make_jobs("protocol", catalog())["jobs"]
    ids = [j["job_id"] for j in jobs if j["stage"] == "diagnostic"]
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"approved": True, "kind": "scoring", "jobs_sha256": "jobs",
        "protocol_sha256": "protocol", "release_sha": "release", "runtime_profile_sha256": "profile",
        "diagnostic_job_ids": ids}))
    records = {job_id: {"state": "succeeded"} for job_id in ids}
    assert d.gate_ok(gate, "scoring", "jobs", "protocol", "release", "profile", jobs, records)
    records[ids[0]]["state"] = "running"
    assert not d.gate_ok(gate, "scoring", "jobs", "protocol", "release", "profile", jobs, records)
    assert not d.gate_ok(gate, "scoring", "other", "protocol", "release", "profile", jobs, records)


def test_scheduler_gate_closed_without_approval(tmp_path):
    base = tmp_path / "darc-v2" / "controller"
    base.mkdir(parents=True)
    state = base / "state"
    jobs = base / "jobs.json"
    spec = d.make_jobs("protocol", catalog())
    spec.update(split_sha256="split", annotations_sha256="ann")
    jobs.write_text(json.dumps(spec))
    hosts = base / "hosts.json"
    hosts.write_text(json.dumps({"hosts": {"comech-2422": {"enabled": True, "data_verified": True,
        "initialization_verified": True, "code_commit": "release", "runtime_profile_sha256": "profile",
        "protocol_sha256": "split", "annotations_sha256": "ann", "result_root": "/tmp/darc-v2",
        "code_path": "/tmp/darc-v2/releases/release", "venv_python": "/p", "data_path": "/data"}}}))
    args = SimpleNamespace(jobs=jobs, hosts=hosts, state=state, diagnostic_gate=base/"diag.json",
        scoring_gate=base/"score.json", release="release", profile="profile", once=True, status=False, poll=60)
    with patch.object(d.fleet, "gpu_free", return_value=[(0, "GPU-a", d.GPU_MODEL)]), patch.object(d, "launch") as launch:
        d.schedule(args)
    launch.assert_not_called()


def test_reconcile_rejects_wrong_receipt():
    record = {"host": "h", "nonce": "n", "attempt": 1}
    with patch.object(d.fleet, "remote_status", return_value={"receipt": {"job_id": "bad", "attempt": 1, "nonce": "n"}, "alive": True}):
        assert d.reconcile("good", record)["state"] == "uncertain"


def test_build_gate_hashes_evidence_and_requires_diagnostics(tmp_path):
    import pytest
    spec = d.make_jobs("protocol", catalog())
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ok":true}')
    entry = {"passed": True, "evidence": [{"path": str(evidence), "sha256": d.digest(evidence)}]}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema": "darc-v2-scoring-audit-1", "jobs_sha256": d.digest(jobs),
        "protocol_sha256": "protocol", "release_sha": "release", "runtime_profile_sha256": "profile",
        "checks": {key: entry for key in d.REQUIRED_CHECKS["scoring"]}}))
    state = tmp_path / "state"
    state.mkdir()
    for job in spec["jobs"]:
        if job["stage"] == "diagnostic":
            (state / (job["job_id"] + ".json")).write_text(json.dumps({"state": "succeeded", "result": {
                "job_id": job["job_id"], "metrics_sha256": "recorded"}}))
    args = SimpleNamespace(jobs=jobs, audit=audit, state=state, output=tmp_path/"gate.json",
                           kind="scoring", release="release", profile="profile")
    d.build_gate(args)
    gate = json.loads(args.output.read_text())
    assert len(gate["diagnostic_job_ids"]) == 12
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="invalid evidence"):
        d.build_gate(args)


def test_wrong_gpu_and_unapproved_scoring_never_launch(tmp_path):
    base = tmp_path / "darc-v2" / "controller"
    base.mkdir(parents=True)
    state = base / "state"
    jobs = base / "jobs.json"
    spec = d.make_jobs("protocol", catalog())
    spec.update(split_sha256="split", annotations_sha256="ann")
    jobs.write_text(json.dumps(spec))
    hosts = base / "hosts.json"
    hosts.write_text(json.dumps({"hosts": {"comech-2422": {"enabled": True, "data_verified": True,
        "initialization_verified": True, "code_commit": "release", "runtime_profile_sha256": "profile",
        "protocol_sha256": "split", "annotations_sha256": "ann", "result_root": "/tmp/darc-v2",
        "code_path": "/tmp/darc-v2/releases/release", "venv_python": "/p", "data_path": "/data"}}}))
    diag_gate = base / "diag.json"
    diag_gate.write_text(json.dumps({"approved": True, "kind": "diagnostic", "jobs_sha256": d.digest(jobs),
        "protocol_sha256": "protocol", "release_sha": "release", "runtime_profile_sha256": "profile"}))
    args = SimpleNamespace(jobs=jobs, hosts=hosts, state=state, diagnostic_gate=diag_gate,
        scoring_gate=base/"score.json", release="release", profile="profile", once=True, status=False, poll=60)
    with patch.object(d.fleet, "gpu_free", return_value=[(0, "GPU-a", "NVIDIA RTX A5000")]), patch.object(d, "launch") as launch:
        d.schedule(args)
    launch.assert_not_called()
    with patch.object(d.fleet, "gpu_free", return_value=[(0, "GPU-a", d.GPU_MODEL)]), patch.object(d, "launch") as launch:
        d.schedule(args)
    assert launch.call_count == 1 and launch.call_args.args[2]["stage"] == "diagnostic"


def test_factory_includes_frozen_protocol_contract():
    job = d.make_jobs("frozen-digest", catalog())["jobs"][0]
    assert job["frozen_protocol"] == "{root}/staging/frozen-inputs-14a06aad/processed-frozen-protocol.json"
    assert job["frozen_protocol_sha256"] == "frozen-digest"


def test_source_seed_block_is_pinned_to_one_host(tmp_path):
    base = tmp_path / "darc-v2" / "controller"
    base.mkdir(parents=True)
    state = base / "state"
    jobs_path = base / "jobs.json"
    spec = d.make_jobs("protocol", catalog())
    spec.update(split_sha256="split", annotations_sha256="ann")
    jobs_path.write_text(json.dumps(spec))
    def cfg(host):
        return {"enabled": True, "data_verified": True, "initialization_verified": True,
            "code_commit": "release", "runtime_profile_sha256": "profile", "threads": 8,
            "protocol_sha256": "split", "annotations_sha256": "ann", "result_root": "/" + host + "/darc-v2",
            "code_path": "/" + host + "/darc-v2/releases/release", "venv_python": "/p", "data_path": "/" + host + "/data"}
    hosts = base / "hosts.json"
    hosts.write_text(json.dumps({"hosts": {"comech-2422": cfg("h1"), "cocesr-1141": cfg("h2")}}))
    diag = base / "diag.json"
    diag.write_text(json.dumps({"approved": True, "kind": "diagnostic", "jobs_sha256": d.digest(jobs_path),
        "protocol_sha256": "protocol", "release_sha": "release", "runtime_profile_sha256": "profile"}))
    args = SimpleNamespace(jobs=jobs_path, hosts=hosts, state=state, diagnostic_gate=diag,
        scoring_gate=base/"score.json", release="release", profile="profile", once=True, status=False, poll=60)
    with patch.object(d.fleet, "gpu_free", return_value=[(0, "GPU-a", d.GPU_MODEL), (1, "GPU-b", d.GPU_MODEL)]), patch.object(d, "launch") as launch:
        d.schedule(args)
    assignments = {}
    for call in launch.call_args_list:
        host, _, job, _, _ = call.args
        assignments.setdefault((job["source"], job["seed"]), set()).add(host)
    assert launch.call_count == 4
    assert all(len(assigned) == 1 for assigned in assignments.values())


def test_explicit_block_restart_archives_whole_block_and_refuses_active(tmp_path):
    import pytest
    base = tmp_path / "darc-v2" / "controller"
    state = base / "state"
    state.mkdir(parents=True)
    spec = d.make_jobs("protocol", catalog())
    spec.update(split_sha256="split", annotations_sha256="ann")
    jobs = base / "jobs.json"
    jobs.write_text(json.dumps(spec))
    hosts = base / "hosts.json"
    hosts.write_text(json.dumps({"hosts": {"cocesr-1141": {"enabled": True, "data_verified": True,
        "initialization_verified": True, "code_commit": "release", "runtime_profile_sha256": "profile",
        "protocol_sha256": "split", "annotations_sha256": "ann", "data_path": "/new/data", "threads": 8,
        "result_root": "/new/darc-v2", "code_path": "/new/darc-v2/releases/release", "venv_python": "/p"}}}))
    block = [j for j in spec["jobs"] if j["source"] == "DAAN" and j["seed"] == 0]
    for job in block[:2]:
        (state / (job["job_id"] + ".json")).write_text(json.dumps({"state": "succeeded", "attempt": 1,
            "host": "comech-2422", "data_path": "/old/data", "runtime_profile_sha256": "profile"}))
    active = state / (block[1]["job_id"] + ".json")
    old = json.loads(active.read_text())
    active.write_text(json.dumps({**old, "state": "running"}))
    args = SimpleNamespace(jobs=jobs, hosts=hosts, state=state, source="DAAN", seed=0,
                           target="cocesr-1141", reason="host unavailable", release="release", profile="profile")
    with pytest.raises(ValueError, match="active"):
        d.restart_block(args)
    assert not (state / "history").exists()
    active.write_text(json.dumps(old))
    d.restart_block(args)
    for job in block[:2]:
        rec = json.loads((state / (job["job_id"] + ".json")).read_text())
        assert rec["state"] == "infra_failed" and rec["attempt"] == 1
        assert rec["host"] == "cocesr-1141" and rec["data_path"] == "/new/data"
    assert len(list((state / "history").glob("*.json"))) == 2
    assert json.loads(next((state / "block-restarts").glob("*.json")).read_text())["status"] == "ready"
