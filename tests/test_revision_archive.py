import io
import json
import subprocess
import sys
import tarfile

import pytest

from src.revision import archive


def remote(*args, data=b"", check=True):
    return subprocess.run(
        [sys.executable, "-c", archive.REMOTE, *map(str, args)],
        input=data,
        capture_output=True,
        check=check,
    )


def test_selection_and_verified_promotion(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "train_sweep").mkdir()
    for name, content in {
        "best.pt": b"weights",
        "data/stage.png": b"data",
        "plot.png": b"figure",
        "train_sweep/candidate.json": b"{}",
        "yolo26n.pt": b"model",
        ".env": b"secret",
    }.items():
        (source / name).write_bytes(content)
    manifest = remote("inventory", source).stdout
    assert set(json.loads(manifest)) == {"best.pt", "plot.png", "train_sweep/candidate.json"}
    stream = remote("stream", source).stdout
    root = tmp_path / "destination"
    stage = root / ".staging" / "job" / "1-unique"
    remote("extract", stage, data=stream)
    remote("finalize", root, "job", "1", stage, data=manifest)
    assert (root / "job/1/best.pt").read_bytes() == b"weights"
    assert json.loads((root / ".ledger/job/1.json").read_text())["files"] == json.loads(manifest)
    remote("finalize", root, "job", "1", root / "missing", data=manifest)
    (root / "job/1/best.pt").write_bytes(b"corrupted")
    assert remote("check", root / "job/1", data=manifest).stdout.strip() == b"no"
    assert remote(
        "finalize", root, "job", "1", root / "missing", data=manifest, check=False
    ).returncode


def test_staging_mismatch_never_promoted(tmp_path):
    stage = tmp_path / "root/.staging/job/1"
    stage.mkdir(parents=True)
    (stage / "weights.pt").write_bytes(b"wrong")
    result = remote(
        "finalize",
        tmp_path / "root",
        "job",
        "1",
        stage,
        data=json.dumps({"weights.pt": {"sha256": "0" * 64, "size": 5}}).encode(),
        check=False,
    )
    assert result.returncode
    assert stage.exists()
    assert not (tmp_path / "root/job/1").exists()
    assert not (tmp_path / "root/.ledger/job/1.json").exists()


def test_symlinks_and_tar_traversal_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path)
    assert remote("inventory", source, check=False).returncode
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    assert remote("extract", tmp_path / "stage", data=payload.getvalue(), check=False).returncode
    assert not (tmp_path / "escape").exists()


def test_only_completed_matching_records_including_history(tmp_path):
    state = tmp_path / "state"
    (state / "history").mkdir(parents=True)
    record = {
        "state": "succeeded",
        "host": "host",
        "out": "/run/1",
        "attempt": 1,
        "nonce": "abc",
        "result": {"job_id": "job", "attempt": 1, "nonce": "abc", "finished": "today"},
    }
    (state / "history/job-attempt1.json").write_text(json.dumps(record))
    (state / "running.json").write_text(json.dumps({"state": "running"}))
    assert [(j, a) for j, a, _ in archive.selected_records(state)] == [("job", "1")]
    record["result"]["nonce"] = "wrong"
    (state / "bad.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        list(archive.selected_records(state))
    with pytest.raises(ValueError):
        archive.safe_id("../bad")


def test_archive_one_uses_remote_manifest_and_skips_verified(monkeypatch):
    calls = []
    monkeypatch.setattr(archive, "ssh", lambda *a, **kw: '{"a":{"size":1,"sha256":"x"}}')
    monkeypatch.setattr(
        archive, "remote_input", lambda host, command, data: calls.append(command) or "yes"
    )
    monkeypatch.setattr(archive, "stream_copy", lambda *args: pytest.fail("must skip transfer"))
    assert (
        archive.archive_one("job", "1", {"host": "source", "out": "/run"}, "target", "/archive")
        == "verified"
    )
    assert len(calls) == 2


def test_client_ledger_skips_only_same_verified_completion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(archive, "archive_one", lambda *args: calls.append(args) or "archived")
    record = {
        "host": "host",
        "out": "/run/1",
        "nonce": "nonce",
        "result": {"job_id": "job", "attempt": 1, "finished": "today", "metrics_sha256": "abc"},
    }
    path = tmp_path / "ledger.json"
    ledger = archive.load_ledger(path)
    assert archive.process_record("job", "1", record, "target", "/root", ledger, path) == "archived"
    ledger = archive.load_ledger(path)
    assert (
        archive.process_record("job", "1", record, "target", "/root", ledger, path)
        == "already verified"
    )
    assert len(calls) == 1
    archive.process_record("job", "1", record, "target", "/root", ledger, path, force=True)
    record["result"]["metrics_sha256"] = "changed"
    archive.process_record("job", "1", record, "target", "/root", ledger, path)
    assert len(calls) == 3


def test_failed_archive_is_not_ledgered(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(archive, "archive_one", fail)
    path = tmp_path / "ledger.json"
    ledger = archive.load_ledger(path)
    record = {"host": "host", "out": "/run", "nonce": "n", "result": {}}
    with pytest.raises(RuntimeError):
        archive.process_record("job", "1", record, "target", "/root", ledger, path)
    assert ledger["verified"] == {}
    assert not path.exists()


def test_remote_inventory_under_python311(tmp_path):
    import shutil

    python = shutil.which("python3.11")
    if not python:
        pytest.skip("Python 3.11 is unavailable")
    (tmp_path / "result.json").write_text("{}")
    completed = subprocess.run(
        [python, "-c", archive.REMOTE, "inventory", str(tmp_path)],
        input=b"",
        capture_output=True,
        check=True,
    )
    assert list(json.loads(completed.stdout)) == ["result.json"]


def test_remote_errors_do_not_log_encoded_commands():
    exc = subprocess.CalledProcessError(
        1, ["ssh", "encoded-secret-command"], stderr=b"remote error"
    )
    assert archive.error_message(exc) == "Remote command failed (exit 1): remote error"


def test_hardlinked_checkpoints_roundtrip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "last_training_state.pt").write_bytes(b"checkpoint contents")
    (source / "best_training_state.pt").hardlink_to(source / "last_training_state.pt")
    manifest = remote("inventory", source).stdout
    stream = remote("stream", source).stdout
    stage = tmp_path / "stage"
    remote("extract", stage, data=stream)
    assert json.loads(remote("inventory", stage).stdout) == json.loads(manifest)
    assert (stage / "best_training_state.pt").read_bytes() == b"checkpoint contents"
    assert (stage / "last_training_state.pt").read_bytes() == b"checkpoint contents"


def test_local_copy_selects_files_and_does_not_link_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").mkdir()
    (source / "data" / "staged.png").write_bytes(b"stage")
    (source / "checkpoint.pt").write_bytes(b"checkpoint")
    stage = tmp_path / "stage"
    remote("localcopy", source, stage)
    assert json.loads(remote("inventory", source).stdout) == json.loads(
        remote("inventory", stage).stdout
    )
    assert not (stage / "data").exists()
    assert (source / "checkpoint.pt").stat().st_ino != (stage / "checkpoint.pt").stat().st_ino
    (source / "checkpoint.pt").write_bytes(b"changed")
    assert (stage / "checkpoint.pt").read_bytes() == b"checkpoint"


def test_same_host_selects_local_copy(monkeypatch):
    calls = []

    def fake_ssh(host, command, **kwargs):
        calls.append(command)
        return '{"file":{"size":1,"sha256":"x"}}'

    monkeypatch.setattr(archive, "ssh", fake_ssh)
    monkeypatch.setattr(archive, "remote_input", lambda *args: "no")
    monkeypatch.setattr(archive, "stream_copy", lambda *args: pytest.fail("must copy locally"))
    assert (
        archive.archive_one(
            "job", "1", {"host": "comech-2080", "out": "/run"}, "comech-2080", "/archive"
        )
        == "archived"
    )
    assert len(calls) == 2


def test_direct_transport_streams_source_to_destination(tmp_path, monkeypatch):
    import shlex

    from src.revision import archive_transport

    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.pt").write_bytes(bytes(range(256)) * 16384)
    stage = tmp_path / "stage"
    popen = subprocess.Popen

    def local_ssh(args, **kwargs):
        return popen(shlex.split(args[-1]) if args[0] == "ssh" else args, **kwargs)

    monkeypatch.setattr(archive_transport.subprocess, "Popen", local_ssh)
    monkeypatch.setattr(archive_transport, "ssh_options", lambda host: [])
    config = {"destination_bind": "127.0.0.1", "sources": {"comech-2422": "127.0.0.1"}}
    metrics = archive_transport.direct_copy(
        "comech-2422", str(source), "comech-2080", str(stage), config, archive.REMOTE
    )
    assert metrics["transport"] == "direct_tls"
    assert metrics["bytes"] >= 4 * 1024 * 1024
    assert json.loads(remote("inventory", source).stdout) == json.loads(
        remote("inventory", stage).stdout
    )


def test_direct_transport_config_rejects_unapproved_hosts(tmp_path):
    from src.revision.archive_transport import load_config

    config = tmp_path / "transport.json"
    config.write_text(
        json.dumps({"destination_bind": "10.1.1.1", "sources": {"unknown": "10.1.1.2"}})
    )
    with pytest.raises(ValueError, match="unapproved"):
        load_config(config)


def test_direct_failure_falls_back_to_fresh_relay_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(archive, "ssh", lambda *args, **kwargs: '{"file":{"size":1,"sha256":"x"}}')
    monkeypatch.setattr(archive, "remote_input", lambda *args: "no")

    def fail(host, source, destination, stage, config, program):
        calls.append(stage)
        raise OSError("failed")

    monkeypatch.setattr(archive, "direct_copy", fail)
    monkeypatch.setattr(
        archive, "stream_copy", lambda host, source, destination, stage: calls.append(stage)
    )
    metrics = {}
    config = {"sources": {"comech-2422": "10.0.0.2"}}
    result = archive.archive_one(
        "job",
        "1",
        {"host": "comech-2422", "out": "/run"},
        "comech-2080",
        "/archive",
        config,
        metrics,
    )
    assert result == "archived" and metrics["transport"] == "relay"
    assert len(calls) == 2 and calls[0] != calls[1]
