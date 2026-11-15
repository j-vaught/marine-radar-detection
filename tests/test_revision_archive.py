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
