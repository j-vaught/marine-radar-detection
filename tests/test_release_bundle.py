import io
import json
import pickle
import tarfile
import zipfile

import pytest

from src.revision.release_bundle import (
    checked_file,
    export_checkpoint,
    public_value,
    rewrite_pickle_metadata,
    safe_relative,
    sha256,
    write_asset,
)


def test_public_metadata_excludes_operational_identity():
    metadata = {
        "gpu_uuid": "GPU-1234567812345678",
        "config": {"lr": 0.01},
        "path": "/home/person/runs/best.pt",
        "address": "100.64.1.2",
        "checkpoint_sha256": "abc",
    }
    cleaned = public_value(metadata)
    assert cleaned == {
        "config": {"lr": 0.01},
        "path": "best.pt",
        "address": "[operational identifier removed]",
        "checkpoint_sha256": "abc",
    }


@pytest.mark.parametrize("value", ["/etc/passwd", "../weights.pt", "a/../b", "a//b", "."])
def test_relative_paths_reject_escape(value):
    with pytest.raises(ValueError):
        safe_relative(value)


def test_pickle_redaction_preserves_values_and_shared_references():
    private = "/home/person/private/weights.pt"
    value = {
        "args": {"model": private, "repeat": private},
        "size": 128,
        "tensor_payload": b"unchanged",
    }
    rewritten, count = rewrite_pickle_metadata(pickle.dumps(value, protocol=2))
    restored = pickle.loads(rewritten)
    assert count == 1
    assert restored["args"] == {
        "model": "release-local/weights.pt",
        "repeat": "release-local/weights.pt",
    }
    assert restored["size"] == value["size"]
    assert restored["tensor_payload"] == value["tensor_payload"]


def test_framed_pickle_refused():
    with pytest.raises(ValueError, match="framed"):
        rewrite_pickle_metadata(pickle.dumps({"path": "/home/private/file"}, protocol=4))


def test_checkpoint_metadata_only_export(tmp_path):
    source, exported = tmp_path / "original.pt", tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "archive/data.pkl", pickle.dumps({"path": "/mnt/private/model.pt"}, protocol=2)
        )
        archive.writestr("archive/data/0", bytes(range(256)))
    before = source.read_bytes()
    result = export_checkpoint(source, exported)
    assert source.read_bytes() == before
    assert result["checkpoint_metadata_strings_rewritten"] == 1
    assert result["original_checkpoint_sha256"] != result["checkpoint_sha256"]
    with zipfile.ZipFile(exported) as archive:
        assert archive.read("archive/data/0") == bytes(range(256))
        assert pickle.loads(archive.read("archive/data.pkl"))["path"] == "release-local/model.pt"


def test_ledger_digest_and_symlink_validation(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"value": 1}))
    ledger = {"files": {"metrics.json": {"size": path.stat().st_size, "sha256": sha256(path)}}}
    assert checked_file(tmp_path, "metrics.json", ledger) == path
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="digest"):
        checked_file(tmp_path, "metrics.json", ledger)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="Symlink"):
        checked_file(tmp_path, "alias", ledger)


def test_asset_is_deterministic_and_budgeted(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    item = root / "run.json"
    item.write_text('{"a": 1}\n')
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    a = write_asset(root, [item], one, 3_000_000)
    b = write_asset(root, [item], two, 3_000_000)
    assert a["sha256"] == b["sha256"]
    with tarfile.open(fileobj=io.BytesIO(one.read_bytes()), mode="r:gz") as archive:
        assert archive.getnames() == ["run.json"]
        assert archive.extractfile("run.json").read() == item.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        write_asset(root, [item], tmp_path / "small.tar.gz", 10)


def test_nested_private_paths_and_auth_fields_are_removed():
    metadata = {
        "note": "saved to /home/person/private/weights.pt",
        "path": "/mnt/GPU-12345678-12345678",
        "api_key": "secretvalue",
        "authorization": "Bearer secretvalue",
    }
    clean = public_value(metadata)
    assert clean == {
        "note": "[operational identifier removed]",
        "path": "[operational identifier removed]",
    }
    payload = pickle.dumps({"note": metadata["note"], "path": metadata["path"]}, protocol=2)
    rewritten, count = rewrite_pickle_metadata(payload)
    assert count == 2
    assert pickle.loads(rewritten) == {"note": "release-local", "path": "release-local"}


def test_sensitive_checkpoint_fields_refused():
    with pytest.raises(ValueError, match="private"):
        rewrite_pickle_metadata(pickle.dumps({"api_key": "opaque"}, protocol=2))


def test_archive_job_symlink_is_rejected(tmp_path):
    original = tmp_path / "original" / "1"
    original.mkdir(parents=True)
    path = original / "metrics.json"
    path.write_text("{}")
    ledger = {"files": {"metrics.json": {"size": 2, "sha256": sha256(path)}}}
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "job").symlink_to(original.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        checked_file(archive / "job" / "1", "metrics.json", ledger)
