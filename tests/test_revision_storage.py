"""Scratch archival safety and failure behavior."""

import json
import shutil
from pathlib import Path

import pytest

from src.revision import storage


def job(tmp_path, method="unet"):
    scratch = tmp_path / "ssd"
    scratch.mkdir()
    return {
        "job_id": "x/1",
        "out_dir": str(tmp_path / "hdd"),
        "checkpoint_scratch_root": str(scratch),
        "method": method,
    }


@pytest.fixture(autouse=True)
def enough_space(monkeypatch):
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(100 * 1024**3, 10 * 1024**3, 90 * 1024**3),
    )


def test_files_archive_and_verify(tmp_path):
    config = job(tmp_path)
    out = Path(config["out_dir"])
    with storage.stage_checkpoint_scratch(config) as report:
        assert (out / "best.pt").is_symlink()
        (out / "best.pt").write_bytes(b"checkpoint")
        (out / "resume.pt").write_bytes(b"state")
        scratch = Path(report["scratch_path"])
        assert scratch.exists()
    assert (out / "best.pt").read_bytes() == b"checkpoint"
    assert not (out / "best.pt").is_symlink()
    assert not (out / "last.pt").exists()
    assert not scratch.exists()
    saved = json.loads((out / "checkpoint_scratch_report.json").read_text())
    assert saved["status"] == "archived" and saved["bytes_archived"] == 15
    assert saved["artifacts"]["best.pt"]["files"]["."]["sha256"] == storage._hash_file(
        out / "best.pt"
    )


def test_directory_archived_on_job_failure(tmp_path):
    config = job(tmp_path, "rtdetr")
    out = Path(config["out_dir"])
    with pytest.raises(ValueError, match="training failed"):  # noqa: SIM117
        with storage.stage_checkpoint_scratch(config):
            (out / "fit" / "weights").mkdir()
            (out / "fit" / "weights" / "last.pt").write_bytes(b"weights")
            (out / "resume" / "state.pt").write_bytes(b"state")
            raise ValueError("training failed")
    assert (out / "fit" / "weights" / "last.pt").read_bytes() == b"weights"
    assert not (out / "fit").is_symlink()
    assert (out / "resume" / "state.pt").read_bytes() == b"state"


def test_existing_path_refused_without_modification(tmp_path):
    config = job(tmp_path)
    out = Path(config["out_dir"])
    out.mkdir()
    (out / "best.pt").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError):  # noqa: SIM117
        with storage.stage_checkpoint_scratch(config):
            pytest.fail("must refuse before running")
    assert (out / "best.pt").is_symlink()
    assert not list(Path(config["checkpoint_scratch_root"]).iterdir())


def test_insufficient_space_falls_back(tmp_path, monkeypatch):
    config = job(tmp_path)
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 95, 5)
    )
    with storage.stage_checkpoint_scratch(config) as report:
        assert report["mode"] == "hdd"
        (Path(config["out_dir"]) / "best.pt").write_bytes(b"hdd")
    assert not (Path(config["out_dir"]) / "best.pt").is_symlink()
    assert not list(Path(config["checkpoint_scratch_root"]).iterdir())


def test_archive_failure_keeps_scratch_and_reports(tmp_path, monkeypatch):
    config = job(tmp_path)
    out = Path(config["out_dir"])
    with pytest.raises(OSError, match="copy failed"):  # noqa: SIM117
        with storage.stage_checkpoint_scratch(config) as report:
            scratch = Path(report["scratch_path"])
            (out / "best.pt").write_bytes(b"checkpoint")
            monkeypatch.setattr(
                storage.shutil, "copy2", lambda *_: (_ for _ in ()).throw(OSError("copy failed"))
            )
    assert scratch.exists() and (out / "best.pt").is_symlink()
    assert (
        json.loads((out / "checkpoint_scratch_report.json").read_text())["status"]
        == "archive_failed"
    )


def test_original_exception_remains_primary_on_archive_failure(tmp_path, monkeypatch):
    config = job(tmp_path)
    with pytest.raises(ValueError, match="primary") as caught:  # noqa: SIM117
        with storage.stage_checkpoint_scratch(config):
            (Path(config["out_dir"]) / "best.pt").write_bytes(b"x")
            monkeypatch.setattr(
                storage.shutil, "copy2", lambda *_: (_ for _ in ()).throw(OSError("copy failed"))
            )
            raise ValueError("primary")
    assert any("archival failed" in note for note in caught.value.__notes__)
