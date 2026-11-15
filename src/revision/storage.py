"""Stage hot checkpoint paths on local scratch and archive them before returning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

_MIN_FREE = 20 * 1024**3


def _write_report(out: Path, report: dict) -> None:
    path = out / "checkpoint_scratch_report.json"
    tmp = out / ".checkpoint_scratch_report.tmp"
    with tmp.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> dict:
    """Hash regular files and record empty directories; never follow symlinks."""
    if path.is_symlink():
        raise RuntimeError(f"Unexpected nested symlink in scratch: {path}")
    if path.is_file():
        return {".": {"sha256": _hash_file(path), "bytes": path.stat().st_size}}
    if not path.is_dir():
        raise RuntimeError(f"Scratch path is not a regular file or directory: {path}")
    items = {".": {"type": "directory"}}
    for child in sorted(path.rglob("*")):
        name = str(child.relative_to(path))
        if child.is_symlink():
            raise RuntimeError(f"Unexpected nested symlink in scratch: {child}")
        if child.is_dir():
            items[name] = {"type": "directory"}
        elif child.is_file():
            items[name] = {"sha256": _hash_file(child), "bytes": child.stat().st_size}
        else:
            raise RuntimeError(f"Unexpected special file in scratch: {child}")
    return items


def _paths(method: str) -> dict[str, str]:
    if method in {"unet", "faster_rcnn"}:
        return {"best.pt": "file", "last.pt": "file", "resume.pt": "file"}
    if method in {"yolo", "rtdetr"}:
        return {"fit": "directory", "resume": "directory", "initial_state.pt": "file"}
    return {}


def _archive(out: Path, scratch: Path, entries: dict, report: dict) -> None:
    started = time.monotonic()
    for name, kind in entries.items():
        link = out / name
        source = scratch / name
        if not link.is_symlink() or os.readlink(link) != str(source):
            raise RuntimeError(f"Staging link changed unexpectedly: {link}")
        if not source.exists():
            # A file checkpoint that has not yet been written is harmless.
            if kind == "directory":
                raise RuntimeError(f"Scratch directory disappeared: {source}")
            link.unlink()
            report["artifacts"][name] = {"status": "not_created"}
            _write_report(out, report)
            continue
        expected = _inventory(source)
        staging = out / f".checkpoint-archive-{scratch.name}-{name}"
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(f"Refusing existing archive staging path: {staging}")
        try:
            if kind == "directory":
                shutil.copytree(source, staging)
            else:
                shutil.copy2(source, staging)
            actual = _inventory(staging)
            if actual != expected or _inventory(source) != expected:
                raise RuntimeError(f"Checkpoint verification failed: {source}")
            if not link.is_symlink() or os.readlink(link) != str(source):
                raise RuntimeError(f"Staging link changed during archive: {link}")
            if kind == "directory":
                link.unlink()
                try:
                    os.replace(staging, link)
                except BaseException:
                    link.symlink_to(source, target_is_directory=True)
                    raise
            else:
                os.replace(staging, link)
            count = sum(item.get("bytes", 0) for item in expected.values())
            report["bytes_archived"] += count
            report["artifacts"][name] = {"status": "verified", "bytes": count, "files": expected}
            _write_report(out, report)
        except BaseException:
            # A partial copy is not authoritative. Keep the original scratch intact.
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            elif staging.exists() or staging.is_symlink():
                staging.unlink()
            raise
    report["archive_seconds"] = time.monotonic() - started


@contextmanager
def stage_checkpoint_scratch(job: dict):
    """Stage fresh checkpoint paths on SSD and verify their HDD copies on exit.

    HDD paths stay stable. An archive failure raises and leaves scratch and an HDD
    report for recovery. If the job itself fails, its exception remains primary.
    """
    out = Path(job["out_dir"]).absolute()
    if out.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for report_name in ("checkpoint_scratch_report.json", ".checkpoint_scratch_report.tmp"):
        report_path = out / report_name
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError(f"Refusing existing scratch report path: {report_path}")
    entries = _paths(str(job.get("method", "")).lower())
    raw_root = job.get("checkpoint_scratch_root")
    report = {
        "mode": "hdd",
        "status": "fallback",
        "reason": None,
        "bytes_archived": 0,
        "archive_seconds": 0.0,
        "scratch_path": None,
        "out_dir": str(out),
        "artifacts": {},
    }
    if not raw_root or not entries:
        report["reason"] = "scratch_not_configured_or_method_unsupported"
        _write_report(out, report)
        yield report
        return
    root = Path(raw_root).absolute()
    if not root.is_dir() or root.is_symlink():
        report["reason"] = "scratch_root_unavailable"
        _write_report(out, report)
        yield report
        return
    usage = shutil.disk_usage(root)
    report["scratch_free_bytes"] = usage.free
    report["scratch_total_bytes"] = usage.total
    if usage.free < _MIN_FREE or usage.used / usage.total >= 0.9:
        report["reason"] = "insufficient_scratch_capacity"
        _write_report(out, report)
        yield report
        return
    for name in entries:
        path = out / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing checkpoint path: {path}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job.get("job_id", "job")))[:80]
    scratch = Path(tempfile.mkdtemp(prefix=f"{safe_id}-", dir=root))
    report.update(mode="scratch", status="active", scratch_path=str(scratch), reason=None)
    created = []
    try:
        for name, kind in entries.items():
            source = scratch / name
            if kind == "directory":
                source.mkdir()
            (out / name).symlink_to(source, target_is_directory=kind == "directory")
            created.append(name)
        _write_report(out, report)
    except BaseException:
        for name in created:
            link = out / name
            if link.is_symlink() and os.readlink(link) == str(scratch / name):
                link.unlink()
        shutil.rmtree(scratch)
        raise
    original = None
    try:
        yield report
    except BaseException as exc:
        original = exc
        raise
    finally:
        try:
            report["status"] = "archiving"
            _write_report(out, report)
            _archive(out, scratch, entries, report)
            report["status"] = "archived"
            _write_report(out, report)
            shutil.rmtree(scratch)
        except BaseException as exc:
            report["status"] = "archive_failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_report(out, report)
            except OSError as report_error:
                print(f"Could not write recovery report: {report_error}", file=sys.stderr)
            if original is not None:
                note = f"Checkpoint archival failed; recovery at {scratch}: {exc}"
                if hasattr(original, "add_note"):
                    original.add_note(note)
                else:
                    print(note, file=sys.stderr)
            else:
                raise
