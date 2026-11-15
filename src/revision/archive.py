"""Copy completed fleet attempts to central storage, with end-to-end verification.

Run on the controller, where both source and destination SSH aliases are available.
The controller's training state is read only; archival state lives on the destination.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .fleet import quote_cmd, ssh, ssh_options

DEFAULT_ROOT = "/mnt/12TB_Drive/dspire/atlantis-master/JACKIE_drone_training/wacv-signal-2027-paper-06/revision/archive"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

# The same implementation runs on each remote host. It accepts only regular files,
# never follows links, hashes contents in bounded chunks, and streams a tar archive.
REMOTE = r"""
import hashlib, json, os, pathlib, shutil, stat, sys, tarfile
from datetime import datetime, timezone

BLOCK = 1024 * 1024

def chosen(path):
    pure = pathlib.PurePosixPath(path)
    parts = pure.parts
    if pure.is_absolute() or not parts or pure.as_posix() != path or any(p in ('', '.', '..') for p in parts):
        return False
    lower = [p.lower() for p in parts]
    if 'data' in lower or lower[-1] == 'yolo26n.pt':
        return False
    # Do not copy common credentials even if they are accidentally placed in a run.
    return not any(p in ('.ssh', '.aws', '.env', 'credentials', 'credentials.json',
                         'id_rsa', 'id_ed25519', 'token.json') or
                   p.endswith(('.pem', '.key', '.p12', '.pfx')) for p in lower)

def files(root):
    root = pathlib.Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('source root must be a real directory')
    for base, dirs, names in os.walk(root, followlinks=False):
        relative = pathlib.Path(base).relative_to(root)
        keep = []
        for d in sorted(dirs):
            rel = (relative / d).as_posix()
            if not chosen(rel):
                continue
            if (pathlib.Path(base) / d).is_symlink():
                raise ValueError('selected directory is a symlink: ' + rel)
            keep.append(d)
        dirs[:] = keep
        for name in sorted(names):
            path = pathlib.Path(base) / name
            rel = (relative / name).as_posix()
            if not chosen(rel):
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('selected path is not a regular file: ' + rel)
            yield rel, path

def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('not a regular file: ' + str(path))
    return os.fdopen(fd, 'rb')

def inventory(root):
    out = {}
    for rel, path in files(root):
        h = hashlib.sha256()
        with open_regular(path) as f:
            while True:
                chunk = f.read(BLOCK)
                if not chunk:
                    break
                h.update(chunk)
        out[rel] = {'sha256': h.hexdigest(), 'size': path.stat().st_size}
    return out

def main():
    mode = sys.argv[1]
    if mode == 'inventory':
        print(json.dumps(inventory(sys.argv[2]), sort_keys=True))
    elif mode == 'stream':
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', dereference=True) as tar:
            for rel, path in files(sys.argv[2]):
                with open_regular(path) as f:
                    info = tar.gettarinfo(fileobj=f, arcname=rel)
                    tar.addfile(info, f)
    elif mode == 'extract':
        root = pathlib.Path(sys.argv[2])
        root.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=sys.stdin.buffer, mode='r|*') as tar:
            for member in tar:
                if not member.isfile() or not chosen(member.name):
                    raise ValueError('unsafe tar member: ' + member.name)
                dest = root.joinpath(*pathlib.PurePosixPath(member.name).parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as f:
                    shutil.copyfileobj(tar.extractfile(member), f, BLOCK)
    elif mode == 'finalize':
        root, job, attempt, staging = map(pathlib.Path, sys.argv[2:6])
        expected = json.loads(sys.stdin.read())
        final = root / job / attempt
        ledger = root / '.ledger' / job / (str(attempt) + '.json')
        lock = root / '.locks' / job / str(attempt)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir()  # A concurrent attempt must retry instead of racing promotion.
        try:
            if final.exists():
                if inventory(final) != expected:
                    raise ValueError('existing archive differs from source inventory')
            else:
                if inventory(staging) != expected:
                    raise ValueError('staging inventory differs from source inventory')
                final.parent.mkdir(parents=True, exist_ok=True)
                os.rename(staging, final)
            ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp = ledger.with_name(ledger.name + '.' + str(os.getpid()) + '.tmp')
            tmp.write_text(json.dumps({'files': expected, 'verified': datetime.now(timezone.utc).isoformat()}, sort_keys=True) + '\n')
            os.replace(tmp, ledger)
        finally:
            lock.rmdir()
    elif mode == 'check':
        final = pathlib.Path(sys.argv[2])
        expected = json.loads(sys.stdin.read())
        print('yes' if final.is_dir() and inventory(final) == expected else 'no')
    else:
        raise ValueError('unknown mode')
main()
"""


def remote_cmd(mode: str, *args: str) -> str:
    program = (
        "import base64;exec(base64.b64decode("
        + repr(base64.b64encode(REMOTE.encode()).decode())
        + "))"
    )
    return quote_cmd(["python3", "-c", program, mode, *args])


def safe_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Unsafe archive identifier: {value!r}")
    return value


def selected_records(state: Path):
    for path in sorted([*state.glob("*.json"), *(state / "history").glob("*.json")]):
        record = json.loads(path.read_text())
        result = record.get("result") or {}
        if record.get("state") not in {"succeeded", "failed"} or not result.get("finished"):
            continue
        job = safe_id(str(result.get("job_id", "")))
        attempt = record.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError(f"Invalid attempt in {path}")
        if result.get("attempt") != attempt or result.get("nonce") != record.get("nonce"):
            raise ValueError(f"Completion identity mismatch in {path}")
        if not record.get("host") or not str(record.get("out", "")).startswith("/"):
            raise ValueError(f"Invalid host or output path in {path}")
        if record["host"].lower().replace("-", "") in {"theia", "delta", "deltaai"}:
            continue
        yield job, str(attempt), record


def remote_input(host: str, command: str, data: bytes) -> str:
    result = subprocess.run(
        ["ssh", *ssh_options(host), host, command],
        input=data,
        capture_output=True,
        check=True,
        timeout=300,
    )
    return result.stdout.decode().strip()


def stream_copy(host: str, source: str, dest_host: str, staging: str):
    # File-backed stderr avoids either process blocking on a full error pipe.
    with tempfile.TemporaryFile() as src_err, tempfile.TemporaryFile() as dst_err:
        source_process = subprocess.Popen(
            ["ssh", *ssh_options(host), host, remote_cmd("stream", source)],
            stdout=subprocess.PIPE,
            stderr=src_err,
        )
        try:
            target_process = subprocess.Popen(
                ["ssh", *ssh_options(dest_host), dest_host, remote_cmd("extract", staging)],
                stdin=source_process.stdout,
                stdout=subprocess.DEVNULL,
                stderr=dst_err,
            )
        except BaseException:
            source_process.kill()
            source_process.wait()
            raise
        assert source_process.stdout is not None
        source_process.stdout.close()
        try:
            target_code = target_process.wait(timeout=86400)
            source_code = source_process.wait(timeout=60)
        except BaseException:
            source_process.kill()
            target_process.kill()
            source_process.wait()
            target_process.wait()
            raise
        if source_code or target_code:
            src_err.seek(0)
            dst_err.seek(0)
            raise RuntimeError(
                f"Archive stream failed ({source_code}, {target_code}): "
                f"{src_err.read().decode(errors='replace')} "
                f"{dst_err.read().decode(errors='replace')}"
            )


def archive_one(job: str, attempt: str, record: dict, dest_host: str, root: str):
    safe_id(job)
    if not attempt.isdigit() or int(attempt) < 1:
        raise ValueError("Invalid attempt")
    source_inventory = json.loads(
        ssh(record["host"], remote_cmd("inventory", record["out"]), timeout=86400)
    )
    if not source_inventory:
        raise ValueError("Refusing empty archive")
    manifest = json.dumps(source_inventory, sort_keys=True).encode()
    final = f"{root}/{job}/{attempt}"
    staging = f"{root}/.staging/{job}/{attempt}-{uuid.uuid4().hex}"
    existing = remote_input(dest_host, remote_cmd("check", final), manifest) == "yes"
    if not existing:
        stream_copy(record["host"], record["out"], dest_host, staging)
    remote_input(dest_host, remote_cmd("finalize", root, job, attempt, staging), manifest)
    return "verified" if existing else "archived"


def archive_identity(job: str, attempt: str, record: dict, dest_host: str, root: str) -> str:
    """Bind a verified archive to the immutable completion and both endpoints."""
    value = {
        "job": job,
        "attempt": attempt,
        "source_host": record["host"],
        "source_out": record["out"],
        "nonce": record.get("nonce"),
        "result": record["result"],
        "destination_host": dest_host,
        "destination_root": root,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "verified": {}}
    value = json.loads(path.read_text())
    if value.get("version") != 1 or not isinstance(value.get("verified"), dict):
        raise ValueError(f"Invalid archive ledger: {path}")
    return value


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x") as f:
            json.dump(ledger, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def process_record(job, attempt, record, dest_host, root, ledger, ledger_path, force=False):
    identity = archive_identity(job, attempt, record, dest_host, root)
    if identity in ledger["verified"]:
        if not force:
            return "already verified"
        verified = dict(ledger["verified"])
        del verified[identity]
        save_ledger(ledger_path, {**ledger, "verified": verified})
        ledger["verified"] = verified
    result = archive_one(job, attempt, record, dest_host, root)
    updated = {
        **ledger,
        "verified": {**ledger["verified"], identity: {"job": job, "attempt": attempt}},
    }
    save_ledger(ledger_path, updated)
    ledger["verified"] = updated["verified"]
    return result


def error_message(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        detail = exc.stderr
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        return f"Remote command failed (exit {exc.returncode}): {(detail or '').strip()}"
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"Remote command timed out after {exc.timeout} seconds"
    return str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dest-host", default="comech-2080")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--ledger",
        type=Path,
        help="Client verification ledger (default: archive-ledger.json beside the state directory)",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Force a full source and destination checksum audit",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.poll < 1 or args.retries < 1:
        parser.error("poll and retries must be positive")
    ledger_path = args.ledger or args.state.parent / "archive-ledger.json"
    if ledger_path.resolve().is_relative_to(args.state.resolve()):
        parser.error("ledger must be outside the controller state directory")
    ledger = load_ledger(ledger_path)
    while True:
        errors = 0
        for job, attempt, record in selected_records(args.state):
            for trial in range(args.retries):
                try:
                    result = process_record(
                        job,
                        attempt,
                        record,
                        args.dest_host,
                        args.root,
                        ledger,
                        ledger_path,
                        args.verify_existing,
                    )
                    print(f"{job}/{attempt}: {result}", flush=True)
                    break
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    if trial + 1 == args.retries:
                        errors += 1
                        print(f"{job}/{attempt}: {error_message(exc)}", file=sys.stderr, flush=True)
                    else:
                        time.sleep(min(2**trial, 10))
        if not args.watch:
            return int(bool(errors))
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
