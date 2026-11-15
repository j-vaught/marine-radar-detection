"""Build portable, ledger-verified checkpoint assets without publishing them."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickletools
import re
import shutil
import struct
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PRIVATE = re.compile(
    r"(?:/(?:home|mnt|tmp|root|Users|Volumes|scratch|data|media|private)(?:/|$)|"
    r"GPU-[0-9a-fA-F-]{16,}|\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b)"
)
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "credential",
    "credentials",
    "secret",
    "token",
    "cookie",
    "cookies",
}
DROP_KEYS = {
    "gpu_uuid",
    "uuid",
    "host",
    "hostname",
    "nonce",
    "attempt_nonce",
    "pid",
    "boot_id",
    "start_ticks",
    "command",
    "runtime_provenance",
    "job",
    "receipt",
    "training_parent",
    "cuda_visible_devices",
    "evaluation_device",
    "device",
    "save_dir",
    "project",
}
ARTIFACT = re.compile(
    r"(?:[A-Za-z_]+_(?:raw_predictions|predictions|raw|operating|gt)|val_predictions|val_raw_predictions|"
    r"history|postprocessing_selection|support_stream|training_audit|subset_ids)\.json\Z"
)
METRICS_FIELDS = (
    "best_epoch",
    "best_update",
    "best_val_loss",
    "best_fitness",
    "training_updates",
    "batch_size",
    "validation_interval",
    "condition",
    "stage",
    "scale",
    "intensity",
    "size_control",
    "annotation_sha256",
    "manifest_sha256",
    "frozen_protocol_sha256",
    "adaptation_protocol_sha256",
    "base_protocol_sha256",
    "source_checkpoint_sha256",
    "target_local_checkpoint_sha256",
    "support_image_ids",
    "support_image_ids_sha256",
    "support_box_count",
    "trainable_parameter_names",
    "evaluation_recipe_version",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or not pure.parts or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*pure.parts)


def public_value(value):
    """Remove operational identifiers and replace original absolute paths."""
    if isinstance(value, dict):
        return {
            key: public_value(item)
            for key, item in value.items()
            if key.lower() not in DROP_KEYS | SENSITIVE_KEYS
            and not key.lower().endswith(
                ("_password", "_credential", "_secret", "_token", "_api_key")
            )
        }
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str) and PRIVATE.search(value):
        if value.startswith("/") and not PRIVATE.search(Path(value).name):
            return Path(value).name
        return "[operational identifier removed]"
    return value


def save_json(path: Path, value):
    if public_value(value) != value:
        raise ValueError("Residual private metadata in release JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked_file(root: Path, relative: str, ledger: dict) -> Path:
    relative_path = safe_relative(relative)
    path = root / relative_path
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError(f"Symlink is not accepted: {relative}")
    expected = ledger["files"].get(relative)
    if not expected or not path.is_file():
        raise ValueError(f"Missing ledger entry or file: {relative}")
    if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Archive digest mismatch: {relative}")
    return path


def rewrite_pickle_metadata(data: bytes) -> tuple[bytes, int]:
    """Replace private string constants, leaving tensor storages untouched.

    PyTorch's protocol-2 checkpoint pickle uses no frames. Refuse newer framed
    formats rather than emitting incorrect FRAME lengths.
    """
    operations = list(pickletools.genops(data))
    edits = []
    for index, (op, value, start) in enumerate(operations):
        if op.name not in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"}:
            continue
        if not isinstance(value, str) or not PRIVATE.search(value):
            continue
        replacement = (
            "release-local/" + Path(value).name if value.startswith("/") else "release-local"
        )
        if PRIVATE.search(replacement):
            replacement = "release-local"
        encoded = replacement.encode("utf-8")
        stop = operations[index + 1][2] if index + 1 < len(operations) else len(data)
        edits.append((start, stop, b"X" + struct.pack("<I", len(encoded)) + encoded))
    if edits and any(op.name == "FRAME" for op, _, _ in operations):
        raise ValueError("Metadata sanitization does not support framed pickle")
    for start, stop, replacement in reversed(edits):
        data = data[:start] + replacement + data[stop:]
    for op, value, _ in pickletools.genops(data):
        if (
            op.name in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"}
            and isinstance(value, str)
            and (PRIVATE.search(value) or value.lower() in SENSITIVE_KEYS)
        ):
            raise ValueError("Residual private checkpoint metadata")
    return data, len(edits)


def export_checkpoint(source: Path, destination: Path) -> dict:
    """Preserve original bytes except required private pickle string redaction."""
    rewrites = 0
    replacements = {}
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if name.endswith("/data.pkl") or name == "data.pkl":
                    rewritten, count = rewrite_pickle_metadata(archive.read(name))
                    if count:
                        replacements[name] = rewritten
                        rewrites += count
        if replacements:
            with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "w") as new:
                for info in old.infolist():
                    with new.open(copy.copy(info), "w") as out:
                        if info.filename in replacements:
                            out.write(replacements[info.filename])
                        else:
                            with old.open(info) as inp:
                                shutil.copyfileobj(inp, out, 1024 * 1024)
        else:
            shutil.copyfile(source, destination)
    else:
        raise ValueError("Only inspectable ZIP checkpoints are accepted")
    original_sha, exported_sha = sha256(source), sha256(destination)
    if not rewrites and original_sha != exported_sha:
        raise ValueError("Checkpoint copy failed end-to-end verification")
    return {
        "original_checkpoint_sha256": original_sha,
        "checkpoint_sha256": exported_sha,
        "checkpoint_metadata_strings_rewritten": rewrites,
        "tensor_storages_changed": False,
    }


def tensor_digest(checkpoint: Path, method: str) -> str:
    import torch

    from .darc_adaptation import _state_digest

    loaded = torch.load(
        checkpoint, map_location="cpu", weights_only=method not in {"yolo", "rtdetr"}
    )
    if method in {"yolo", "rtdetr"}:
        network = loaded.get("ema")
        if network is None:
            network = loaded["model"]
        loaded = network.state_dict()
    return _state_digest(loaded)


def package_run(entry: dict, staging: Path) -> dict:
    job_id, scope = entry["job_id"], entry["scope"]
    if not SAFE_ID.fullmatch(job_id) or not SAFE_ID.fullmatch(entry["asset_group"]):
        raise ValueError("Unsafe run or asset identifier")
    archive = Path(entry["archive_root"])
    attempt = int(entry["attempt"])
    run = archive / job_id / str(attempt)
    ledger = json.loads((archive / ".ledger" / job_id / f"{attempt}.json").read_text())
    if not ledger.get("verified"):
        raise ValueError("Archive ledger is not verified")
    metrics_path = checked_file(run, "metrics.json", ledger)
    metrics_digest = sha256(metrics_path)
    if metrics_digest != entry["metrics_sha256"]:
        raise ValueError("Selected state and archive metrics differ")
    metrics = json.loads(metrics_path.read_text())
    completion_path = checked_file(run, "completion.json", ledger)
    completion = json.loads(completion_path.read_text())
    if (
        completion["state"] != "succeeded"
        or completion["job_id"] != job_id
        or completion["attempt"] != attempt
        or completion["metrics_sha256"] != metrics_digest
        or metrics["job_id"] != job_id
    ):
        raise ValueError("Selected attempt does not have a matching successful completion")
    method = metrics.get("method", "faster_rcnn")
    if scope == "adaptation":
        checkpoint, expected = "adapted_final.pt", metrics["final_checkpoint_sha256"]
    elif method in {"yolo", "rtdetr"}:
        checkpoint, expected = "fit/weights/best.pt", metrics["best_sha256"]
    else:
        checkpoint, expected = "best.pt", metrics["best_weights_sha256"]
    original = checked_file(run, checkpoint, ledger)
    if sha256(original) != expected:
        raise ValueError("Selected checkpoint does not match metrics")
    destination = staging / "runs" / job_id
    destination.mkdir(parents=True, exist_ok=False)
    export = destination / "checkpoint.pt"
    checkpoint_meta = export_checkpoint(original, export)
    original_tensor = tensor_digest(original, method)
    if (
        checkpoint_meta["checkpoint_metadata_strings_rewritten"]
        and tensor_digest(export, method) != original_tensor
    ):
        raise ValueError("Export changed checkpoint tensor state")
    if scope in {"darc", "adaptation"}:
        from .darc_train import _cfg

        config, initialization = _cfg(), "pretrained"
    elif method in {"faster_rcnn", "unet"}:
        resolved_path = checked_file(run, "resolved_config.json", ledger)
        if sha256(resolved_path) != metrics["resolved_config_sha256"]:
            raise ValueError("Resolved configuration differs from selected metrics")
        resolved = json.loads(resolved_path.read_text())
        config, initialization = resolved["config"], resolved["initialization"]
    else:
        config, initialization = metrics["config"], metrics["initialization"]
    numerical_policy = public_value(completion.get("runtime_provenance", {}).get("torch", {}))
    result = {
        "schema_version": 1,
        "scope": scope,
        "job_id": job_id,
        "attempt": attempt,
        "method": method,
        "source": metrics["source"],
        "seed": metrics["seed"],
        "target": metrics.get("target"),
        "checkpoint": "checkpoint.pt",
        **checkpoint_meta,
        "tensor_state_sha256": original_tensor,
        "config": public_value(config),
        "initialization": initialization,
        "pixel_threshold": metrics.get("pixel_threshold"),
        "threshold": metrics.get(
            "score_threshold", metrics.get("threshold", metrics.get("selected_threshold"))
        ),
        "intensity": metrics.get("intensity", "I0"),
        "evaluation_scale": "S0",
        "numerical_policy": numerical_policy,
        "training_or_repair_code_commit": completion["code_commit"],
        "source_metrics_sha256": metrics_digest,
        "source_completion_sha256": sha256(completion_path),
        "archive_ledger_verified": ledger["verified"],
        "selection": {key: metrics[key] for key in METRICS_FIELDS if key in metrics},
        "artifacts": {},
    }
    for relative in sorted(ledger["files"]):
        if "/" not in relative and ARTIFACT.fullmatch(relative):
            source = checked_file(run, relative, ledger)
            # COCO filenames are relative; reject unexpected operational content.
            value = json.loads(source.read_text())
            if public_value(value) != value:
                raise ValueError(f"Unexpected private metadata in scientific artifact: {relative}")
            shutil.copyfile(source, destination / relative)
            if sha256(destination / relative) != ledger["files"][relative]["sha256"]:
                raise ValueError("Scientific artifact copy failed verification")
            result["artifacts"][relative] = ledger["files"][relative]
    public_metrics = {key: metrics[key] for key in METRICS_FIELDS if key in metrics}
    public_metrics.update(
        {
            key: metrics[key]
            for key in (
                "job_id",
                "method",
                "source",
                "target",
                "seed",
                "validation",
                "source_validation",
                "tests",
                "pixel_threshold",
                "threshold",
                "selected_threshold",
                "score_threshold",
            )
            if key in metrics
        }
    )
    save_json(destination / "metrics.json", public_value(public_metrics))
    save_json(destination / "run.json", result)
    return {
        "job_id": job_id,
        "asset_group": entry["asset_group"],
        "run": f"runs/{job_id}/run.json",
        **checkpoint_meta,
    }


def inventory(root: Path) -> dict:
    return {
        file.relative_to(root).as_posix(): {"sha256": sha256(file), "size": file.stat().st_size}
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def write_asset(root: Path, paths: list[Path], destination: Path, max_bytes: int) -> dict:
    raw_bytes = sum(path.stat().st_size for path in paths)
    # Worst-case gzip overhead is small but permit a fixed margin and reject actual overflow.
    if raw_bytes > max_bytes - 2 * 1024 * 1024:
        raise ValueError("Asset group exceeds uncompressed safety budget; split the plan")
    with (
        destination.open("xb") as out,
        gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0, compresslevel=1) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for path in sorted(paths):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as content:
                archive.addfile(info, content)
    if destination.stat().st_size > max_bytes:
        raise ValueError("Compressed asset exceeds budget")
    return {
        "file": destination.name,
        "sha256": sha256(destination),
        "size": destination.stat().st_size,
        "uncompressed_size": raw_bytes,
    }


def build(plan: dict, output: Path) -> dict:
    import torch

    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    staging = output / "package"
    staging.mkdir()
    selected = []
    ids = [entry["job_id"] for entry in plan["runs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Plan contains duplicate job IDs")
    for entry in plan["runs"]:
        selected.append(package_run(entry, staging))
        print(json.dumps({"packaged": entry["job_id"]}), flush=True)
    files = inventory(staging)
    scope = plan["scope"]
    if not SAFE_ID.fullmatch(scope):
        raise ValueError("Unsafe scope identifier")
    manifest_dir = staging / "manifests"
    inventory_path = manifest_dir / f"{scope}-inventory.json"
    selected_path = manifest_dir / f"{scope}-selected-runs.json"
    save_json(inventory_path, {"schema_version": 1, "files": files})
    save_json(selected_path, {"schema_version": 1, "runs": selected})
    assets_dir = output / "assets"
    assets_dir.mkdir()
    groups: dict[str, list[Path]] = {}
    for run in selected:
        groups.setdefault(run["asset_group"], []).extend(
            file for file in (staging / "runs" / run["job_id"]).rglob("*") if file.is_file()
        )
    max_bytes = int(plan.get("max_asset_bytes", 1_800_000_000))
    assets = [
        write_asset(staging, paths, assets_dir / f"{group}.tar.gz", max_bytes)
        for group, paths in sorted(groups.items())
    ]
    assets.append(
        write_asset(
            staging,
            [inventory_path, selected_path],
            assets_dir / f"{scope}-manifest.tar.gz",
            max_bytes,
        )
    )
    result = {
        "schema_version": 1,
        "published": False,
        "scope": plan["scope"],
        "run_count": len(selected),
        "assets": assets,
    }
    save_json(output / "release-index.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Private archive-selection plan")
    parser.add_argument("--output", type=Path, required=True, help="New HDD directory")
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.plan.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
