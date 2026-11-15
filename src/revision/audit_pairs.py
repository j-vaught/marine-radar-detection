"""Audit matched core/adjacency metadata without emitting target accuracy.

Run locally against a collected archive, or use --ssh to read the runs directory
on a remote host. State and artifacts are never modified. Example:
python -m src.revision.audit_pairs --state STATE_DIR --runs-dir ARCHIVE_DIR --ssh HOST
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

METHODS = ("yolo", "faster_rcnn")
SITES = ("DAAN", "DARC", "MANV")
TARGETS = set(SITES)
# Conservative inventory: differences require review, not an assumption of equivalence.
TRAIN_FILES = {
    "yolo": (
        "src/revision/ultralytics_models.py",
        "src/revision/protocol.py",
        "src/yolo/config.yaml",
        "src/data/radar_mask.py",
    ),
    "faster_rcnn": (
        "src/revision/torch_models.py",
        "src/revision/protocol.py",
        "src/faster_rcnn/model.py",
        "src/faster_rcnn/config.yaml",
        "src/data/radar_mask.py",
    ),
}


def read_records(states: dict, root: Path) -> dict:
    """Return only metadata, digests and history length; never return metrics."""
    records: dict[str, Any] = {}
    for name, state in states.items():
        if state.get("state") != "succeeded":
            continue
        p = root / name / str(state["attempt"])
        try:
            data = (p / "metrics.json").read_bytes()
            m = json.loads(data)
            j = json.loads((p / "job.json").read_text())
            c = json.loads((p / "completion.json").read_text())
            cfg = p / "resolved_config.json"
            log = p / ("fit/results.csv" if j["method"] == "yolo" else "history.json")
            rows = None
            if log.is_file():
                if j["method"] == "yolo":
                    with log.open(newline="") as stream:
                        rows = sum(1 for _ in csv.DictReader(stream))
                else:
                    rows = len(json.loads(log.read_text()))
            keys = (
                "job_id",
                "stage",
                "condition",
                "method",
                "source",
                "seed",
                "manifest_sha256",
                "annotation_sha256",
                "validation_gt_sha256",
                "initialization",
                "initial_sha256",
                "initial_state_sha256",
                "initialized_state_sha256",
                "initial_model",
                "architecture",
                "best_epoch",
                "best_sha256",
                "best_weights_sha256",
                "config",
                "train_args",
                "resolved_train_args",
                "training_parent",
                "evaluation_recipe_version",
            )
            records[name] = {
                "attempt": state["attempt"],
                "completion": c,
                "job": j,
                "metrics_sha256_actual": hashlib.sha256(data).hexdigest(),
                "metadata": {k: m.get(k) for k in keys},
                "test_digests": {
                    k: {
                        z: v.get(z)
                        for z in ("checkpoint_sha256", "gt_sha256", "ground_truth_sha256")
                    }
                    for k, v in m.get("tests", {}).items()
                },
                "resolved_config": json.loads(cfg.read_text()) if cfg.exists() else None,
                "history_rows": rows,
            }
        except (OSError, ValueError, KeyError) as exc:
            records[name] = {"attempt": state["attempt"], "unavailable": str(exc)}
    return records


def code_evidence(method: str, left: str | None, right: str | None) -> dict:
    """Compare tracked training-related file contents when both commits are available."""
    if not left or not right:
        return {"status": "unverified", "reason": "missing training commit"}
    if left == right:
        return {"status": "same_commit", "commit": left}
    hashes: dict[str, list[str]] = {}
    for file in TRAIN_FILES[method]:
        pair = []
        for commit in (left, right):
            result = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "show", f"{commit}:{file}"],
                capture_output=True,
                check=False,
            )
            if result.returncode:
                return {
                    "status": "unverified",
                    "reason": f"cannot read {file} at {commit}",
                    "hashes": hashes,
                }
            pair.append(hashlib.sha256(result.stdout).hexdigest())
        hashes[file] = pair
    return {
        "status": "files_equal" if all(a == b for a, b in hashes.values()) else "unverified",
        "hashes": hashes,
    }


def compare_pair(method: str, site: str, seed: int, states: dict, records: dict) -> dict:
    names = [f"core-{method}-{site}-s{seed}", f"adjacency-{method}-{site}-s{seed}"]
    row: dict[str, Any] = {
        "method": method,
        "source": site,
        "seed": seed,
        "runs": {},
        "critical": [],
        "unverified": [],
        "hardware": {},
        "differences": [],
    }
    for name in names:
        state, rec = states.get(name, {}), records.get(name)
        if not rec or "unavailable" in rec:
            row["runs"][name] = {"state": state.get("state", "not_dispatched"), "archive": rec}
            continue
        m, j, c = rec["metadata"], rec["job"], rec["completion"]
        parent = m.get("training_parent") or c
        row["runs"][name] = {
            "state": "available",
            "attempt": rec["attempt"],
            "training_code": parent.get("code_commit"),
            "evaluation_code": c.get("code_commit"),
            "checkpoint_selection": "source-validation fitness"
            if method == "yolo"
            else "minimum source-validation loss",
        }
        if c.get("state") != "succeeded" or c.get("metrics_sha256") != rec["metrics_sha256_actual"]:
            row["critical"].append(f"{name}: completion/metrics identity failure")
        expected = {
            "job_id": name,
            "method": method,
            "source": site,
            "seed": seed,
            "stage": "core" if name.startswith("core-") else "adjacency",
            "condition": "buffered" if name.startswith("core-") else "adjacency",
        }
        for key, value in expected.items():
            if j.get(key) != value or m.get(key) != value:
                row["critical"].append(f"{name}: identity mismatch {key}")
        ck = m.get("best_sha256") or m.get("best_weights_sha256")
        tests = rec["test_digests"]
        if (
            set(tests) != TARGETS
            or not ck
            or any(t.get("checkpoint_sha256") != ck for t in tests.values())
        ):
            row["critical"].append(f"{name}: inconsistent target checkpoint identity")
        if m.get("training_parent") and parent.get("checkpoint_sha256") != ck:
            row["critical"].append(f"{name}: repair parent checkpoint mismatch")
        epoch, rows = m.get("best_epoch"), rec.get("history_rows")
        if (
            not isinstance(epoch, int)
            or epoch < (0 if method == "yolo" else 1)
            or not rows
            or epoch > rows
        ):
            row["unverified"].append(f"{name}: selected epoch/history missing or inconsistent")
        if method == "yolo" and m.get("evaluation_recipe_version") != "yolo-exact-letterbox-v1":
            row["unverified"].append(f"{name}: corrected evaluation version missing")
        runtime = parent.get("runtime_provenance") or {}
        row["hardware"][name] = {
            "gpu": runtime.get("gpu"),
            "torch": runtime.get("torch"),
            "evaluation_device": j.get("device"),
        }
    if all(records.get(n) and "unavailable" not in records[n] for n in names):
        a, b = (records[n] for n in names)
        ma, mb, ja, jb = a["metadata"], b["metadata"], a["job"], b["job"]
        for key in (
            "method",
            "source",
            "seed",
            "initialization",
            "epochs",
            "patience",
            "config_overrides",
            "eval_targets",
        ):
            if ja.get(key) != jb.get(key):
                row["critical"].append(f"job mismatch: {key}")
        for key in ("manifest_sha256", "annotation_sha256", "validation_gt_sha256"):
            if not ma.get(key) or ma.get(key) != mb.get(key):
                row["critical"].append(f"reference mismatch or missing: {key}")
        for target in SITES:
            x, y = (r["test_digests"].get(target, {}) for r in (a, b))
            da = x.get("gt_sha256") or x.get("ground_truth_sha256")
            db = y.get("gt_sha256") or y.get("ground_truth_sha256")
            if not da or da != db:
                row["critical"].append(f"target reference mismatch or missing: {target}")
        if method == "yolo":
            for key in ("initial_sha256", "architecture", "config"):
                if not ma.get(key) or ma.get(key) != mb.get(key):
                    row["critical"].append(f"initialization/config mismatch or missing: {key}")
            if (
                PurePosixPath(ma.get("initial_model") or "").name
                != PurePosixPath(mb.get("initial_model") or "").name
            ):
                row["critical"].append("initial model name mismatch")
            for key in ("train_args", "resolved_train_args"):
                va, vb = ma.get(key), mb.get(key)
                excluded = {"data", "project", "save_dir", "model", "device"}
                if not va or not vb:
                    row["unverified"].append(f"missing {key}")
                elif {k: v for k, v in va.items() if k not in excluded} != {
                    k: v for k, v in vb.items() if k not in excluded
                }:
                    row["critical"].append(f"effective training recipe mismatch: {key}")
            if not ma.get("initial_state_sha256") or ma.get("initial_state_sha256") != mb.get(
                "initial_state_sha256"
            ):
                row["unverified"].append(
                    "serialized initial-state hashes differ or missing; compare tensor contents"
                )
        else:
            ra, rb = a["resolved_config"], b["resolved_config"]
            if not ra or not rb:
                row["unverified"].append("resolved config missing")
            elif ra.get("config") != rb.get("config"):
                row["critical"].append("effective training/config mismatch")
            if not ma.get("initialized_state_sha256") or ma.get(
                "initialized_state_sha256"
            ) != mb.get("initialized_state_sha256"):
                row["critical"].append("initialized model tensor hash mismatch or missing")
        row["code_evidence"] = code_evidence(
            method, row["runs"][names[0]]["training_code"], row["runs"][names[1]]["training_code"]
        )
        if row["code_evidence"]["status"] != "same_commit":
            row["unverified"].append("training-code equivalence requires review")
        if row["hardware"][names[0]].get("gpu") != row["hardware"][names[1]].get("gpu"):
            row["differences"].append(
                "training GPU identity differs; inspect hardware and backend metadata"
            )
    row["status"] = (
        "critical"
        if row["critical"]
        else (
            "incomplete"
            if any(v["state"] != "available" for v in row["runs"].values())
            else ("unverified" if row["unverified"] else "matched")
        )
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--runs-dir", required=True, help="Archive root, local or on --ssh host")
    parser.add_argument("--ssh", help="SSH host for read-only archive access")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--internal-read", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_read:
        print(json.dumps(read_records(json.load(sys.stdin), Path(args.runs_dir))))
        return
    if not args.state.is_dir():
        parser.error("--state must be an existing directory")
    states = {
        p.stem: json.loads(p.read_text())
        for p in args.state.glob("*.json")
        if p.stem.startswith(
            tuple(f"{stage}-{method}-" for stage in ("core", "adjacency") for method in METHODS)
        )
    }
    if args.ssh:
        # Run this standard-library module remotely; only filtered metadata crosses SSH.
        remote = [
            "python3",
            "-c",
            "exec(__import__('base64').b64decode(__import__('sys').argv.pop(1)))",
            base64.b64encode(Path(__file__).read_bytes()).decode(),
            "--state",
            ".",
            "--runs-dir",
            args.runs_dir,
            "--internal-read",
        ]
        command = ["ssh", "-o", "BatchMode=yes", args.ssh, shlex.join(remote)]
        process = subprocess.run(
            command, input=json.dumps(states), text=True, capture_output=True, check=True
        )
        records = json.loads(process.stdout)
    else:
        records = read_records(states, Path(args.runs_dir))
    report = {
        "scope": "Metadata only. Target accuracy excluded. Hardware differences are descriptive. Selection criterion is recorded, not independently recomputed.",
        "pairs": [
            compare_pair(m, s, seed, states, records)
            for m in METHODS
            for s in SITES
            for seed in range(3)
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
