"""Isolated DARC processed-image experiment scheduler.

Factories and dispatch do not create or modify the historical benchmark state.
Both diagnostics and target-scoring fits require explicit, manifest-bound gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import uuid

from src.revision import fleet

SCHEMA = "darc-v2-jobs-1"
SOURCES = ("DAAN", "DARC", "MANV")
ARMS = (("S0", "I0"), ("S0", "I1"), ("S1", "I0"), ("S1", "I1"))
ALLOWED = {"comech-2422", "cocesr-1141", "comech-2080", "cocesr-1034", "comech-1176"}
GPU_MODEL = "NVIDIA RTX 6000 Ada Generation"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_jobs(protocol, catalog):
    """Catalog: {SOURCE: {str(seed): {path, sha256, tensor_sha256}}}."""
    jobs = []
    for source in SOURCES:
        for seed in range(3):
            init = catalog[source][str(seed)]
            if not all(init.get(k) for k in ("path", "sha256", "tensor_sha256")):
                raise ValueError(f"missing initialization provenance for {source} seed {seed}")
            common = {"method": "faster_rcnn", "source": source, "seed": seed,
                      "condition": "buffered", "ann": "{data}/annotations/all.json",
                      "images": "{data}/images", "manifest": "{root}/protocol/split_manifest.json",
                      "initial_state": init["path"], "initial_state_sha256": init["sha256"],
                      "initial_tensor_sha256": init["tensor_sha256"]}
            for scale, intensity in ARMS:
                base = {**common, "scale": scale, "intensity": intensity, "size_control": False}
                jobs.append({**base, "job_id": f"darc-v2-fit-{source}-s{seed}-{scale}{intensity}",
                             "stage": "core", "eval_targets": list(SOURCES)})
                if seed == 0:
                    jobs.append({**base, "job_id": f"darc-v2-diag-{source}-s{seed}-{scale}{intensity}",
                                 "stage": "diagnostic", "eval_targets": [],
                                 "diagnostic_updates": 32, "diagnostic_val_interval": 16})
            if source in {"DAAN", "MANV"}:
                jobs.append({**common, "job_id": f"darc-v2-fit-{source}-s{seed}-S0I0-n316",
                             "stage": "core", "scale": "S0", "intensity": "I0",
                             "size_control": True, "eval_targets": list(SOURCES)})
    return {"schema": SCHEMA, "protocol_sha256": protocol, "jobs": jobs}


def gate_ok(path, kind, manifest_hash, protocol_hash, release, profile, jobs, records):
    if not path.is_file():
        return False
    try:
        gate = json.loads(path.read_text())
        same = (gate.get("approved") is True and gate.get("kind") == kind
                and gate.get("jobs_sha256") == manifest_hash
                and gate.get("protocol_sha256") == protocol_hash
                and gate.get("release_sha") == release
                and gate.get("runtime_profile_sha256") == profile)
        if not same:
            return False
        if kind == "scoring":
            diag = {j["job_id"] for j in jobs if j["stage"] == "diagnostic"}
            return bool(diag) and set(gate.get("diagnostic_job_ids", [])) == diag and all(
                records.get(job_id, {}).get("state") == "succeeded" for job_id in diag)
        return True
    except (ValueError, TypeError):
        return False


def worker(job_path, completion_path, attempt):
    job = json.loads(Path(job_path).read_text())
    receipt = {"job_id": job["job_id"], "attempt": attempt, "nonce": job["attempt_nonce"],
               **fleet.process_identity()}
    fleet.atomic_json(Path(job["out_dir"]) / "receipt.json", receipt)
    result = {**receipt, "state": "failed", "started": fleet.now(), "code_commit": job["code_commit"]}
    try:
        if job["stage"] == "diagnostic" and job.get("eval_targets"):
            raise ValueError("diagnostics must not score targets")
        for field, expected in (("manifest", job["split_sha256"]),
                                ("ann", job["annotations_sha256"]),
                                ("initial_state", job["initial_state_sha256"])):
            if digest(job[field]) != expected:
                raise ValueError(f"{field} checksum mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu_uuid"]:
            raise ValueError("CUDA device assignment mismatch")
        if fleet.cuda_uuid().lower() != job["gpu_uuid"].lower():
            raise ValueError("CUDA process UUID mismatch")
        import torch
        if torch.cuda.get_device_properties(0).name != GPU_MODEL:
            raise ValueError("GPU model does not match frozen experiment")
        from src.revision.darc_train import run_job
        run_job(job)
        metrics = Path(job["out_dir"]) / "metrics.json"
        data = json.loads(metrics.read_text())
        if data.get("job_id") != job["job_id"]:
            raise ValueError("metrics job identity mismatch")
        if set(data.get("tests", {})) != set(job["eval_targets"]):
            raise ValueError("metrics target set mismatch")
        result.update(state="succeeded", exit_code=0, metrics_sha256=digest(metrics))
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
    try:
        result["runtime_provenance"] = fleet.runtime_provenance(job)
    except Exception as exc:  # noqa: BLE001
        result["provenance_error"] = f"{type(exc).__name__}: {exc}"
    result["finished"] = fleet.now()
    fleet.atomic_json(completion_path, result)
    return result["exit_code"]


def launch(host, cfg, job, record, spec):
    out, code = record["out"], cfg["code_path"]
    values = {"root": cfg["result_root"], "code": code, "data": cfg["data_path"],
              "source": job["source"], "seed": job["seed"]}
    resolved = fleet.resolve(job, values)
    resolved.update(out_dir=out, device="cuda:0", gpu_uuid=record["gpu_uuid"],
                    attempt_nonce=record["nonce"], code_commit=cfg["code_commit"],
                    split_sha256=spec["split_sha256"], annotations_sha256=spec["annotations_sha256"])
    run = [cfg["venv_python"], "-m", "src.revision.darc_fleet", "worker",
           out + "/job.json", record["completion"], str(record["attempt"])]
    if cfg.get("wrapper"):
        run.insert(0, cfg["wrapper"])
    script = '''import json,os,subprocess,sys
out,code,payload,cmd,gpu,threads,torch_home=sys.argv[1:]
os.makedirs(os.path.dirname(out),exist_ok=True)
os.mkdir(out)
open(os.path.join(out,'job.json'),'w').write(payload)
args=json.loads(cmd)
open(os.path.join(out,'command.txt'),'w').write(json.dumps(args))
f=open(os.path.join(out,'worker.log'),'ab',buffering=0)
env=os.environ.copy()
env.update(PYTHONPATH=code,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,
           OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,RAYON_NUM_THREADS=threads,TORCH_HOME=torch_home)
p=subprocess.Popen(args,cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
'''
    fleet.ssh(host, fleet.quote_cmd([cfg["venv_python"], "-c", script, out, code,
        json.dumps(resolved), json.dumps(run), record["gpu_uuid"], str(cfg.get("threads", 4)),
        cfg.get("torch_home", cfg["result_root"] + "/torch-cache")]))


def reconcile(job_id, record):
    status = fleet.remote_status(record["host"], record)
    receipt = status.get("receipt")
    if receipt and not fleet.matches(record, receipt, job_id):
        return {**record, "state": "uncertain", "error": "receipt identity mismatch"}
    if "completion" in status:
        done = status["completion"]
        if not receipt or not fleet.matches(record, done, job_id) or any(
            done.get(k) != receipt.get(k) for k in ("pid", "start_ticks", "boot_id")):
            return {**record, "state": "uncertain", "error": "completion identity mismatch"}
        if done.get("state") == "succeeded" and done.get("exit_code") == 0:
            return {**record, "state": "succeeded", "result": done}
        if done.get("state") == "failed" and done.get("exit_code") != 0:
            return {**record, "state": "failed", "result": done}
        return {**record, "state": "uncertain", "error": "invalid completion"}
    if receipt and status.get("alive") is False:
        return {**record, "state": "infra_failed", "error": "verified worker exited without completion"}
    if receipt and status.get("alive") is True:
        return {**record, "state": "running"}
    return {**record, "state": "uncertain", "error": status.get("unknown", "receipt pending")}


def schedule(args):
    spec = json.loads(args.jobs.read_text())
    jobs = spec["jobs"]
    if spec.get("schema") != SCHEMA or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("invalid or duplicate DARC manifest")
    if any(not j["job_id"].startswith("darc-v2-") for j in jobs):
        raise ValueError("historical job IDs are not allowed")
    if (sum(j.get("stage") == "core" for j in jobs) != 42
            or sum(j.get("stage") == "diagnostic" for j in jobs) != 12
            or len(jobs) != 54
            or any((j.get("stage") == "diagnostic" and j.get("eval_targets") != [])
                   or (j.get("stage") == "core" and j.get("eval_targets") != list(SOURCES)) for j in jobs)):
        raise ValueError("DARC manifest has unexpected stages, counts, or target sets")
    if args.state.name != "state" or args.state.parent.name != "controller" or args.state.parent.parent.name != "darc-v2":
        raise ValueError("state must be isolated under darc-v2/controller/state")
    if args.status:
        for file in sorted(args.state.glob("*.json")):
            print(file.read_text().strip())
        return
    args.state.mkdir(parents=True, exist_ok=True)
    lock = (args.state / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs_hash = digest(args.jobs)
    while True:
        hosts = json.loads(args.hosts.read_text())["hosts"]
        if not set(hosts) <= ALLOWED:
            raise ValueError("host configuration contains a host outside the DARC allowlist")
        records = {}
        pending = {}
        for job in jobs:
            path = args.state / (job["job_id"] + ".json")
            if path.is_file():
                record = json.loads(path.read_text())
                records[job["job_id"]] = record
                if record["state"] in {"running", "launching", "uncertain"}:
                    pending[job["job_id"]] = record
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(pending)))) as pool:
            futures = {job_id: pool.submit(reconcile, job_id, record) for job_id, record in pending.items()}
        for job_id, future in futures.items():
            try:
                records[job_id] = future.result()
                fleet.atomic_json(args.state / (job_id + ".json"), records[job_id])
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f"{fleet.now()} {job_id} status unavailable: {fleet.concise_error(exc)}", flush=True)
        diag_ok = gate_ok(args.diagnostic_gate, "diagnostic", jobs_hash, spec["protocol_sha256"],
                          args.release, args.profile, jobs, records)
        scoring_ok = diag_ok and gate_ok(args.scoring_gate, "scoring", jobs_hash, spec["protocol_sha256"],
                             args.release, args.profile, jobs, records)
        eligible = {name: cfg for name, cfg in hosts.items() if cfg.get("enabled") is True
                    and cfg.get("data_verified") is True and cfg.get("initialization_verified") is True
                    and cfg.get("code_commit") == args.release and cfg.get("runtime_profile_sha256") == args.profile
                    and int(cfg.get("threads", 4)) == 4
                    and cfg.get("protocol_sha256") == spec["split_sha256"]
                    and cfg.get("annotations_sha256") == spec["annotations_sha256"]
                    and Path(cfg.get("result_root", "/")).name == "darc-v2"
                    and str(cfg.get("code_path", "")).startswith(str(cfg["result_root"]) + "/releases/")
                    and cfg.get("venv_python") and cfg.get("data_path")}
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(eligible)))) as pool:
            futures = {host: pool.submit(fleet.gpu_free, host, cfg, include_models=True)
                       for host, cfg in eligible.items()}
        used = {(r.get("host"), r.get("device")) for r in records.values()
                if r.get("state") in {"running", "launching", "uncertain"}}
        for host, cfg in eligible.items():
            try:
                free = futures[host].result()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f"{fleet.now()} {host} GPU check failed: {fleet.concise_error(exc)}", flush=True)
                continue
            for device in free:
                if len(device) != 3 or device[2] != GPU_MODEL or (host, device[0]) in used:
                    continue
                candidates = [job for job in jobs
                    if ((job["stage"] == "diagnostic" and diag_ok) or
                        (job["stage"] == "core" and scoring_ok))
                    and (job["job_id"] not in records or
                         (records[job["job_id"]]["state"] == "infra_failed"
                          and records[job["job_id"]]["attempt"] < 3))]
                if not candidates:
                    continue
                job = min(candidates, key=lambda j: j["stage"] != "diagnostic")
                try:
                    if device not in fleet.gpu_free(host, cfg, include_models=True):
                        continue
                except (OSError, ValueError, subprocess.SubprocessError):
                    continue
                attempt = records.get(job["job_id"], {}).get("attempt", 0) + 1
                out = f"{cfg['result_root']}/runs/{job['job_id']}/{attempt}"
                record = {"state": "launching", "host": host, "python": cfg["venv_python"], "out": out,
                          "receipt": out + "/receipt.json", "completion": out + "/completion.json",
                          "device": device[0], "gpu_uuid": device[1], "attempt": attempt,
                          "nonce": uuid.uuid4().hex, "started": fleet.now()}
                path = args.state / (job["job_id"] + ".json")
                fleet.atomic_json(path, record)
                try:
                    launch(host, cfg, job, record, spec)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    record.update(state="uncertain", error=fleet.concise_error(exc))
                fleet.atomic_json(path, record)
                records[job["job_id"]] = record
                used.add((host, device[0]))
                print(f"{fleet.now()} {job['job_id']} {record['state']} {host} GPU {device[0]}", flush=True)
        counts = {}
        for job in jobs:
            state = records.get(job["job_id"], {}).get("state", "pending")
            counts[state] = counts.get(state, 0) + 1
        print(f"{fleet.now()} {json.dumps(counts, sort_keys=True)} diagnostics={bool(diag_ok)} scoring={bool(scoring_ok)}", flush=True)
        if args.once:
            break
        time.sleep(args.poll)


REQUIRED_CHECKS = {
    "diagnostic": {"release_bytes", "data_and_split_hashes", "initialization_hashes", "unit_tests"},
    "scoring": {"diagnostic_artifacts", "shared_initialization", "spatial_and_box_geometry",
                "finite_gradients_and_valid_predictions", "source_only_selection", "runtime_precision",
                "source_scale_area_audit"},
}


def build_gate(args):
    spec = json.loads(args.jobs.read_text())
    if spec.get("schema") != SCHEMA:
        raise ValueError("invalid DARC jobs schema")
    jobs_hash = digest(args.jobs)
    report = json.loads(args.audit.read_text())
    if (report.get("schema") != f"darc-v2-{args.kind}-audit-1"
        or report.get("jobs_sha256") != jobs_hash
        or report.get("protocol_sha256") != spec["protocol_sha256"]
        or report.get("release_sha") != args.release
        or report.get("runtime_profile_sha256") != args.profile):
        raise ValueError("audit identity does not match jobs, protocol, release and runtime profile")
    checks = report.get("checks", {})
    for name in REQUIRED_CHECKS[args.kind]:
        check = checks.get(name, {})
        evidence = check.get("evidence", [])
        if check.get("passed") is not True or not evidence:
            raise ValueError(f"missing passed evidence for {name}")
        for item in evidence:
            path = Path(item["path"])
            if not path.is_absolute() or not path.is_file() or digest(path) != item.get("sha256"):
                raise ValueError(f"invalid evidence for {name}: {path}")
    gate = {"approved": True, "kind": args.kind, "jobs_sha256": jobs_hash,
            "protocol_sha256": spec["protocol_sha256"], "release_sha": args.release,
            "runtime_profile_sha256": args.profile, "audit_sha256": digest(args.audit)}
    if args.kind == "scoring":
        diagnostics = [j["job_id"] for j in spec["jobs"] if j["stage"] == "diagnostic"]
        if len(diagnostics) != 12:
            raise ValueError("expected exactly 12 diagnostics")
        for job_id in diagnostics:
            rec = json.loads((args.state / (job_id + ".json")).read_text())
            if rec.get("state") != "succeeded" or rec.get("result", {}).get("job_id") != job_id or not rec.get("result", {}).get("metrics_sha256"):
                raise ValueError(f"missing validated diagnostic completion: {job_id}")
        gate["diagnostic_job_ids"] = diagnostics
    fleet.atomic_json(args.output, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("factory")
    f.add_argument("--protocol", type=Path, required=True)
    f.add_argument("--catalog", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("schedule")
    for name in ("jobs", "hosts", "state", "diagnostic-gate", "scoring-gate"):
        s.add_argument("--" + name, type=Path, required=True)
    s.add_argument("--release", required=True)
    s.add_argument("--profile", required=True)
    s.add_argument("--poll", type=int, default=60)
    s.add_argument("--once", action="store_true")
    s.add_argument("--status", action="store_true")
    g = sub.add_parser("build-gate")
    g.add_argument("--kind", choices=["diagnostic", "scoring"], required=True)
    for name in ("jobs", "audit", "state", "output"):
        g.add_argument("--" + name, type=Path, required=True)
    g.add_argument("--release", required=True)
    g.add_argument("--profile", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--state", type=Path, required=True)
    c.add_argument("--dest", type=Path, required=True)
    c.add_argument("--metadata-only", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("job")
    w.add_argument("completion")
    w.add_argument("attempt", type=int)
    args = parser.parse_args()
    if args.mode == "factory":
        protocol = json.loads(args.protocol.read_text())
        spec = make_jobs(digest(args.protocol), json.loads(args.catalog.read_text()))
        spec.update(split_sha256=protocol["inputs"]["split_sha256"],
                    annotations_sha256=protocol["inputs"]["annotations_sha256"],
                    initial_catalog_sha256=digest(args.catalog))
        fleet.atomic_json(args.output, spec)
    elif args.mode == "worker":
        sys.exit(worker(args.job, args.completion, args.attempt))
    elif args.mode == "build-gate":
        build_gate(args)
    elif args.mode == "collect":
        for path in sorted(args.state.glob("darc-v2-*.json")):
            record = json.loads(path.read_text())
            if record.get("state") not in {"succeeded", "failed", "infra_failed"}:
                continue
            dest = args.dest / path.stem / str(record["attempt"])
            dest.mkdir(parents=True, exist_ok=True)
            command = ["rsync", "-a", "-e", fleet.quote_cmd(["ssh", *fleet.ssh_options(record["host"])])]
            if args.metadata_only:
                command += ["--include=*/", "--include=*.json", "--include=*.csv", "--include=*.log", "--exclude=*"]
            subprocess.run([*command, f"{record['host']}:{record['out']}/", str(dest) + "/"], check=True)
    else:
        schedule(args)


if __name__ == "__main__":
    main()
