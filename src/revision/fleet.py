"""Conservative, resumable SSH job scheduler. See --help for usage."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import stat
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GPU_METHODS = {"faster_rcnn", "unet", "yolo", "rtdetr"}


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def ssh_options(host):
    # Short fixed path avoids platform-specific UNIX socket length limits.
    directory = Path(f"/tmp/revision-ssh-{os.getuid()}")
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise OSError(f"Unsafe SSH control directory: {directory}")
    socket = directory / hashlib.sha256(host.encode()).hexdigest()[:20]
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=120",
        "-o",
        f"ControlPath={socket}",
    ]


def ssh(host, command, timeout=30):
    return subprocess.run(
        ["ssh", *ssh_options(host), host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout


def concise_error(exc):
    """Avoid logging embedded remote scripts or other full subprocess arguments."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"command timed out after {exc.timeout}s"
    if isinstance(exc, subprocess.CalledProcessError):
        return f"command exited with status {exc.returncode}"
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def quote_cmd(parts):
    return shlex.join([str(x) for x in parts])


def gpu_free(host, config):
    """GPU must have no compute process, irrespective of process ownership."""
    prefix = [config["wrapper"]] if config.get("wrapper") else []
    inventory = prefix + ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    processes = prefix + [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]
    marker = "__REVISION_PROCESS_LIST__"
    command = quote_cmd(inventory) + " && printf '\n" + marker + "\n' && " + quote_cmd(processes)
    output = ssh(host, command)
    parts = output.split(marker)
    if len(parts) != 2:
        raise ValueError("GPU query output missing delimiter")
    all_gpu, active = (part.strip().splitlines() for part in parts)
    busy = {line.split(",")[0].strip() for line in active if line.strip()}
    allowed = config.get("allowed_gpus")
    return [
        (int(index.strip()), gpu_uuid.strip())
        for index, gpu_uuid in (line.split(",", 1) for line in all_gpu)
        if gpu_uuid.strip() not in busy and (allowed is None or int(index.strip()) in allowed)
    ]


def resolve(value, values):
    if isinstance(value, str):
        return value.format_map(values)
    if isinstance(value, list):
        return [resolve(v, values) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, values) for k, v in value.items()}
    return value


def cuda_uuid():
    """Initialize logical device zero and locate this process in the driver process table."""
    import torch

    torch.cuda.set_device(0)
    probe = torch.empty(1, device="cuda:0")
    rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    found = {
        parts[0].strip()
        for row in rows
        if len(parts := row.split(",")) == 2 and parts[1].strip() == str(os.getpid())
    }
    del probe
    if len(found) != 1:
        raise RuntimeError(f"Cannot verify this worker's GPU UUID: {sorted(found)}")
    return found.pop()


def process_identity():
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    return {
        "pid": os.getpid(),
        "start_ticks": stat[21],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def validate_metrics(job, data):
    if not isinstance(data, dict) or data.get("job_id") != job["job_id"]:
        raise ValueError("metrics job identity mismatch")
    if (
        data.get("manifest_sha256", data.get("paths", {}).get("manifest_sha256"))
        != job["protocol_sha256"]
    ):
        raise ValueError("metrics manifest digest mismatch")
    for key in ("method", "stage", "condition"):
        if data.get(key) != job.get(key):
            raise ValueError(f"metrics {key} mismatch")
    if data.get("annotation_sha256") != hashlib.sha256(Path(job["ann"]).read_bytes()).hexdigest():
        raise ValueError("metrics annotation digest mismatch")
    tests = data.get("tests")
    if not isinstance(tests, dict) or set(tests) != set(job.get("eval_targets", [])):
        raise ValueError("metrics evaluation targets mismatch")
    for target, result in tests.items():
        raw = result.get("raw", result.get("raw_metrics", result))
        ap = raw.get("ap", raw.get("AP", raw.get("mAP")))
        if not isinstance(ap, (float, int)) or not math.isfinite(ap):
            raise ValueError(f"missing or nonfinite AP for {target}")
        prediction = result.get("raw_path") or data.get("paths", {}).get("subsets", {}).get(
            "test_" + target, {}
        ).get("predictions")
        if not prediction:
            prediction = str(Path(job["out_dir"]) / f"{target}_raw_predictions.json")
        if not Path(prediction).is_file():
            raise ValueError(f"missing predictions for {target}: {prediction}")
    if tests and job["method"] in GPU_METHODS:
        digest = (
            data.get("checkpoint_sha256")
            or data.get("best_sha256")
            or data.get("best_weights_sha256")
        )
        checkpoint = data.get("best_checkpoint") or data.get("checkpoint")
        if not digest or not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("missing checkpoint path or digest")
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != digest:
            raise ValueError("checkpoint digest mismatch")


def runtime_provenance(job):
    """Capture the effective process environment without changing backend settings."""
    result = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        result["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        result["cpu_affinity_count"] = None
    if not result["processor"]:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            result["processor"] = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.splitlines()
                    if line.startswith("model name") and ":" in line
                ),
                "",
            )
        except OSError:
            pass
    # Do not import torch for classical CPU jobs merely to collect metadata.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            result["torch"] = {
                "version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
            if job.get("gpu_uuid"):
                props = torch.cuda.get_device_properties(0)
                result["gpu"] = {
                    "uuid": job["gpu_uuid"],
                    "model": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                }
        except Exception as exc:  # noqa: BLE001 - provenance cannot break a completed job
            result["torch_provenance_error"] = f"{type(exc).__name__}: {exc}"
    return result


def worker(job_path, completion_path, attempt):
    job = json.loads(Path(job_path).read_text())
    nonce = job["attempt_nonce"]
    # Receipt is written by the worker itself, before imports or training.
    receipt = {"job_id": job["job_id"], "attempt": attempt, "nonce": nonce, **process_identity()}
    atomic_json(Path(job["out_dir"]) / "receipt.json", receipt)
    completion = {
        **receipt,
        "started": now(),
        "state": "failed",
        "failure_kind": "model",
        "code_commit": job.get("code_commit"),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_uuid": job.get("gpu_uuid"),
    }
    manifest = Path(job.get("manifest", ""))
    if manifest.is_file():
        completion["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    try:
        if not manifest.is_file():
            raise RuntimeError(f"Missing protocol manifest: {manifest}")
        if (
            not job.get("protocol_sha256")
            or completion["manifest_sha256"] != job["protocol_sha256"]
        ):
            raise RuntimeError("Protocol manifest SHA-256 mismatch or expected digest missing")
        if job.get("gpu_uuid"):
            if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu_uuid"]:
                raise RuntimeError("CUDA visibility differs from assigned GPU UUID")
            actual = cuda_uuid()
            completion["observed_gpu_uuid"] = actual
            if actual.lower() != job["gpu_uuid"].lower():
                raise RuntimeError(
                    f"CUDA device mismatch: expected {job['gpu_uuid']}, got {actual}"
                )
        from src.revision.run import run_job
        if job.get("evaluation_from"):
            run_job(job)
        else:
            from src.revision.storage import stage_checkpoint_scratch

            with stage_checkpoint_scratch(job):
                run_job(job)
        metrics = Path(job["out_dir"]) / "metrics.json"
        if not metrics.is_file():
            raise RuntimeError(f"Missing metrics: {metrics}")
        data = json.loads(metrics.read_text())
        validate_metrics(job, data)
        completion.update(
            state="succeeded",
            exit_code=0,
            failure_kind=None,
            metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
        )
    except BaseException as exc:  # noqa: BLE001 - record interrupted jobs before exiting
        traceback.print_exc()
        completion.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
    try:
        completion["runtime_provenance"] = runtime_provenance(job)
    except Exception as exc:  # noqa: BLE001 - provenance cannot break a completed job
        completion["runtime_provenance_error"] = f"{type(exc).__name__}: {exc}"
    completion["finished"] = now()
    storage_report = Path(job["out_dir"]) / "checkpoint_scratch_report.json"
    if storage_report.is_file():
        completion["checkpoint_storage"] = json.loads(storage_report.read_text())
        if completion["checkpoint_storage"].get("status") == "archive_failed":
            print("Checkpoint recovery required; completion withheld", flush=True)
            return 3
    atomic_json(completion_path, completion)
    return completion["exit_code"]


def remote_status(host, state):
    # The worker receipt binds PID to boot ID and kernel start ticks.
    script = """import json,os,sys
receipt,completion=sys.argv[1:]
if not os.path.isfile(receipt):
 print(json.dumps({'unknown':'receipt not present'})); sys.exit()
r=json.load(open(receipt))
storage=os.path.join(os.path.dirname(receipt),'checkpoint_scratch_report.json')
if os.path.isfile(storage) and json.load(open(storage)).get('status')=='archive_failed':
 print(json.dumps({'receipt':r,'unknown':'checkpoint archival requires recovery'})); sys.exit()
if os.path.isfile(completion):
 print(json.dumps({'receipt':r,'completion':json.load(open(completion))})); sys.exit()
pid=r['pid']
try:
 boot=open('/proc/sys/kernel/random/boot_id').read().strip()
 stat=open('/proc/%s/stat'%pid).read().split()
 alive=(boot==r['boot_id'] and stat[21]==str(r['start_ticks']) and stat[2]!='Z')
 print(json.dumps({'receipt':r,'alive':alive}))
except FileNotFoundError: print(json.dumps({'receipt':r,'alive':False}))
except Exception as e: print(json.dumps({'receipt':r,'unknown':str(e)}))
"""
    return json.loads(
        ssh(host, quote_cmd([state["python"], "-c", script, state["receipt"], state["completion"]]))
    )


def prepare_launch(host, cfg, job, attempt, device):
    root, _code = cfg["result_root"], cfg["code_path"]
    out = f"{root}/runs/{job['job_id']}/{attempt}"
    index, gpu_uuid = device if device is not None else (None, None)
    return {
        "state": "launching",
        "host": host,
        "python": cfg["venv_python"],
        "out": out,
        "receipt": out + "/receipt.json",
        "completion": out + "/completion.json",
        "device": index,
        "gpu_uuid": gpu_uuid,
        "attempt": attempt,
        "nonce": uuid.uuid4().hex,
        "started": now(),
    }


def launch(host, cfg, job, record):
    root, code, out = cfg["result_root"], cfg["code_path"], record["out"]
    values = {
        "root": root,
        "code": code,
        "data": cfg["data_path"],
        "source": job.get("source", ""),
        "job_id": job["job_id"],
        "attempt": record["attempt"],
        "weights": cfg.get("weights_path", code + "/weights"),
    }
    resolved = resolve(job, values)
    resolved.update(
        out_dir=out,
        device="cuda:0" if record["gpu_uuid"] else "cpu",
        gpu_uuid=record["gpu_uuid"],
        attempt_nonce=record["nonce"],
        weights_dir=cfg.get("weights_path", root + "/weights"),
        checkpoint_scratch_root=cfg.get("checkpoint_scratch_root"),
        code_commit=cfg.get("code_commit"),
        protocol_sha256=cfg.get("protocol_sha256"),
    )
    job_file, completion = out + "/job.json", record["completion"]
    run = [
        cfg["venv_python"],
        "-m",
        "src.revision.fleet",
        "worker",
        job_file,
        completion,
        str(record["attempt"]),
    ]
    if cfg.get("wrapper"):
        run.insert(0, cfg["wrapper"])
    script = """import json,os,subprocess,sys
out,code,payload,cmd,gpu,threads,torch_home=sys.argv[1:]
os.makedirs(os.path.dirname(out),exist_ok=True)
os.mkdir(out)
open(os.path.join(out,'job.json'),'w').write(payload)
args=json.loads(cmd)
open(os.path.join(out,'command.txt'),'w').write(json.dumps(args))
f=open(os.path.join(out,'worker.log'),'ab',buffering=0)
env=os.environ.copy()
env['PYTHONPATH']=code
env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
env['OMP_NUM_THREADS']=threads
env['MKL_NUM_THREADS']=threads
env['RAYON_NUM_THREADS']=threads
env['TORCH_HOME']=torch_home
env['CUDA_VISIBLE_DEVICES']=gpu if gpu else ''
p=subprocess.Popen(args,cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
    ssh(
        host,
        quote_cmd(
            [
                cfg["venv_python"],
                "-c",
                script,
                out,
                code,
                json.dumps(resolved),
                json.dumps(run),
                record["gpu_uuid"] or "",
                str(cfg.get("cpu_threads", cfg.get("threads", 4))
                    if record["device"] is None else cfg.get("threads", 4)),
                cfg.get("torch_home", root + "/torch-cache"),
            ]
        ),
    )


def matches(record, obj, job_id):
    return (
        obj.get("job_id") == job_id
        and obj.get("attempt") == record["attempt"]
        and obj.get("nonce") == record["nonce"]
    )


def approved_gate_methods(gate, jobs, records, jobs_sha256):
    """Validate global gate identity, then return methods whose diagnostics passed."""
    diagnostics = [job for job in jobs if job.get("stage") == "diagnostic"]
    if (
        gate.get("approved") is not True
        or not diagnostics
        or gate.get("jobs_sha256") != jobs_sha256
        or set(gate.get("diagnostic_job_ids", [])) != {job["job_id"] for job in diagnostics}
    ):
        return set()
    if "approved_methods" not in gate:
        if not all(
            records.get(job["job_id"], {}).get("state") == "succeeded" for job in diagnostics
        ):
            return set()
        return {job["method"] for job in jobs}
    methods = gate["approved_methods"]
    if not isinstance(methods, list) or any(not isinstance(method, str) for method in methods):
        return set()
    return {
        method
        for method in methods
        if any(job["method"] == method for job in diagnostics)
        and all(
            records.get(job["job_id"], {}).get("state") == "succeeded"
            for job in diagnostics
            if job["method"] == method
        )
    }


def schedule(args):
    state_dir = args.state
    if args.status:
        for path in sorted(state_dir.glob("*.json")):
            print(path.read_text().strip())
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = json.loads(args.manifest.read_text())
    if isinstance(jobs, dict):
        jobs = jobs["jobs"]
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)) or any(
        not x
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in x
        )
        for x in ids
    ):
        raise ValueError("job IDs must be unique and contain only letters, digits, _, - or .")
    for job in jobs:
        resource = job.get("resource")
        if resource not in (None, "cpu", "gpu"):
            raise ValueError(f"Invalid job resource: {resource!r}")
        if job.get("evaluation_from"):
            if resource != "cpu" or not job.get("required_host") or not Path(job["evaluation_from"]).is_absolute():
                raise ValueError("Evaluation repair requires CPU, a required host, and an absolute parent path")
        elif resource == "cpu" and job["method"] in GPU_METHODS:
            raise ValueError("Learned CPU jobs require evaluation_from")
    while True:
        hosts = json.loads(args.hosts.read_text())["hosts"]
        approved_methods = set()
        records = {}
        pending_status = {}
        for job in jobs:
            path = state_dir / (job["job_id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if record["state"] in {"running", "launching", "uncertain"}:
                    pending_status[job["job_id"]] = record
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(pending_status)))) as executor:
            futures = {
                job_id: executor.submit(remote_status, record["host"], record)
                for job_id, record in pending_status.items()
            }
        for job in jobs:
            path = state_dir / (job["job_id"] + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["state"] in {"running", "launching", "uncertain"}:
                try:
                    status = futures[job["job_id"]].result()
                    receipt = status.get("receipt")
                    if receipt and not matches(record, receipt, job["job_id"]):
                        record.update(state="uncertain", error="receipt identity mismatch")
                    elif "completion" in status:
                        completion = status["completion"]
                        if not matches(record, completion, job["job_id"]) or any(
                            completion.get(k) != receipt.get(k)
                            for k in ("pid", "start_ticks", "boot_id")
                        ):
                            record.update(state="uncertain", error="completion identity mismatch")
                        elif (
                            completion.get("state") == "succeeded"
                            and completion.get("exit_code") == 0
                        ):
                            record.update(state="succeeded", result=completion)
                        elif (
                            completion.get("state") == "failed" and completion.get("exit_code") != 0
                        ):
                            record.update(state="failed", result=completion)
                        else:
                            record.update(state="uncertain", error="invalid completion state")
                    elif receipt and status.get("alive") is False:
                        record.update(
                            state="infra_failed", error="verified worker exited without completion"
                        )
                    elif receipt and status.get("alive") is True:
                        record.update(state="running", receipt_data=receipt)
                    else:
                        record.update(
                            state="uncertain", error=status.get("unknown", "receipt pending")
                        )
                    atomic_json(path, record)
                except (
                    subprocess.SubprocessError,
                    OSError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    print(f"{now()} {job['job_id']}: host/status unavailable: {concise_error(exc)}", flush=True)
            records[job["job_id"]] = record
        if args.approval.is_file():
            gate = json.loads(args.approval.read_text())
            approved_methods = approved_gate_methods(
                gate, jobs, records, hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            )
        else:
            gate = {}
        if not args.status:
            used = {
                (r.get("host"), r.get("device"))
                for r in records.values()
                if r["state"] in {"running", "uncertain", "launching"}
            }
            eligible = {
                host: cfg
                for host, cfg in hosts.items()
                if cfg.get("data_verified")
                and cfg.get("code_path")
                and cfg.get("protocol_sha256")
                and cfg.get("code_commit")
                and cfg.get("enabled", True)
            }
            with ThreadPoolExecutor(max_workers=min(16, max(1, len(eligible)))) as executor:
                gpu_queries = {
                    host: executor.submit(gpu_free, host, cfg) for host, cfg in eligible.items()
                }
            for host, cfg in eligible.items():
                try:
                    free = gpu_queries[host].result()
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    print(f"{now()} {host}: GPU check failed: {concise_error(exc)}", flush=True)
                    free = []
                slots = [(host, i) for i in free if (host, i[0]) not in used]
                slots += [(host, None)] * max(
                    0,
                    int(cfg.get("cpu_workers", 0))
                    - sum(
                        1
                        for r in records.values()
                        if r.get("host") == host
                        and r.get("device") is None
                        and r["state"] in {"running", "uncertain", "launching"}
                    ),
                )
                for _, device in slots:
                    candidates = [
                        j
                        for j in jobs
                        if (j.get("resource", "gpu" if j["method"] in GPU_METHODS else "cpu") == "gpu") == (device is not None)
                        and (not j.get("required_host") or j["required_host"] == host)
                        and (
                            j.get("stage") == "diagnostic"
                            or (
                                j["method"] in approved_methods
                                and gate.get("split_sha256") == cfg.get("protocol_sha256")
                                and gate.get("release_sha") == cfg.get("code_commit")
                            )
                        )
                        and (
                            device is None
                            or j["method"]
                            in cfg.get("gpu_methods", {}).get(str(device[0]), list(GPU_METHODS))
                        )
                        and (
                            j["job_id"] not in records
                            or (
                                (records[j["job_id"]]["state"] == "infra_failed"
                                 or (records[j["job_id"]]["state"] == "repair_pending"
                                     and j.get("evaluation_from")))
                                and records[j["job_id"]]["attempt"] < 3
                            )
                        )
                    ]
                    if not candidates:
                        continue
                    job = min(
                        candidates, key=lambda candidate: (
                            0 if candidate.get("stage") == "diagnostic" else
                            1 if records.get(candidate["job_id"], {}).get("state") == "repair_pending" else 2
                        )
                    )
                    if device is not None:
                        try:
                            if device not in gpu_free(host, cfg):
                                continue
                        except (subprocess.SubprocessError, OSError, ValueError):
                            continue
                    attempt = records.get(job["job_id"], {}).get("attempt", 0) + 1
                    path = state_dir / (job["job_id"] + ".json")
                    record = prepare_launch(host, cfg, job, attempt, device)
                    atomic_json(path, record)
                    try:
                        launch(host, cfg, job, record)
                        record.update(state="launching")
                    except (subprocess.SubprocessError, OSError, ValueError) as exc:
                        record.update(state="uncertain", error=f"launch outcome unknown: {concise_error(exc)}")
                    atomic_json(path, record)
                    records[job["job_id"]] = record
                    used.add((host, record["device"]))
                    print(
                        f"{now()} {job['job_id']}: {record['state']} on {host} device={device} attempt={attempt}",
                        flush=True,
                    )
        counts = {}
        for job in jobs:
            state = records.get(job["job_id"], {}).get("state", "pending")
            counts[state] = counts.get(state, 0) + 1
        print(
            f"{now()} {json.dumps(counts, sort_keys=True)} approved_methods={sorted(approved_methods)}",
            flush=True,
        )
        if args.once or args.status:
            break
        time.sleep(args.poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("job")
    w.add_argument("completion")
    w.add_argument("attempt", type=int)
    s = sub.add_parser("schedule")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--hosts", type=Path, required=True)
    s.add_argument("--state", type=Path, required=True)
    s.add_argument("--approval", type=Path, required=True)
    s.add_argument("--once", action="store_true")
    s.add_argument("--status", action="store_true")
    s.add_argument("--poll", type=int, default=60)
    c = sub.add_parser("collect", help="Collect finished attempts independently of scheduling")
    c.add_argument("--state", type=Path, required=True)
    c.add_argument("--dest", type=Path, required=True)
    c.add_argument(
        "--metadata-only",
        action="store_true",
        help="Collect JSON, CSV and logs without model weights or image data",
    )
    args = p.parse_args()
    if args.mode == "worker":
        sys.exit(worker(args.job, args.completion, args.attempt))
    if args.mode == "collect":
        for path in sorted(args.state.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("state") not in {"succeeded", "failed", "infra_failed"}:
                continue
            dest = args.dest / path.stem / str(record["attempt"])
            dest.mkdir(parents=True, exist_ok=True)
            rsync = ["rsync", "-a", "-e", quote_cmd(["ssh", *ssh_options(record["host"])])]
            if args.metadata_only:
                rsync += [
                    "--include=*/",
                    "--include=*.json",
                    "--include=*.csv",
                    "--include=*.log",
                    "--include=command.txt",
                    "--exclude=*",
                ]
            subprocess.run(
                [*rsync, f"{record['host']}:{record['out']}/", str(dest) + "/"], check=True
            )
        return
    schedule(args)


if __name__ == "__main__":
    main()
