"""Conservative, resumable SSH job scheduler. See --help for usage."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import platform
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

GPU_METHODS = {"faster_rcnn", "unet", "yolo", "rtdetr"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def ssh(host, command, timeout=30):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
                          capture_output=True, text=True, timeout=timeout, check=True).stdout


def quote_cmd(parts):
    return shlex.join([str(x) for x in parts])


def gpu_free(host, config):
    """GPU must have no compute process, irrespective of process ownership."""
    prefix = [config["wrapper"]] if config.get("wrapper") else []
    cmd = prefix + ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    all_gpu = ssh(host, quote_cmd(cmd)).strip().splitlines()
    cmd = prefix + ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    active = ssh(host, quote_cmd(cmd)).strip().splitlines()
    busy = {line.split(",")[0].strip() for line in active if line.strip()}
    allowed = config.get("allowed_gpus")
    return [int(index.strip()) for index, uuid in (line.split(",", 1) for line in all_gpu)
            if uuid.strip() not in busy and (allowed is None or int(index.strip()) in allowed)]


def resolve(value, values):
    if isinstance(value, str):
        return value.format_map(values)
    if isinstance(value, list):
        return [resolve(v, values) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, values) for k, v in value.items()}
    return value


def worker(job_path, completion_path, attempt):
    job = json.loads(Path(job_path).read_text())
    completion = {"job_id": job["job_id"], "attempt": attempt, "started": now(),
                  "state": "failed", "failure_kind": "model", "code_commit": job.get("code_commit"),
                  "python": sys.version, "platform": platform.platform()}
    manifest = Path(job.get("manifest", ""))
    if manifest.is_file():
        completion["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    try:
        from src.revision.run import run_job
        run_job(job)
        metrics = Path(job["out_dir"]) / "metrics.json"
        if not metrics.is_file():
            raise RuntimeError(f"Missing metrics: {metrics}")
        json.loads(metrics.read_text())
        completion.update(state="succeeded", exit_code=0, failure_kind=None)
    except BaseException as exc:
        traceback.print_exc()
        completion.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
    completion["finished"] = now()
    atomic_json(completion_path, completion)
    return completion["exit_code"]


def remote_status(host, state):
    # A missing PID or unreachable host is ambiguous and must never trigger reassignment.
    script = """import json,os,sys
p,pid=sys.argv[1],int(sys.argv[2])
if os.path.isfile(p):
 print(json.dumps({'completion':json.load(open(p))}))
else:
 try:
  os.kill(pid,0)
  stat=open('/proc/%s/stat'%pid).read().split()
  cmd=open('/proc/%s/cmdline'%pid,'rb').read().replace(b'\\0',b' ').decode(errors='replace')
  print(json.dumps({'alive':stat[2]!='Z','cmd':cmd}))
 except ProcessLookupError: print(json.dumps({'alive':False}))
 except Exception as e: print(json.dumps({'unknown':str(e)}))
"""
    return json.loads(ssh(host, quote_cmd([state["python"], "-c", script,
                                          state["completion"], state["pid"]])))


def launch(host, cfg, job, attempt, device):
    root, code = cfg["result_root"], cfg["code_path"]
    out = f"{root}/runs/{job['job_id']}/{attempt}"
    values = {"root": root, "code": code, "data": cfg["data_path"],
              "source": job.get("source", ""), "job_id": job["job_id"], "attempt": attempt}
    resolved = resolve(job, values)
    resolved.update(out_dir=out, device=f"cuda:{device}" if device is not None else "cpu",
                    code_commit=cfg.get("code_commit"))
    payload = json.dumps(resolved)
    job_file, completion = out + "/job.json", out + "/completion.json"
    run = [cfg["venv_python"], "-m", "src.revision.fleet", "worker", job_file, completion, str(attempt)]
    if cfg.get("wrapper"):
        run.insert(0, cfg["wrapper"])
    script = """import json,os,subprocess,sys
out,code,payload,cmd=sys.argv[1:]
os.makedirs(os.path.dirname(out),exist_ok=True)
os.mkdir(out)
open(os.path.join(out,'job.json'),'w').write(payload)
args=json.loads(cmd)
open(os.path.join(out,'command.txt'),'w').write(json.dumps(args))
f=open(os.path.join(out,'worker.log'),'ab',buffering=0)
p=subprocess.Popen(args,cwd=code,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
    pid = int(ssh(host, quote_cmd([cfg["venv_python"], "-c", script, out, code, payload, json.dumps(run)])).strip())

    return {"host": host, "python": cfg["venv_python"], "out": out,
            "completion": completion, "pid": pid, "device": device, "attempt": attempt}


def schedule(args):
    state_dir = args.state
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = json.loads(args.manifest.read_text())
    if isinstance(jobs, dict):
        jobs = jobs["jobs"]
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)) or any(not x or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in x) for x in ids):
        raise ValueError("job IDs must be unique and contain only letters, digits, _, - or .")
    while True:
        hosts = json.loads(args.hosts.read_text())["hosts"]
        approved = args.approval.is_file() and json.loads(args.approval.read_text()).get("approved") is True
        records = {}
        for job in jobs:
            path = state_dir / (job["job_id"] + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["state"] == "running":
                try:
                    status = remote_status(record["host"], record)
                    if "completion" in status:
                        record.update(state=status["completion"]["state"], result=status["completion"])
                        if args.collect:
                            dest = args.collect / job["job_id"] / str(record["attempt"])
                            dest.mkdir(parents=True, exist_ok=True)
                            subprocess.run(["rsync", "-a", f"{record['host']}:{record['out']}/", str(dest) + "/"], check=True, timeout=300)
                    elif status.get("alive") is False:
                        record.update(state="infra_failed", error="worker process exited without completion record")
                    elif status.get("alive") and "src.revision.fleet" not in status.get("cmd", ""):
                        record.update(state="uncertain", error="PID reused or unexpected process; manual inspection required")
                    atomic_json(path, record)
                except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as exc:
                    print(f"{now()} {job['job_id']}: host/status unavailable: {exc}", flush=True)
            records[job["job_id"]] = record
        if not args.status:
            used = {(r.get("host"), r.get("device")) for r in records.values() if r["state"] in {"running", "uncertain", "launching"}}
            for host, cfg in hosts.items():
                if not cfg.get("data_verified") or not cfg.get("code_path") or not cfg.get("enabled", True):
                    continue
                try:
                    free = gpu_free(host, cfg)
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    print(f"{now()} {host}: GPU check failed: {exc}", flush=True)
                    free = []
                slots = [(host, i) for i in free if (host, i) not in used]
                slots += [(host, None)] * max(0, int(cfg.get("cpu_workers", 0)) - sum(1 for r in records.values() if r.get("host") == host and r.get("device") is None and r["state"] in {"running", "uncertain", "launching"}))
                for _, device in slots:
                    candidates = [j for j in jobs if (j["method"] in GPU_METHODS) == (device is not None)
                                  and (j.get("stage") == "diagnostic" or approved)
                                  and (j["job_id"] not in records or (records[j["job_id"]]["state"] == "infra_failed" and records[j["job_id"]]["attempt"] < 3))]
                    if not candidates:
                        continue
                    job = candidates[0]
                    attempt = records.get(job["job_id"], {}).get("attempt", 0) + 1
                    path = state_dir / (job["job_id"] + ".json")
                    record = {"state": "launching", "host": host, "device": device, "attempt": attempt, "started": now()}
                    atomic_json(path, record)
                    try:
                        record.update(launch(host, cfg, job, attempt, device), state="running")
                    except (subprocess.SubprocessError, OSError, ValueError) as exc:
                        record.update(state="uncertain", error=f"launch outcome unknown: {exc}")
                    atomic_json(path, record)
                    records[job["job_id"]] = record
                    used.add((host, device))
                    print(f"{now()} {job['job_id']}: {record['state']} on {host} device={device} attempt={attempt}", flush=True)
        counts = {}
        for job in jobs:
            state = records.get(job["job_id"], {}).get("state", "pending")
            counts[state] = counts.get(state, 0) + 1
        print(f"{now()} {json.dumps(counts, sort_keys=True)} gate={'open' if approved else 'closed'}", flush=True)
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
    s.add_argument("--collect", type=Path)
    s.add_argument("--once", action="store_true")
    s.add_argument("--status", action="store_true")
    s.add_argument("--poll", type=int, default=60)
    args = p.parse_args()
    if args.mode == "worker":
        sys.exit(worker(args.job, args.completion, args.attempt))
    schedule(args)


if __name__ == "__main__":
    main()
