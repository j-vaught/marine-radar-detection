#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_LOG_DIR = ROOT / "logs" / "plateau"


@dataclass(frozen=True)
class JobSpec:
    name: str
    module: str
    output_dir: str
    extra_args: tuple[str, ...]


JOBS: tuple[JobSpec, ...] = (
    JobSpec(
        name="yolo26",
        module="src.yolo.experiments",
        output_dir="results/yolo26_matrix_plateau",
        extra_args=("--epochs", "100", "--patience", "12"),
    ),
    JobSpec(
        name="rtdetr",
        module="src.rtdetr.experiments",
        output_dir="results/rtdetr_matrix_plateau",
        extra_args=("--epochs", "100", "--patience", "12"),
    ),
    JobSpec(
        name="faster_rcnn",
        module="src.faster_rcnn.experiments",
        output_dir="results/faster_rcnn_matrix_plateau",
        extra_args=("--epochs", "60", "--patience", "8", "--lr-step-size", "20"),
    ),
    JobSpec(
        name="unet",
        module="src.unet.experiments",
        output_dir="results/unet_matrix_plateau",
        extra_args=("--epochs", "60", "--patience", "8", "--t-max", "60"),
    ),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--gpus", default="auto", help='Comma-separated GPU ids or "auto"')
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-busy-mem-mb", type=int, default=1024)
    ap.add_argument("--max-busy-util", type=int, default=10)
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--only", nargs="*", default=None, help="Subset of jobs: unet faster_rcnn rtdetr yolo26")
    return ap.parse_args()


def query_gpus() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows: list[dict[str, int]] = []
    for line in proc.stdout.strip().splitlines():
        idx_raw, mem_raw, util_raw = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(idx_raw),
                "memory_used_mb": int(mem_raw),
                "utilization": int(util_raw),
            }
        )
    return rows


def select_gpus(args: argparse.Namespace) -> list[int]:
    if args.gpus != "auto":
        return [int(part.strip()) for part in args.gpus.split(",") if part.strip()]

    free = []
    for row in query_gpus():
        if row["memory_used_mb"] <= args.max_busy_mem_mb and row["utilization"] <= args.max_busy_util:
            free.append(row["index"])
    return free


def filter_jobs(only: list[str] | None) -> list[JobSpec]:
    if not only:
        return list(JOBS)
    wanted = {raw.strip().lower() for raw in only}
    return [job for job in JOBS if job.name.lower() in wanted]


def build_command(python_bin: str, job: JobSpec, gpu_id: int, resume: bool) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        job.module,
        "--device",
        str(gpu_id),
        "--output-dir",
        job.output_dir,
        *job.extra_args,
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def spawn_job(
    python_bin: str,
    job: JobSpec,
    gpu_id: int,
    log_dir: Path,
    resume: bool,
) -> tuple[subprocess.Popen[str], Path, list[str], object]:
    log_path = log_dir / f"{job.name}.log"
    cmd = build_command(python_bin, job, gpu_id, resume)
    handle = log_path.open("a")
    handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launch gpu={gpu_id}\n")
    handle.write(f"{' '.join(shlex.quote(part) for part in cmd)}\n\n")
    handle.flush()
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_path, cmd, handle


def main() -> int:
    args = parse_args()
    python_bin = args.python
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = filter_jobs(args.only)
    if not jobs:
        print("[plateau] no jobs selected")
        return 1

    gpu_ids = select_gpus(args)
    if not gpu_ids:
        print("[plateau] no free GPUs found")
        return 1

    print(f"[plateau] using GPUs: {gpu_ids}")
    print(f"[plateau] jobs: {[job.name for job in jobs]}")

    queued = list(jobs)
    running: dict[int, tuple[JobSpec, subprocess.Popen[str], Path, list[str], object]] = {}
    failures: list[tuple[JobSpec, int, Path]] = []

    while queued or running:
        while queued and len(running) < len(gpu_ids):
            gpu_id = next(gpu for gpu in gpu_ids if gpu not in running)
            job = queued.pop(0)
            proc, log_path, cmd, handle = spawn_job(python_bin, job, gpu_id, log_dir, args.resume)
            running[gpu_id] = (job, proc, log_path, cmd, handle)
            print(f"[plateau] launched {job.name} on gpu {gpu_id}: {' '.join(shlex.quote(part) for part in cmd)}")
            print(f"[plateau] log: {log_path}")

        finished_gpu: int | None = None
        for gpu_id, (job, proc, log_path, _, handle) in list(running.items()):
            return_code = proc.poll()
            if return_code is None:
                continue
            finished_gpu = gpu_id
            handle.close()
            if return_code == 0:
                print(f"[plateau] finished {job.name} on gpu {gpu_id}")
            else:
                print(f"[plateau] FAILED {job.name} on gpu {gpu_id} with code {return_code}; log: {log_path}")
                failures.append((job, return_code, log_path))
            del running[gpu_id]
            break

        if finished_gpu is None:
            time.sleep(args.poll_seconds)

    if failures:
        print("[plateau] failures detected:")
        for job, return_code, log_path in failures:
            print(f"  - {job.name}: exit={return_code}, log={log_path}")
        return 1

    print("[plateau] all jobs completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
