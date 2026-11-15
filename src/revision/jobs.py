"""Generate deterministic jobs from an explicit experiment matrix.

The input JSON contains ``defaults`` and a list of ``jobs``. Each entry is
expanded over list-valued ``seeds`` and ``sources``. Other fields are passed
unchanged to the runner. This keeps dataset splits and evaluation targets
explicit and reviewable rather than inferring them from directory names.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path


def generate(matrix: dict) -> list[dict]:
    result = []
    for row in matrix["jobs"]:
        row = {**matrix.get("defaults", {}), **row}
        seeds = row.pop("seeds", [row.get("seed", 0)])
        sources = row.pop("sources", [row.get("source", "")])
        for source, seed in itertools.product(sources, seeds):
            job = {**row, "source": source, "seed": seed}
            if "job_id" not in job:
                parts = [job.get("stage", "core"), job["method"], source,
                         job.get("condition", "buffered"), f"s{seed}"]
                job["job_id"] = re.sub(r"[^A-Za-z0-9_.-]+", "-", "-".join(map(str, parts)))
            result.append(job)
    ids = [j["job_id"] for j in result]
    if len(ids) != len(set(ids)):
        raise ValueError("job IDs must be unique")
    return result


def standard_jobs() -> list[dict]:
    """The frozen review matrix; split membership is supplied by the protocol manifest."""
    jobs = []
    common = {"ann": "{data}/annotations/all.json", "images": "{data}/images",
              "manifest": "{root}/protocol/split_manifest.json",
              "eval_targets": ["DAAN", "DARC", "MANV"], "config_overrides": {}}
    for stage, condition, methods, seeds in (
        ("core", "buffered", ("unet", "faster_rcnn", "yolo", "rtdetr"), range(5)),
        ("adjacency", "adjacency", ("faster_rcnn", "yolo"), range(3)),
        ("core", "buffered", ("ca_cfar", "os_cfar", "otsu", "ss_otsu", "wavelet"), (0,)),
        ("diagnostic", "buffered", ("unet", "faster_rcnn", "yolo", "rtdetr"), (0,)),
    ):
        sources = ("DAAN",) if stage == "diagnostic" else ("DAAN", "DARC", "MANV")
        for method, source, seed in itertools.product(methods, sources, seeds):
            deep = method in {"unet", "faster_rcnn", "yolo", "rtdetr"}
            job = {**common, "job_id": f"{stage}-{method}-{source}-s{seed}",
                   "method": method, "source": source, "seed": seed,
                   "condition": condition, "stage": stage,
                   "initialization": "scratch" if method == "unet" or not deep else "pretrained",
                   "epochs": 60 if method in {"unet", "faster_rcnn"} else 100,
                   "patience": 8 if method in {"unet", "faster_rcnn"} else 12,
                   "overfit": stage == "diagnostic"}
            if stage == "diagnostic":
                job.update(train_limit=32, eval_targets=[])
            jobs.append(job)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="Matrix JSON, or - for the standard review matrix")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    jobs = standard_jobs() if str(args.matrix) == "-" else generate(json.loads(args.matrix.read_text()))
    args.output.write_text(json.dumps(jobs, indent=2) + "\n")


if __name__ == "__main__":
    main()
