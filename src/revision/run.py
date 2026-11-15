"""Execute a single revision job from a JSON specification."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

MODULES = {
    "faster_rcnn": "torch_models", "unet": "torch_models",
    "yolo": "ultralytics_models", "rtdetr": "ultralytics_models",
    "ca_cfar": "classical", "os_cfar": "classical", "otsu": "classical",
    "ss_otsu": "classical", "wavelet": "classical",
}


def run_job(job: dict) -> object:
    method = job.get("method")
    if method not in MODULES:
        raise ValueError(f"Unsupported method: {method!r}")
    return importlib.import_module(f"src.revision.{MODULES[method]}").run_job(job)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    run_job(job)
    metrics = Path(job["out_dir"]) / "metrics.json"
    if not metrics.is_file():
        raise RuntimeError(f"Job returned without writing {metrics}")
    json.loads(metrics.read_text())


if __name__ == "__main__":
    main()
