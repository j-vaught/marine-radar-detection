"""Shared-manifest classical detector selection and evaluation.

Rust binaries must be built separately in release mode. The explicit ``run``
command consumes only the subset JSON supplied here; no detector-specific split
is used. All candidate ranking uses the shared COCO evaluator.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import time
from pathlib import Path

from .protocol import prepare_job, save_json, set_seed, sha256_file
from .evaluation import evaluate_coco

ROOT = Path(__file__).resolve().parents[2]


def _grid(method: str, overrides: dict) -> list[dict]:
    if method in ("ca_cfar", "os_cfar"):
        from src.cfar.experiments.grid import enumerate_grid
        from dataclasses import asdict
        grid = [asdict(c) for c in enumerate_grid("CA" if method == "ca_cfar" else "OS")]
    else:
        if method == "otsu":
            axes = dict(min_area=[5, 10, 20, 30, 50], max_area=[3000, 5000, 10000],
                        dilation=[0, 1, 2], box_padding=[0, 1, 2], max_eccentricity=[1.0])
        elif method == "ss_otsu":
            axes = dict(block_size=[64, 96], seed_threshold_factor=[1.25, 1.5, 1.75],
                        region_grow_threshold=[0.6, 0.7, 0.8], min_area=[10, 20, 30, 50],
                        max_area=[3000, 5000, 10000], dilation=[0, 1], box_padding=[0, 1, 2],
                        max_eccentricity=[1.0])
        elif method == "wavelet":
            axes = dict(level=[2, 3], threshold_sigma=[4.5, 6.0, 7.5], min_area=[20, 30, 50],
                        max_area=[3000], dilation=[0, 1], box_padding=[0])
        else:
            raise ValueError(f"Unknown method: {method}")
        if method == "wavelet" and overrides.get("wavelet_extended"):
            axes["threshold_sigma"] = [3.0, 4.5, 6.0, 7.5, 9.0]
        grid = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    # Overrides fix specific axes without silently removing the default grid.
    # Explicit candidate list permits reproducible diagnostic runs.
    if "candidates" in overrides:
        grid = overrides["candidates"]
    fixed = overrides.get("fixed", {})
    return [{**cfg, **fixed} for cfg in grid]


def _native_predictions(method, config, ann_path, image_dir, output, device, binary=None):
    crate = "wavelet" if method == "wavelet" else "otsu"
    exe = Path(binary) if binary else ROOT / "src" / crate / "target" / "release" / crate
    if not exe.is_file():
        raise FileNotFoundError(f"Build detector with cargo build --release --manifest-path "
                                f"{ROOT / 'src' / crate / 'Cargo.toml'}")
    cmd = [str(exe), "run", "--ann", str(ann_path), "--images", str(image_dir),
           "--out", str(output), "--backend", "cuda" if str(device).startswith("cuda") else "cpu"]
    if crate == "otsu":
        cmd += ["--method", "ssotsu" if method == "ss_otsu" else "otsu"]
    for key, value in config.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.read_text())


def _cfar_predictions(method, config, ann_path, image_dir, device, caches):
    import numpy as np
    from src.cfar.gpu.data import load_corpus
    from src.cfar.experiments.grid import BackgroundCache, predict_subset
    if ann_path not in caches:
        corpus = load_corpus(str(ann_path), str(image_dir), device=device)
        caches[ann_path] = (corpus, BackgroundCache(corpus))
    corpus, cache = caches[ann_path]
    k = config.get("os_k")
    preds = predict_subset(corpus, cache, np.arange(corpus.N), config["method"],
                           config["guard"], config["ref"], pfas=[config["pfa"]],
                           min_areas=[config["min_area"]], os_ks=[k] if k else [0.5, 0.75, 0.9])
    key = (config["pfa"], config["min_area"], k) if k else (config["pfa"], config["min_area"])
    return preds[key].to_coco()


def run_job(job: dict) -> dict:
    """Tune on train, choose top five on validation, then evaluate frozen tests."""
    set_seed(job["seed"])
    data = prepare_job(job)
    out = Path(job["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    method = job["method"]
    overrides = job.get("config_overrides") or {}
    train, val = data["train"], data["validation"]
    datasets = {"train": train, "validation": val}
    for site, coco in data["tests"].items():
        if site in (job.get("eval_targets") or []):
            datasets[f"test_{site}"] = coco
    paths = {}
    for name, coco in datasets.items():
        paths[name] = out / f"{name}_annotations.json"
        save_json(paths[name], coco)

    cfar_caches = {}

    def predict(config, name, label):
        destination = out / f"{label}_predictions.json"
        if method in ("ca_cfar", "os_cfar"):
            pred = _cfar_predictions(method, config, paths[name], data["image_dir"], job.get("device", "cpu"), cfar_caches)
            save_json(destination, pred)
        else:
            pred = _native_predictions(method, config, paths[name], data["image_dir"], destination,
                                       job.get("device", "cpu"), overrides.get("binary"))
        return pred

    start = time.monotonic()
    candidates = []
    grid = _grid(method, overrides)
    if not grid:
        raise ValueError("Empty candidate grid")
    native_sweep = method in ("otsu", "ss_otsu") and not overrides.get("candidates") and not overrides.get("fixed")
    if native_sweep:
        exe = Path(overrides.get("binary", ROOT / "src/otsu/target/release/otsu"))
        if not exe.is_file():
            raise FileNotFoundError(f"Build detector with cargo build --release --manifest-path {ROOT / 'src/otsu/Cargo.toml'}")
        sweep_dir = out / "train_sweep"
        subprocess.run([str(exe), "sweep", "--ann", str(paths["train"]), "--images", str(data["image_dir"]),
                        "--out-dir", str(sweep_dir), "--method", "ssotsu" if method == "ss_otsu" else "otsu",
                        "--backend", "cuda" if str(job.get("device", "cpu")).startswith("cuda") else "cpu"], check=True)
        grid = [{k: v for k, v in cfg.items() if v is not None}
                for cfg in json.loads((sweep_dir / "configs.json").read_text())]
    for i, cfg in enumerate(grid):
        tick = time.monotonic()
        if native_sweep:
            pred = json.loads((sweep_dir / f"candidate_{i:04d}.json").read_text())
        else:
            pred = predict(cfg, "train", f"candidate_{i:04d}_train")
        score = evaluate_coco(train, pred)
        candidates.append(dict(index=i, config=cfg, train=score, seconds=time.monotonic()-tick))
        if (i + 1) % 25 == 0 or i + 1 == len(grid):
            save_json(out / "candidates.json", candidates)
    ranked = sorted(candidates, key=lambda c: (-c["train"]["ap50"], c["index"]))[:5]
    for candidate in ranked:
        pred = predict(candidate["config"], "validation", f"candidate_{candidate['index']:04d}_validation")
        candidate["validation"] = evaluate_coco(val, pred)
    winner = sorted(ranked, key=lambda c: (-c["validation"]["ap50"], c["index"]))[0]
    selection = dict(config=winner["config"], index=winner["index"], source=job["source"],
                     train=winner["train"], validation=winner["validation"],
                     grid_seconds=time.monotonic()-start)
    save_json(out / "selected_config.json", selection)
    save_json(out / "candidates.json", candidates)
    metrics = {}
    for name, coco in datasets.items():
        if name.startswith("test_"):
            pred = predict(winner["config"], name, name)
            metrics[name[5:]] = evaluate_coco(coco, pred)
    result = dict(job_id=job["job_id"], method=method, selected_config=selection,
                  candidates=candidates, validation=winner["validation"], tests=metrics,
                  paths={"selection": str(out / "selected_config.json"), "candidates": str(out / "candidates.json"),
                         "manifest": str(job["manifest"]), "manifest_sha256": sha256_file(job["manifest"]),
                         "subsets": {name: {"annotations": str(path), "annotations_sha256": sha256_file(path),
                                            **({"predictions": str(out / f"{name}_predictions.json")}
                                               if name.startswith("test_") else {})}
                                     for name, path in paths.items()}})
    save_json(out / "result.json", result)
    save_json(out / "metrics.json", result)
    return result
