"""Fixed-update processed-raster Faster R-CNN sensitivity experiments.

The source-only checkpoint and threshold are frozen before destination scoring.
Initialization artifacts are produced separately, once per source and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.version
import yaml
from torch.utils.data import DataLoader

from . import evaluation, protocol
from .torch_models import _restore_rng, _rng_state, _state_hash

PROTOCOL_ID = "darc-processed-v1"
PROTOCOL_SHA = "14a06aad755ef18a9fb4b90371ea036057af22e57f28e7a892278ac3e721ea16"
MANIFEST_SHA = "03b7d1d48e5faac7872a04f8fb23580a748ca1521539e99b0e54bf45c7d8f4aa"
ANNOTATION_SHA = "29d602bb9a5d764691e44fe697ba3449348a7e9a9cd05b8c19a415b1fa8d5911"
TOTAL_UPDATES = 3432
VAL_INTERVAL = 286
DECAY_AFTER = 2288
BATCH_SIZE = 2


def _cfg():
    return yaml.safe_load((Path(__file__).parents[1] / "faster_rcnn/config.yaml").read_text())


def _validate_job(job):
    if job.get("method", "faster_rcnn") != "faster_rcnn":
        raise ValueError("Only Faster R-CNN is supported")
    if job["source"] not in ("DAAN", "DARC", "MANV") or int(job["seed"]) not in (0, 1, 2):
        raise ValueError("Unrecognized source or seed")
    if job["condition"] != "buffered" or job["stage"] not in ("core", "diagnostic"):
        raise ValueError("Expected buffered core or diagnostic job")
    if job["intensity"] not in ("I0", "I1") or job["scale"] not in ("S0", "S1"):
        raise ValueError("Invalid intensity or scale factor")
    if job.get("size_control") and (
        job["source"] == "DARC" or job["scale"] != "S0" or job["intensity"] != "I0"
    ):
        raise ValueError("Count control only applies to DAAN/MANV S0I0")
    if job["stage"] == "diagnostic":
        if job.get("eval_targets"):
            raise ValueError("Diagnostics must not evaluate target holdouts")
        updates = int(job.get("diagnostic_updates", TOTAL_UPDATES))
        interval = int(job.get("diagnostic_val_interval", VAL_INTERVAL))
    else:
        if (
            set(job.get("eval_targets", [])) != {"DAAN", "DARC", "MANV"}
            or len(job["eval_targets"]) != 3
        ):
            raise ValueError("Core jobs require each fixed target exactly once")
        if "diagnostic_updates" in job or "diagnostic_val_interval" in job:
            raise ValueError("Budget overrides are diagnostic-only")
        updates, interval = TOTAL_UPDATES, VAL_INTERVAL
    if updates <= 0 or interval <= 0 or updates % interval:
        raise ValueError("Updates must be a positive multiple of validation interval")
    if job.get("config_overrides") or job.get("train_limit") or job.get("overfit"):
        raise ValueError("This protocol does not permit split or configuration overrides")
    return updates, interval


def _atomic_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _learning_rate(update):
    if update < 1:
        raise ValueError("Updates are one-indexed")
    return 0.0025 if update <= DECAY_AFTER else 0.00025


def _device_policy(job):
    from ..faster_rcnn.model import resolve_device

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("TF32 override is incompatible with this protocol")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Default tensor dtype must be float32")
    device = resolve_device(job["device"])
    if job["stage"] == "core" and device.type != "cuda":
        raise ValueError("Core jobs require a pinned CUDA GPU")
    if job["stage"] == "core" and not re.fullmatch(
        r"[0-9a-f]{40}", str(job.get("code_commit", ""))
    ):
        raise ValueError("Core jobs require a full immutable code commit")
    if device.type == "cuda":
        uuid = job.get("gpu_uuid")
        if (
            not uuid
            or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid
            or torch.cuda.device_count() != 1
        ):
            raise ValueError("A single assigned GPU UUID is required")
        torch.cuda.set_device(device)
        actual = "GPU-" + str(torch.cuda.get_device_properties(device).uuid)
        if actual.lower() != uuid.lower():
            raise RuntimeError("Assigned and observed GPU UUID differ")
        torch.cuda.reset_peak_memory_stats(device)
    return device


def _name_digest(names):
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _architecture_identity(model):
    return {
        "state_keys_sha256": _name_digest(sorted(model.state_dict())),
        "trainable_parameters_sha256": _name_digest(
            sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _verified_pretrained_asset(expected):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    url = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url
    asset = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    if not asset.is_file() or protocol.sha256_file(asset) != expected:
        raise ValueError("Verified pretrained asset is missing or has the wrong SHA-256")
    return url, asset


def initialize(job):
    """Create one portable seeded initial state using a verified local asset."""
    from ..faster_rcnn.model import build_model

    if job["source"] not in ("DAAN", "DARC", "MANV") or int(job["seed"]) not in (0, 1, 2):
        raise ValueError("Unrecognized initialization source or seed")
    if not re.fullmatch(r"[0-9a-f]{40}", str(job.get("code_commit", ""))):
        raise ValueError("Initialization requires a full immutable code commit")
    url, asset = _verified_pretrained_asset(job["pretrained_weights_sha256"])
    path = Path(job["initial_state"])
    meta_path = Path(str(path) + ".json")
    if path.exists() or path.is_symlink() or meta_path.exists() or meta_path.is_symlink():
        raise FileExistsError("Refusing existing initial-state artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol.set_seed(int(job["seed"]))
    model = build_model(_cfg(), pretrained=True, box_score_thresh=0.0, box_detections_per_img=300)
    tensor_hash = _state_hash(model)
    with path.open("xb") as stream:
        torch.save(model.state_dict(), stream)
    import torchvision

    meta = {
        "source": job["source"],
        "seed": int(job["seed"]),
        "protocol_id": PROTOCOL_ID,
        "frozen_protocol_sha256": PROTOCOL_SHA,
        "code_commit": job.get("code_commit"),
        "model_config_sha256": _json_digest(_cfg()),
        "pretrained_asset": str(asset),
        "pretrained_url": url,
        "pretrained_weights_sha256": job["pretrained_weights_sha256"],
        "initial_tensor_sha256": tensor_hash,
        "initial_state_sha256": protocol.sha256_file(path),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "architecture": _architecture_identity(model),
    }
    protocol.save_json(meta_path, meta)
    return meta


def _val_seed(source, seed, index):
    raw = f"{PROTOCOL_ID}:validation:{source}:{seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _collate(batch):
    return tuple(zip(*batch))


def _loader(ds, batch_size, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _validation_loss(model, loader, device, seed):
    """Score source validation without changing any training RNG or buffers."""
    saved_rng = _rng_state()
    buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
    prior_mode = model.training
    try:
        protocol.set_seed(seed)
        model.train()
        total = 0.0
        count = 0
        with torch.no_grad():
            for images, targets, _stats in loader:
                moved_images = [x.to(device) for x in images]
                moved_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = sum(model(moved_images, moved_targets).values())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite source validation loss")
                total += float(loss.item()) * len(images)
                count += len(images)
        if not count:
            raise ValueError("Empty source validation")
        return total / count
    finally:
        for name, tensor in model.named_buffers():
            tensor.copy_(buffers[name])
        model.train(prior_mode)
        _restore_rng(saved_rng)


def _predictions(model, dataset, device):
    preds, audits = [], []
    model.eval()
    with torch.inference_mode():
        for i in range(len(dataset)):
            image, target, stats = dataset[i]
            with torch.autocast(device_type=device.type, enabled=False):
                out = model([image.to(device)])[0]
            boxes, scores, labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise RuntimeError("Nonfinite detector output")
            h, w = image.shape[-2:]
            if not (
                (
                    (boxes[:, 0] >= 0)
                    & (boxes[:, 1] >= 0)
                    & (boxes[:, 2] <= w)
                    & (boxes[:, 3] <= h)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                ).all()
            ):
                raise RuntimeError("Invalid native-coordinate detector box")
            for box, score, label in zip(boxes, scores, labels, strict=True):
                if int(label) != 1:
                    raise ValueError("Unexpected detector category")
                x1, y1, x2, y2 = map(float, box.tolist())
                preds.append(
                    {
                        "image_id": int(target["image_id"]),
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
            audits.append(stats)
    return preds, audits


def _center_recall(gt, predictions, *, radius=8.0, max_detections=30):
    """Maximum one-to-one center matches under the fixed rank and distance limits."""
    from scipy.optimize import linear_sum_assignment

    refs, preds = {}, {}
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        refs.setdefault(ann["image_id"], []).append((x + w / 2, y + h / 2))
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        preds.setdefault(pred["image_id"], []).append((float(pred["score"]), x + w / 2, y + h / 2))
    matched, total = 0, 0
    for info in gt["images"]:
        ref = refs.get(info["id"], [])
        det = sorted(preds.get(info["id"], []), key=lambda x: -x[0])[:max_detections]
        total += len(ref)
        if not ref or not det:
            continue
        distances = np.linalg.norm(
            np.asarray(ref)[:, None, :] - np.asarray([p[1:] for p in det])[None, :, :], axis=2
        )
        # Infeasible matches cost more than all feasible matching costs combined.
        cost = np.where(distances <= radius, distances / (radius + 1), len(ref) + len(det) + 1)
        rows, cols = linear_sum_assignment(cost)
        matched += int(np.count_nonzero(distances[rows, cols] <= radius))
    return {
        "matched": matched,
        "references": total,
        "recall": matched / total if total else None,
        "tolerance_px": radius,
        "max_detections": max_detections,
    }


def _signature(job, stream, initial_file_hash, runtime_identity):
    stable = {
        key: job.get(key)
        for key in (
            "job_id",
            "source",
            "seed",
            "intensity",
            "scale",
            "size_control",
            "stage",
            "condition",
            "diagnostic_updates",
            "diagnostic_val_interval",
        )
    }
    stable.update(
        initial_state_sha256=initial_file_hash,
        presentation_stream=stream,
        runtime_identity=runtime_identity,
        model_config=_cfg(),
        ann=str(job["ann"]),
        images=str(job["images"]),
        manifest=str(job["manifest"]),
        code_commit=job.get("code_commit"),
        frozen_protocol_sha256=job.get("frozen_protocol_sha256"),
    )
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_audit(stats, effective_hw):
    scales = [float(row["scale"]) for row in stats]
    before = [float(area) for row in stats for area in row["box_areas_before"]]
    after = [float(area) for row in stats for area in row["box_areas_after"]]
    ratio = effective_hw[0] * effective_hw[1] / (1024 * 1050)

    def summary(values):
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "presentations": len(stats),
        "intensity_fallbacks": sum(bool(x["intensity_fallback"]) for x in stats),
        "scale_identity_reasons": {
            reason: sum(x["scale_identity_reason"] == reason for x in stats)
            for reason in ("baseline", "gate", "rejected")
        },
        "scale_rejections": sum(int(x["scale_rejections"]) for x in stats),
        "accepted_shrink": sum(scale < 1 for scale in scales),
        "accepted_grow": sum(scale > 1 for scale in scales),
        "scale_summary": summary(scales),
        "box_area_before_native_px2": summary(before),
        "box_area_after_native_px2": summary(after),
        "box_area_after_model_resize_px2": summary([area * ratio for area in after]),
        "native_hw": [1024, 1050],
        "model_resize_hw": list(effective_hw),
        "coordinate_mapping": "resize scales x and y independently from native canvas to model_resize_hw",
    }


def run(job):
    """Run one fixed-update fit and, after source-only selection, fixed-target scoring."""
    import torchvision

    from ..faster_rcnn.model import build_model
    from .darc_data import DARCData, select_count_subset, train_stream

    updates, interval = _validate_job(job)
    if (
        job.get("frozen_protocol_sha256") != PROTOCOL_SHA
        or protocol.sha256_file(job["frozen_protocol"]) != PROTOCOL_SHA
    ):
        raise ValueError("Frozen protocol digest mismatch")
    if (
        protocol.sha256_file(job["manifest"]) != MANIFEST_SHA
        or protocol.sha256_file(job["ann"]) != ANNOTATION_SHA
    ):
        raise ValueError("Frozen split or annotation digest mismatch")
    prepared = protocol.prepare_job(job)
    manifest = json.loads(Path(job["manifest"]).read_text())
    expected_hashes = {int(row["id"]): row["sha256"] for row in manifest["images"]}
    train = prepared["train"]
    subset_ids = None
    if job.get("size_control"):
        subset_ids = select_count_subset(
            train["images"], job["source"], int(job["seed"]), count=316
        )
        if len(set(subset_ids)) != 316:
            raise ValueError("Invalid count-control sample")
        train = protocol._subset(train, subset_ids)
    stream = train_stream(
        train["images"], job["source"], int(job["seed"]), presentations=updates * BATCH_SIZE
    )
    if len(stream) != updates * BATCH_SIZE or [row[0] for row in stream] != list(
        range(updates * BATCH_SIZE)
    ):
        raise RuntimeError("Invalid presentation stream")
    output = Path(job["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    initial_path = Path(job["initial_state"])
    initial_file_hash = protocol.sha256_file(initial_path)
    if initial_file_hash != job["initial_state_sha256"]:
        raise ValueError("Initial-state file digest mismatch")
    meta = json.loads(Path(str(initial_path) + ".json").read_text())
    if (
        meta["source"],
        meta["seed"],
        meta["initial_tensor_sha256"],
        meta["initial_state_sha256"],
    ) != (job["source"], int(job["seed"]), job["initial_tensor_sha256"], initial_file_hash):
        raise ValueError("Initialization metadata mismatch")
    if (meta["frozen_protocol_sha256"], meta["model_config_sha256"], meta["code_commit"]) != (
        PROTOCOL_SHA,
        _json_digest(_cfg()),
        job.get("code_commit"),
    ):
        raise ValueError("Initialization protocol, model config, or release differs")
    if (meta["torch_version"], meta["torchvision_version"]) != (
        torch.__version__,
        torchvision.__version__,
    ):
        raise ValueError("Initialization and fit framework versions differ")
    _verified_pretrained_asset(meta["pretrained_weights_sha256"])
    device = _device_policy(job)
    protocol.set_seed(int(job["seed"]))
    cfg = _cfg()
    # Pretrained constructor is essential for the same FrozenBatchNorm and
    # trainable backbone policy used when creating the portable initial state.
    model = build_model(cfg, pretrained=True, box_score_thresh=0.0, box_detections_per_img=300)
    if _architecture_identity(model) != meta["architecture"]:
        raise ValueError("Initial and fit model architectures differ")
    model.load_state_dict(
        torch.load(initial_path, map_location="cpu", weights_only=True), strict=True
    )
    if _state_hash(model) != job["initial_tensor_sha256"]:
        raise ValueError("Initial-state tensor digest mismatch")
    model.to(device)
    # Probe the exact fixed internal resize once, without changing training RNG.
    previous_rng = _rng_state()
    transform = model.transform
    transform.eval()
    resized, _ = transform.resize(torch.zeros((3, 1024, 1050)), None)
    effective_hw = tuple(int(x) for x in resized.shape[-2:])
    _restore_rng(previous_rng)
    del resized
    protocol.set_seed(int(job["seed"]))
    workers = int(job.get("workers", 2))
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    train_ds = DARCData(
        train,
        prepared["image_dir"],
        intensity=job["intensity"],
        scale=job["scale"],
        source=job["source"],
        seed=int(job["seed"]),
        presentations=stream,
        expected_hashes=expected_hashes,
    )
    val_ds = DARCData(
        prepared["validation"],
        prepared["image_dir"],
        intensity=job["intensity"],
        scale="S0",
        source=job["source"],
        seed=int(job["seed"]),
        expected_hashes=expected_hashes,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)
    runtime_identity = {
        "device": str(device),
        "gpu_uuid": job.get("gpu_uuid"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "python_version": platform.python_version(),
        "workers": workers,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    run_signature = _signature(job, stream, initial_file_hash, runtime_identity)
    history, audit, best_update, best_loss, first_update = [], [], 0, float("inf"), 0
    elapsed_prior = 0.0
    if job.get("resume"):
        checkpoint = torch.load(output / "resume.pt", map_location=device, weights_only=False)
        if checkpoint["run_signature"] != run_signature:
            raise ValueError("Resume checkpoint does not match the frozen run")
        if checkpoint["best_update"] and (
            not (output / "best.pt").is_file()
            or protocol.sha256_file(output / "best.pt") != checkpoint["best_sha256"]
        ):
            raise ValueError("Resume best checkpoint missing or mismatched")
        elapsed_prior = float(checkpoint["elapsed_seconds"])
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_rng(checkpoint["rng"])
        first_update, best_update, best_loss, history, audit = (
            checkpoint[k] for k in ("update", "best_update", "best_loss", "history", "audit")
        )
    elif any(
        (output / name).exists() for name in ("resume.pt", "best.pt", "last.pt", "metrics.json")
    ):
        raise FileExistsError("Fresh run refuses existing checkpoint/output artifacts")
    remaining_ds = torch.utils.data.Subset(
        train_ds, range(first_update * BATCH_SIZE, len(train_ds))
    )
    train_loader = _loader(remaining_ds, BATCH_SIZE, workers, int(job["seed"]) + 10000)
    val_loader = _loader(val_ds, BATCH_SIZE, workers, int(job["seed"]) + 20000)
    protocol.save_json(
        output / "job_resolved.json",
        {
            "job": job,
            "protocol_id": PROTOCOL_ID,
            "config": cfg,
            "updates": updates,
            "validation_interval": interval,
            "batch_size": BATCH_SIZE,
            "decay_after_update": DECAY_AFTER,
            "presentation_stream_sha256": _json_digest(stream),
            "subset_ids": subset_ids,
            "subset_ids_sha256": _json_digest(subset_ids) if subset_ids else None,
            "initial_architecture": meta["architecture"],
            "model_resize_hw": effective_hw,
        },
    )
    protocol.save_json(output / "train_gt.json", train)
    protocol.save_json(output / "validation_gt.json", prepared["validation"])
    started = time.perf_counter()
    actual_calls = first_update
    interval_loss_sum = 0.0
    for update, batch in enumerate(train_loader, start=first_update + 1):
        images, targets, stats = batch
        if len(images) != BATCH_SIZE:
            raise RuntimeError("Incomplete training batch")
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(update)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = model(
                [x.to(device) for x in images],
                [{k: v.to(device) for k, v in target.items()} for target in targets],
            )
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {update}")
        loss.backward()
        grad_checks = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not grad_checks or not torch.stack(grad_checks).all():
            raise RuntimeError(f"Missing or nonfinite gradient at update {update}")
        optimizer.step()
        actual_calls += 1
        interval_loss_sum += float(loss.item())
        audit.extend(stats)
        if update % interval == 0:
            val_index = update // interval - 1
            val_loss = _validation_loss(
                model, val_loader, device, _val_seed(job["source"], int(job["seed"]), val_index)
            )
            history.append(
                {
                    "update": update,
                    "train_loss_last_batch": float(loss.item()),
                    "train_loss_mean": interval_loss_sum / interval,
                    "source_val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "validation_seed": _val_seed(job["source"], int(job["seed"]), val_index),
                }
            )
            interval_loss_sum = 0.0
            if val_loss < best_loss:
                best_update, best_loss = update, val_loss
                _atomic_save(model.state_dict(), output / "best.pt")
            _atomic_save(model.state_dict(), output / "last.pt")
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(),
                    "update": update,
                    "best_update": best_update,
                    "best_loss": best_loss,
                    "best_sha256": protocol.sha256_file(output / "best.pt"),
                    "history": history,
                    "audit": audit,
                    "run_signature": run_signature,
                    "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
                },
                output / "resume.pt",
            )
            protocol.save_json(output / "history.json", history)
    train_seconds = elapsed_prior + time.perf_counter() - started
    if (
        actual_calls != updates
        or len(audit) != updates * BATCH_SIZE
        or len(history) != updates // interval
    ):
        raise RuntimeError("Fixed-update, presentation, or validation budget did not complete")
    model.load_state_dict(
        torch.load(output / "best.pt", map_location=device, weights_only=True), strict=True
    )
    source_audit = _source_audit(audit, effective_hw)
    protocol.save_json(
        output / "source_training_audit.json", {"summary": source_audit, "presentations": audit}
    )
    val_gt = prepared["validation"]
    # The prediction dataset contains no labels; labels enter the common evaluator only.
    val_pred_ds = DARCData(
        {**val_gt, "annotations": []},
        prepared["image_dir"],
        intensity=job["intensity"],
        scale="S0",
        source=job["source"],
        seed=int(job["seed"]),
        expected_hashes=expected_hashes,
    )
    val_raw, val_input_audit = _predictions(model, val_pred_ds, device)
    protocol.save_json(output / "validation_input_audit.json", val_input_audit)
    protocol.save_json(output / "val_raw_predictions.json", val_raw)
    grid = [float(x) for x in cfg["postprocessing"]["score_threshold_grid"]]
    choices = []
    for threshold in grid:
        filtered = [p for p in val_raw if p["score"] >= threshold]
        metrics = evaluation.evaluate_coco(val_gt, filtered)
        choices.append((metrics["ap50"], metrics["ap"], -threshold, threshold, metrics))
    _, _, _, threshold, val_metrics = max(choices, key=lambda x: x[:3])
    protocol.save_json(
        output / "postprocessing_selection.json",
        {
            "score_threshold": threshold,
            "selection_split": "source_validation",
            "metrics": val_metrics,
        },
    )
    protocol.save_json(
        output / "val_predictions.json", [p for p in val_raw if p["score"] >= threshold]
    )
    checkpoint_hash = protocol.sha256_file(output / "best.pt")
    test_results = {}
    for site in job.get("eval_targets", []):
        gt = prepared["tests"][site]
        gt_path = output / f"{site}_gt.json"
        raw_path = output / f"{site}_raw_predictions.json"
        filtered_path = output / f"{site}_predictions.json"
        protocol.save_json(gt_path, gt)
        test_ds = DARCData(
            {**gt, "annotations": []},
            prepared["image_dir"],
            intensity=job["intensity"],
            scale="S0",
            source=job["source"],
            seed=int(job["seed"]),
            expected_hashes=expected_hashes,
        )
        raw, input_audit = _predictions(model, test_ds, device)
        filtered = [p for p in raw if p["score"] >= threshold]
        protocol.save_json(raw_path, raw)
        protocol.save_json(filtered_path, filtered)
        protocol.save_json(output / f"{site}_input_audit.json", input_audit)
        test_results[site] = {
            "gt_path": str(gt_path),
            "gt_sha256": protocol.sha256_file(gt_path),
            "checkpoint_sha256": checkpoint_hash,
            "raw_predictions_path": str(raw_path),
            "raw_predictions_sha256": protocol.sha256_file(raw_path),
            "predictions_path": str(filtered_path),
            "filtered_predictions_path": str(filtered_path),
            "predictions_sha256": protocol.sha256_file(filtered_path),
            "raw": evaluation.evaluate_coco(gt, raw),
            "filtered": evaluation.evaluate_coco(gt, filtered),
            "center_recall_8px_top30": _center_recall(gt, raw),
        }
    runtime = {
        **runtime_identity,
        "device": str(device),
        "gpu_uuid": job.get("gpu_uuid"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_resize_hw": list(effective_hw),
        "training_updates": actual_calls,
        "validation_count": len(history),
        "lr_first": 0.0025,
        "lr_after_update": DECAY_AFTER,
        "lr_after": 0.00025,
        "training_rng_seed": int(job["seed"]),
        "precision": "fp32",
        "autocast": False,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "train_seconds": train_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    protocol.save_json(output / "runtime.json", runtime)
    result = {
        "job_id": job["job_id"],
        "source": job["source"],
        "seed": int(job["seed"]),
        "stage": job["stage"],
        "condition": job["condition"],
        "intensity": job["intensity"],
        "scale": job["scale"],
        "size_control": bool(job.get("size_control")),
        "protocol_id": PROTOCOL_ID,
        "frozen_protocol_sha256": PROTOCOL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "annotation_sha256": ANNOTATION_SHA,
        "initial_state_sha256": initial_file_hash,
        "initial_tensor_sha256": job["initial_tensor_sha256"],
        "checkpoint": str(output / "best.pt"),
        "best_weights_sha256": checkpoint_hash,
        "best_update": best_update,
        "best_val_loss": best_loss,
        "training_updates": actual_calls,
        "validation_interval": interval,
        "batch_size": BATCH_SIZE,
        "presentation_stream_sha256": _json_digest(stream),
        "subset_ids_sha256": _json_digest(subset_ids) if subset_ids else None,
        "train_gt_sha256": protocol.sha256_file(output / "train_gt.json"),
        "validation_gt_sha256": protocol.sha256_file(output / "validation_gt.json"),
        "training_audit_sha256": protocol.sha256_file(output / "source_training_audit.json"),
        "source_validation": val_metrics,
        "score_threshold": threshold,
        "runtime_path": str(output / "runtime.json"),
        "runtime_sha256": protocol.sha256_file(output / "runtime.json"),
        "tests": test_results,
    }
    protocol.save_json(output / "metrics.json", result)
    return result


def run_job(job):
    """Fleet-compatible entry point."""
    return run(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "run"))
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    result = initialize(job) if args.action == "initialize" else run(job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
