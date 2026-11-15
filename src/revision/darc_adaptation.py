"""Frozen-label, head-only DARC↔MANV adaptation pilot.

A support-only diagnostic uses target-training labels without selecting held-out
labels for computation, inference, or scoring. The monolithic annotation file is
parsed. Pilot scoring is explicitly gated after the final update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import torch.version
import torchvision

from . import evaluation, protocol
from .darc_data import DARCData
from .darc_train import (
    ANNOTATION_SHA,
    MANIFEST_SHA,
    PROTOCOL_SHA,
    _atomic_save,
    _center_recall,
    _cfg,
    _configure_threads,
    _device_policy,
    _loader,
    _predictions,
    _verified_pretrained_asset,
)

ADAPT_ID = "darc-adapt-v1"
ADAPT_SHA = "d7c4d4f2395bdc5fa8bb00f9badf12fdd57c94f3c48ae5d1428ed2a6352dc549"
SUPPORT_COUNT = 16
UPDATES = 64
LR = 0.00025
PREDICTOR_PREFIX = "roi_heads.box_predictor."
PREDICTOR_NAMES = frozenset(
    PREDICTOR_PREFIX + name
    for name in ("cls_score.weight", "cls_score.bias", "bbox_pred.weight", "bbox_pred.bias")
)


def _digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _ordered_hash(prefix, image_id):
    return hashlib.sha256(f"{prefix}:{image_id}".encode()).digest(), image_id


def support_ids(images, target, seed):
    """Choose one existing target-training frame from each index stratum."""
    if target not in ("DARC", "MANV") or seed not in (0, 1, 2):
        raise ValueError("Invalid target or seed")
    ordered = sorted(images, key=lambda row: (int(row["frame"]), int(row["id"])))
    if len(ordered) < SUPPORT_COUNT or len({int(x["id"]) for x in ordered}) != len(ordered):
        raise ValueError("Invalid source training inventory")
    selected = []
    for k in range(SUPPORT_COUNT):
        group = ordered[k * len(ordered) // SUPPORT_COUNT : (k + 1) * len(ordered) // SUPPORT_COUNT]
        prefix = f"{ADAPT_ID}:support:{target}:{seed}"
        selected.append(
            min((int(x["id"]) for x in group), key=lambda value: _ordered_hash(prefix, value))
        )
    return selected


def presentation_stream(ids, source, target, seed, updates=UPDATES):
    """Concatenate deterministic support permutations and form consecutive pairs."""
    if len(ids) != SUPPORT_COUNT or len(set(ids)) != SUPPORT_COUNT or updates < 1:
        raise ValueError("Invalid support or update budget")
    result = []
    cycle = 0
    while len(result) < updates * 2:
        prefix = f"{ADAPT_ID}:train-order:{source}:{target}:{seed}:{cycle}"
        order = sorted(ids, key=lambda image_id: _ordered_hash(prefix, image_id))
        for image_id in order:
            if len(result) == updates * 2:
                break
            result.append((len(result), image_id))
        cycle += 1
    return result


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _partition_state(model):
    state = model.state_dict()
    head = {name: value for name, value in state.items() if name.startswith(PREDICTOR_PREFIX)}
    other = {name: value for name, value in state.items() if not name.startswith(PREDICTOR_PREFIX)}
    if set(head) != PREDICTOR_NAMES:
        raise ValueError("Unexpected box predictor architecture")
    return _state_digest(head), _state_digest(other)


def _freeze_predictor_only(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in PREDICTOR_NAMES)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable != PREDICTOR_NAMES:
        raise ValueError("Unexpected trainable parameters")
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _verify_checkpoint(path, expected_sha, metrics_path, expected_metrics_sha, source, seed):
    if (
        protocol.sha256_file(path) != expected_sha
        or protocol.sha256_file(metrics_path) != expected_metrics_sha
    ):
        raise ValueError("Checkpoint or its metrics digest differs from the frozen input")
    metrics = json.loads(Path(metrics_path).read_text())
    if (
        metrics.get("source"),
        metrics.get("seed"),
        metrics.get("stage"),
        metrics.get("condition"),
        metrics.get("protocol_id"),
        metrics.get("scale"),
        metrics.get("intensity"),
        metrics.get("size_control"),
        metrics.get("training_updates"),
        metrics.get("batch_size"),
        metrics.get("validation_interval"),
        metrics.get("manifest_sha256"),
        metrics.get("annotation_sha256"),
        metrics.get("frozen_protocol_sha256"),
        metrics.get("best_weights_sha256"),
    ) != (
        source,
        seed,
        "core",
        "buffered",
        "darc-processed-v1",
        "S0",
        "I0",
        False,
        3432,
        2,
        286,
        MANIFEST_SHA,
        ANNOTATION_SHA,
        PROTOCOL_SHA,
        expected_sha,
    ):
        raise ValueError("Checkpoint is not the matching completed full S0I0 source fit")
    if metrics.get("best_update") not in range(286, 3433, 286):
        raise ValueError("Source checkpoint selection update is invalid")
    return metrics


def _check_score_gate(job, protocol_sha):
    if not job.get("score_gate") or not job.get("score_gate_sha256"):
        raise ValueError("Pilot scoring requires a frozen, verified diagnostic gate")
    path = Path(job["score_gate"])
    if protocol.sha256_file(path) != job["score_gate_sha256"]:
        raise ValueError("Score gate digest mismatch")
    gate = json.loads(path.read_text())
    if (
        gate.get("schema"),
        gate.get("protocol_sha256"),
        gate.get("code_commit"),
        gate.get("passed"),
    ) != ("darc-adaptation-score-gate-v1", protocol_sha, job["code_commit"], True):
        raise ValueError("Score gate does not approve this protocol and release")
    diagnostics = gate.get("diagnostics", [])
    if len(diagnostics) != 2:
        raise ValueError("Score gate requires two direction-specific diagnostics")
    directions = set()
    for item in diagnostics:
        path = Path(item["metrics_path"])
        if protocol.sha256_file(path) != item["metrics_sha256"]:
            raise ValueError("Diagnostic metrics digest mismatch")
        m = json.loads(path.read_text())
        if (
            m.get("job_id"),
            m.get("stage"),
            m.get("seed"),
            m.get("training_updates"),
            m.get("adaptation_protocol_sha256"),
            m.get("code_commit"),
            m.get("gradients_finite"),
            m.get("target_test_accessed"),
            m.get("target_local_checkpoint_loaded"),
        ) != (
            item["job_id"],
            "diagnostic",
            0,
            2,
            protocol_sha,
            job["code_commit"],
            True,
            False,
            False,
        ):
            raise ValueError("Invalid diagnostic evidence")
        if not m.get("support_ids_verified") or m.get("frozen_state_sha256_before") != m.get(
            "frozen_state_sha256_after"
        ):
            raise ValueError("Diagnostic support or frozen-state check failed")
        directions.add((m["source"], m["target"]))
    if directions != {("DARC", "MANV"), ("MANV", "DARC")}:
        raise ValueError("Diagnostic evidence must cover both directions")
    return gate


def _source_protocol(job):
    path = Path(job["adaptation_protocol"])
    digest = protocol.sha256_file(path)
    if digest != ADAPT_SHA or job["adaptation_protocol_sha256"] != ADAPT_SHA:
        raise ValueError("Adaptation protocol digest mismatch")
    doc = json.loads(path.read_text())
    if doc.get("protocol_id") != ADAPT_ID:
        raise ValueError("Unexpected adaptation protocol")
    jobs = [
        entry
        for entry in doc["jobs"]
        if (entry["source"], entry["target"], entry["seed"])
        == (job["source"], job["target"], job["seed"])
    ]
    if len(jobs) != 1:
        raise ValueError("Pilot identity is missing or duplicated in protocol")
    return doc, jobs[0], digest


def run_job(job):
    """Fit a fixed head-only pilot or run a two-update source-only diagnostic."""
    from ..faster_rcnn.model import build_model

    thread_policy = _configure_threads()
    source, target, seed = job["source"], job["target"], int(job["seed"])
    if (source, target) not in (("DARC", "MANV"), ("MANV", "DARC")) or seed not in (0, 1, 2):
        raise ValueError("The pilot covers only DARC↔MANV and seeds 0–2")
    stage = job["stage"]
    if stage not in ("diagnostic", "pilot"):
        raise ValueError("Invalid adaptation stage")
    if not re.fullmatch(r"[0-9a-f]{40}", str(job.get("code_commit", ""))):
        raise ValueError("A full immutable code commit is required")
    if stage == "diagnostic" and any(
        job.get(key)
        for key in ("score_gate", "target_local_checkpoint", "target_local_metrics", "eval_targets")
    ):
        raise ValueError("Diagnostics cannot access a score gate or local-reference checkpoint")
    _doc, row, adaptation_sha = _source_protocol(job)
    if stage == "pilot" and job["job_id"] != row["job_id"]:
        raise ValueError("Pilot job ID differs from frozen protocol")
    if stage == "pilot":
        _check_score_gate(job, adaptation_sha)
    if (
        protocol.sha256_file(job["manifest"]) != MANIFEST_SHA
        or protocol.sha256_file(job["ann"]) != ANNOTATION_SHA
    ):
        raise ValueError("Frozen split or annotation digest mismatch")
    if (
        job.get("base_protocol_sha256") != PROTOCOL_SHA
        or protocol.sha256_file(job["base_protocol"]) != PROTOCOL_SHA
    ):
        raise ValueError("Frozen base study protocol mismatch")
    base = _verify_checkpoint(
        job["source_checkpoint"],
        job["source_checkpoint_sha256"],
        job["source_metrics"],
        job["source_metrics_sha256"],
        source,
        seed,
    )
    if base["job_id"] != row["source_fit_job_id"]:
        raise ValueError("Source checkpoint job differs from frozen protocol")
    manifest = json.loads(Path(job["manifest"]).read_text())
    all_coco = json.loads(Path(job["ann"]).read_text())
    train_ids = set(manifest["sites"][target]["partitions"]["train"])
    train_images = [image for image in all_coco["images"] if image["id"] in train_ids]
    ids = support_ids(train_images, target, seed)
    if ids != row["support_image_ids"] or _digest_ids(ids) != row["support_image_ids_sha256"]:
        raise ValueError("Support IDs do not match the frozen target-training selection")
    frames = {int(image["id"]): int(image["frame"]) for image in train_images}
    if [frames[image_id] for image_id in ids] != row["support_frame_indices"]:
        raise ValueError("Support frame indices differ from frozen protocol")
    selected = protocol._subset(all_coco, ids)
    if (
        len(selected["images"]) != SUPPORT_COUNT
        or len(selected["annotations"]) != row["support_box_count"]
    ):
        raise ValueError("Support image or label inventory mismatch")
    expected_hashes = {int(image["id"]): image["sha256"] for image in manifest["images"]}
    updates = 2 if stage == "diagnostic" else UPDATES
    full_stream = presentation_stream(ids, source, target, seed, UPDATES)
    if (
        _digest_ids([image_id for _, image_id in full_stream])
        != row["train_presentation_ids_sha256"]
    ):
        raise ValueError("Support presentation stream differs from frozen protocol")
    stream = full_stream[: updates * 2]
    output = Path(job["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ("metrics.json", "adapted_final.pt", "history.json")
    ):
        raise FileExistsError("Refusing to overwrite an adaptation attempt")
    _verified_pretrained_asset(job["pretrained_weights_sha256"])
    device = _device_policy({**job, "stage": "core" if stage == "pilot" else "diagnostic"})
    if device.type == "cuda" and torch.cuda.get_device_name(device) != "NVIDIA RTX A5000":
        raise ValueError("Adaptation pilot requires the designated RTX A5000 hardware")
    protocol.set_seed(seed)
    model = build_model(_cfg(), pretrained=True, box_score_thresh=0.0, box_detections_per_img=300)
    model.load_state_dict(
        torch.load(job["source_checkpoint"], map_location="cpu", weights_only=True), strict=True
    )
    model.to(device)
    head_before, frozen_before = _partition_state(model)
    params = _freeze_predictor_only(model)
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    dataset = DARCData(
        selected,
        job["images"],
        intensity="I0",
        scale="S0",
        source=source,
        seed=seed,
        presentations=stream,
        expected_hashes=expected_hashes,
    )
    loader = _loader(dataset, 2, int(job.get("workers", 2)), seed + 10000)
    protocol.set_seed(seed)
    started = time.perf_counter()
    history = []
    optimizer_calls = 0
    for update, (images, targets, stats) in enumerate(loader, 1):
        if len(images) != 2:
            raise ValueError("Adaptation batch must contain exactly two frames")
        model.train()
        model.backbone.eval()
        model.roi_heads.box_head.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = model(
                [image.to(device) for image in images],
                [{k: v.to(device) for k, v in item.items()} for item in targets],
            )
            loss = losses["loss_classifier"] + losses["loss_box_reg"]
        if any(not torch.isfinite(item) for item in losses.values()) or not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adaptation loss")
        loss.backward()
        checks = [torch.isfinite(p.grad).all() for p in params if p.grad is not None]
        if len(checks) != len(PREDICTOR_NAMES) or not torch.stack(checks).all():
            raise RuntimeError("Missing or nonfinite predictor gradient")
        if any(
            p.grad is not None
            for name, p in model.named_parameters()
            if name not in PREDICTOR_NAMES
        ):
            raise RuntimeError("Frozen parameter received a gradient")
        optimizer.step()
        optimizer_calls += 1
        history.append(
            {
                "update": update,
                "classifier_loss": float(losses["loss_classifier"].item()),
                "box_reg_loss": float(losses["loss_box_reg"].item()),
                "lr": LR,
                "image_ids": [int(t["image_id"]) for t in targets],
            }
        )
    if optimizer_calls != updates:
        raise RuntimeError("Adaptation update budget did not complete")
    head_after, frozen_after = _partition_state(model)
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen non-predictor tensor or buffer changed")
    if head_before == head_after:
        raise RuntimeError("Adaptation did not change the predictor")
    fit_seconds = time.perf_counter() - started
    final_path = output / ("diagnostic_final.pt" if stage == "diagnostic" else "adapted_final.pt")
    _atomic_save(model.state_dict(), final_path)
    protocol.save_json(output / "history.json", history)
    protocol.save_json(output / "support_gt.json", selected)
    protocol.save_json(output / "support_stream.json", stream)
    runtime = {
        "device": str(device),
        "gpu_uuid": job.get("gpu_uuid"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "thread_policy": thread_policy,
        "workers": int(job.get("workers", 2)),
        "precision": "fp32",
        "autocast": False,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fit_seconds": fit_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    protocol.save_json(output / "runtime.json", runtime)
    result = {
        "job_id": job["job_id"],
        "stage": stage,
        "source": source,
        "target": target,
        "seed": seed,
        "code_commit": job["code_commit"],
        "adaptation_protocol_sha256": adaptation_sha,
        "base_protocol_sha256": PROTOCOL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "annotation_sha256": ANNOTATION_SHA,
        "source_checkpoint_sha256": job["source_checkpoint_sha256"],
        "source_selected_update": base["best_update"],
        "source_metrics_sha256": job["source_metrics_sha256"],
        "support_image_ids": ids,
        "support_image_ids_sha256": _digest_ids(ids),
        "support_box_count": len(selected["annotations"]),
        "support_ids_verified": True,
        "support_stream_sha256": protocol.sha256_file(output / "support_stream.json"),
        "train_presentation_ids_sha256": _digest_ids([image_id for _, image_id in full_stream]),
        "training_updates": optimizer_calls,
        "batch_size": 2,
        "lr": LR,
        "gradients_finite": True,
        "trainable_parameter_names": sorted(PREDICTOR_NAMES),
        "predictor_sha256_before": head_before,
        "predictor_sha256_after": head_after,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": protocol.sha256_file(final_path),
        "target_test_accessed": False,
        "target_test_accessed_definition": "Held-out IDs and labels selected for computation or scoring; the monolithic annotation JSON is parsed in full.",
        "target_local_checkpoint_loaded": False,
        "runtime_sha256": protocol.sha256_file(output / "runtime.json"),
        "tests": {},
    }
    if stage == "pilot":
        # Destination references and the local comparator are opened only after
        # the diagnostic gate and the final adaptation checkpoint are frozen.
        _check_score_gate(job, adaptation_sha)
        local = _verify_checkpoint(
            job["target_local_checkpoint"],
            job["target_local_checkpoint_sha256"],
            job["target_local_metrics"],
            job["target_local_metrics_sha256"],
            target,
            seed,
        )
        if local["job_id"] != row["target_local_fit_job_id"]:
            raise ValueError("Local reference checkpoint differs from frozen protocol")
        partitions = manifest["sites"][target]["partitions"]
        for partition in ("validation", "validation_buffer", "test_buffer", "test"):
            if set(ids) & set(partitions[partition]):
                raise ValueError("Support overlaps a protected target partition")
        test_ids = partitions["test"]
        if (
            len(test_ids) != row["target_test_image_count"]
            or _digest_ids(test_ids) != row["target_test_image_ids_sha256"]
        ):
            raise ValueError("Target test inventory differs from frozen protocol")
        gt = protocol._subset(all_coco, test_ids)
        protocol.save_json(output / "target_gt.json", gt)
        dataset = DARCData(
            {**gt, "annotations": []},
            job["images"],
            intensity="I0",
            scale="S0",
            source=source,
            seed=seed,
            expected_hashes=expected_hashes,
        )
        result["target_test_accessed"] = True
        result["target_local_checkpoint_loaded"] = True
        result["target_gt_sha256"] = protocol.sha256_file(output / "target_gt.json")
        result["target_test_image_ids_sha256"] = _digest_ids(test_ids)
        result["target_local_checkpoint_sha256"] = job["target_local_checkpoint_sha256"]
        result["target_local_selected_update"] = local["best_update"]
        result["score_gate_sha256"] = job["score_gate_sha256"]
        for name, path in (
            ("source", job["source_checkpoint"]),
            ("adapted", final_path),
            ("target_local", job["target_local_checkpoint"]),
        ):
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True), strict=True
            )
            score_started = time.perf_counter()
            raw, _audit = _predictions(model, dataset, device)
            score_seconds = time.perf_counter() - score_started
            raw_path = output / f"{name}_raw_predictions.json"
            protocol.save_json(raw_path, raw)
            result["tests"][name] = {
                "checkpoint_sha256": protocol.sha256_file(path),
                "inference_seconds": score_seconds,
                "raw_predictions_path": str(raw_path),
                "raw_predictions_sha256": protocol.sha256_file(raw_path),
                "raw": evaluation.evaluate_coco(gt, raw),
                "center_recall_8px_top30": _center_recall(gt, raw),
            }
    protocol.save_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_job(json.loads(args.job.read_text())), indent=2))


if __name__ == "__main__":
    main()
