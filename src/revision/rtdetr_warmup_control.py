"""Bounded paired, source-only RT-DETR bias-warmup control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import protocol
from .ultralytics_models import stage_split

MANIFEST_SHA = "03b7d1d48e5faac7872a04f8fb23580a748ca1521539e99b0e54bf45c7d8f4aa"
GT_SHA = "407fb14aaa77641175ef6ca7a54f51c062f0c9d2044af436c0a5d1671aff6188"
WEIGHTS_SHA = "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
INITIAL_SHA = "2021349d41cc1e72c5a33eadcc8a202a275819d01d125a080f672f28411ca01a"


def _json(path):
    return json.loads(Path(path).read_text())


def control_arguments(data: Path, output: Path, seed: int, bias: float) -> dict:
    if seed not in (0, 1) or bias not in (0.0, 0.1):
        raise ValueError("only locked seeds and bias rates are allowed")
    return {
        "data": str(data),
        "epochs": 100,
        "patience": 0,
        "batch": 4,
        "imgsz": 1024,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "warmup_epochs": 2.0,
        "warmup_bias_lr": bias,
        "weight_decay": 0.0001,
        "device": "",
        "workers": 2,
        "amp": True,
        "cache": False,
        "seed": seed,
        "deterministic": False,
        "project": str(output),
        "name": "fit",
        "exist_ok": True,
        "pretrained": True,
        "save": True,
        "cos_lr": True,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.0,
    }


def comparable_arguments(args: dict) -> dict:
    """Remove only the intended bias intervention and attempt-local paths."""
    return {
        k: v
        for k, v in args.items()
        if k not in ("warmup_bias_lr", "project", "name", "save_dir", "data")
    }


def tensor_digest(state: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        )
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _gpu(uuid):
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or torch.cuda.device_count() != 1:
        raise ValueError("one explicitly pinned GPU is required")
    props = torch.cuda.get_device_properties(0)
    if ("GPU-" + str(props.uuid)).lower() != uuid.lower() or props.name != "NVIDIA RTX A5000":
        raise ValueError("assigned GPU must be the verified RTX A5000")
    return props


def prepare(
    manifest: Path,
    ann: Path,
    images: Path,
    weights: Path,
    initial: Path,
    output: Path,
    gpu_uuid: str,
) -> dict:
    """Verify exact source subset/assets and perform a disposable GPU gradient gate."""
    import torch
    import ultralytics
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if output.exists() and any(output.iterdir()):
        raise ValueError("gate output must be empty")
    for path, digest in ((manifest, MANIFEST_SHA), (weights, WEIGHTS_SHA), (initial, INITIAL_SHA)):
        if protocol.sha256_file(path) != digest:
            raise ValueError(f"locked input digest mismatch: {path}")
    props = _gpu(gpu_uuid)
    job = {
        "ann": str(ann),
        "images": str(images),
        "manifest": str(manifest),
        "source": "DAAN",
        "seed": 0,
        "condition": "buffered",
        "stage": "diagnostic",
        "eval_targets": [],
        "train_limit": 32,
        "overfit": True,
    }
    prepared = protocol.prepare_job(job)
    if (
        prepared["tests"]
        or prepared["train"] != prepared["validation"]
        or len(prepared["train"]["images"]) != 32
    ):
        raise ValueError("gate requires exactly 32 source training images and no test targets")
    output.mkdir(parents=True, exist_ok=True)
    gt_path = output / "source_gt.json"
    protocol.save_json(gt_path, prepared["validation"])
    if protocol.sha256_file(gt_path) != GT_SHA:
        raise ValueError("source subset differs from locked historical 32 frames")
    records = {x["id"]: x for x in _json(manifest)["images"]}
    original_hashes = {}
    for im in prepared["train"]["images"]:
        record = records[im["id"]]
        if record["site"] != "DAAN" or record["file_name"] != im["file_name"]:
            raise ValueError("non-source frame in subset")
        actual = protocol.sha256_file(images / im["file_name"])
        if actual != record["sha256"]:
            raise ValueError("source image differs from manifest")
        original_hashes[str(im["id"])] = actual
    train_dir = stage_split(prepared["train"], images, output / "data/train")
    stage_split(prepared["validation"], images, output / "data/val")
    staged = {
        str(p.relative_to(output)): protocol.sha256_file(p)
        for p in sorted((output / "data").rglob("*"))
        if p.is_file()
    }
    reference = torch.load(initial, map_location="cpu", weights_only=False)
    expected_digest = tensor_digest(reference["model"])
    # Disposable forward/backward gate uses the exact one-class initialized model, without an optimizer step.
    model = RTDETRDetectionModel(reference["architecture"], nc=1, ch=3, verbose=False)
    model.load_state_dict(reference["model"], strict=True)
    model = model.to("cuda").eval()
    from PIL import Image

    image = np.asarray(
        Image.open(next(train_dir.glob("*.png"))).convert("RGB").resize((1024, 1024))
    ).copy()
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255
    prediction = model(x)
    tensor = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    loss = tensor[..., :4].float().sum()
    if not torch.isfinite(loss):
        raise ValueError("GPU gate forward output is nonfinite")
    loss.backward()
    grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not grads or not bool(torch.stack(grads).all().item()):
        raise ValueError("GPU gate gradients are nonfinite or absent")
    del model, x, prediction, tensor, loss, grads
    torch.cuda.empty_cache()
    gate = {
        "scope": "source training frames only; no test data",
        "source": "DAAN",
        "image_ids": [i["id"] for i in prepared["train"]["images"]],
        "gt_sha256": GT_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "weights_sha256": WEIGHTS_SHA,
        "initial_file_sha256": INITIAL_SHA,
        "initial_tensor_sha256": expected_digest,
        "weights": str(weights.resolve()),
        "initial": str(initial.resolve()),
        "data_dir": str((output / "data").resolve()),
        "original_image_sha256": original_hashes,
        "staged_file_sha256": staged,
        "gpu_uuid": gpu_uuid,
        "gpu_model": props.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "forward_backward_gate": "passed",
        "seed_schedule": [0, 1],
        "bias_values": [0.0, 0.1],
        "scheduler_horizon_epochs": 100,
        "stop_after_epochs": 42,
        "early_stopping": "disabled for both arms",
        "input_preflight_comparable": comparable_arguments(
            control_arguments(Path("data.yaml"), Path("out"), 0, 0.0)
        )
        == comparable_arguments(control_arguments(Path("data.yaml"), Path("out"), 0, 0.1)),
    }
    protocol.save_json(output / "gate.json", gate)
    return gate


def _finite_gradients(model) -> bool:
    import torch

    parts = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    return not parts or bool(torch.stack(parts).all().item())


def observe_optimizer_step(trainer, control_state: dict, execute_original) -> None:
    """Record native scaler skip/backoff without changing its update sequence."""
    import torch

    # Observe the framework's unmodified scaler/optimizer sequence. AMP
    # overflows are expected to skip the step and back off the scale.
    if trainer.loss is None or not bool(torch.isfinite(trainer.loss).all().item()):
        raise FloatingPointError("nonfinite training loss before optimizer step")
    row = trainer.control_batch
    row["optimizer_step"] = True
    row["scaled_gradient_finite"] = _finite_gradients(trainer.model)
    row["scale_before"] = float(trainer.scaler.get_scale())
    observed: dict[str, Any] = {"executed": False}
    original_step, original_unscale = trainer.optimizer.step, trainer.scaler.unscale_

    def observe_step(*step_args, **step_kwargs):
        if not observed.get("unscaled_gradient_finite", False) or any(
            observed.get("found_inf", {}).values()
        ):
            raise FloatingPointError("optimizer attempted an update with nonfinite gradients")
        observed["executed"] = True
        return original_step(*step_args, **step_kwargs)

    def observe_unscale(optimizer):
        value = original_unscale(optimizer)
        observed["unscaled_gradient_finite"] = _finite_gradients(trainer.model)
        observed["found_inf"] = (
            {
                str(device): float(value.item())
                for device, value in trainer.scaler._found_inf_per_device(optimizer).items()
            }
            if trainer.scaler.is_enabled()
            else {}
        )
        if not observed["unscaled_gradient_finite"] and not any(observed["found_inf"].values()):
            raise FloatingPointError("nonfinite unscaled gradient was not registered by scaler")
        return value

    trainer.optimizer.step = observe_step
    trainer.scaler.unscale_ = observe_unscale
    try:
        execute_original()
    finally:
        trainer.optimizer.step = original_step
        trainer.scaler.unscale_ = original_unscale
    control_state["optimizer_attempts"] += 1
    control_state["optimizer_steps"] += int(observed["executed"])
    control_state["skipped_steps"] += int(not observed["executed"])
    row.update(
        {
            "unscaled_gradient_finite": observed["unscaled_gradient_finite"],
            "found_inf": observed["found_inf"],
            "optimizer_step_executed": observed["executed"],
            "scale_after": float(trainer.scaler.get_scale()),
        }
    )


def fit(
    gate_path: Path,
    output: Path,
    seed: int,
    bias: float,
    gpu_uuid: str,
    probe_first_step: bool = False,
) -> dict:
    """Run a locked arm, recording per-batch losses, gradients, LR and augmentations."""
    import torch
    import ultralytics
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    gate = _json(gate_path)
    if gate.get("forward_backward_gate") != "passed" or gate["gpu_uuid"] != gpu_uuid:
        raise ValueError("a passing gate on the same GPU is required")
    props = _gpu(gpu_uuid)
    if output.exists() and any(output.iterdir()):
        raise ValueError("arm output must be empty")
    for key, digest in (("weights", WEIGHTS_SHA), ("initial", INITIAL_SHA)):
        if protocol.sha256_file(Path(gate[key])) != digest:
            raise ValueError(f"{key} changed after gate")
    data_source = Path(gate["data_dir"])
    for relative, digest in gate["staged_file_sha256"].items():
        if protocol.sha256_file(gate_path.parent / relative) != digest:
            raise ValueError(f"staged input changed: {relative}")
    output.mkdir(parents=True)
    shutil.copytree(data_source, output / "data")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output / "data"),
                "train": str(output / "data/train/images"),
                "val": str(output / "data/val/images"),
                "names": {0: "vessel"},
            }
        )
    )
    args = control_arguments(data_yaml, output, seed, bias)
    reference = torch.load(gate["initial"], map_location="cpu", weights_only=False)
    original_digest = tensor_digest(reference["model"])
    if original_digest != gate["initial_tensor_sha256"]:
        raise ValueError("initial model tensor digest mismatch")
    control_state = {
        "batches": 0,
        "optimizer_attempts": 0,
        "optimizer_steps": 0,
        "skipped_steps": 0,
        "complete_epoch": 0,
    }
    trace_path = output / "batches.jsonl"

    class ControlTrainer(RTDETRTrainer):
        def preprocess_batch(self, batch):
            if self.batch_size != 4:
                raise RuntimeError("batch size changed, including OOM retry")

            def hash_tensor(value):
                return hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()

            self.control_batch = {
                "epoch": int(self.epoch) + 1,
                "index": control_state["batches"],
                "image_ids": [Path(x).stem for x in batch["im_file"]],
                "image_tensor_sha256": hash_tensor(batch["img"]),
                "boxes_sha256": hash_tensor(batch["bboxes"]),
                "classes_sha256": hash_tensor(batch["cls"]),
                "lr": [float(g["lr"]) for g in self.optimizer.param_groups],
                "accumulate": int(self.accumulate),
                "optimizer_step": False,
            }
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            return observe_optimizer_step(self, control_state, super().optimizer_step)

    def start(trainer):
        if trainer.batch_size != 4 or trainer.args.epochs != 100 or trainer.args.patience != 0:
            raise ValueError("training settings differ from locked control")
        trainer.model.load_state_dict(reference["model"], strict=True)
        trainer.ema.ema.load_state_dict(reference["model"], strict=True)
        observed = tensor_digest(trainer.model.state_dict())
        if observed != original_digest:
            raise ValueError("restored initial model differs")
        trainer.validator.args.device = ""
        trainer.args.device = ""
        names = {id(p): name for name, p in trainer.model.named_parameters()}
        groups = [
            {
                "param_group": g.get("param_group"),
                "parameters": sorted(names[id(p)] for p in g["params"]),
            }
            for g in trainer.optimizer.param_groups
        ]
        state = {
            "initial_tensor_sha256": observed,
            "resolved_args": vars(trainer.args),
            "optimizer_groups": groups,
            "effective_amp": bool(trainer.amp),
            "gpu_model": props.name,
            "gpu_uuid": gpu_uuid,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "threads": torch.get_num_threads(),
            "backend": {
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
            },
            "original_fit_settings": args,
        }
        protocol.save_json(output / "initial_provenance.json", state)

    class ProbeComplete(Exception):
        pass

    def batch_end(trainer):
        if not bool(torch.isfinite(trainer.loss).all().item()):
            raise FloatingPointError("nonfinite training loss")
        row = trainer.control_batch
        row["loss"] = float(trainer.loss.detach().cpu())
        row["loss_items"] = [float(x) for x in trainer.loss_items.detach().cpu().reshape(-1)]
        row["gradient_finite_after_batch"] = _finite_gradients(trainer.model)
        if not trainer.scaler.is_enabled() and not row["gradient_finite_after_batch"]:
            raise FloatingPointError("nonfinite gradient with AMP disabled")
        row["optimizer_step_index"] = control_state["optimizer_steps"]
        row["optimizer_attempt_index"] = control_state["optimizer_attempts"]
        with trace_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        control_state["batches"] += 1
        if probe_first_step and control_state["optimizer_attempts"] >= 1:
            raise ProbeComplete("first optimizer attempt recorded")

    def saved(trainer):
        if int(trainer.epoch) == 41:
            shutil.copy2(trainer.last, output / "epoch42_resumable.pt")

    def epoch_end(trainer):
        control_state["complete_epoch"] = int(trainer.epoch) + 1
        if trainer.epoch == 41:
            trainer.stop = True

    protocol.set_seed(seed)
    model = RTDETR(gate["weights"])
    model.add_callback("on_pretrain_routine_end", start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_model_save", saved)
    model.add_callback("on_fit_epoch_end", epoch_end)
    t0 = time.perf_counter()
    try:
        model.train(trainer=ControlTrainer, **args)
        if control_state["complete_epoch"] != 42 or not (output / "epoch42_resumable.pt").exists():
            raise RuntimeError("arm did not complete exactly 42 epochs")
        result = {
            "status": "succeeded",
            "seed": seed,
            "warmup_bias_lr": bias,
            "gate_sha256": protocol.sha256_file(gate_path),
            "gpu_uuid": gpu_uuid,
            "executed_epochs": control_state["complete_epoch"],
            "batches": control_state["batches"],
            "optimizer_attempts": control_state["optimizer_attempts"],
            "optimizer_steps": control_state["optimizer_steps"],
            "skipped_steps": control_state["skipped_steps"],
            "elapsed_seconds": time.perf_counter() - t0,
            "trace_sha256": protocol.sha256_file(trace_path),
            "results_sha256": protocol.sha256_file(output / "fit/results.csv"),
            "epoch42_checkpoint_sha256": protocol.sha256_file(output / "epoch42_resumable.pt"),
        }
    except ProbeComplete:
        result = {
            "status": "probe-complete",
            "seed": seed,
            "warmup_bias_lr": bias,
            "gpu_uuid": gpu_uuid,
            "progress": control_state,
            "trace_sha256": protocol.sha256_file(trace_path),
        }
    except BaseException as exc:
        protocol.save_json(
            output / "completion.json",
            {
                "status": "failed",
                "seed": seed,
                "warmup_bias_lr": bias,
                "error": f"{type(exc).__name__}: {exc}",
                "progress": control_state,
                "last_batch": getattr(getattr(model, "trainer", None), "control_batch", None),
            },
        )
        raise
    protocol.save_json(output / "completion.json", result)
    return result


def compare(left: Path, right: Path, output: Path) -> dict:
    """Check pair integrity before reporting training trajectories."""
    a, b = _json(left / "completion.json"), _json(right / "completion.json")
    if (
        a["status"] != "succeeded"
        or b["status"] != "succeeded"
        or a["seed"] != b["seed"]
        or {a["warmup_bias_lr"], b["warmup_bias_lr"]} != {0.0, 0.1}
        or a["gpu_uuid"] != b["gpu_uuid"]
    ):
        raise ValueError("arms are not a completed matched pair")
    x, y = _json(left / "initial_provenance.json"), _json(right / "initial_provenance.json")
    if (
        x["initial_tensor_sha256"] != y["initial_tensor_sha256"]
        or x["optimizer_groups"] != y["optimizer_groups"]
    ):
        raise ValueError("initial tensors or optimizer groups differ")
    if comparable_arguments(x["resolved_args"]) != comparable_arguments(y["resolved_args"]):
        raise ValueError("resolved arguments differ beyond intervention or paths")
    left_rows = [json.loads(line) for line in (left / "batches.jsonl").read_text().splitlines()]
    right_rows = [json.loads(line) for line in (right / "batches.jsonl").read_text().splitlines()]
    fields = (
        "epoch",
        "index",
        "image_ids",
        "image_tensor_sha256",
        "boxes_sha256",
        "classes_sha256",
    )
    if len(left_rows) != len(right_rows) or any(
        any(i[k] != j[k] for k in fields) for i, j in zip(left_rows, right_rows, strict=True)
    ):
        raise ValueError("paired batch order or augmentation differs")
    curves = []
    for path in (left, right):
        with (path / "fit/results.csv").open(newline="") as stream:
            curves.append(list(csv.DictReader(stream, skipinitialspace=True)))
    if any(len(c) != 42 for c in curves):
        raise ValueError("expected 42 per-epoch measurements")
    result = {
        "seed": a["seed"],
        "gpu_uuid": a["gpu_uuid"],
        "matched_initial_tensors": True,
        "matched_optimizer_groups": True,
        "matched_batch_order_and_augmentations": True,
        "batches_compared": len(left_rows),
        "arms": [a, b],
        "final_epoch_metrics": [c[-1] for c in curves],
    }
    protocol.save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gate = sub.add_parser("prepare")
    for key in ("manifest", "ann", "images", "weights", "initial", "output"):
        gate.add_argument("--" + key, type=Path, required=True)
    gate.add_argument("--gpu-uuid", required=True)
    arm = sub.add_parser("fit")
    arm.add_argument("--gate", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--seed", type=int, choices=(0, 1), required=True)
    arm.add_argument("--bias", type=float, choices=(0.0, 0.1), required=True)
    arm.add_argument("--gpu-uuid", required=True)
    arm.add_argument("--probe-first-step", action="store_true")
    paired = sub.add_parser("compare")
    paired.add_argument("--left", type=Path, required=True)
    paired.add_argument("--right", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.manifest,
            args.ann,
            args.images,
            args.weights,
            args.initial,
            args.output,
            args.gpu_uuid,
        )
    elif args.action == "fit":
        result = fit(
            args.gate, args.output, args.seed, args.bias, args.gpu_uuid, args.probe_first_step
        )
    else:
        result = compare(args.left, args.right, args.output)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
