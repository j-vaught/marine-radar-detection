"""Protocol guards and state preservation for the fixed-update trainer."""

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.revision import darc_train as trainer


def base_job():
    return {
        "source": "DAAN",
        "seed": 0,
        "stage": "core",
        "condition": "buffered",
        "intensity": "I0",
        "scale": "S0",
        "eval_targets": ["DAAN", "DARC", "MANV"],
    }


def test_core_budget_and_diagnostic_target_guard():
    job = base_job()
    assert trainer._validate_job(job) == (3432, 286)
    job["diagnostic_updates"] = 32
    with pytest.raises(ValueError):
        trainer._validate_job(job)
    job.update(stage="diagnostic", diagnostic_val_interval=16)
    with pytest.raises(ValueError):
        trainer._validate_job(job)
    job["eval_targets"] = []
    assert trainer._validate_job(job) == (32, 16)


def test_validation_restores_all_rng_and_mutable_buffers():
    class LossModel(torch.nn.Module):
        counter: torch.Tensor

        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.tensor(0.0))

        def forward(self, images, targets):
            self.counter.add_(1)
            random.random()
            np.random.random()
            return {"loss": torch.rand(()) + len(images)}

    model = LossModel()
    model.eval()
    loader = [([torch.ones(1)], [{"boxes": torch.zeros((0, 4))}], [{}])]
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    snapshot = trainer._rng_state()
    before = copy.deepcopy(snapshot)
    first = trainer._validation_loss(model, loader, torch.device("cpu"), 123)
    second = trainer._validation_loss(model, loader, torch.device("cpu"), 123)
    after = trainer._rng_state()
    assert first == second
    assert model.counter.item() == 0 and not model.training
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])


def test_init_uses_pretrained_architecture_and_records_identity(tmp_path, monkeypatch):
    from src.faster_rcnn import model as faster_model

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)
            self.backbone.weight.requires_grad = False

    calls = []

    def build(cfg, **kwargs):
        calls.append(kwargs)
        return Toy()

    monkeypatch.setattr(faster_model, "build_model", build)
    monkeypatch.setattr(
        trainer, "_verified_pretrained_asset", lambda _: ("asset-url", tmp_path / "asset")
    )
    state = tmp_path / "init.pt"
    job = {
        "source": "DAAN",
        "seed": 1,
        "initial_state": str(state),
        "pretrained_weights_sha256": "a" * 64,
        "code_commit": "a" * 40,
    }
    metadata = trainer.initialize(job)
    assert calls == [{"pretrained": True, "box_score_thresh": 0.0, "box_detections_per_img": 300}]
    toy = Toy()
    toy.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
    assert trainer._state_hash(toy) == metadata["initial_tensor_sha256"]
    assert trainer._architecture_identity(toy) == metadata["architecture"]
    assert metadata["architecture"]["trainable_parameter_count"] == sum(
        p.numel() for p in toy.parameters() if p.requires_grad
    )
    with pytest.raises(FileExistsError):
        trainer.initialize(job)


def test_center_matching_maximizes_unique_matches():
    gt = {
        "images": [{"id": 1}],
        "annotations": [
            {"image_id": 1, "bbox": [0, 0, 2, 2]},
            {"image_id": 1, "bbox": [8, 0, 2, 2]},
        ],
    }
    preds = [
        {"image_id": 1, "bbox": [3, 0, 2, 2], "score": 0.9},
        {"image_id": 1, "bbox": [-6, 0, 2, 2], "score": 0.8},
    ]
    result = trainer._center_recall(gt, preds)
    assert result["matched"] == 2 and result["recall"] == 1.0


def test_resume_identity_binds_runtime_and_stream():
    job = base_job() | {
        "job_id": "x",
        "ann": "/a",
        "images": "/i",
        "manifest": "/m",
        "code_commit": "a" * 40,
    }
    runtime = {"gpu_uuid": "first", "workers": 2, "precision": "fp32"}
    original = trainer._signature(job, [(0, 5), (1, 9)], "a", runtime)
    assert original != trainer._signature(job, [(0, 9), (1, 5)], "a", runtime)
    assert original != trainer._signature(
        job, [(0, 5), (1, 9)], "a", runtime | {"gpu_uuid": "second"}
    )


def test_core_disallows_cpu_before_training():
    job = base_job() | {"device": "cpu", "code_commit": "a" * 40}
    with pytest.raises(ValueError, match="pinned CUDA"):
        trainer._device_policy(job)


def test_diagnostic_runs_exact_updates_without_target_scoring(tmp_path, monkeypatch):
    import json
    import sys
    import types

    import torchvision

    from src.faster_rcnn import model as faster_model

    class Transform(torch.nn.Module):
        def resize(self, image, target):
            return image, target

    class TinyModel(torch.nn.Module):
        counter: torch.Tensor

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.transform = Transform()
            self.register_buffer("counter", torch.tensor(0))

        def forward(self, images, targets=None):
            if self.training:
                self.counter.add_(1)
                return {"loss": self.weight.square()}
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                    "scores": torch.tensor([0.8]),
                    "labels": torch.tensor([1]),
                }
                for _ in images
            ]

    class TinyDataset:
        def __init__(
            self,
            coco,
            image_dir,
            *,
            intensity,
            scale,
            source,
            seed,
            presentations=None,
            expected_hashes=None,
        ):
            self.rows = (
                presentations
                if presentations is not None
                else [(None, row["id"]) for row in coco["images"]]
            )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            presentation, image_id = self.rows[index]
            stats = {
                "image_id": image_id,
                "presentation_index": presentation,
                "intensity_fallback": False,
                "scale": 1.0,
                "scale_identity_reason": "baseline",
                "scale_rejections": 0,
                "box_areas_before": [4.0],
                "box_areas_after": [4.0],
            }
            target = {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "labels": torch.tensor([1]),
                "image_id": torch.tensor(image_id),
            }
            return torch.zeros((3, 4, 4)), target, stats

    fake_data = types.ModuleType("src.revision.darc_data")
    fake_data.__dict__["DARCData"] = TinyDataset
    fake_data.__dict__["train_stream"] = lambda images, source, seed, presentations: [
        (i, images[i % len(images)]["id"]) for i in range(presentations)
    ]
    fake_data.__dict__["select_count_subset"] = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "src.revision.darc_data", fake_data)
    calls = []

    def build(cfg, **kwargs):
        calls.append(kwargs)
        return TinyModel()

    monkeypatch.setattr(faster_model, "build_model", build)
    monkeypatch.setattr(
        trainer, "_verified_pretrained_asset", lambda _: ("url", tmp_path / "asset")
    )
    frozen = tmp_path / "protocol.json"
    frozen.write_text("protocol")
    ann = tmp_path / "ann.json"
    ann.write_text("ann")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"id": 1, "sha256": "x"}]}))
    monkeypatch.setattr(trainer, "PROTOCOL_SHA", trainer.protocol.sha256_file(frozen))
    monkeypatch.setattr(trainer, "ANNOTATION_SHA", trainer.protocol.sha256_file(ann))
    monkeypatch.setattr(trainer, "MANIFEST_SHA", trainer.protocol.sha256_file(manifest))
    coco = {"images": [{"id": 1}], "annotations": [], "categories": [{"id": 1}]}
    monkeypatch.setattr(
        trainer.protocol,
        "prepare_job",
        lambda _: {"train": coco, "validation": coco, "tests": {}, "image_dir": tmp_path},
    )
    monkeypatch.setattr(
        trainer.evaluation, "evaluate_coco", lambda gt, preds: {"ap": 0.5, "ap50": 0.5}
    )
    model = TinyModel()
    initial = tmp_path / "init.pt"
    torch.save(model.state_dict(), initial)
    initial_hash, tensor_hash = trainer.protocol.sha256_file(initial), trainer._state_hash(model)
    meta = {
        "source": "DAAN",
        "seed": 0,
        "initial_tensor_sha256": tensor_hash,
        "initial_state_sha256": initial_hash,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pretrained_weights_sha256": "x",
        "architecture": trainer._architecture_identity(model),
        "frozen_protocol_sha256": trainer.PROTOCOL_SHA,
        "model_config_sha256": trainer._json_digest(trainer._cfg()),
        "code_commit": "a" * 40,
    }
    Path(str(initial) + ".json").write_text(json.dumps(meta))
    job = base_job() | {
        "stage": "diagnostic",
        "code_commit": "a" * 40,
        "job_id": "test",
        "eval_targets": [],
        "diagnostic_updates": 4,
        "diagnostic_val_interval": 2,
        "ann": str(ann),
        "manifest": str(manifest),
        "images": str(tmp_path),
        "frozen_protocol": str(frozen),
        "frozen_protocol_sha256": trainer.PROTOCOL_SHA,
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
        "initial_state": str(initial),
        "initial_state_sha256": initial_hash,
        "initial_tensor_sha256": tensor_hash,
        "workers": 0,
    }
    result = trainer.run_job(job)
    assert result["training_updates"] == 4 and result["tests"] == {}
    assert len(json.loads((tmp_path / "out/history.json").read_text())) == 2
    assert calls == [{"pretrained": True, "box_score_thresh": 0.0, "box_detections_per_img": 300}]
    assert not list((tmp_path / "out").glob("DAAN_*"))


def test_decay_occurs_after_exactly_2288_updates():
    assert trainer._learning_rate(1) == trainer._learning_rate(2288) == 0.0025
    assert trainer._learning_rate(2289) == trainer._learning_rate(3432) == 0.00025
    with pytest.raises(ValueError):
        trainer._learning_rate(0)


def test_validation_disables_inherited_autocast():
    class AutocastCheck(torch.nn.Module):
        def forward(self, images, targets):
            assert not torch.is_autocast_enabled("cpu")
            return {"loss": torch.tensor(1.0)}

    loader = [([torch.ones(1)], [{"boxes": torch.zeros((0, 4))}], [{}])]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert trainer._validation_loss(AutocastCheck(), loader, torch.device("cpu"), 3) == 1.0
