"""Locked adaptation selection, gradient scope and scoring gate checks."""

import json

import pytest
import torch

from src.revision import darc_adaptation as adapt


def test_frozen_presentation_stream_matches_protocol():
    ids = [
        1504,
        1520,
        1569,
        1605,
        1632,
        1686,
        1695,
        1739,
        1772,
        1833,
        1845,
        1892,
        1911,
        1953,
        1990,
        2015,
    ]
    stream = adapt.presentation_stream(ids, "DARC", "MANV", 0)
    assert len(stream) == 128
    assert (
        adapt._digest_ids([image_id for _, image_id in stream])
        == "922492223cbadb95bbe197425b391e8b4856fc8cda04bf59bb9124e99613d71c"
    )
    assert all(
        {image_id for _, image_id in stream[k : k + 16]} == set(ids) for k in range(0, 128, 16)
    )
    assert adapt.presentation_stream(ids, "DARC", "MANV", 0, 2) == stream[:4]


def test_support_is_one_per_stratum_and_deterministic():
    images = [{"id": i + 100, "frame": i} for i in reversed(range(32))]
    selected = adapt.support_ids(images, "DARC", 2)
    assert len(set(selected)) == 16
    assert all(100 + 2 * k <= image_id < 102 + 2 * k for k, image_id in enumerate(selected))
    assert selected == adapt.support_ids(list(reversed(images)), "DARC", 2)


def test_predictor_only_freezes_all_other_state():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.roi_heads = torch.nn.Module()
            self.roi_heads.box_predictor = torch.nn.Module()
            self.roi_heads.box_predictor.cls_score = torch.nn.Linear(2, 2)
            self.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(2, 4)
            self.register_buffer("fixed_buffer", torch.ones(1))

    model = Toy()
    head, frozen = adapt._partition_state(model)
    params = adapt._freeze_predictor_only(model)
    optimizer = torch.optim.SGD(params, lr=0.1)
    loss = sum(p.square().sum() for p in params)
    loss.backward()
    optimizer.step()
    head_after, frozen_after = adapt._partition_state(model)
    assert head != head_after and frozen == frozen_after
    assert len(params) == 4


def test_gate_requires_two_hashed_source_only_diagnostics(tmp_path):
    entries = []
    for source, target in [("DARC", "MANV"), ("MANV", "DARC")]:
        data = {
            "job_id": source,
            "stage": "diagnostic",
            "seed": 0,
            "training_updates": 2,
            "adaptation_protocol_sha256": "protocol",
            "code_commit": "a" * 40,
            "gradients_finite": True,
            "target_test_accessed": False,
            "target_local_checkpoint_loaded": False,
            "support_ids_verified": True,
            "frozen_state_sha256_before": "same",
            "frozen_state_sha256_after": "same",
            "source": source,
            "target": target,
        }
        p = tmp_path / f"{source}.json"
        p.write_text(json.dumps(data))
        entries.append(
            {
                "job_id": source,
                "metrics_path": str(p),
                "metrics_sha256": adapt.protocol.sha256_file(p),
            }
        )
    gate = {
        "schema": "darc-adaptation-score-gate-v1",
        "protocol_sha256": "protocol",
        "code_commit": "a" * 40,
        "passed": True,
        "diagnostics": entries,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    job = {
        "score_gate": str(path),
        "score_gate_sha256": adapt.protocol.sha256_file(path),
        "code_commit": "a" * 40,
    }
    adapt._check_score_gate(job, "protocol")
    (tmp_path / "DARC.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        adapt._check_score_gate(job, "protocol")


def test_diagnostic_fits_predictor_without_test_or_local_access(tmp_path, monkeypatch):
    from src.faster_rcnn import model as faster_model

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.roi_heads = torch.nn.Module()
            self.roi_heads.box_head = torch.nn.Linear(2, 2)
            self.roi_heads.box_predictor = torch.nn.Module()
            self.roi_heads.box_predictor.cls_score = torch.nn.Linear(2, 2)
            self.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(2, 4)
            self.register_buffer("frozen_buffer", torch.tensor(3.0))

        def forward(self, images, targets):
            predictor = self.roi_heads.box_predictor
            return {
                "loss_classifier": sum(p.square().sum() for p in predictor.cls_score.parameters()),
                "loss_box_reg": sum(p.square().sum() for p in predictor.bbox_pred.parameters()),
                "loss_objectness": torch.tensor(1.0),
            }

    class Dataset:
        def __init__(self, coco, image_dir, *, presentations, **kwargs):
            self.stream = presentations

        def __len__(self):
            return len(self.stream)

        def __getitem__(self, index):
            _, image_id = self.stream[index]
            target = {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "labels": torch.tensor([1]),
                "image_id": torch.tensor(image_id),
            }
            return torch.ones((3, 4, 4)), target, {}

    images = [
        {"id": i + 1, "frame": i, "height": 4, "width": 4, "file_name": f"{i}.png"}
        for i in range(32)
    ]
    ids = adapt.support_ids(images, "DARC", 0)
    full = adapt.presentation_stream(ids, "MANV", "DARC", 0)
    row = {
        "source_fit_job_id": "base",
        "support_image_ids": ids,
        "support_image_ids_sha256": adapt._digest_ids(ids),
        "support_box_count": 16,
        "support_frame_indices": [i - 1 for i in ids],
        "train_presentation_ids_sha256": adapt._digest_ids([i for _, i in full]),
    }
    monkeypatch.setattr(adapt, "_source_protocol", lambda job: ({}, row, "adapt-protocol"))
    monkeypatch.setattr(adapt, "_configure_threads", lambda: {"test": True})
    monkeypatch.setattr(adapt, "_device_policy", lambda job: torch.device("cpu"))
    monkeypatch.setattr(adapt, "_verified_pretrained_asset", lambda sha: None)
    monkeypatch.setattr(
        adapt, "_verify_checkpoint", lambda *a: {"job_id": "base", "best_update": 286}
    )
    monkeypatch.setattr(adapt, "DARCData", Dataset)
    monkeypatch.setattr(
        adapt, "_predictions", lambda *a: pytest.fail("diagnostic must not infer target")
    )
    monkeypatch.setattr(faster_model, "build_model", lambda *a, **k: Toy())
    annotation = tmp_path / "all.json"
    annotation.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": [
                    {"id": i + 1, "image_id": i + 1, "bbox": [0, 0, 1, 1]} for i in range(32)
                ],
                "categories": [{"id": 1}],
            }
        )
    )
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [{"id": row["id"], "sha256": "x"} for row in images],
                "sites": {"DARC": {"partitions": {"train": [row["id"] for row in images]}}},
            }
        )
    )
    core_protocol = tmp_path / "base-protocol.json"
    core_protocol.write_text("core")
    monkeypatch.setattr(adapt, "ANNOTATION_SHA", adapt.protocol.sha256_file(annotation))
    monkeypatch.setattr(adapt, "MANIFEST_SHA", adapt.protocol.sha256_file(manifest))
    monkeypatch.setattr(adapt, "PROTOCOL_SHA", adapt.protocol.sha256_file(core_protocol))
    checkpoint = tmp_path / "source.pt"
    torch.save(Toy().state_dict(), checkpoint)
    job = {
        "job_id": "diag",
        "stage": "diagnostic",
        "source": "MANV",
        "target": "DARC",
        "seed": 0,
        "code_commit": "a" * 40,
        "ann": str(annotation),
        "manifest": str(manifest),
        "base_protocol": str(core_protocol),
        "base_protocol_sha256": adapt.PROTOCOL_SHA,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": "stub",
        "source_metrics": "stub",
        "source_metrics_sha256": "stub",
        "pretrained_weights_sha256": "stub",
        "images": str(tmp_path),
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
        "workers": 0,
    }
    result = adapt.run_job(job)
    assert result["training_updates"] == 2 and result["tests"] == {}
    assert result["frozen_state_sha256_before"] == result["frozen_state_sha256_after"]
    assert result["predictor_sha256_before"] != result["predictor_sha256_after"]
    assert not result["target_test_accessed"] and not result["target_local_checkpoint_loaded"]
    assert not (tmp_path / "out/adapted_final.pt").exists()


def test_checkpoint_verification_accepts_matching_core_and_rejects_changes(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = adapt.protocol.sha256_file(checkpoint)
    path = tmp_path / "metrics.json"
    metrics = {
        "source": "DARC",
        "seed": 1,
        "stage": "core",
        "condition": "buffered",
        "protocol_id": "darc-processed-v1",
        "scale": "S0",
        "intensity": "I0",
        "size_control": False,
        "training_updates": 3432,
        "batch_size": 2,
        "validation_interval": 286,
        "manifest_sha256": adapt.MANIFEST_SHA,
        "annotation_sha256": adapt.ANNOTATION_SHA,
        "frozen_protocol_sha256": adapt.PROTOCOL_SHA,
        "best_weights_sha256": digest,
        "best_update": 572,
    }
    path.write_text(json.dumps(metrics))
    verified = adapt._verify_checkpoint(
        checkpoint, digest, path, adapt.protocol.sha256_file(path), "DARC", 1
    )
    assert verified == metrics
    metrics["stage"] = "diagnostic"
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="matching completed"):
        adapt._verify_checkpoint(
            checkpoint, digest, path, adapt.protocol.sha256_file(path), "DARC", 1
        )
