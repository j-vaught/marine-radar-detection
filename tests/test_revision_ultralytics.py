from src.revision.ultralytics_models import _model_spec, coco_yolo_lines


def test_empty_and_box_roundtrip():
    image = {"id": 4, "width": 1062, "height": 1030}
    coco = {"images": [image], "categories": [{"id": 7, "name": "target"}], "annotations": []}
    assert coco_yolo_lines(coco, image) == []
    box = [123.25, 456.75, 87.5, 52.25]
    coco["annotations"].append({"image_id": 4, "category_id": 7, "bbox": box})
    cls, cx, cy, width, height = map(float, coco_yolo_lines(coco, image)[0].split())
    assert cls == 0
    actual = [(cx - width / 2) * 1062, (cy - height / 2) * 1030, width * 1062, height * 1030]
    assert all(abs(a - b) < 1e-10 for a, b in zip(actual, box))


def test_scratch_uses_yaml_even_when_pretrained_is_configured():
    for method, variant in [("yolo", "yolo26m"), ("rtdetr", "rtdetr-l")]:
        cfg = {"model": {"variant": variant, "pretrained": True}}
        assert _model_spec(method, "scratch", cfg) == variant + ".yaml"
        assert _model_spec(method, "pretrained", cfg) == variant + ".pt"


def test_letterbox_inverse_preserves_edge_before_clipping():
    from types import SimpleNamespace

    import numpy as np
    import torch

    from src.revision.ultralytics_models import _coordinate_predictor, _letterbox_ratio_pad

    gain = 1024 / 1050
    assert _letterbox_ratio_pad((1024, 1024), (1024, 1050)) == ((gain, gain), (0, 12))
    predictor = _coordinate_predictor().__new__(_coordinate_predictor())
    predictor.model = SimpleNamespace(names={0: "vessel"})
    # Top edge is just inside content, but the incorrect 13-pixel pad clips it to zero.
    pred = torch.tensor([[100.0, 12.25, 200.0, 20.0, 0.8, 0.0]])
    result = predictor.construct_result(
        pred, torch.zeros((1, 3, 1024, 1024)), np.zeros((1024, 1050, 3), dtype=np.uint8), "edge.png"
    )
    assert abs(result.boxes.xyxy[0, 1].item() - 0.25 / gain) < 1e-5


def test_repair_validates_parent_and_never_trains(tmp_path, monkeypatch):
    import json

    import ultralytics

    from src.revision import protocol, yolo_repair

    parent, output = tmp_path / "parent", tmp_path / "new"
    (parent / "fit/weights").mkdir(parents=True)
    (parent / "resume").mkdir()
    for path in (
        "fit/weights/best.pt",
        "fit/weights/last.pt",
        "initial_state.pt",
        "fit/results.csv",
    ):
        (parent / path).write_bytes(path.encode())
    manifest, ann = tmp_path / "manifest.json", tmp_path / "ann.json"
    manifest.write_text("{}")
    ann.write_text("{}")
    gt = {
        "images": [{"id": 1, "width": 1050, "height": 1024}],
        "categories": [{"id": 1, "name": "vessel"}],
        "annotations": [],
    }
    protocol.save_json(parent / "validation_gt.json", gt)
    protocol.save_json(parent / "DAAN_gt.json", gt)
    job = {
        "job_id": "core-yolo-DAAN-s0",
        "method": "yolo",
        "source": "DAAN",
        "seed": 0,
        "condition": "buffered",
        "stage": "core",
        "initialization": "pretrained",
        "eval_targets": ["DAAN"],
        "train_limit": None,
        "overfit": False,
        "config_overrides": {},
        "manifest": str(manifest),
        "ann": str(ann),
        "protocol_sha256": protocol.sha256_file(manifest),
        "device": "cpu",
        "out_dir": str(output),
        "evaluation_from": str(parent),
        "code_commit": "new",
        "attempt_nonce": "new-nonce",
    }
    original = {**job, "out_dir": str(parent), "attempt_nonce": "old-nonce"}
    protocol.save_json(parent / "job.json", original)
    identity = {
        "job_id": job["job_id"],
        "attempt": 1,
        "nonce": "old-nonce",
        "pid": 100,
        "start_ticks": "1",
        "boot_id": "b",
    }
    protocol.save_json(parent / "receipt.json", identity)
    metrics = {
        k: job[k]
        for k in ("job_id", "method", "source", "seed", "condition", "stage", "initialization")
    }
    metrics.update(
        {
            "manifest_sha256": protocol.sha256_file(manifest),
            "annotation_sha256": protocol.sha256_file(ann),
            "best_checkpoint": str(parent / "fit/weights/best.pt"),
            "best_sha256": protocol.sha256_file(parent / "fit/weights/best.pt"),
            "last_sha256": protocol.sha256_file(parent / "fit/weights/last.pt"),
            "initial_state_sha256": protocol.sha256_file(parent / "initial_state.pt"),
            "tests": {
                "DAAN": {"ground_truth_sha256": protocol.sha256_file(parent / "DAAN_gt.json")}
            },
            "config": {
                "training": {"imgsz": 1024},
                "postprocessing": {
                    "score_floor": 0.001,
                    "score_threshold": 0.05,
                    "score_threshold_grid": [0.01, 0.05],
                },
            },
        }
    )
    protocol.save_json(parent / "metrics.json", metrics)
    completion = {
        **identity,
        "state": "succeeded",
        "exit_code": 0,
        "manifest_sha256": protocol.sha256_file(manifest),
        "metrics_sha256": protocol.sha256_file(parent / "metrics.json"),
        "runtime_provenance": {"gpu": "original"},
    }
    protocol.save_json(parent / "completion.json", completion)
    monkeypatch.setattr(
        protocol,
        "prepare_job",
        lambda j: {"train": gt, "validation": gt, "tests": {"DAAN": gt}, "image_dir": tmp_path},
    )
    monkeypatch.setattr(yolo_repair, "stage_split", lambda *args: tmp_path)
    monkeypatch.setattr(yolo_repair, "_predict", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        yolo_repair.evaluation, "evaluate_coco", lambda *args: {"ap": 0.0, "ap50": 0.0}
    )

    class Model:
        def __init__(self, path):
            self.path = path

        def train(self, *a, **k):
            raise AssertionError("training must never run")

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    result = yolo_repair.evaluate_from_parent(job)
    assert result["training_parent"]["runtime_provenance"] == {"gpu": "original"}
    assert result["best_sha256"] == protocol.sha256_file(output / "fit/weights/best.pt")
    assert result["tests"]["DAAN"]["checkpoint_sha256"] == result["best_sha256"]
    assert (
        json.loads((output / "metrics.json").read_text())["evaluation_recipe"]
        == "yolo-exact-letterbox-v1"
    )
    (parent / "fit/weights/best.pt").write_text("tampered")
    import pytest

    with pytest.raises(ValueError, match="checkpoint digest"):
        yolo_repair.verify_parent(job)
