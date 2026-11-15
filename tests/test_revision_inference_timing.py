import numpy as np
import pytest
from PIL import Image

from src.revision import inference_timing as timing
from src.revision import protocol


def test_stage_keeps_first_32_masked_shared_pngs(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    # Pixels include the mask center and the left and upper excluded margins.
    Image.fromarray(np.full((516, 532), 19, np.uint8)).save(originals / "common.png")
    digest = protocol.sha256_file(originals / "common.png")
    ids = list(range(1, 35))
    metadata = [
        {"id": image_id, "file_name": "common.png", "width": 532, "height": 516} for image_id in ids
    ]
    annotation = tmp_path / "annotations.json"
    protocol.save_json(annotation, {"images": metadata})
    manifest = {
        "annotation_sha256": protocol.sha256_file(annotation),
        "sites": {"DAAN": {"partitions": {"validation": ids[:-1], "test": ids[-1:]}}},
        "images": [
            {
                "id": image_id,
                "site": "DAAN",
                "frame": image_id * 2,
                "file_name": "common.png",
                "sha256": digest,
            }
            for image_id in ids
        ],
    }
    directory, coco, frames = timing.stage_sample(
        manifest, annotation, originals, tmp_path / "stage"
    )
    assert [row["id"] for row in frames] == ids[:32]
    assert [row["id"] for row in coco["images"]] == sorted(ids[:32], key=lambda x: f"{x}.png")
    assert len(list(directory.glob("*.png"))) == 32
    assert coco["annotations"] == []
    mask = np.asarray(Image.open(directory / "1.png"))
    assert (mask[0, 0] == 0).all()
    assert (mask[515, 531] == 19).all()
    assert frames[0]["staged_png_sha256"] == protocol.sha256_file(directory / "1.png")
    manifest["images"][0]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="Source PNG digest"):
        timing.stage_sample(manifest, annotation, originals, tmp_path / "invalid")


def test_measure_pass_brackets_model_with_device_sync_and_counts(monkeypatch):
    calls = []
    ticks = iter([4.0, 4.25])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))

    def predict():
        calls.append("predict")
        return [{"image_id": 1, "category_id": 1, "bbox": [1, 2, 5, 6], "score": 0.5}]

    record = timing.measure_pass(predict, lambda: calls.append("sync"), [1, 2])
    assert calls == ["sync", "predict", "sync"]
    assert record == {
        "seconds": 0.25,
        "ms_per_frame": 125.0,
        "fps": 8.0,
        "boxes": 1,
        "boxes_per_image": [1, 0],
        "degenerate_boxes": 0,
    }
    with pytest.raises(ValueError, match="Invalid raw prediction"):
        timing.check_predictions(
            [{"image_id": 1, "category_id": 1, "bbox": [float("nan"), 2, 5, 6], "score": 0.5}],
            [1, 2],
        )


def test_selected_yolo_requires_corrected_attempt_same_weights(tmp_path):
    job = "core-yolo-DAAN-s0"
    run = tmp_path / job / "2"
    weights = run / "fit/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"frozen")
    manifest_hash = "manifest-digest"
    digest = protocol.sha256_file(weights)
    metrics = {
        "job_id": job,
        "method": "yolo",
        "source": "DAAN",
        "seed": 0,
        "stage": "core",
        "condition": "buffered",
        "initialization": "pretrained",
        "config": {},
        "manifest_sha256": manifest_hash,
        "best_sha256": digest,
        "evaluation_recipe_version": "yolo-exact-letterbox-v1",
        "training_parent": {"checkpoint_sha256": digest},
    }
    completion = {
        "job_id": job,
        "state": "succeeded",
        "attempt": 2,
        "code_commit": "commit",
        "manifest_sha256": manifest_hash,
    }

    def save():
        protocol.save_json(run / "metrics.json", metrics)
        completion["metrics_sha256"] = protocol.sha256_file(run / "metrics.json")
        protocol.save_json(run / "completion.json", completion)

    save()
    assert timing.read_run(tmp_path, "yolo", manifest_hash).checkpoint == weights
    metrics["training_parent"]["checkpoint_sha256"] = "wrong"
    save()
    with pytest.raises(ValueError, match="original training checkpoint"):
        timing.read_run(tmp_path, "yolo", manifest_hash)
