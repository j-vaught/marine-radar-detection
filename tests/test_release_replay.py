"""Portable replay integrity, path containment and comparison checks."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from src.revision import protocol
from src.revision import release_replay as replay


def make_run(tmp_path, **updates):
    checkpoint = tmp_path / "checkpoint.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"checkpoint")
    run = {
        "schema_version": 1,
        "scope": "primary",
        "job_id": "core-faster_rcnn-DAAN-s0",
        "method": "faster_rcnn",
        "source": "DAAN",
        "seed": 0,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": protocol.sha256_file(checkpoint),
        "config": {},
        "initialization": "pretrained",
        "pixel_threshold": None,
        "threshold": 0.5,
        "intensity": "I0",
        "numerical_policy": {},
        **updates,
    }
    path = tmp_path / "run.json"
    protocol.save_json(path, run)
    return path


def make_inputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for image_id in (2, 10, 1):
        name = f"frame-{image_id}.png"
        (images / name).write_bytes(f"image-{image_id}".encode())
        rows.append({"id": image_id, "file_name": name, "width": 1070, "height": 1040})
    coco = {
        "images": rows,
        "categories": [{"id": 1, "name": "boat"}],
        "annotations": [
            {"id": 1, "image_id": 2, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16}
        ],
    }
    annotations = tmp_path / "annotations.json"
    protocol.save_json(annotations, coco)
    manifest = tmp_path / "split.json"
    protocol.save_json(
        manifest,
        {
            "images": [
                {**row, "sha256": protocol.sha256_file(images / row["file_name"])} for row in rows
            ]
        },
    )
    return annotations, images, manifest


def prediction(image_id=1, score=0.75):
    return {"image_id": image_id, "category_id": 1, "bbox": [3.0, 4.0, 5.0, 6.0], "score": score}


def test_checkpoint_hash_and_containment(tmp_path):
    path = make_run(tmp_path)
    run, checkpoint = replay.read_run(path)
    assert checkpoint == (tmp_path / "checkpoint.pt").resolve()
    assert run["scope"] == "primary"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        replay.read_run(path)
    make_run(tmp_path, checkpoint="../checkpoint.pt")
    with pytest.raises(ValueError, match="Unsafe relative path"):
        replay.read_run(path)


@pytest.mark.parametrize("name", ["/tmp/file", "../file", "sub/../file", "sub\\file", "./file"])
def test_rejects_unsafe_relative_paths(tmp_path, name):
    with pytest.raises(ValueError):
        replay._relative_file(tmp_path, name)


def test_rejects_symlink_escape(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (tmp_path / "outside").write_text("private")
    (root / "checkpoint.pt").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="unsafe"):
        replay._relative_file(root, "checkpoint.pt")


@pytest.mark.parametrize(
    "updates",
    [
        {"scope": "darc", "method": "unet", "pixel_threshold": 0.5},
        {"scope": "adaptation", "intensity": "I1"},
        {"method": "unet", "pixel_threshold": None},
        {"threshold": float("inf")},
    ],
)
def test_rejects_incompatible_run_recipes(tmp_path, updates):
    path = make_run(tmp_path)
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, **updates}))
    with pytest.raises(ValueError):
        replay.read_run(path)


def test_input_subset_is_hashed_and_keeps_native_dimensions(tmp_path):
    annotations, images, manifest = make_inputs(tmp_path)
    coco, observed = replay.prepare_inputs(annotations, images, manifest, limit=2)
    assert [row["id"] for row in coco["images"]] == [1, 10]
    assert coco["annotations"] == []
    assert {(row["width"], row["height"]) for row in observed} == {(1070, 1040)}
    (images / "frame-1.png").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Input image SHA-256 mismatch"):
        replay.prepare_inputs(annotations, images, manifest)


def test_input_rejects_path_escape_even_without_manifest(tmp_path):
    annotations, images, _manifest = make_inputs(tmp_path)
    coco = json.loads(annotations.read_text())
    coco["images"][0]["file_name"] = "../annotations.json"
    protocol.save_json(annotations, coco)
    with pytest.raises(ValueError, match="Unsafe relative path"):
        replay.prepare_inputs(annotations, images)


def test_tensor_digest_keeps_saved_dtype_and_prefers_ema(tmp_path):
    from src.revision.darc_adaptation import _state_digest

    state = {"weights": torch.tensor([1.0, 2.0], dtype=torch.float16)}
    path = tmp_path / "state.pt"
    torch.save(state, path)
    assert replay.tensor_digest(path, "unet") == _state_digest(state)
    assert replay.tensor_digest(path, "unet") != _state_digest(
        {"weights": state["weights"].float()}
    )
    model = torch.nn.Linear(2, 1).half()
    ema = torch.nn.Linear(2, 1).half()
    with torch.no_grad():
        ema.weight.fill_(3)
    torch.save({"model": model, "ema": ema}, path)
    assert replay.tensor_digest(path, "yolo") == _state_digest(ema.state_dict())


def test_prediction_parity_reports_drift_counts_and_limit():
    actual = [prediction()]
    reference = [prediction(2), prediction()]
    exact = replay.compare_predictions(actual, reference, [1])
    assert exact["exact"] and exact["within_tolerance"]
    reference[1]["bbox"][0] += 0.001
    drift = replay.compare_predictions(actual, reference, [1], atol=0.002)
    assert not drift["exact"] and drift["within_tolerance"]
    assert drift["max_abs_box_difference"] == pytest.approx(0.001)
    assert not replay.compare_predictions(actual, reference, [1])["within_tolerance"]
    assert not replay.compare_predictions(actual, [], [1])["counts_equal"]


def test_invalid_prediction_is_rejected():
    coco = {"images": [{"id": 1, "width": 10, "height": 10}], "categories": [{"id": 1}]}
    replay.validate_predictions([prediction()], coco)
    out_of_bounds = prediction()
    out_of_bounds["bbox"][2] = 8
    with pytest.raises(ValueError, match="native raster"):
        replay.validate_predictions([out_of_bounds], coco)
    degenerate = prediction()
    degenerate["bbox"][2] = 0
    assert replay.validate_predictions([degenerate], coco)["degenerate_boxes"] == 1
    degenerate["bbox"][2] = -1
    with pytest.raises(ValueError, match="native raster"):
        replay.validate_predictions([degenerate], coco)


def test_replay_verifies_before_predict_and_refuses_overwrite(tmp_path, monkeypatch):
    run_path = make_run(tmp_path, tensor_state_sha256="a" * 64)
    annotations, images, manifest = make_inputs(tmp_path)
    reference = tmp_path / "reference.json"
    protocol.save_json(reference, [prediction()])
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        run=run_path,
        annotations=annotations,
        images=images,
        output=output,
        split_manifest=manifest,
        limit=1,
        reference_predictions=reference,
        atol=0.0,
        rtol=0.0,
        device="cpu",
    )
    monkeypatch.setattr(replay, "tensor_digest", lambda *_: "a" * 64)
    monkeypatch.setattr(replay, "configure_runtime", lambda *_: ("cpu", {"device": "cpu"}))
    calls = []

    def predict(*args):
        calls.append(args)
        assert Path(args[-1]).parent == tmp_path
        return [prediction()], {"precision": "float32"}

    monkeypatch.setattr(replay, "predict", predict)
    result = replay.run(args)
    assert result["parity"]["exact"]
    assert result["tensor_state_verified"]
    assert result["input_images_verified_against_split"]
    assert result["operating_predictions"] == result["predictions"]
    assert not list(tmp_path.glob(".release-replay-*"))
    with pytest.raises(FileExistsError):
        replay.run(args)
    assert len(calls) == 1
    args.output = tmp_path / "mismatch.json"
    monkeypatch.setattr(replay, "tensor_digest", lambda *_: "b" * 64)
    with pytest.raises(ValueError, match="tensor-state"):
        replay.run(args)
    assert len(calls) == 1 and not args.output.exists()


def test_primary_reuses_staging_and_selected_predictor(tmp_path, monkeypatch):
    from src.revision import inference_timing, ultralytics_models

    calls = []
    monkeypatch.setattr(replay, "_verify_bootstrap", lambda _: None)
    monkeypatch.setattr(
        ultralytics_models, "stage_split", lambda *args: calls.append(args) or tmp_path
    )

    def factory(method, selected, directory, coco, device):
        assert method == "unet" and selected.pixel_threshold == 0.7
        assert directory == tmp_path
        return lambda: [prediction()], {"precision": "CUDA autocast"}

    monkeypatch.setattr(inference_timing, "make_predictor", factory)
    run = {
        "scope": "primary",
        "method": "unet",
        "config": {},
        "initialization": "scratch",
        "pixel_threshold": 0.7,
    }
    predictions, settings = replay.predict(
        run, tmp_path / "checkpoint.pt", {}, tmp_path, [], torch.device("cpu"), tmp_path
    )
    assert predictions == [prediction()] and calls
    assert settings["precision"] == "float32 (CPU, no CUDA autocast)"


def test_darc_reuses_intensity_loader_without_evaluation_jitter(tmp_path, monkeypatch):
    from src.faster_rcnn import model as faster
    from src.revision import darc_data, darc_train

    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    calls = []
    monkeypatch.setattr(replay, "_verify_bootstrap", lambda _: None)
    monkeypatch.setattr(faster, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(darc_data, "DARCData", lambda *args, **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(darc_train, "_predictions", lambda *args: ([prediction()], []))
    run = {
        "scope": "darc",
        "method": "faster_rcnn",
        "config": {},
        "intensity": "I1",
        "source": "DARC",
        "seed": 2,
    }
    predictions, settings = replay.predict(
        run, checkpoint, {}, tmp_path, [], torch.device("cpu"), tmp_path
    )
    assert predictions == [prediction()]
    assert calls[0]["intensity"] == "I1" and calls[0]["scale"] == "S0"
    assert settings["precision"] == "float32"


def test_runtime_uses_public_numerical_policy_keys(monkeypatch):
    import cv2

    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _: None)
    algorithms = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", algorithms.append)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    for backend, attribute in [
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    ]:
        monkeypatch.setattr(backend, attribute, False)
    run = {
        "scope": "darc",
        "numerical_policy": {
            "deterministic_algorithms": True,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
    }
    device, observed = replay.configure_runtime(run, "cpu")
    assert device.type == "cpu" and algorithms == [True]
    assert observed["deterministic_algorithms"] and observed["cudnn_deterministic"]
    assert observed["matmul_allow_tf32"] is False
    run["numerical_policy"]["matmul_allow_tf32"] = True
    with pytest.raises(ValueError, match="numerical policy"):
        replay.configure_runtime(run, "cpu")
