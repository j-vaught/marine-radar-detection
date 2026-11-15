import json
from pathlib import Path
from unittest import mock

import pytest

from src.revision import wavelet_study as study
from src.revision.protocol import _subset, save_json, sha256_file


def test_neighbors_are_prescribed_one_axis_and_deduplicated():
    parent = {
        "level": 1,
        "threshold_sigma": 9.0,
        "min_area": 30,
        "max_area": 3000,
        "dilation": 0,
        "box_padding": 1,
    }
    grid = study.neighbors(parent)
    assert grid[0] == parent
    assert len(grid) == 5
    assert all(sum(c[k] != parent[k] for k in parent) == 1 for c in grid[1:])
    assert {c["threshold_sigma"] for c in grid} == {9.0, 10.5}


def test_selection_ties_and_proposal_diagnostics():
    rows = [
        {"index": 0, "metrics": {"ap50": 0.8, "ap": 0.5}},
        {"index": 1, "metrics": {"ap50": 0.8, "ap": 0.6}},
        {"index": 2, "metrics": {"ap50": 0.8, "ap": 0.6}},
    ]
    assert study.choose(rows)["index"] == 1
    with pytest.raises(ValueError):
        study.choose([{"index": 0, "metrics": {"ap50": float("nan"), "ap": 0.1}}])
    gt = {
        "images": [{"id": 1}, {"id": 2}],
        "categories": [{"id": 1}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
    }
    preds = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 1.0},
        {"image_id": 1, "bbox": [0, 0, 5, 5], "score": 1.0},
    ]
    result = study.proposal_summary(gt, preds)
    assert result["recall_at_iou"] == {"0.25": 0.5, "0.5": 0.5, "0.75": 0.5}
    assert result["mean_best_iou"] == 0.5
    assert result["largest_score_tie"] == 2


def make_parent(tmp_path):
    parent, images = tmp_path / "parent", tmp_path / "images"
    parent.mkdir()
    images.mkdir()
    ims, anns, records, sites = [], [], [], {}
    for i, site in enumerate(("a", "b", "c")):
        ids = [2 * i + 1, 2 * i + 2]
        sites[site] = {"partitions": {"validation": [ids[0]], "test": [ids[1]]}}
        for image_id in ids:
            name = f"{image_id}.png"
            (images / name).write_bytes(name.encode())
            ims.append(
                {"id": image_id, "file_name": name, "dataset": site, "width": 10, "height": 10}
            )
            anns.append(
                {"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [0, 0, 5, 5]}
            )
            records.append(
                {
                    "id": image_id,
                    "file_name": name,
                    "site": site,
                    "sha256": sha256_file(images / name),
                }
            )
    coco = {"images": ims, "annotations": anns, "categories": [{"id": 1}]}
    ann = tmp_path / "ann.json"
    save_json(ann, coco)
    manifest = tmp_path / "manifest.json"
    save_json(manifest, {"annotation_sha256": sha256_file(ann), "sites": sites, "images": records})
    config = {
        "level": 2,
        "threshold_sigma": 6.0,
        "min_area": 30,
        "max_area": 3000,
        "dilation": 0,
        "box_padding": 0,
    }
    selection = {"source": "a", "config": config}
    save_json(parent / "selected_config.json", selection)
    targets = {}
    for name, ids in [("validation", [1]), ("test_a", [2]), ("test_b", [4]), ("test_c", [6])]:
        p = parent / f"{name}_annotations.json"
        save_json(p, _subset(coco, ids))
        targets[name] = {"annotations_sha256": sha256_file(p)}
        if name.startswith("test_"):
            save_json(
                parent / f"{name}_predictions.json",
                [{"image_id": ids[0], "category_id": 1, "bbox": [0, 0, 5, 5], "score": 1.0}],
            )
    metric = {"ap": 1.0, "ap50": 1.0, "predictions": 1}
    metrics = {
        "job_id": "j",
        "method": "wavelet",
        "annotation_sha256": sha256_file(ann),
        "selected_config": selection,
        "tests": {s: metric for s in sites},
        "paths": {"manifest_sha256": sha256_file(manifest), "subsets": targets},
    }
    save_json(parent / "metrics.json", metrics)
    receipt = {
        "job_id": "j",
        "attempt": 1,
        "nonce": "n",
        "pid": 1,
        "start_ticks": "1",
        "boot_id": "b",
    }
    save_json(parent / "receipt.json", receipt)
    save_json(
        parent / "completion.json",
        {
            **receipt,
            "state": "succeeded",
            "exit_code": 0,
            "metrics_sha256": sha256_file(parent / "metrics.json"),
            "manifest_sha256": sha256_file(manifest),
        },
    )
    save_json(
        parent / "job.json",
        {
            "job_id": "j",
            "attempt_nonce": "n",
            "method": "wavelet",
            "stage": "core",
            "condition": "buffered",
            "source": "a",
            "protocol_sha256": sha256_file(manifest),
            "eval_targets": list(sites),
        },
    )
    binary = tmp_path / "native"
    binary.write_bytes(b"binary")
    return parent, ann, images, manifest, binary


def test_run_freezes_validation_selection_before_holdout_and_checks_parent(tmp_path):
    parent, ann, images, manifest, binary = make_parent(tmp_path)
    out = tmp_path / "out"

    def predict(method, cfg, ann_path, image_dir, path, device, binary_path):
        gt = json.loads(Path(ann_path).read_text())
        if Path(ann_path).name.startswith("test_"):
            assert (out / "selection.json").is_file()
        predictions = [
            {
                "image_id": gt["images"][0]["id"],
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "score": 1.0,
            }
        ]
        save_json(path, predictions)
        return predictions

    def evaluate(gt, preds):
        return {"ap": 1.0, "ap50": 1.0, "predictions": len(preds)}

    original_read = study._read

    def checked_read(path):
        path = Path(path)
        if path == ann or path.name.startswith("test_") or path.name == "metrics.json":
            assert (out / "selection.json").is_file()
        return original_read(path)

    with (
        mock.patch.object(study, "_read", checked_read),
        mock.patch.object(study, "_native_predictions", predict),
        mock.patch.object(study, "evaluate_coco", evaluate),
    ):
        result = study.run(parent, ann, images, manifest, binary, out, 2)
    assert len(json.loads((out / "candidates.json").read_text())) == 8
    assert json.loads((out / "selection.json").read_text())["winner"]["index"] == 0
    assert set(result["tests"]) == {"a", "b", "c"}
    assert result["selection_sha256"] == sha256_file(out / "selection.json")
    (parent / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="metrics hash"):
        study.validate_parent(parent, ann, manifest, images)


def test_real_coco_evaluator_and_geometry():
    pytest.importorskip("pycocotools")
    from src.revision.evaluation import evaluate_coco

    coco = {
        "images": [{"id": 1, "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "target"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }
    pred = [{"image_id": 1, "category_id": 1, "bbox": [4, 4, 10, 10], "score": 0.75}]
    assert evaluate_coco(coco, pred)["ap50"] == pytest.approx(1.0)
    assert study.proposal_summary(coco, pred)["recall_at_iou"]["0.75"] == 1
