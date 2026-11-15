import json

import pytest

from src.revision.evaluation import evaluate_coco
from src.revision.protocol import build_manifest, prepare_job, save_json


def dataset(tmp_path, n=100):
    images = []
    for i in range(n):
        name = f"SITE_{i:03}.png"
        (tmp_path / name).write_bytes(str(i).encode())
        images.append(
            {
                "id": i + 1,
                "file_name": name,
                "frame": i,
                "dataset": "SITE",
                "width": 50,
                "height": 50,
            }
        )
    ann = {
        "images": images,
        "categories": [{"id": 1, "name": "vessel"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
            for i in range(n)
        ],
    }
    path = tmp_path / "all.json"
    path.write_text(json.dumps(ann))
    return path, ann


def test_partition_and_preparation(tmp_path):
    ann_path, ann = dataset(tmp_path)
    manifest = build_manifest(ann_path, tmp_path)
    site = manifest["sites"]["SITE"]
    assert site["counts"] == {
        "validation": 15,
        "validation_buffer": 5,
        "train": 60,
        "test_buffer": 5,
        "test": 15,
    }
    parts = site["partitions"]
    assert len(set().union(*map(set, parts.values()))) == 100
    for seed in range(5):
        train = set(site["adjacency"][str(seed)])
        assert len(train) == 60
        assert train & set(parts["test_buffer"]) == set(parts["test_buffer"])
        assert not train & (set(parts["test"]) | set(parts["validation"]))
        assert len(set(parts["train"]) - train) == 5
    path = tmp_path / "manifest.json"
    save_json(path, manifest)
    job = {
        "ann": str(ann_path),
        "images": str(tmp_path),
        "manifest": str(path),
        "source": "SITE",
        "condition": "buffered",
        "seed": 0,
        "stage": "diagnostic",
        "train_limit": 3,
        "overfit": True,
        "eval_targets": [],
    }
    prepared = prepare_job(job)
    assert [x["id"] for x in prepared["train"]["images"]] == parts["train"][:3]
    assert prepared["train"] == prepared["validation"]
    assert prepared["annotations"] == ann
    assert prepared["tests"] == {}
    job.update(stage="core", overfit=False, train_limit=None, eval_targets=["SITE"])
    assert len(prepare_job(job)["tests"]["SITE"]["images"]) == 15
    job["overfit"] = True
    with pytest.raises(ValueError):
        prepare_job(job)
    ann_path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        prepare_job(job)


def test_coco_empty_duplicates_ties_and_max_dets(tmp_path):
    _, gt = dataset(tmp_path, n=2)
    assert evaluate_coco(gt, [])["ap"] == 0

    def pred(image, box, score):
        return {"image_id": image, "category_id": 1, "bbox": box, "score": score}

    correct = [pred(i, [0, 0, 10, 10], 0.9) for i in (1, 2)]
    assert evaluate_coco(gt, correct)["ap50"] == pytest.approx(1)
    duplicate = [correct[0], correct[0], correct[1]]
    assert evaluate_coco(gt, duplicate)["ap50"] < 1
    crowded = [pred(1, [30, 30, 5, 5], 1.0) for _ in range(100)] + [correct[0], correct[1]]
    result = evaluate_coco(gt, crowded)
    assert result["ar100"] == pytest.approx(0.5)
    assert result["predictions"] == 102 and result["preds_per_image"] == 51
