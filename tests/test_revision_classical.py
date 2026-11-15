import importlib
import json
import sys
import types
from typing import Any


def test_selection_uses_train_then_validation_and_freezes_tests(tmp_path, monkeypatch):
    protocol: Any = types.ModuleType("src.revision.protocol")
    datasets = {
        "train": {"name": "train"},
        "validation": {"name": "validation"},
        "tests": {"A": {"name": "A"}, "B": {"name": "B"}},
        "image_dir": tmp_path,
    }
    protocol.prepare_job = lambda job: datasets
    protocol.save_json = lambda p, d: p.write_text(json.dumps(d))
    protocol.set_seed = lambda seed: None
    protocol.sha256_file = lambda p: "sha"
    evaluation: Any = types.ModuleType("src.revision.evaluation")
    evaluation.evaluate_coco = lambda coco, preds: {"ap50": preds[0]["score"]}
    monkeypatch.setitem(sys.modules, "src.revision.protocol", protocol)
    monkeypatch.setitem(sys.modules, "src.revision.evaluation", evaluation)
    sys.modules.pop("src.revision.classical", None)
    runner = importlib.import_module("src.revision.classical")
    calls = []

    def predict(method, config, ann, image_dir, output, device, binary):
        split = json.loads(ann.read_text())["name"]
        calls.append((split, config["id"]))
        score = config["id"] if split == "train" else -config["id"]
        pred = [{"score": score}]
        output.write_text(json.dumps(pred))
        return pred

    monkeypatch.setattr(runner, "_native_predictions", predict)
    result = runner.run_job(
        {
            "seed": 1,
            "method": "otsu",
            "job_id": "job",
            "source": "A",
            "out_dir": tmp_path,
            "manifest": "manifest.json",
            "device": "cpu",
            "eval_targets": ["A", "B"],
            "config_overrides": {"candidates": [{"id": n} for n in range(7)]},
        }
    )
    assert [n for split, n in calls if split == "validation"] == [6, 5, 4, 3, 2]
    assert [(split, n) for split, n in calls if split in ("A", "B")] == [("A", 2), ("B", 2)]
    assert result["selected_config"]["config"] == {"id": 2}
    assert set(result["tests"]) == {"A", "B"}


def test_cfar_cpu_integral_box_sum_matches_pool():
    import torch
    import torch.nn.functional as F

    from src.cfar.gpu.os_cfar import _box_sum

    torch.set_num_threads(2)
    x = torch.randint(0, 2, (2, 3, 128, 128), generator=torch.Generator().manual_seed(17)).float()
    for k in (21, 31, 41, 51, 61, 81):
        padded = F.pad(x, (k // 2,) * 4, mode="reflect")
        expected = F.avg_pool2d(padded, k, 1) * (k * k)
        assert torch.equal(_box_sum(x, k), expected)


def test_cfar_group_predictions_match_serial(monkeypatch):
    import numpy as np
    import torch

    protocol: Any = types.ModuleType("src.revision.protocol")
    protocol.prepare_job = protocol.save_json = protocol.set_seed = protocol.sha256_file = (
        lambda *a: None
    )
    evaluation: Any = types.ModuleType("src.revision.evaluation")
    evaluation.evaluate_coco = lambda *a: None
    legacy: Any = types.ModuleType("src.cfar.gpu.eval_coco")
    legacy.build_gt_index = legacy.evaluate = legacy.fast_ap50 = lambda *a: None
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    monkeypatch.setitem(sys.modules, evaluation.__name__, evaluation)
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    sys.modules.pop("src.revision.classical", None)
    runner = importlib.import_module("src.revision.classical")
    from src.cfar.experiments.grid import BackgroundCache
    from src.cfar.gpu.data import Corpus

    torch.set_num_threads(2)
    images = (
        torch.randint(0, 256, (2, 1, 16, 16), generator=torch.Generator().manual_seed(5)).float()
        / 255
    )
    corpus = Corpus(
        images, torch.ones(1, 1, 16, 16), np.array([1, 2]), ["a", "b"], np.array([0, 0]), "", {}
    )
    for method in ("CA", "OS"):
        configs = [
            {"method": method, "guard": 1, "ref": 2, "pfa": p, "min_area": a, "os_k": k}
            for p in (0.01, 0.001)
            for a in (1, 3)
            for k in ([None] if method == "CA" else [0.5, 0.75])
        ]
        caches = {"fixture": (corpus, BackgroundCache(corpus))}
        grouped = runner._cfar_group_predictions(configs, "fixture", "", "cpu", caches)
        for config in configs:
            serial = runner._cfar_predictions("unused", config, "fixture", "", "cpu", caches)
            assert grouped[runner._cfar_key(config)].to_coco() == serial
