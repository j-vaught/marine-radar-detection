import importlib
import json
import sys
import types


def test_selection_uses_train_then_validation_and_freezes_tests(tmp_path, monkeypatch):
    protocol = types.ModuleType("src.revision.protocol")
    datasets = {"train": {"name": "train"}, "validation": {"name": "validation"},
                "tests": {"A": {"name": "A"}, "B": {"name": "B"}}, "image_dir": tmp_path}
    protocol.prepare_job = lambda job: datasets
    protocol.save_json = lambda p, d: p.write_text(json.dumps(d))
    protocol.set_seed = lambda seed: None
    protocol.sha256_file = lambda p: "sha"
    evaluation = types.ModuleType("src.revision.evaluation")
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
    result = runner.run_job({"seed": 1, "method": "otsu", "job_id": "job", "source": "A",
                             "out_dir": tmp_path, "manifest": "manifest.json", "device": "cpu",
                             "eval_targets": ["A", "B"],
                             "config_overrides": {"candidates": [{"id": n} for n in range(7)]}})
    assert [n for split, n in calls if split == "validation"] == [6, 5, 4, 3, 2]
    assert [(split, n) for split, n in calls if split in ("A", "B")] == [("A", 2), ("B", 2)]
    assert result["selected_config"]["config"] == {"id": 2}
    assert set(result["tests"]) == {"A", "B"}
