"""Behavioral checks for paired metadata auditing without target score access."""

from copy import deepcopy

from src.revision.audit_pairs import compare_pair


def pair():
    states, records = {}, {}
    for stage, condition in (("core", "buffered"), ("adjacency", "adjacency")):
        name = f"{stage}-faster_rcnn-DAAN-s0"
        job = {
            "job_id": name,
            "method": "faster_rcnn",
            "source": "DAAN",
            "seed": 0,
            "stage": stage,
            "condition": condition,
            "initialization": "pretrained",
            "epochs": 60,
            "patience": 8,
            "config_overrides": {},
            "eval_targets": ["DAAN", "DARC", "MANV"],
            "device": "cuda:0",
        }
        meta = {
            **{k: job[k] for k in ("job_id", "method", "source", "seed", "stage", "condition")},
            "manifest_sha256": "manifest",
            "annotation_sha256": "ann",
            "validation_gt_sha256": "validation",
            "initialized_state_sha256": "initial",
            "best_weights_sha256": "best",
            "best_epoch": 2,
        }
        states[name] = {"state": "succeeded", "attempt": 1}
        records[name] = {
            "attempt": 1,
            "job": job,
            "metadata": meta,
            "completion": {
                "state": "succeeded",
                "metrics_sha256": "sha",
                "code_commit": "same",
                "runtime_provenance": {"gpu": {"model": "Ada", "uuid": stage}},
            },
            "metrics_sha256_actual": "sha",
            "resolved_config": {"config": {"training": {"batch_size": 2, "optimizer": "SGD"}}},
            "history_rows": 5,
            "test_digests": {
                target: {"checkpoint_sha256": "best", "gt_sha256": target}
                for target in ("DAAN", "DARC", "MANV")
            },
        }
    return states, records


def test_hardware_difference_is_descriptive():
    states, records = pair()
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records)
    assert result["status"] == "matched"
    assert result["differences"]
    assert not result["critical"]


def test_batch_and_initialization_mismatch_are_critical():
    states, records = pair()
    adjacency = records["adjacency-faster_rcnn-DAAN-s0"]
    adjacency["resolved_config"]["config"]["training"]["batch_size"] = 4
    adjacency["metadata"]["initialized_state_sha256"] = "different"
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records)
    assert result["status"] == "critical"
    assert any("training/config mismatch" in x for x in result["critical"])
    assert any("initialized model" in x for x in result["critical"])


def test_incomplete_pair_and_missing_archive():
    states, records = pair()
    records.pop("adjacency-faster_rcnn-DAAN-s0")
    states["adjacency-faster_rcnn-DAAN-s0"] = {"state": "running"}
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records)
    assert result["status"] == "incomplete"
    assert result["runs"]["adjacency-faster_rcnn-DAAN-s0"]["state"] == "running"


def test_checkpoint_identity_failure():
    states, records = pair()
    records = deepcopy(records)
    records["core-faster_rcnn-DAAN-s0"]["test_digests"]["DARC"]["checkpoint_sha256"] = "other"
    assert compare_pair("faster_rcnn", "DAAN", 0, states, records)["status"] == "critical"
