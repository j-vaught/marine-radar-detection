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


def flagged_pair():
    states, records = pair()
    left = "core-faster_rcnn-DAAN-s0"
    right = "adjacency-faster_rcnn-DAAN-s0"
    records[right]["job"]["match_core_gpu_model"] = True
    records[right]["job"]["paired_environment_sha256"] = "attest"
    for name, host in ((left, "host-a"), (right, "host-b")):
        states[name]["host"] = host
        completion = records[name]["completion"]
        completion["python"] = "3.11.15 (build info)"
        runtime = completion["runtime_provenance"]
        runtime["thread_environment"] = {
            key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
        }
        runtime["torch"] = {
            "version": "2.5.1",
            "cuda_version": "12.1",
            "cudnn_version": 90100,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": True,
            "matmul_allow_tf32": False,
            "deterministic_algorithms": False,
        }
    hosts = {
        "host-a": {"paired_environment_sha256": "attest"},
        "host-b": {"paired_environment_sha256": "attest"},
    }
    return states, records, hosts


def test_flagged_policy_model_and_precision_mismatch_are_critical():
    states, records, hosts = flagged_pair()
    right = records["adjacency-faster_rcnn-DAAN-s0"]["completion"]["runtime_provenance"]
    right["gpu"]["model"] = "A5000"
    right["torch"]["matmul_allow_tf32"] = True
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)
    assert result["status"] == "critical"
    assert "hardware policy mismatch: GPU model" in result["critical"]
    assert "hardware policy mismatch: torch.matmul_allow_tf32" in result["critical"]


def test_flagged_equal_model_different_uuid_and_missing_cudnn():
    states, records, hosts = flagged_pair()
    assert compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)["status"] == "matched"
    del records["adjacency-faster_rcnn-DAAN-s0"]["completion"]["runtime_provenance"]["torch"][
        "cudnn_version"
    ]
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)
    assert result["status"] == "unverified"
    assert "hardware policy evidence missing: torch.cudnn_version" in result["unverified"]
    assert result["differences"]


def test_flagged_cross_host_attestation_mismatch():
    states, records, hosts = flagged_pair()
    hosts["host-b"]["paired_environment_sha256"] = "different"
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)
    assert result["status"] == "critical"
    assert "cross-host package attestation mismatch" in result["critical"]


def test_repair_uses_training_parent_precision_and_python():
    states, records, hosts = flagged_pair()
    name = "core-faster_rcnn-DAAN-s0"
    record = records[name]
    original = deepcopy(record["completion"])
    original.update(job_id=name, attempt=1, nonce="old")
    record["metadata"]["training_parent"] = {
        "checkpoint_sha256": "best",
        "code_commit": "same",
        "runtime_provenance": deepcopy(original["runtime_provenance"]),
    }
    record["parent_completion"] = original
    record["parent_completion_verified"] = True
    record["completion"]["runtime_provenance"]["torch"]["matmul_allow_tf32"] = True
    record["completion"]["python"] = "3.12.0 repair"
    states[name]["training_state"] = {"host": "host-a", "result": original}
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)
    assert result["status"] == "matched"
    assert result["hardware"][name]["torch"]["matmul_allow_tf32"] is False
    assert result["hardware"][name]["python"].startswith("3.11.15")


def test_flagged_thread_and_amp_mismatch_are_critical():
    states, records, hosts = flagged_pair()
    left = records["core-faster_rcnn-DAAN-s0"]
    right = records["adjacency-faster_rcnn-DAAN-s0"]
    left["resolved_config"]["config"]["training"]["amp"] = True
    right["resolved_config"]["config"]["training"]["amp"] = False
    right["completion"]["runtime_provenance"]["thread_environment"]["OMP_NUM_THREADS"] = "8"
    result = compare_pair("faster_rcnn", "DAAN", 0, states, records, hosts)
    assert result["status"] == "critical"
    assert "hardware policy mismatch: OMP_NUM_THREADS" in result["critical"]
    assert "effective training/config mismatch" in result["critical"]
