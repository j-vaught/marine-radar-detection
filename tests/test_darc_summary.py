import hashlib
import itertools

import pytest

from src.revision import darc_summary as analysis
from src.revision import protocol


def _manifest():
    rows = []
    for source, seed, arm in itertools.product(analysis.SITES, analysis.SEEDS, analysis.ARMS):
        base = {
            "source": source,
            "seed": seed,
            "scale": arm[:2],
            "intensity": arm[2:],
            "method": "faster_rcnn",
            "condition": "buffered",
            "initial_tensor_sha256": "a" * 64,
            "initial_state_sha256": "b" * 64,
            "size_control": False,
        }
        rows.append(
            {
                **base,
                "job_id": f"darc-v2-fit-{source}-s{seed}-{arm}",
                "stage": "core",
                "eval_targets": list(analysis.SITES),
            }
        )
        if seed == 0:
            rows.append(
                {
                    **base,
                    "job_id": f"darc-v2-diag-{source}-s0-{arm}",
                    "stage": "diagnostic",
                    "eval_targets": [],
                }
            )
        if source != "DARC" and arm == "S0I0":
            rows.append(
                {
                    **base,
                    "job_id": f"darc-v2-fit-{source}-s{seed}-S0I0-n316",
                    "stage": "core",
                    "eval_targets": list(analysis.SITES),
                    "size_control": True,
                }
            )
    return {"schema": "darc-v2-jobs-1", "jobs": rows}


def test_gate_rejects_missing_fits_before_opening_scores(tmp_path):
    state = tmp_path / "state"
    jobs = _manifest()
    plan, diagnostics = analysis._fit_plan(jobs)
    assert len(plan) == 42
    assert len(diagnostics) == 12
    protocol_path = tmp_path / "protocol.json"
    protocol.save_json(protocol_path, {"protocol_id": "darc-processed-v1"})
    jobs["protocol_sha256"] = protocol.sha256_file(protocol_path)
    job_path = tmp_path / "jobs.json"
    protocol.save_json(job_path, jobs)
    gate_path = tmp_path / "gate.json"
    gate = {
        "approved": True,
        "kind": "scoring",
        "jobs_sha256": protocol.sha256_file(job_path),
        "protocol_sha256": jobs["protocol_sha256"],
        "release_sha": "e" * 40,
        "runtime_profile_sha256": "d" * 64,
        "audit_sha256": "c" * 64,
        "diagnostic_job_ids": sorted(diagnostics),
    }
    protocol.save_json(gate_path, gate)

    for job_id in diagnostics:
        success = {
            "job_id": job_id,
            "attempt": 1,
            "nonce": "demo",
            "state": "succeeded",
            "exit_code": 0,
            "metrics_sha256": "a" * 64,
        }
        protocol.save_json(state / f"{job_id}.json", {**success, "result": success})

    with pytest.raises(ValueError, match="Expected 42 completed DARC fits, missing 42"):
        analysis._preflight(
            tmp_path / "unreadable_scores", state, job_path, gate_path, protocol_path
        )
    protocol.save_json(gate_path, {**gate, "approved": False})
    with pytest.raises(ValueError, match="approved scoring gate"):
        analysis._preflight(
            tmp_path / "unreadable_scores", state, job_path, gate_path, protocol_path
        )


def test_exact_fit_matrix_rejects_missing_factorial_arm():
    jobs = _manifest()
    jobs["jobs"] = [row for row in jobs["jobs"] if row["job_id"] != "darc-v2-fit-MANV-s2-S1I1"]
    with pytest.raises(ValueError, match="42 fits"):
        analysis._fit_plan(jobs)


def test_factorial_interaction_and_count_effect_keep_local_reference_visible():
    rows = []
    for source, target, seed, arm in itertools.product(
        analysis.SITES, analysis.SITES, analysis.SEEDS, (*analysis.ARMS, analysis.REDUCED)
    ):
        if arm == analysis.REDUCED and source == "DARC":
            continue
        s, t = analysis.SITES.index(source), analysis.SITES.index(target)
        i, z = arm[3] == "1", arm[1] == "1"
        value = 0.1 + 0.01 * s + 0.03 * t + 0.002 * seed
        value += i * (0.02 + 0.01 * s) + z * (0.04 + 0.01 * t) + i * z * 0.01 * s
        if arm == analysis.REDUCED:
            value += -0.07 + seed * 0.01
        rows.append(
            {"source": source, "target": target, "seed": seed, "arm": arm}
            | dict.fromkeys(analysis.OUTCOMES, value)
        )
    gaps, seeds, estimates = analysis._paired_effects(rows)
    direction = [
        row
        for row in estimates
        if row["source"] == "MANV" and row["target"] == "DARC" and row["metric"] == "ap"
    ]
    by = {row["contrast"]: row for row in direction}
    interaction = by["scale_by_intensity"]
    assert interaction["score_change_by_seed"] == pytest.approx([0.02] * 3)
    assert interaction["local_reference_change_by_seed"] == pytest.approx([0.01] * 3)
    assert interaction["gap_change_mean"] == pytest.approx(-0.01)
    reduced = by["reduced_minus_full"]
    assert reduced["score_change_by_seed"] == pytest.approx([-0.07, -0.06, -0.05])
    assert reduced["score_change_sample_sd"] == pytest.approx(0.01)
    assert reduced["local_reference_change_mean"] == pytest.approx(0)
    assert reduced["gap_change_mean"] == pytest.approx(0.06)
    assert all(
        row["local_reference_arm"] == "S0I0" for row in gaps if row["arm"] == analysis.REDUCED
    )
    assert sum(row["contrast"] == "reduced_minus_full" for row in seeds) == 6 * 3 * 5
    with pytest.raises(ValueError, match="126 unique"):
        analysis._paired_effects(rows[:-1])


def test_fixed_update_budget_learning_rate_and_earliest_checkpoint(tmp_path):
    fit = analysis.Receipt(
        {"job_id": "demo", "source": "DARC", "seed": 1}, {}, tmp_path, b"", "gpu"
    )
    policy = {
        "optimizer_updates": 6,
        "batch_size": 2,
        "updates_per_validation": 2,
        "lr_first_update": 0.0025,
        "lr_decay_after_update": 4,
        "lr_decay_multiplier": 0.1,
    }
    metrics = {
        "training_updates": 6,
        "batch_size": 2,
        "validation_interval": 2,
        "best_update": 2,
        "best_val_loss": 0.4,
    }
    history = [
        {
            "update": update,
            "lr": lr,
            "source_val_loss": loss,
            "train_loss_last_batch": 0.5,
            "validation_seed": int.from_bytes(
                hashlib.sha256(f"darc-processed-v1:validation:DARC:1:{i}".encode()).digest()[:4],
                "big",
            ),
        }
        for i, (update, lr, loss) in enumerate(
            zip((2, 4, 6), (0.0025, 0.0025, 0.00025), (0.4, 0.4, 0.6), strict=True)
        )
    ]
    analysis._fit_schedule(fit, metrics, history, policy, "darc-processed-v1")
    with pytest.raises(ValueError, match="checkpoint selection"):
        analysis._fit_schedule(
            fit, {**metrics, "best_update": 4}, history, policy, "darc-processed-v1"
        )
    history[-1]["lr"] = 0.0025
    with pytest.raises(ValueError, match="fixed learning rate"):
        analysis._fit_schedule(fit, metrics, history, policy, "darc-processed-v1")


def test_gpu_receipt_must_match_fp32_with_tf32_disabled(tmp_path):
    flags = {
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "deterministic_algorithms": False,
    }
    receipt = analysis.Receipt(
        {"job_id": "demo", "device": "cuda:0", "gpu_uuid": "GPU-example", "images": "/data/images"},
        {
            "runtime_provenance": {
                "gpu": {
                    "uuid": "GPU-example",
                    "model": analysis.GPU_MODEL,
                    "compute_capability": [8, 9],
                    "total_memory_bytes": 1234,
                },
                "torch": {
                    "version": "2.5.1+cu121",
                    "cuda_version": "12.1",
                    "matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    **flags,
                },
                "thread_environment": dict.fromkeys(
                    (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "RAYON_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    ),
                    "8",
                ),
            }
        },
        tmp_path,
        b"",
        "gpu",
    )
    runtime = {
        "device": "cuda:0",
        "gpu_uuid": "GPU-example",
        "gpu_name": analysis.GPU_MODEL,
        "precision": "fp32",
        "autocast": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "torch_version": "2.5.1+cu121",
        "cuda_version": "12.1",
        "torchvision_version": "0.20.1+cu121",
        "cudnn_version": 90100,
        "model_resize_hw": [640, 655],
        "gpu_capability": [8, 9],
        "workers": 2,
        "torch_threads": 8,
        "torch_interop_threads": 8,
        "numpy_version": "2.4.6",
        "python_version": "3.11.15",
        **flags,
    }
    assert analysis._runtime_signature(receipt, runtime)[0] == analysis.GPU_MODEL
    with pytest.raises(ValueError, match="FP32 policy differs"):
        analysis._runtime_signature(receipt, {**runtime, "cudnn_allow_tf32": True})
    with pytest.raises(ValueError, match="numerical policy differs"):
        analysis._runtime_signature(receipt, {**runtime, "cudnn_benchmark": True})


def test_relocated_prediction_artifacts_are_confined_and_checksummed(tmp_path):
    path = tmp_path / "raw.json"
    protocol.save_json(path, [])
    digest = protocol.sha256_file(path)
    receipt = analysis.Receipt(
        {"job_id": "demo", "out_dir": "/worker/path/to/attempt"}, {}, tmp_path, b"", "gpu"
    )
    assert analysis._artifact(receipt, "/worker/path/to/attempt/raw.json", digest) == path
    with pytest.raises(ValueError, match="not in the subpath"):
        analysis._artifact(receipt, "/worker/path/to/another/attempt/raw.json", digest)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        analysis._artifact(receipt, "raw.json", "f" * 64)
