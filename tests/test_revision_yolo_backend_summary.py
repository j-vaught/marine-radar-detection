import itertools

import pytest

from src.revision import protocol
from src.revision import yolo_backend_summary as backend


def test_adjacency_substitution_uses_same_target_block_and_seed_pairs():
    primary = {}
    for stage, source, seed, target in itertools.product(
        backend.STAGES, backend.TARGETS, backend.SEEDS, backend.TARGETS
    ):
        score = 0.5 if source == target else 0.3
        if stage == "adjacency":
            score += 0.1 if source == target else 0.05
        primary[(stage, "yolo", source, seed), target] = {
            metric: [score] * 6 for metric in backend.METRICS
        }

    altered = primary.copy()
    key = ("core", "yolo", "DAAN", 0), "DARC"
    altered[key] = {metric: [0.39, 0.39, 0.39, 0.48, 0.39, 0.39] for metric in backend.METRICS}

    def relative_effect(target, block, measure):
        return next(
            row["gpu_matched_minus_primary"]["mean"]
            for row in backend._compare(primary, altered, target, block)
            if row["statistic"] == measure and row["metric"] == "ap"
        )

    assert relative_effect(None, None, "macro_transfer") == pytest.approx(-0.005)
    assert relative_effect(None, None, "macro_target_local_minus_incoming") == pytest.approx(0.005)
    assert relative_effect(None, None, "macro_local") == pytest.approx(0.0)
    assert relative_effect("DARC", 2, "macro_transfer") == pytest.approx(-0.01)
    assert relative_effect("DAAN", 2, "macro_transfer") == pytest.approx(-0.005)


def test_gpu_artifact_verifies_bytes_before_load(tmp_path):
    path = tmp_path / "DAAN_gpu_raw.json"
    raw = [{"image_id": 3, "score": 0.5}]
    protocol.save_json(path, raw)
    digest = protocol.sha256_file(path)
    assert backend._verified_json(path, digest) == raw

    protocol.save_json(path, [{"image_id": 3, "score": 0.4}])
    with pytest.raises(ValueError, match="Digest differs"):
        backend._verified_json(path, digest)


def test_staged_input_and_observed_inference_audit_is_bound_to_study():
    counts = dict.fromkeys(backend.TARGETS, 10)
    replay = {
        "job_id": "core-yolo-DAAN-s0",
        "gpu_uuid": "GPU-demo",
        "manifest_sha256": "manifest",
        "batch": 1,
        "imgsz": 1024,
        "torch_threads": 4,
        "cudnn_allow_tf32": True,
        "matmul_allow_tf32": False,
    }
    audit = {
        **{key: replay[key] for key in ("job_id", "gpu_uuid", "manifest_sha256")},
        "study_sha256": "study",
        "images_verified": 30,
        "per_target": {
            target: {
                "images": frames,
                "manifest_hash_mismatches": 0,
                "mask_pixel_mismatches": 0,
                "staged_hash_mismatches": 0,
            }
            for target, frames in counts.items()
        },
        "predictor_observed": {
            "actual_results": 1,
            "args_batch": 1,
            "args_imgsz": 1024,
            "args_rect": True,
            "args_half": False,
            "last_batch_size": 1,
            **{
                key: replay[key]
                for key in ("torch_threads", "cudnn_allow_tf32", "matmul_allow_tf32")
            },
        },
    }
    backend._verify_input_audit(audit, replay, "study", counts)
    with pytest.raises(ValueError, match="input or predictor audit differs"):
        backend._verify_input_audit(audit, replay, "another-study", counts)
    audit["per_target"]["DARC"]["mask_pixel_mismatches"] = 1
    with pytest.raises(ValueError, match="staged image audit differs"):
        backend._verify_input_audit(audit, replay, "study", counts)
