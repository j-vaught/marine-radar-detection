import pytest

from src.revision.yolo_backend_study import prediction_comparison, verify_cpu_metrics


def test_prediction_matching_is_one_to_one_and_reports_unmatched():
    cpu = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [40, 40, 5, 5], "score": 0.3},
    ]
    gpu = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 0, 10, 10], "score": 0.7},
        {"image_id": 2, "category_id": 1, "bbox": [40, 40, 5, 5], "score": 0.9},
    ]
    result = prediction_comparison(cpu, gpu)
    assert result["matched_iou_at_least_0_5"] == 1
    assert result["unmatched_cpu"] == result["unmatched_gpu"] == 1
    assert result["absolute_xywh_delta_median_p95_max"][0] == [1.0, 0.0, 0.0, 0.0]
    assert result["absolute_score_delta_median_p95_max"][0] == pytest.approx(0.1)


def test_primary_metrics_must_replay_before_comparison():
    result = {
        "ap": 0.5,
        "ap50": 0.8,
        "ap75": 0.6,
        "ar100": 0.7,
        "predictions": 2,
        "preds_per_image": 1.0,
    }
    verify_cpu_metrics(result, dict(result))
    with pytest.raises(ValueError, match="ap75"):
        verify_cpu_metrics(result, {**result, "ap75": 0.61})
