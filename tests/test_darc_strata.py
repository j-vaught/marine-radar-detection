from pathlib import Path

import pytest

from src.revision import darc_strata as strata


@pytest.fixture
def spec():
    root = Path(__file__).resolve().parents[1]
    return strata.load_spec(root / "experiments/review-2026-11/darc-strata-v1.json")


def reference(boxes):
    return {
        "images": [{"id": 1, "width": 1050, "height": 1024}],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": 1, "bbox": box, "iscrowd": 0}
            for i, box in enumerate(boxes, 1)
        ],
    }


def detection(box, score):
    return {"image_id": 1, "category_id": 1, "bbox": box, "score": score}


def test_matching_is_one_to_one_and_prioritizes_cardinality(spec):
    truth = reference([[10, 20, 10, 10], [14, 20, 10, 10]])
    a_flexible = detection([11, 20, 10, 10], 0.9)
    a_constrained = detection([7, 20, 10, 10], 0.5)
    matches = strata.match_predictions(truth, [a_flexible, a_constrained], spec["scoring"])
    assert matches == {"iou_0.5": {1, 2}, "center_8px": {1, 2}}
    limited = {**spec["scoring"], "top_predictions_per_image": 1}
    single = strata.match_predictions(truth, [a_constrained, a_flexible], limited)
    assert single == {"iou_0.5": {1}, "center_8px": {1}}
    center_truth = reference([[100, 100, 10, 10], [110, 100, 10, 10]])
    center_hits = strata.match_predictions(
        center_truth,
        [detection([102, 100, 10, 10], 0.9), detection([95, 100, 10, 10], 0.8)],
        spec["scoring"],
    )
    assert center_hits["center_8px"] == {1, 2}


def test_thresholds_include_boundary_and_tied_rank_preserves_archive_order(spec):
    truth = reference([[20, 20, 9, 10]])
    scoring = {**spec["scoring"], "top_predictions_per_image": 1}
    # The first proposal overlaps exactly 60 of 120 combined square pixels.
    hits = strata.match_predictions(
        truth, [detection([23, 20, 9, 10], 0.5), detection([80, 20, 9, 10], 0.5)], scoring
    )
    assert hits["iou_0.5"] == {1}
    boundary = strata.match_predictions(truth, [detection([28, 20, 9, 10], 0.5)], scoring)
    assert boundary == {"iou_0.5": set(), "center_8px": {1}}
    with pytest.raises(ValueError, match="different target IDs"):
        strata.match_predictions(
            truth, [{**detection([80, 20, 9, 10], 0.5), "image_id": 2}], scoring
        )


def test_half_open_strata_and_missing_rendered_contrast(spec):
    for value, expected in [
        (127.9, "lt128"),
        (128, "128_lt512"),
        (511.9, "128_lt512"),
        (512, "ge512"),
    ]:
        assert strata.bin_value(value, strata.AREA, spec) == expected
    for value, expected in [
        (-255, "lt64"),
        (63.9, "lt64"),
        (64, "64_lt128"),
        (128, "128_lt192"),
        (192, "ge192"),
        (255, "ge192"),
        (None, "missing"),
    ]:
        assert strata.bin_value(value, strata.CONTRAST, spec) == expected
    with pytest.raises(ValueError, match="Out-of-domain"):
        strata.bin_value(-255.01, strata.CONTRAST, spec)
    with pytest.raises(ValueError, match="Nonfinite"):
        strata.bin_value(float("nan"), strata.CONTRAST, spec)


def test_stratification_counts_matched_labels_without_rematching_and_zero_support(spec):
    truth = reference([[10, 20, 8, 10], [14, 20, 16, 10]])
    features = [
        {"annotation_id": 1, "image_id": 1, strata.AREA: 80, strata.CONTRAST: None},
        {"annotation_id": 2, "image_id": 1, strata.AREA: 160, strata.CONTRAST: -1.0},
    ]
    groups = strata.reference_strata(truth, features, spec)
    rows = []
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            matched_ids = {1} if arm == "S0I0" else {2}
            rows.extend(
                strata.seed_rows(
                    {"iou_0.5": matched_ids},
                    groups,
                    {"source": "MANV", "target": "DARC", "seed": seed, "arm": arm},
                )
            )
    paired = {
        (row["family"], row["bin"], row["contrast"]): row for row in strata.paired_rows(rows, spec)
    }
    area = paired[strata.AREA, "128_lt512", "I1_minus_I0_at_S0"]
    assert (area["seed0"], area["seed1"], area["seed2"], area["effect_mean"]) == (1, 1, 1, 1)
    missing = paired[strata.CONTRAST, "missing", "I1_minus_I0_at_S0"]
    assert missing["reference_count"] == 1 and missing["effect_mean"] == -1
    empty = paired[strata.AREA, "ge512", "I1_minus_I0_at_S0"]
    assert empty["reference_count"] == 0 and empty["effect_mean"] is None
    assert empty["effect_sample_sd"] is None
    rows = [row for row in rows if row["seed"] != 2]
    with pytest.raises(ValueError, match="Incomplete paired strata cell"):
        strata.paired_rows(rows, spec)
