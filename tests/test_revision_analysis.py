import hashlib
import itertools
import json

import pytest

from src.revision.summarize import ADJACENCY, CLASSICAL, LEARNED, SITES, summarize


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    image_count = (2, 3, 5)
    sites = {}
    images = []
    boxes = []
    for index, (site, frames) in enumerate(zip(SITES, image_count)):
        first_id = index * 100
        test_ids = list(range(first_id + 3, first_id + 3 + frames))
        sites[site] = {
            "partitions": {"validation": [first_id], "train": [first_id + 1], "test": test_ids},
            "adjacency": {str(seed): [first_id + 2] for seed in range(3)},
        }
        for image_id in range(first_id, first_id + 3 + frames):
            images.append(
                {
                    "id": image_id,
                    "site": site,
                    "frame": image_id - first_id,
                    "sha256": hashlib.sha256(str(image_id).encode()).hexdigest(),
                }
            )
            boxes.append(
                {
                    "id": image_id + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [4, 5, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            )
    coco = {"categories": [{"id": 1, "name": "object"}], "images": images, "annotations": boxes}
    ann = tmp_path / "annotations.json"
    manifest_data = {"sites": sites, "images": images, "annotation_sha256": _json(ann, coco)}
    manifest = tmp_path / "split.json"
    digest = _json(manifest, manifest_data)
    return {
        "ann": str(ann),
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "manifest_data": manifest_data,
        "coco": coco,
        "runs": tmp_path / "runs",
    }


def _job(frozen, stage, method, source, seed):
    return {
        "job_id": f"{stage}-{method}-{source}-s{seed}",
        "method": method,
        "source": source,
        "seed": seed,
        "condition": "adjacency" if stage == "adjacency" else "buffered",
        "stage": stage,
        "eval_targets": list(SITES),
        "ann": frozen["ann"],
        "manifest": frozen["manifest"],
    }


def _metrics(value, count=1):
    return {
        "ap": value,
        "ap50": value,
        "ap75": value,
        "ar100": value,
        "predictions": 1,
        "preds_per_image": 1 / count,
    }


def _ap(source, target, seed, stage):
    s, t = SITES.index(source), SITES.index(target)
    if s == t:
        return 0.6 + 0.05 * t + 0.01 * seed + (0.12 if stage == "adjacency" else 0)
    return 0.1 + 0.05 * s + 0.07 * t + 0.02 * seed - (0.03 if stage == "adjacency" else 0)


def _write_attempt(frozen, job, *, failed=False):
    out = frozen["runs"] / job["job_id"] / "1"
    _json(
        out / "completion.json",
        {
            "job_id": job["job_id"],
            "attempt": 1,
            "state": "failed" if failed else "succeeded",
            "exit_code": int(failed),
        },
    )
    _json(out / "job.json", job)
    if failed:
        return out
    metrics = {
        **job,
        "manifest_sha256": frozen["manifest_sha256"],
        "checkpoint_sha256": hashlib.sha256(job["job_id"].encode()).hexdigest(),
        "validation": _metrics(0.5),
        "tests": {
            target: _metrics(
                _ap(job["source"], target, job["seed"], job["stage"]),
                len(frozen["manifest_data"]["sites"][target]["partitions"]["test"]),
            )
            for target in SITES
        },
    }
    if job["method"] in CLASSICAL:
        metrics["selected_config"] = {"config": {"threshold": 2}}
    else:
        # Exercise both supported learned payload shapes.
        key = "raw" if job["method"] in {"unet", "faster_rcnn"} else "raw_metrics"
        metrics["tests"] = {site: {key: record} for site, record in metrics["tests"].items()}
    _json(out / "metrics.json", metrics)
    return out


def _stat(summary, statistic, source=None, target=None, group="core"):
    return next(
        row
        for row in summary[group]
        if (row["method"], row["metric"], row["statistic"], row["source"], row["target"])
        == ("faster_rcnn", "ap", statistic, source, target)
    )


def test_complete_matrix_equal_edge_weights_and_three_paired_seeds(frozen):
    jobs = [
        _job(frozen, stage, method, source, seed)
        for stage, methods, seeds in (
            ("core", LEARNED, range(5)),
            ("core", CLASSICAL, (0,)),
            ("adjacency", ADJACENCY, range(3)),
        )
        for method, source, seed in itertools.product(methods, SITES, seeds)
    ]
    for job in jobs:
        _write_attempt(frozen, job)
    summary, rows = summarize(jobs, frozen["runs"])
    assert len(rows) == 279
    assert summary["missing_jobs"] == []
    assert summary["undeclared_primary_jobs"] == []
    assert summary["counts"]["learned_core"]["complete_jobs"] == 60
    assert summary["counts"]["classical_core"]["complete_jobs"] == 15
    assert summary["counts"]["learned_adjacency"]["complete_jobs"] == 18
    assert summary["unique_learned_checkpoint_hashes"] == 78
    checkpoint_rows = [row for row in rows if row["job_id"] == "core-faster_rcnn-DAAN-s0"]
    assert [row["objects"] for row in checkpoint_rows] == [2, 3, 5]
    assert len({row["checkpoint_sha256"] for row in checkpoint_rows}) == 1

    edges = list(itertools.permutations(SITES, 2))
    expected = [sum(_ap(s, t, seed, "core") for s, t in edges) / 6 for seed in range(5)]
    macro = _stat(summary, "macro_transfer")
    assert [seed["value"] for seed in macro["seed_values"]] == pytest.approx(expected)
    assert macro["mean"] == pytest.approx(0.26)
    # Support differs by target. A frame-weighted average would change this result.
    holdout_frames = {site: size for site, size in zip(SITES, (2, 3, 5))}
    weighted = sum(_ap(s, t, 0, "core") * holdout_frames[t] for s, t in edges) / 20
    assert weighted != pytest.approx(expected[0])
    assert macro["sample_sd"] == pytest.approx(0.0316227766)
    assert macro["approximate_t95"] == pytest.approx([0.220735, 0.299265], abs=1e-6)
    assert _stat(summary, "target_local_minus_incoming", "DAAN", "DARC")["mean"] == pytest.approx(
        0.46
    )
    assert _stat(summary, "reciprocal_forward_minus_reverse", "DAAN", "DARC")[
        "mean"
    ] == pytest.approx(0.02)
    assert _stat(summary, "macro_target_local_minus_incoming", group="adjacency")[
        "mean"
    ] == pytest.approx(0.15)
    assert _stat(summary, "macro_transfer", group="adjacency")["expected_seeds"] == [0, 1, 2]
    classical = next(row for row in summary["core"] if row["method"] == "otsu")
    assert classical["deterministic"]
    assert classical["approximate_t95"] is None


def test_failed_seed_cannot_produce_complete_macro(frozen):
    method = "faster_rcnn"
    jobs = [
        _job(frozen, "core", method, source, seed)
        for source, seed in itertools.product(SITES, range(5))
    ]
    omitted, failed = ("MANV", 4), ("DARC", 3)
    for job in jobs:
        pair = (job["source"], job["seed"])
        if pair != omitted:
            _write_attempt(frozen, job, failed=pair == failed)
    summary, rows = summarize(jobs, frozen["runs"])
    assert len(rows) == 39
    assert sorted(report["state"] for report in summary["missing_jobs"]) == ["failed", "missing"]
    macro = _stat(summary, "macro_transfer")
    assert macro["missing_seeds"] == [3, 4]
    assert macro["mean"] is None
    assert macro["sample_sd"] is None
    assert macro["approximate_t95"] is None
    assert [seed["seed"] for seed in macro["seed_values"]] == [0, 1, 2]
    # An unaffected cell still has a complete group.
    assert _stat(summary, "cell", "DAAN", "MANV")["n_complete"] == 5


def test_refuse_legacy_holdout_mix_and_detect_unverified_digest(frozen, tmp_path):
    method = "faster_rcnn"
    core = _job(frozen, "core", method, "DAAN", 0)
    bad = _job(frozen, "adjacency", method, "DAAN", 0)
    _write_attempt(frozen, core)

    different = tmp_path / "different.json"
    data = frozen["manifest_data"]
    data["sites"]["DARC"]["partitions"]["test"] = [101, 103, 104, 105]
    bad["manifest"] = str(different)
    digest = _json(different, data)
    _write_attempt({**frozen, "manifest_sha256": digest}, bad)
    with pytest.raises(ValueError, match="mismatched fixed target holdout for DARC"):
        summarize([core, bad], frozen["runs"])

    _write_attempt(frozen, bad)
    summary, rows = summarize([core, bad], frozen["runs"])
    assert len(rows) == 3
    assert summary["missing_jobs"][0]["state"] == "invalid"
    assert "split manifest SHA-256" in summary["missing_jobs"][0]["error"]


def test_block_deletion_replays_paired_gap(frozen):
    evaluation = pytest.importorskip("src.revision.evaluation")
    pytest.importorskip("pycocotools")
    from src.revision.summarize import _subset_coco

    jobs = [_job(frozen, "core", "faster_rcnn", site, 0) for site in ("DAAN", "DARC")]
    for job in jobs:
        out = _write_attempt(frozen, job)
        payload = json.loads((out / "metrics.json").read_text())
        for site in SITES:
            ids = frozen["manifest_data"]["sites"][site]["partitions"]["test"]
            gt = _subset_coco(frozen["coco"], set(ids))
            predicted_ids = ids if job["source"] == site else ids[:1]
            predictions = [
                {"image_id": id_, "bbox": [4, 5, 10, 10], "score": 0.8, "category_id": 1}
                for id_ in predicted_ids
            ]
            payload["tests"][site] = {"raw": evaluation.evaluate_coco(gt, predictions)}
            _json(out / f"{site}_raw_predictions.json", predictions)
        _json(out / "metrics.json", payload)

    report, _ = summarize(jobs, frozen["runs"], blocks={"DARC": [[103], [104, 105]]})
    gap = report["block_deletion"]["DARC"]["target_local_minus_incoming"][0]
    assert gap["source"] == "DAAN"
    scores = gap["ap_ranges"]["ap"]
    assert scores["full_holdout"] == pytest.approx(1 - 34 / 101)
    assert scores["by_deleted_block"] == pytest.approx([1.0, 0.0])
    assert [scores["min"], scores["max"]] == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="disjoint partition"):
        summarize(jobs, frozen["runs"], blocks={"DARC": [[103, 104], [104, 105]]})
