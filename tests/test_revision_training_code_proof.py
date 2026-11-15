"""Check that the proof ignores only the authorized repair branch."""

from unittest import TestCase

from src.revision.training_code_proof import parts

BASE = """import os

def coco_yolo_lines(coco, image):
    return []

def stage_split(coco, image_dir, root):
    return root

def _model_spec(method, initialization, cfg):
    return cfg

def run_job(job):
    args = {"epochs": 100}
    model.train(**args)
    return 1
"""
GUARD = """    if job.get("evaluation_from"):
        from .yolo_repair import evaluate_from_parent
        return evaluate_from_parent(job)
"""


def candidate():
    return BASE.replace('    args = {"epochs": 100}\n', GUARD + '    args = {"epochs": 100}\n')


def test_guard_only_and_post_training_change_match():
    new = candidate().replace("    return 1\n", "    return 2\n")
    assert parts(BASE, remove_guard=False) == parts(new, remove_guard=True)


def test_training_or_helper_changes_are_detected():
    old = parts(BASE, remove_guard=False)
    new = (
        candidate()
        .replace('"epochs": 100', '"epochs": 50')
        .replace("return root", "return image_dir")
    )
    changed = parts(new, remove_guard=True)
    assert old["run_job_through_model_train"] != changed["run_job_through_model_train"]
    assert old["stage_split"] != changed["stage_split"]


def test_modified_or_missing_guard_is_rejected():
    with TestCase().assertRaisesRegex(ValueError, "repair guard"):
        parts(
            candidate().replace('job.get("evaluation_from")', 'job.get("other")'), remove_guard=True
        )
    with TestCase().assertRaisesRegex(ValueError, "repair guard"):
        parts(BASE, remove_guard=True)


def test_missing_training_call_is_rejected():
    with TestCase().assertRaisesRegex(ValueError, "model.train"):
        parts(candidate().replace("model.train(**args)", "model.train()"), remove_guard=True)
