from pathlib import Path

import pytest
import torch

from src.revision.rtdetr_warmup_control import (
    comparable_arguments,
    control_arguments,
    tensor_digest,
)


def test_pair_changes_only_bias_and_keeps_full_cosine_schedule():
    a = control_arguments(Path("/a/data.yaml"), Path("/a/out"), 0, 0.0)
    b = control_arguments(Path("/b/data.yaml"), Path("/b/out"), 0, 0.1)
    assert comparable_arguments(a) == comparable_arguments(b)
    assert (a["epochs"], a["cos_lr"], a["patience"]) == (100, True, 0)
    assert (a["warmup_bias_lr"], b["warmup_bias_lr"]) == (0.0, 0.1)
    with pytest.raises(ValueError):
        control_arguments(Path("a"), Path("b"), 2, 0.0)


def test_tensor_digest_detects_changed_values_and_shapes():
    original = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    assert tensor_digest(original) == tensor_digest(dict(reversed(list(original.items()))))
    assert tensor_digest(original) != tensor_digest({**original, "a": torch.tensor([1.0, 3.0])})
    assert tensor_digest(original) != tensor_digest({**original, "a": torch.tensor([[1.0, 2.0]])})
