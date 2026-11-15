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


@pytest.mark.parametrize("overflow", [False, True])
def test_scaler_instrumentation_preserves_native_skip_and_backoff(overflow):
    from types import SimpleNamespace

    from src.revision.rtdetr_warmup_control import observe_optimizer_step

    events = []
    model = torch.nn.Linear(1, 1)
    for p in model.parameters():
        p.grad = torch.full_like(p, float("inf") if overflow else 1.0)

    class Optimizer:
        def step(self):
            events.append("optimizer_step")

    class Scaler:
        def __init__(self):
            self.scale = 8.0

        def get_scale(self):
            return self.scale

        def is_enabled(self):
            return True

        def unscale_(self, optimizer):
            events.append("unscale")

        def _found_inf_per_device(self, optimizer):
            return {"cpu": torch.tensor(float(overflow))}

        def step(self, optimizer):
            events.append("scaler_step")
            if not overflow:
                optimizer.step()

        def update(self):
            events.append("update")
            if overflow:
                self.scale /= 2

    trainer = SimpleNamespace(
        loss=torch.tensor(1.0),
        control_batch={},
        model=model,
        optimizer=Optimizer(),
        scaler=Scaler(),
    )

    def native_step():
        trainer.scaler.unscale_(trainer.optimizer)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()

    state = {"optimizer_attempts": 0, "optimizer_steps": 0, "skipped_steps": 0}
    observe_optimizer_step(trainer, state, native_step)
    assert events == (
        ["unscale", "scaler_step", "update"]
        if overflow
        else ["unscale", "scaler_step", "optimizer_step", "update"]
    )
    assert state == {
        "optimizer_attempts": 1,
        "optimizer_steps": int(not overflow),
        "skipped_steps": int(overflow),
    }
    assert trainer.control_batch["found_inf"] == {"cpu": float(overflow)}
    assert trainer.control_batch["scale_after"] == (4.0 if overflow else 8.0)
    assert trainer.control_batch["unscaled_gradient_finite"] is not overflow
