"""Enumerate the U-Net P/S/T experiment matrix."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .dataset import SegmentationItem, indices_for_site, pooled_split, site_internal_split


@dataclass
class ExperimentSpec:
    exp_id: str
    condition: str
    tune_label: str
    val_label: str
    test_label: str
    seed: int
    split_fn: Callable[[list[SegmentationItem]], tuple[list[int], list[int], list[int]]] = field(repr=False)

    def split(self, items: list[SegmentationItem]) -> tuple[list[int], list[int], list[int]]:
        return self.split_fn(items)


def _pooled_factory(seed: int):
    def fn(items: list[SegmentationItem]) -> tuple[list[int], list[int], list[int]]:
        return pooled_split(len(items), seed=seed)

    return fn


def _site_perm_factory(tune: str, val: str, test: str):
    def fn(items: list[SegmentationItem]) -> tuple[list[int], list[int], list[int]]:
        return (
            indices_for_site(items, tune),
            indices_for_site(items, val),
            indices_for_site(items, test),
        )

    return fn


def _transfer_factory(tune: str, test: str):
    def fn(items: list[SegmentationItem]) -> tuple[list[int], list[int], list[int]]:
        train_idx, val_idx, held_idx = site_internal_split(items, tune, seed=0)
        if tune == test:
            return train_idx, val_idx, held_idx
        return train_idx, val_idx, indices_for_site(items, test)

    return fn


def build_all() -> list[ExperimentSpec]:
    exps: list[ExperimentSpec] = []

    for seed in range(5):
        exps.append(
            ExperimentSpec(
                exp_id=f"P-UNET-{seed}",
                condition="P",
                tune_label=f"pool70@{seed}",
                val_label=f"pool15@{seed}",
                test_label=f"pool15@{seed}",
                seed=seed,
                split_fn=_pooled_factory(seed),
            )
        )

    perms = [
        ("MANV", "DARC", "DAAN"),
        ("MANV", "DAAN", "DARC"),
        ("DARC", "MANV", "DAAN"),
        ("DARC", "DAAN", "MANV"),
        ("DAAN", "MANV", "DARC"),
        ("DAAN", "DARC", "MANV"),
    ]
    for idx, (tune, val, test) in enumerate(perms, start=1):
        exps.append(
            ExperimentSpec(
                exp_id=f"S-UNET-{idx}",
                condition="S",
                tune_label=tune,
                val_label=val,
                test_label=test,
                seed=0,
                split_fn=_site_perm_factory(tune, val, test),
            )
        )

    code = {"MANV": "Mv", "DARC": "Dc", "DAAN": "Da"}
    sites = ("MANV", "DARC", "DAAN")
    for tune in sites:
        for test in sites:
            exps.append(
                ExperimentSpec(
                    exp_id=f"T-UNET-{code[tune]}{code[test]}",
                    condition="T",
                    tune_label=f"{tune}[70]",
                    val_label=f"{tune}[15]",
                    test_label=(f"{test}[15]" if tune == test else f"{test}[all]"),
                    seed=0,
                    split_fn=_transfer_factory(tune, test),
                )
            )

    return exps
