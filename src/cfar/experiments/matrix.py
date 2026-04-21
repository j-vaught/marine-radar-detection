"""Enumerate the 40 CFAR experiments (P × 10, S × 12, T × 18)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..gpu.data import Corpus, pooled_split, site_internal_split


@dataclass
class Experiment:
    exp_id: str
    condition: str              # 'P', 'S', or 'T'
    method: str                 # 'CA' or 'OS'
    tune_label: str
    val_label: str
    test_label: str
    seed: int
    split_fn: Callable[[Corpus], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(repr=False)


def _pooled_factory(seed: int):
    def fn(c: Corpus):
        return pooled_split(c.N, seed)
    return fn


def _site_perm_factory(tune: str, val: str, test: str):
    def fn(c: Corpus):
        return (c.indices_for_site(tune), c.indices_for_site(val), c.indices_for_site(test))
    return fn


def _transfer_factory(tune: str, test: str):
    def fn(c: Corpus):
        t, v, held = site_internal_split(c, tune, seed=0)
        if tune == test:
            return t, v, held               # diagonal: evaluate on held-out 15%
        test_idx = c.indices_for_site(test)  # off-diagonal: full other site
        return t, v, test_idx
    return fn


def build_all() -> list[Experiment]:
    exps: list[Experiment] = []

    # ── Condition P: pooled random split, 5 seeds ─────────────────────────
    for method in ("CA", "OS"):
        for seed in range(5):
            exps.append(Experiment(
                exp_id=f"P-{method}-{seed}", condition="P", method=method,
                tune_label=f"pool70@{seed}", val_label=f"pool15@{seed}",
                test_label=f"pool15@{seed}", seed=seed,
                split_fn=_pooled_factory(seed),
            ))

    # ── Condition S: 6 site permutations ──────────────────────────────────
    perms = [
        ("MANV", "DARC", "DAAN"),
        ("MANV", "DAAN", "DARC"),
        ("DARC", "MANV", "DAAN"),
        ("DARC", "DAAN", "MANV"),
        ("DAAN", "MANV", "DARC"),
        ("DAAN", "DARC", "MANV"),
    ]
    for method in ("CA", "OS"):
        for i, (t, v, te) in enumerate(perms, start=1):
            exps.append(Experiment(
                exp_id=f"S-{method}-{i}", condition="S", method=method,
                tune_label=t, val_label=v, test_label=te, seed=0,
                split_fn=_site_perm_factory(t, v, te),
            ))

    # ── Condition T: 3×3 transfer ─────────────────────────────────────────
    sites = ["MANV", "DARC", "DAAN"]
    for method in ("CA", "OS"):
        for tune in sites:
            for test in sites:
                # 2-char unique codes to avoid DARC/DAAN first-letter collision
                code = {"MANV": "Mv", "DARC": "Dc", "DAAN": "Da"}
                tag = f"{code[tune]}{code[test]}"
                exps.append(Experiment(
                    exp_id=f"T-{method}-{tag}", condition="T", method=method,
                    tune_label=f"{tune}[70]",
                    val_label=f"{tune}[15]",
                    test_label=(f"{test}[15]" if tune == test else f"{test}[all]"),
                    seed=0,
                    split_fn=_transfer_factory(tune, test),
                ))

    return exps
