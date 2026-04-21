"""Run the 40-experiment CFAR matrix and write ``results/cfar/matrix.csv``.

Usage:
    python -m src.cfar.experiments.run_matrix [--only P|S|T|CA|OS|<exp_id>]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..gpu.data import load_corpus
from ..gpu.eval_coco import build_gt_index
from .grid import BackgroundCache, run_experiment
from .matrix import build_all

CSV_COLS = [
    "exp_id", "condition", "method", "tune_set", "val_set", "test_set", "seed",
    "win_window_guard", "win_window_ref", "win_pfa", "win_min_area", "win_os_k",
    "tune_AP50", "val_AP50",
    "test_AP", "test_AP50", "test_AP75", "test_recall_at_p90", "test_fa_per_image",
    "total_runtime_sec",
]


def _match_filter(exp, only: str | None) -> bool:
    if not only:
        return True
    if only in {"P", "S", "T"}:
        return exp.condition == only
    if only in {"CA", "OS"}:
        return exp.method == only
    return exp.exp_id == only


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Run subset: P | S | T | CA | OS | <exp_id>")
    ap.add_argument("--resume", action="store_true",
                    help="Skip experiments with existing per-experiment JSON outputs and append to matrix.csv")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ann", default="data/processed/annotations/all.json")
    ap.add_argument("--images", default="data/processed/images")
    ap.add_argument("--out-dir", default="results/cfar")
    args = ap.parse_args()

    print(f"[load] corpus from {args.ann}")
    corpus = load_corpus(args.ann, args.images, device=args.device)
    print(f"[load] {corpus.N} images on {args.device}")

    cache = BackgroundCache(corpus)
    gt_index = build_gt_index(corpus.coco)
    exps = [e for e in build_all() if _match_filter(e, args.only)]
    print(f"[matrix] running {len(exps)} experiments")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_exp_dir = out_dir / "per_experiment"
    per_exp_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "matrix.csv"
    completed: set[str] = set()
    if args.resume:
        completed = {p.stem for p in per_exp_dir.glob("*.json")}
        if completed:
            print(f"[resume] found {len(completed)} completed experiments")

    exps = [e for e in exps if e.exp_id not in completed]
    if not exps:
        print("[done] nothing to run")
        return

    csv_mode = "a" if args.resume and csv_path.exists() else "w"
    with csv_path.open(csv_mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if csv_mode == "w":
            w.writeheader()
        for exp in exps:
            print(f"\n=== {exp.exp_id}  [{exp.condition}]  {exp.tune_label} → "
                  f"{exp.val_label} → {exp.test_label} ===")
            row = run_experiment(exp, corpus, cache, gt_index=gt_index)
            w.writerow(row); f.flush()
            (per_exp_dir / f"{exp.exp_id}.json").write_text(json.dumps(row, indent=2))
            print(f"  winner: g={row['win_window_guard']} r={row['win_window_ref']} "
                  f"pfa={row['win_pfa']:.0e} min_area={row['win_min_area']} "
                  f"os_k={row['win_os_k']}")
            print(f"  test AP50={row['test_AP50']}  AP={row['test_AP']}  "
                  f"FA/img={row['test_fa_per_image']}  ({row['total_runtime_sec']}s)")

    print(f"\n[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
