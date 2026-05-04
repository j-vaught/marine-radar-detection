#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

MATRIX_FILES = [
    Path("results/cfar/matrix.csv"),
    Path("results/otsu/matrix.csv"),
    Path("results/wavelet/matrix.csv"),
    Path("results/unet_matrix/matrix.csv"),
    Path("results/faster_rcnn_matrix/matrix.csv"),
    Path("results/rtdetr_matrix/matrix.csv"),
    Path("results/yolo26_matrix/matrix.csv"),
]

PER_EXPERIMENT_DIRS = [
    Path("results/cfar/per_experiment"),
    Path("results/otsu/per_experiment"),
    Path("results/wavelet/per_experiment"),
    Path("results/unet_matrix/per_experiment"),
    Path("results/faster_rcnn_matrix/per_experiment"),
    Path("results/rtdetr_matrix/per_experiment"),
    Path("results/yolo26_matrix/per_experiment"),
]

DIRECTORY_SPECS = [
    Path("results/figures/paper"),
    Path("results/tide"),
]

FAILURE_PAYLOAD_FILES = [
    Path("results/unet_matrix/T-UNET-MvDc/test_gt.json"),
    Path("results/unet_matrix/T-UNET-MvDc/test_predictions.json"),
    Path("results/faster_rcnn_matrix/T-FRCNN-MvDc/test_gt.json"),
    Path("results/faster_rcnn_matrix/T-FRCNN-MvDc/test_predictions.json"),
    Path("results/yolo26_matrix/T-Y26-MvDc/test_gt.json"),
    Path("results/yolo26_matrix/T-Y26-MvDc/test_predictions.json"),
    Path("results/rtdetr_matrix/T-RTDETR-DcMv/test_gt.json"),
    Path("results/rtdetr_matrix/T-RTDETR-DcMv/test_predictions.json"),
    Path("results/yolo26_matrix/T-Y26-DcMv/test_gt.json"),
    Path("results/yolo26_matrix/T-Y26-DcMv/test_predictions.json"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-name", default="paper-artifacts-v1")
    return ap.parse_args()


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def git_metadata() -> dict[str, object]:
    try:
        status_lines = run_text(["git", "status", "--short"]).splitlines()
        branch = run_text(["git", "branch", "--show-current"])
        head = run_text(["git", "rev-parse", "HEAD"])
        return {
            "branch": branch,
            "head": head,
            "dirty": bool(status_lines),
            "status": status_lines,
        }
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        return {"error": str(exc)}


def require_file(rel_path: Path) -> Path:
    path = ROOT / rel_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(rel_path: Path) -> Path:
    path = ROOT / rel_path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def copy_file(
    src: Path,
    dest_root: Path,
    rel_dest: Path,
    category: str,
    manifest_entries: list[dict[str, object]],
) -> None:
    dest = dest_root / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest_entries.append(
        {
            "category": category,
            "path": rel_dest.as_posix(),
            "size_bytes": dest.stat().st_size,
            "sha256": sha256_hex(dest),
        }
    )


def copy_dir(
    src_root: Path,
    dest_root: Path,
    rel_prefix: Path,
    category: str,
    manifest_entries: list[dict[str, object]],
) -> None:
    for src in sorted(path for path in src_root.rglob("*") if path.is_file()):
        rel_dest = rel_prefix / src.relative_to(src_root)
        copy_file(src, dest_root, rel_dest, category, manifest_entries)


def write_bundle_readme(bundle_root: Path) -> None:
    text = """# Paper Artifacts Bundle

This tarball supplements the marine-radar-detection Git repo with the ignored result files needed to reproduce the paper-style figure pack and support manuscript writing.

## Layout

- `repo_overlay/results/*/matrix.csv`: experiment-level result tables
- `repo_overlay/results/*/per_experiment/*.json`: winning parameter summaries
- `repo_overlay/results/figures/paper/`: rendered paper figures, zoom panels, and caches
- `repo_overlay/results/tide/`: pooled TIDE summaries
- selected learned-model `test_gt.json` and `test_predictions.json` files for the failure-analysis panels

## Apply To A Clone

```bash
rsync -a repo_overlay/ /path/to/marine-radar-detection/
```

Then regenerate the figure pack with:

```bash
./.venv/bin/python tools/plot_paper_figures.py
```
"""
    (bundle_root / "README.md").write_text(text)


def build_bundle(bundle_name: str) -> tuple[Path, Path, Path]:
    DIST.mkdir(parents=True, exist_ok=True)
    bundle_root = DIST / bundle_name
    overlay_root = bundle_root / "repo_overlay"
    archive_path = DIST / f"{bundle_name}.tar.zst"
    checksum_path = DIST / f"{bundle_name}.tar.zst.sha256"

    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    if archive_path.exists():
        archive_path.unlink()
    if checksum_path.exists():
        checksum_path.unlink()

    overlay_root.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []

    for rel_path in MATRIX_FILES:
        copy_file(
            require_file(rel_path),
            bundle_root,
            Path("repo_overlay") / rel_path,
            "matrix",
            manifest_entries,
        )

    for rel_dir in PER_EXPERIMENT_DIRS:
        copy_dir(
            require_dir(rel_dir),
            bundle_root,
            Path("repo_overlay") / rel_dir,
            "per_experiment",
            manifest_entries,
        )

    for rel_dir in DIRECTORY_SPECS:
        copy_dir(
            require_dir(rel_dir),
            bundle_root,
            Path("repo_overlay") / rel_dir,
            "supporting_outputs",
            manifest_entries,
        )

    for rel_path in FAILURE_PAYLOAD_FILES:
        copy_file(
            require_file(rel_path),
            bundle_root,
            Path("repo_overlay") / rel_path,
            "failure_payload",
            manifest_entries,
        )

    write_bundle_readme(bundle_root)

    category_counts: dict[str, int] = {}
    for entry in manifest_entries:
        category = str(entry["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    manifest = {
        "bundle_name": bundle_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "category_counts": category_counts,
        "files": manifest_entries,
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    proc = subprocess.run(
        ["tar", "--zstd", "-cf", str(archive_path), bundle_name],
        cwd=DIST,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"tar failed ({proc.returncode})\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    checksum_path.write_text(f"{sha256_hex(archive_path)}  {archive_path.name}\n")
    return bundle_root, archive_path, checksum_path


def main() -> int:
    args = parse_args()
    bundle_root, archive_path, checksum_path = build_bundle(args.bundle_name)
    manifest = json.loads((bundle_root / "manifest.json").read_text())
    print(json.dumps(
        {
            "bundle_root": str(bundle_root),
            "archive": str(archive_path),
            "checksum": str(checksum_path),
            "category_counts": manifest["category_counts"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
