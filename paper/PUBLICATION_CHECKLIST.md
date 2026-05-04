# Publication Checklist

Use this checklist before telling someone that "everything is on GitHub."

1. Commit and push the current working-tree changes that are still local-only.
2. Build `dist/paper-artifacts-v1.tar.zst` with `./.venv/bin/python tools/package_paper_artifacts.py`.
3. Upload these release assets to the GitHub tag that corresponds to the published code snapshot:
   - `processed-v1.tar.zst`
   - `processed-v1.tar.zst.sha256`
   - `paper-artifacts-v1.tar.zst`
   - `paper-artifacts-v1.tar.zst.sha256`
4. In the release notes, state the commit SHA used to build the bundle and mention that the tarball overlays the repo at `repo_overlay/`.
5. Verify from a clean clone:
   - checkout the release tag
   - download the dataset asset
   - download the paper-artifacts asset
   - extract `repo_overlay/` into the clone
   - rerun `./.venv/bin/python tools/plot_paper_figures.py`
6. Confirm that the collaborator can find the manuscript inputs:
   - [paper/README.md](README.md)
   - [results/figures/paper/README.md](../results/figures/paper/README.md)
   - [results/figures/paper/DISCUSSION.md](../results/figures/paper/DISCUSSION.md)

## Why The Extra Asset Exists

The repo intentionally ignores most of `results/`, `runs/`, and `dist/`. That is fine for day-to-day development, but it means the following material is not automatically public unless it is uploaded as an asset:

- experiment matrices
- per-experiment summary JSON
- TIDE outputs
- additional paper figures and zoom panels generated under `results/figures/paper/`
- selected prediction payloads used by the failure-analysis figures

The paper bundle exists to publish that evidence without committing multi-gigabyte training outputs directly into Git.
