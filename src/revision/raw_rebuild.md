# Historical raw preprocessing candidate

This harness reconstructs a candidate historical dataset from original DLR image captures and `polarMeas.json`. It writes a separate output version and preserves the released dataset. Real raw inputs were unavailable when the harness was added. Validation uses synthetic captures; no real-raw reproduction or DARC performance result is claimed.

The archived modules under `vendor/ec6d088/` are byte-preserved from commit `ec6d088018ff255e82cfdd629e2a458c0f7d4172`. `SOURCE.json` records their original paths, Git blob identifiers, and SHA-256 hashes. The harness checks those identifiers before executing the conversion. Historical references to AIS and ground truth remain inside those unchanged modules as source evidence. The harness consumes no independent AIS file and does not validate those historical descriptions. `referenceMeas.mat`, if present, is inventoried but is not used to construct labels.

Commit `562fb39` records 7,647 regenerated annotations, matching the released total. Its exporter is byte-identical to the recovered candidate. This count agreement motivates the reconstruction; the release comparison establishes whether individual pixels and labels actually agree. The preserved `make preprocess-dlr` recipe enables CLAHE, but no execution manifest binds that invocation to the released archive. Therefore the harness generates both `clahe_off` and `clahe_on` image variants with identical pre-CLAHE labels.

## Run after locating the raw package

Use the existing repository checkout on an approved compute server. Stage the raw data on an SSD only when its capacity permits, and choose a new output directory. The raw root must contain `DAAN/polarMeas.json`, `DAAN/radarImages/000.png`, and the corresponding DARC and MANV directories. File naming follows numeric JSON frame keys. The harness refuses missing requested sites, ambiguous frame keys, incompatible included-image geometry, symlinks within site directories, and every existing output root. It never resumes a partial reconstruction into the same directory.

Create a dedicated environment using the small, hash-locked dependency set. This avoids installing detector and training packages.

```sh
uv venv --python 3.11 .venv-raw-rebuild
uv pip sync --python .venv-raw-rebuild/bin/python \
  src/revision/raw_rebuild.requirements.txt
.venv-raw-rebuild/bin/python -m src.revision.raw_rebuild \
  --raw-root /path/to/original-dlr-captures \
  --output-root /path/to/new/historical-ec6d088-replay-v1 \
  --release-annotations /path/to/released/annotations/all.json \
  --release-images /path/to/released/images
```

The release arguments are optional as a pair. A partial-site diagnostic can use `--datasets DARC`; site order otherwise remains DAAN, DARC, MANV. A partial-site comparison against a three-site release explicitly reports the omitted sites as missing from the rebuild. Archive the completed reconstruction to HDD after active SSD experiments finish. This command creates full-frame images and COCO annotations; it does not assign training splits, tile images, or launch a benchmark.

## Interpretation and evidence

Each output contains `manifest.json`, `raw_inventory.json`, and `frame_ledger.json`. The inventory hashes every ordinary file beneath the selected site directories, including unused inputs, and is recomputed after processing to detect input changes. The frame ledger distinguishes included, missing-image, unreadable-image, null-entry, and image-without-entry cases. Empty JSON lists remain included frames with no matched annotations. `manifest.json` records source and driver hashes, dependency versions, settings, counts, and artifact hashes. Only `status: complete` establishes a finished reconstruction; a failed run retains its partial output and error record.

Each `variants/clahe_off` and `variants/clahe_on` directory contains `annotations/all.json`, `images/`, and `image_manifest.json`. Images use the historical site extraction, circular mask, and optional CLAHE order. The historical code uses strict intensity greater than 25, binary connected-component centroids, a 5 by 5 elliptical closing kernel, inclusive component area 10–8,000, and strict association distance less than 20 pixels. It does not implement intensity-weighted centroids or an Otsu peak gate. CLAHE occurs after masking, without remasking, and does not change the labels.

When a release is supplied, each variant adds `comparison.json`. Frames are keyed by site and frame number. Pixels are compared after decoding and also receive file hashes, so PNG re-encoding does not create a false pixel mismatch. Labels are compared as multisets with only `id` and `image_id` removed; all other annotation fields, including peak intensity, remain significant. Numerically integral floats and integers compare equally. The report separately records label ordering and category-table equality. Missing frames and unreadable release images remain visible in the report. Neither total-count equality nor a successful process exit establishes exact release reproduction.

Run the synthetic contracts with the dedicated environment.

```sh
.venv-raw-rebuild/bin/python -m unittest discover -s tests -p 'test_raw_rebuild.py' -v
```

Keep the vendored module bytes unchanged. Format and lint only the new harness and tests; formatting the archived files would invalidate their source hashes.
