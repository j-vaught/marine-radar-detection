# Marine radar revision reproduction

The revision evaluates each source-trained detector on the same fixed destination holdouts. One learned checkpoint or one classical parameter selection per source run supplies all three test evaluations. The submitted paper is preserved in the submission archive. This guide covers the detector code, run matrix, and evidence archive.

## Release and environment

Set `RELEASE_SHA` to worker commit `3ed180804db2a3dd24420b51099ee0771ef2db43`. Use the controller checkout containing `experiments/review-2026-11/jobs.json` and the fleet, archive, and finalization commands below. Record both commits. Set `RESULT_ROOT` to the permanent run directory on each worker and `DATA_ROOT` to the processed release containing `annotations/all.json` and `images/`. Keep the processed inputs read-only and compare their hashes with the frozen manifest when staging each worker.

Create an empty release directory on each worker and extract the committed tree there. Run the locked install from that directory. [Revision environment](revision-environment.md). Use Python 3.11 and a compatible NVIDIA driver for the CUDA pilot. The Linux x86-64 lock uses the CUDA 12.1 builds of PyTorch 2.5.1 and torchvision 0.20.1, alongside Ultralytics 8.4.14 and pycocotools 2.0.11. Keep the lock, interpreter version, and installed package versions with the worker evidence.

```sh
git archive "$RELEASE_SHA" | tar -x -C "$WORKER_RELEASE"
(
  cd "$WORKER_RELEASE"
  UV_PROJECT_ENVIRONMENT="$WORKER_VENV" uv sync --locked
)
```

Place `yolo26m.pt`, `rtdetr-l.pt`, and the Ultralytics automatic mixed-precision probe `yolo26n.pt` in the directory specified by the host's `weights_path`. Stage the checkpoint for `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` in the torchvision hub cache under the configured `torch_home`. Record SHA-256 digests of the staged files. The learned core initializes Faster R-CNN, YOLO26, and the Real-Time Detection Transformer (RT-DETR) from pretrained weights and initializes U-Net from scratch. The runners seed the available random number generators before model construction and record initialization evidence with each attempt.

## Native detectors

Build the CPU Otsu and Wavelet binaries at the locations declared by the frozen job matrix. The Otsu binary runs both Otsu and SS-Otsu. The matrix uses the following native build directory key. The native sources at this key match the worker release.

```sh
NATIVE_DIR="$RESULT_ROOT/native/6e475f71f4a6c7a8760f67f52965d3b66d53fbcb"

cargo build --locked --release --manifest-path src/otsu/Cargo.toml \
  --target-dir "$NATIVE_DIR/otsu-cpu"
cargo build --locked --release --manifest-path src/wavelet/Cargo.toml \
  --target-dir "$NATIVE_DIR/wavelet-cpu"
```

The frozen fleet matrix runs classical methods on CPU slots. For a Wavelet CUDA build on RTX 6000 Ada, use `sm_89`. The build script invokes `nvcc` with GNU C++ 12 at `/usr/bin/g++-12`. Record the Rust and CUDA toolchain versions and hash the resulting binaries. Use a separate directory for the CUDA build so the matrix retains its CPU executable.

```sh
RADAR_CUDA_ARCH=sm_89 cargo build --locked --release --features cuda \
  --manifest-path src/wavelet/Cargo.toml --target-dir "$NATIVE_DIR/wavelet-cuda"
```

## Freeze the split and jobs

Build the common image-ID manifest once from the released annotations and image files.

```sh
SPLIT="$RESULT_ROOT/protocol/split_manifest.json"
JOBS=experiments/review-2026-11/jobs.json

uv run --locked python -m src.revision.protocol \
  --ann "$DATA_ROOT/annotations/all.json" \
  --images "$DATA_ROOT/images" \
  --output "$SPLIT"
```

The split sorts each site by its numeric frame index. At the floor-rounded 15%, 20%, 80%, and 85% boundaries, it defines validation, a validation buffer, training, a test buffer, and test. It records the image and annotation hashes, partition IDs, and nearest cross-partition frame-index distances. The adjacency arm replaces an equal number of core training images with the pre-test buffer images, using a recorded seed-dependent SHA-256 ordering for removal. The validation and test images remain fixed. The archive establishes frame-index separation. Verified recording chronology and persistent vessel tracks are unavailable.

Use the checked-in job manifest directly. It declares 110 jobs, comprising 17 source-only diagnostics and 93 benchmark jobs. Four learned models use five seeds across three sources for 60 core fits. Five deterministic classical methods use one source parameter selection per site for 15 core jobs. Faster R-CNN and YOLO26 each run the adjacency arm on three sources with seeds 0–2 for another 18 learned jobs. The 93 benchmark jobs produce 279 destination evaluations. The diagnostic jobs have empty `eval_targets`.

## Gate and run the fleet

The controller uses a host JSON file that maps SSH aliases to staged paths and worker capabilities. Fill this example with the verified settings for each host. Use `allowed_gpus` and `gpu_methods` when a device is available for only part of the matrix. A worker that needs a user-local driver wrapper supplies its path in `wrapper`. Set `data_verified` after checking the staged data and split against the freeze.

```json
{
  "hosts": {
    "worker-alias": {
      "code_path": "/path/to/clean-worker-release",
      "code_commit": "3ed180804db2a3dd24420b51099ee0771ef2db43",
      "data_path": "/path/to/processed",
      "result_root": "/path/to/permanent-revision-storage",
      "venv_python": "/path/to/locked-environment/bin/python",
      "weights_path": "/path/to/staged-weights",
      "torch_home": "/path/to/torch-cache",
      "checkpoint_scratch_root": "/path/to/temporary-ssd",
      "protocol_sha256": "<SHA-256 of split_manifest.json>",
      "data_verified": true,
      "cpu_workers": 1,
      "threads": 4,
      "allowed_gpus": [0]
    }
  }
}
```

Launch the controller with a fresh state directory and a gate path. With the gate absent, it schedules diagnostic jobs. The scheduler checks for active compute processes, assigns a GPU by UUID and isolates it through `CUDA_VISIBLE_DEVICES`. The worker checks that the effective device UUID matches the assignment. Use the status command to read the saved attempt records.

```sh
uv run --locked python -m src.revision.fleet schedule \
  --manifest "$JOBS" --hosts "$HOSTS" --state "$STATE" --approval "$GATE"

uv run --locked python -m src.revision.fleet schedule \
  --manifest "$JOBS" --hosts "$HOSTS" --state "$STATE" --approval "$GATE" --status
```

After the source-only diagnostics and recipe review, the gate JSON records `approved: true`, the `approved_methods` list, SHA-256 digests of the exact jobs and split files under `jobs_sha256` and `split_sha256`, the worker release commit under `release_sha`, and every diagnostic job ID under `diagnostic_job_ids`. Each method becomes eligible when all its diagnostics have succeeded and its gate matches the worker release and split. The scheduler preserves separate attempt directories and retries confirmed infrastructure failures up to the third attempt. A model failure remains visible in the state and attempt record.

## Select and evaluate each source model

The Torch runners select the learned checkpoint by source-validation loss. The Ultralytics runners record their source-validation fitness and selected epoch. Learned score thresholds use source-validation COCO metrics. U-Net also selects its pixel threshold there. The classical runners score candidates on source training and rerank the five finalists on source validation before applying the selected configuration to all three holdouts. The test pass uses the frozen source selection.

The common evaluator follows Common Objects in Context (COCO) bounding-box evaluation with one foreground class and `maxDets=[1,10,100]`. Prediction rows contain `image_id`, `category_id`, `bbox` in pixel `[x,y,width,height]`, and `score`. The evaluator records `ap`, `ap50`, `ap75`, `ar100`, `predictions`, and `preds_per_image`. Raw predictions support ranking metrics. The learned runners also archive their operating-point predictions after applying the source-selected score threshold. Average precision (AP) and average recall (AR) measure agreement with the available reference labels.

Each attempt preserves the resolved job, worker receipt and completion record, metrics, and predictions. Learned attempts also preserve their checkpoint and epoch logs. The completion records the metrics digest. Per-run metadata ties the results to the annotation and split digests and the selected model or classical parameters. Per-target metadata identifies the COCO subset digest and, for learned runs, the selected checkpoint digest. The finalizer checks these identities against the reconstructed holdouts and the archive.

## Archive and replay

Where configured and available, the worker stages frequent checkpoint writes on the local solid-state drive. It verifies the file inventory and SHA-256 digests after copying to the permanent hard drive. A successful copy replaces temporary paths with verified archive files. Checkpoint staging status, bytes, and copy time are recorded separately. The worker records a recoverable archive failure when a copy cannot be verified.

Archive finished attempts at `ARCHIVE_ROOT/job_id/attempt` on a permanent storage host. The archive command compares source and destination inventories and writes a verification ledger. Pass `--verify-existing` when a full checksum scan of previously verified attempts is required.

```sh
uv run --locked python -m src.revision.archive \
  --state "$STATE" --dest-host "$ARCHIVE_HOST" --root "$ARCHIVE_ROOT" --watch

uv run --locked python -m src.revision.fleet collect \
  --state "$STATE" --dest "$COLLECTED" --metadata-only
```

Worker completions and metrics carry runtime provenance. Report training and inference times using the timers recorded by the particular runner. Torch records training time and the full target evaluation-loop time. Ultralytics records training time and inference time per target. Classical `grid_seconds` covers the candidate search and validation rerank. These boundaries differ. Device model, GPU UUID, compute capability, memory, driver and runtime provenance, and CPU thread settings identify the hardware and process behind each timing where recorded. Peak GPU allocation is a framework measurement. Archive copies and scheduling time are recorded separately. RT-DETR's CUDA `grid_sample` backward pass is a recorded source of nondeterminism.

Run the finalizer after collecting all declared attempts. The JSON, CSV, and log collection suffices for report generation. Preserve the `job_id/attempt` layout and include the three raw prediction files for each core job. The finalizer writes progress for incomplete archives and opens the held-out references for analysis only after all expected attempts pass preflight.

```sh
uv run --locked python -m src.revision.finalize \
  --jobs "$JOBS" --runs-dir "$COLLECTED" \
  --split-manifest "$SPLIT" --annotations "$DATA_ROOT/annotations/all.json" \
  --output-dir "$REPORT_DIR"
```

The report includes full test rows, seed and adjacency estimates, learning-curve metadata, and reference-geometry estimates. The finalizer replays the core raw predictions against the released references, then scores the same predictions on 11 geometry variants. The fixed source envelope uses the median width and height of the source training labels. Its comparisons stay within the same source and destination. The geometry study uses the released object inventory and requires zero manual labeling. The automated provenance audit documents annotation-construction agreement and discrepancies. The geometry study leaves association accuracy and omitted vessels unresolved where independently synchronized reference positions are unavailable.

Run the automated annotation-provenance audit on the released inputs. The audit records coordinate and eligibility agreement with the implemented candidate extraction rules.

```sh
uv run --locked python -m src.revision.provenance_audit \
  --ann "$DATA_ROOT/annotations/all.json" --images "$DATA_ROOT/images" \
  --output "$REPORT_DIR/annotation_provenance.json"
```

The second-worker replay of `core-unet-DAAN-s0` loads the archived checkpoint into the resolved U-Net architecture with strict state-dictionary matching on CPU and runs one source-validation image. It then evaluates the archived raw predictions with the common evaluator on all three frozen target subsets. The reproducibility audit records exact-match status, content hashes, the runtime, and separate CPU evaluation times. The audit covers checkpoint load and frozen-prediction replay on a second host. The finalization marker `finalized.json` identifies checksummed reports tied to their input and implementation digests.
