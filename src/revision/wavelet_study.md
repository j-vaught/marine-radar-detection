# Wavelet source-only extension

Run only after a buffered core wavelet attempt has completed. Supply its original
attempt directory and the same full COCO annotations, image root, and split
manifest. Build the native wavelet CPU binary separately. Choose an output
directory that does not already exist.

```sh
python -m src.revision.wavelet_study \
  --parent /path/to/attempt --ann /path/to/all.json \
  --images /path/to/images --manifest /path/to/split_manifest.json \
  --binary /path/to/wavelet --out /path/to/new/study --threads 8
```

The parent receipt, completion, metrics, selected configuration, subset files,
image hashes, and frozen partition identities are checked first. At most eight
configurations are evaluated on the source validation subset. The parent is
index zero, followed by single-axis neighbors for level 1, sigma 9 and 10.5,
minimum area 100 and 150, and box padding 1 and 2. Duplicates are removed. The
selection record is written before any holdout evaluation, using AP50, AP, then
stable index. Candidate predictions, metrics, times, and hashes are retained.

Each held-out site is then evaluated with the frozen winner, and the parent raw
predictions are re-evaluated against the same annotations and checked against
receipt-bound metrics. The parent receipt binds the metrics file, not prediction
bytes; their hashes are recorded here and aggregate metric agreement is checked.
These results are a separate prespecified extension and do not replace the
original fixed-grid results.

Source validation component diagnostics compare each selected configuration
with its minimum area set to one and maximum area set to the largest frame's
pixel count. They report proposal count, best-IoU recall, and score ties. The
native detector quantizes component scores and normalizes them per image; this
study does not implement a continuous-score ablation. Total elapsed time and
per-candidate, component, and holdout execution times are recorded.

When coordinating multiple studies, pass `--selection-barrier /path/to/ready`
to make each process wait after writing its selection. Create the barrier file
only after all source selections have been recorded. Full COCO and held-out
metrics are decoded after the barrier. The initial hash checks scan their input
files as bytes, while selection uses only receipt-bound source validation data.
