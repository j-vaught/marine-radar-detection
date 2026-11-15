# Common-GPU inference timing

The study uses the selected DAAN seed-zero checkpoints and the first 32 source validation frames in the frozen manifest. Preserve the archive layout under the study directory's `inputs/runs` directory. Provide `metrics.json`, `completion.json`, and the selected `best.pt` for each run. Include `resolved_config.json` for Faster R-CNN and U-Net. Use YOLO attempt 2 and attempt 1 for the other methods. The module verifies the checkpoint and manifest digests and the corrected YOLO attempt's link to its parent weights.

Run on one idle RTX 6000 Ada GPU using the installed revision environment. Set `TORCH_HOME` to the staged revision cache containing the Faster R-CNN bootstrap weights. The original Faster R-CNN constructor uses those weights to build the matching normalization layers before strictly loading the selected radar checkpoint.

```sh
CUDA_VISIBLE_DEVICES="$timing_gpu_uuid" TORCH_HOME="$timing_torch_cache" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$timing_nvidia_wrapper" "$timing_python" -m src.revision.inference_timing \
  --gpu-uuid "$timing_gpu_uuid" --threads 2 --runs "$timing_root/inputs/runs" \
  --manifest "$timing_manifest" --ann "$timing_data/annotations/all.json" \
  --images "$timing_data/images" --output "$timing_root/timings.json"
```

The program records staged PNG digests and elapsed staging time. Each model loads once, runs one complete warmup pass, and runs three timed passes. The iterator visits the same PNGs in lexicographic filename order. Timed passes cover PNG decoding, native resizing, device copies, forward inference, native postprocessing, and creation of the in-memory raw detection list at batch size one. CUDA is synchronized immediately outside each pass. The report records checkpoint load time and the warmup separately. It also records finite-box and count checks, parameter dtypes, warmup and timed peak PyTorch allocations, driver and package versions, and observed GPU occupancy before and after each model. The raw floors are 0.0 for Faster R-CNN and 0.001 for YOLO26 and RT-DETR. U-Net uses the frozen source validation pixel threshold with connected-component box extraction, and retains all component scores. No target partition or accuracy metric enters the study.
