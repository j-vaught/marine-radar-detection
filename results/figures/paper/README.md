# Paper Figure Pack

This directory contains the consolidated figure set for the classical and learned detector study.

To regenerate these figures from a clean clone, you need the ignored result inputs bundled by [paper/README.md](../../../paper/README.md). The figure script reads the method matrix CSVs, per-experiment JSON summaries, `results/tide/`, and selected experiment prediction payloads in addition to the tracked source code.

Regenerate with:

```bash
./.venv/bin/python tools/plot_paper_figures.py
```

## Core Comparison Figures

- `paper_ap50_by_condition_families.png`: mean AP50 for classical and learned methods across pooled (`P`), site (`S`), and transfer (`T`) settings.
- `paper_ap_by_condition_families.png`: mean AP across the same settings.
- `paper_ap75_by_condition_families.png`: mean AP75 across the same settings.
- `paper_method_mean_ap50.png`: mean AP50 over the full matrix, aggregated by method.
- `paper_method_mean_ap50_bottom_legend.png`: same plot with a bottom legend layout for paper composition.
- `paper_method_mean_ap50_daan_bottom_legend.png`: method mean AP50 restricted to `DAAN` test cases, with the same bottom legend layout.
- `paper_localization_retention.png`: AP75/AP50 retention, used as a compact localization-quality summary.
- `paper_method_mean_rank_by_condition.png`: mean AP50 rank within each scenario.
- `paper_pareto_ap50_runtime.png`: accuracy vs experiment runtime tradeoff.

## Generalization And Stability

- `paper_transfer_heatmaps_all_methods.png`: 3x3 transfer matrices for every method, averaged over tune/test site pairs.
- `paper_transfer_generalization_gap.png`: same-site transfer vs cross-site transfer AP50.
- `paper_transfer_target_site_breakdown.png`: transfer AP50 grouped by target test site.
- `paper_pooled_seed_variability.png`: pooled-split AP50 spread over seeds.
- `paper_site_permutation_ap50_families.png`: AP50 across all 6 site permutations.
- `paper_detection_volume_by_condition.png`: false alarms per image for classical methods and predictions per image for learned methods.

## Ablation / Winner-Selection Figures

- `paper_cfar_ca_winner_parameter_counts.png`
- `paper_cfar_os_winner_parameter_counts.png`
- `paper_otsu_winner_parameter_counts.png`
- `paper_ssotsu_winner_parameter_counts.png`
- `paper_wavelet_winner_parameter_counts.png`
- `paper_unet_winner_parameter_counts.png`

These plots summarize how often each tuned parameter value was selected as the final winner across the matrix.

## Failure Analysis Figures

- `paper_failure_hard_transfer_mvdc.png`: hard `MANV -> DARC` transfer examples comparing U-Net, Faster R-CNN, and YOLO26.
- `paper_failure_rtdetr_collapse_dcmv.png`: examples showing RT-DETR collapse on `DARC -> MANV`, contrasted against YOLO26.
- `failure_zoomed/`: panel-by-panel crops plus the small classical cache used to regenerate the zoom strips without rerunning every classical method.

## Tables

- `method_summary.csv`: compact per-method summary table across `ALL`, `P`, `S`, and `T`.
- `transfer_gap_summary.csv`: same-site vs cross-site transfer AP50 summary.

See [DISCUSSION.md](DISCUSSION.md) for manuscript-style interpretation.
