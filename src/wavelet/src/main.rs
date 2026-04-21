use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use wavelet::config::{Backend, DetectorConfig};
use wavelet::cuda;
use wavelet::eval::{build_gt_index, evaluate_predictions_with_index, fast_ap50, Metrics};
use wavelet::image_io::{
    load_coco_dataset, load_gray_image, load_predictions, save_predictions, CircleMask,
    CocoDataset, CocoImageInfo, CocoPrediction, GrayImage,
};
use wavelet::matrix::{
    build_all_experiments, build_search_grid, filter_experiment, subset_image_ids, Experiment,
    GridSize, SearchConfig, StructuralKey,
};
use wavelet::{
    components_for_dilation, detect, detections_from_components, preprocess, preprocess_batch,
};

const CUDA_PREPROCESS_BATCH: usize = 32;

#[derive(Parser)]
#[command(name = "wavelet")]
#[command(about = "Native Rust/CUDA wavelet detector")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Eval(EvalArgs),
    Bench(BenchArgs),
    SweepBench(SweepBenchArgs),
    Matrix(MatrixArgs),
}

#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[arg(long)]
    limit: Option<usize>,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Args)]
struct EvalArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    pred: PathBuf,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value_t = 200)]
    count: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    repeat: usize,
    #[arg(long)]
    preload: bool,
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Args)]
struct SweepBenchArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value_t = 64)]
    count: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    repeat: usize,
    #[arg(long)]
    preload: bool,
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[arg(long, value_enum, default_value_t = GridSize::Small)]
    grid: GridSize,
}

#[derive(Args)]
struct MatrixArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value = "results/wavelet")]
    out_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[arg(long, value_enum, default_value_t = GridSize::Small)]
    grid: GridSize,
    #[arg(long)]
    only: Option<String>,
    #[arg(long)]
    resume: bool,
    #[arg(long)]
    preload: bool,
}

#[derive(Args, Clone)]
struct ConfigArgs {
    #[arg(long, default_value_t = 3)]
    level: usize,
    #[arg(long, default_value_t = 6.0)]
    threshold_sigma: f32,
    #[arg(long, default_value_t = 30)]
    min_area: u32,
    #[arg(long, default_value_t = 3_000)]
    max_area: u32,
    #[arg(long, default_value_t = 0)]
    dilation: usize,
    #[arg(long, default_value_t = 0)]
    box_padding: i32,
}

#[derive(Clone, Copy)]
struct BenchSummary {
    images: usize,
    ms_per_image: f64,
}

#[derive(Clone, Copy)]
enum SweepVariant {
    Naive,
    Grouped,
}

#[derive(Clone, Debug, Serialize)]
struct MatrixRow {
    exp_id: String,
    condition: String,
    method: String,
    tune_set: String,
    val_set: String,
    test_set: String,
    seed: u64,
    win_level: usize,
    win_threshold_sigma: f32,
    win_min_area: u32,
    win_max_area: u32,
    win_dilation: usize,
    win_box_padding: i32,
    tune_ap50: f32,
    val_ap50: f32,
    test_ap: f32,
    test_ap50: f32,
    test_ap75: f32,
    test_recall_at_p90: f32,
    test_fa_per_image: f32,
    total_runtime_sec: f32,
}

impl From<ConfigArgs> for DetectorConfig {
    fn from(args: ConfigArgs) -> Self {
        DetectorConfig {
            level: args.level,
            threshold_sigma: args.threshold_sigma,
            min_area: args.min_area,
            max_area: args.max_area,
            dilation: args.dilation,
            box_padding: args.box_padding,
            ..DetectorConfig::default()
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run(args),
        Commands::Eval(args) => eval_predictions(args),
        Commands::Bench(args) => bench(args),
        Commands::SweepBench(args) => sweep_bench(args),
        Commands::Matrix(args) => run_matrix(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let infos = slice_infos(&dataset.images, args.limit);
    let config = DetectorConfig::from(args.config);
    let mut predictions = Vec::<CocoPrediction>::new();
    let mut circle_mask = None::<CircleMask>;

    for info in infos {
        let image = load_gray_image(&args.images.join(&info.file_name))?;
        let mask = circle_mask
            .get_or_insert_with(|| CircleMask::new(image.width, image.height, config.circle));
        let output = detect(&image, mask, &config, args.backend)?;
        predictions.extend(format_predictions(info.id, &output.detections));
    }

    save_predictions(&args.out, &predictions)?;
    let gt_index = build_gt_index(&dataset);
    let image_ids = infos.iter().map(|info| info.id).collect::<Vec<_>>();
    let metrics = evaluate_predictions_with_index(&predictions, &gt_index, &image_ids);
    println!(
        "saved {} predictions to {}",
        predictions.len(),
        args.out.display()
    );
    print_metrics(&metrics);
    Ok(())
}

fn eval_predictions(args: EvalArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let predictions = load_predictions(&args.pred)?;
    let gt_index = build_gt_index(&dataset);
    let image_ids = dataset
        .images
        .iter()
        .map(|info| info.id)
        .collect::<Vec<_>>();
    let metrics = evaluate_predictions_with_index(&predictions, &gt_index, &image_ids);
    print_metrics(&metrics);
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let infos = slice_infos(&dataset.images, Some(args.count));
    let config = DetectorConfig::from(args.config);
    let summary = benchmark_native(
        &args.images,
        infos,
        &config,
        args.backend,
        args.warmup,
        args.repeat,
        args.preload,
    )?;
    println!(
        "native {:?}{}: {:.3} ms/image over {} images (best of {})",
        args.backend,
        if args.preload { " preload" } else { "" },
        summary.ms_per_image,
        summary.images,
        args.repeat,
    );
    if args.backend == Backend::Cuda {
        println!("cuda available: {}", cuda::is_available());
    }
    Ok(())
}

fn sweep_bench(args: SweepBenchArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let count = args.count.min(dataset.images.len());
    if count == 0 {
        return Err(anyhow!("sweep benchmark image set is empty"));
    }

    let indices = (0..count).collect::<Vec<_>>();
    let search_grid = build_search_grid(args.grid);
    let preloaded = if args.preload {
        Some(preload_images(&args.images, &dataset.images)?)
    } else {
        None
    };

    let naive = benchmark_sweep_variant(
        &dataset,
        &args.images,
        preloaded.as_deref(),
        &indices,
        &search_grid,
        args.backend,
        args.warmup,
        args.repeat,
        SweepVariant::Naive,
    )?;
    let grouped = benchmark_sweep_variant(
        &dataset,
        &args.images,
        preloaded.as_deref(),
        &indices,
        &search_grid,
        args.backend,
        args.warmup,
        args.repeat,
        SweepVariant::Grouped,
    )?;

    println!(
        "sweep {:?} {:?}{} over {} images and {} configs",
        args.backend,
        args.grid,
        if args.preload { " preload" } else { "" },
        count,
        search_grid.len()
    );
    println!("naive:   {:.3} s total", naive);
    println!("grouped: {:.3} s total", grouped);
    if grouped > 0.0 {
        println!("speedup: {:.2}x", naive / grouped);
    }
    Ok(())
}

fn run_matrix(args: MatrixArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let gt_index = build_gt_index(&dataset);
    let search_grid = build_search_grid(args.grid);
    let mut experiments = build_all_experiments()
        .into_iter()
        .filter(|exp| filter_experiment(exp, args.only.as_deref()))
        .collect::<Vec<_>>();
    if experiments.is_empty() {
        return Err(anyhow!("no experiments matched the requested filter"));
    }

    let preloaded = if args.preload {
        Some(preload_images(&args.images, &dataset.images)?)
    } else {
        None
    };

    let out_dir = args.out_dir;
    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let per_exp_dir = out_dir.join("per_experiment");
    std::fs::create_dir_all(&per_exp_dir)
        .with_context(|| format!("create {}", per_exp_dir.display()))?;

    let mut completed = std::collections::HashSet::<String>::new();
    if args.resume {
        for entry in std::fs::read_dir(&per_exp_dir)? {
            let entry = entry?;
            if let Some(stem) = entry.path().file_stem().and_then(|name| name.to_str()) {
                completed.insert(stem.to_string());
            }
        }
    }
    experiments.retain(|exp| !completed.contains(&exp.exp_id));
    if experiments.is_empty() {
        println!("[done] nothing to run");
        return Ok(());
    }

    let csv_path = out_dir.join("matrix.csv");
    let csv_mode_append = args.resume && csv_path.exists();
    let csv_file = std::fs::OpenOptions::new()
        .create(true)
        .append(csv_mode_append)
        .write(true)
        .truncate(!csv_mode_append)
        .open(&csv_path)
        .with_context(|| format!("open {}", csv_path.display()))?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(!csv_mode_append)
        .from_writer(csv_file);

    for exp in experiments {
        println!(
            "\n=== {}  [{}]  {} -> {} -> {} ===",
            exp.exp_id, exp.condition, exp.tune_label, exp.val_label, exp.test_label
        );
        let row = run_experiment(
            &exp,
            &dataset,
            &args.images,
            preloaded.as_deref(),
            args.backend,
            &gt_index,
            &search_grid,
        )?;
        writer.serialize(&row)?;
        writer.flush()?;
        let payload = serde_json::to_vec_pretty(&row)?;
        std::fs::write(per_exp_dir.join(format!("{}.json", exp.exp_id)), payload)?;
        println!(
            "  winner: level={} sigma={:.2} min_area={} max_area={} dilation={} box_padding={}",
            row.win_level,
            row.win_threshold_sigma,
            row.win_min_area,
            row.win_max_area,
            row.win_dilation,
            row.win_box_padding,
        );
        println!(
            "  test AP50={:.4}  AP={:.4}  FA/img={:.2}  ({:.1}s)",
            row.test_ap50, row.test_ap, row.test_fa_per_image, row.total_runtime_sec
        );
    }

    println!("\n[done] wrote {}", csv_path.display());
    Ok(())
}

fn run_experiment(
    exp: &Experiment,
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    backend: Backend,
    gt_index: &wavelet::eval::GtIndex,
    search_grid: &[SearchConfig],
) -> Result<MatrixRow> {
    let (tune_idx, val_idx, test_idx) = exp.split(dataset)?;
    let tune_ids = subset_image_ids(dataset, &tune_idx);
    let val_ids = subset_image_ids(dataset, &val_idx);
    let test_ids = subset_image_ids(dataset, &test_idx);
    let t0 = Instant::now();

    let mut scored = Vec::<(f32, SearchConfig)>::with_capacity(search_grid.len());
    for (_, configs) in group_configs_by_structural_key(search_grid) {
        let predictions =
            predict_configs_grouped(dataset, image_dir, preloaded, &tune_idx, &configs, backend)?;
        for (predictions, config) in predictions.into_iter().zip(configs.into_iter()) {
            let ap50 = fast_ap50(&predictions, gt_index, &tune_ids);
            scored.push((ap50, config));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = scored.into_iter().take(5).collect::<Vec<_>>();
    if top_k.is_empty() {
        return Err(anyhow!("search grid produced no candidates"));
    }

    let mut best_cfg = None::<SearchConfig>;
    let mut best_tune_ap50 = 0.0f32;
    let mut best_val_ap50 = -1.0f32;
    for (_, grouped) in group_scored_configs_by_structural_key(&top_k) {
        let configs = grouped
            .iter()
            .map(|(_, config)| config.clone())
            .collect::<Vec<_>>();
        let predictions =
            predict_configs_grouped(dataset, image_dir, preloaded, &val_idx, &configs, backend)?;
        for ((tune_ap50, config), predictions) in grouped.into_iter().zip(predictions.into_iter()) {
            let ap50 = fast_ap50(&predictions, gt_index, &val_ids);
            if ap50 > best_val_ap50 {
                best_val_ap50 = ap50;
                best_tune_ap50 = tune_ap50;
                best_cfg = Some(config);
            }
        }
    }
    let best_cfg = best_cfg.ok_or_else(|| anyhow!("no validation winner"))?;

    let predictions = predict_subset(dataset, image_dir, preloaded, &test_idx, &best_cfg, backend)?;
    let metrics = evaluate_predictions_with_index(&predictions, gt_index, &test_ids);

    Ok(MatrixRow {
        exp_id: exp.exp_id.clone(),
        condition: exp.condition.clone(),
        method: exp.method.clone(),
        tune_set: exp.tune_label.clone(),
        val_set: exp.val_label.clone(),
        test_set: exp.test_label.clone(),
        seed: exp.seed,
        win_level: best_cfg.level,
        win_threshold_sigma: round4(best_cfg.threshold_sigma),
        win_min_area: best_cfg.min_area,
        win_max_area: best_cfg.max_area,
        win_dilation: best_cfg.dilation,
        win_box_padding: best_cfg.box_padding,
        tune_ap50: round4(best_tune_ap50),
        val_ap50: round4(best_val_ap50),
        test_ap: round4(metrics.ap),
        test_ap50: round4(metrics.ap50),
        test_ap75: round4(metrics.ap75),
        test_recall_at_p90: round4(metrics.recall_at_p90),
        test_fa_per_image: round2(metrics.fa_per_image),
        total_runtime_sec: round1(t0.elapsed().as_secs_f32()),
    })
}

fn benchmark_native(
    image_dir: &Path,
    infos: &[CocoImageInfo],
    config: &DetectorConfig,
    backend: Backend,
    warmup: usize,
    repeat: usize,
    preload: bool,
) -> Result<BenchSummary> {
    if infos.is_empty() {
        return Err(anyhow!("benchmark image set is empty"));
    }

    let preloaded = if preload {
        Some(
            infos
                .iter()
                .map(|info| load_gray_image(&image_dir.join(&info.file_name)))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    let mut timings = Vec::<std::time::Duration>::with_capacity(repeat);
    for round in 0..(warmup + repeat) {
        let mut circle_mask = None::<CircleMask>;
        let start = Instant::now();
        for (idx, info) in infos.iter().enumerate() {
            let owned;
            let image = if let Some(cache) = preloaded.as_ref() {
                &cache[idx]
            } else {
                owned = load_gray_image(&image_dir.join(&info.file_name))?;
                &owned
            };
            let mask = circle_mask
                .get_or_insert_with(|| CircleMask::new(image.width, image.height, config.circle));
            let _ = detect(image, mask, config, backend)?;
        }
        if round >= warmup {
            timings.push(start.elapsed());
        }
    }

    let best = timings
        .into_iter()
        .min()
        .ok_or_else(|| anyhow!("benchmark did not run"))?;
    Ok(BenchSummary {
        images: infos.len(),
        ms_per_image: best.as_secs_f64() * 1000.0 / infos.len() as f64,
    })
}

fn predict_subset(
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    search: &SearchConfig,
    backend: Backend,
) -> Result<Vec<CocoPrediction>> {
    Ok(predict_configs_grouped(
        dataset,
        image_dir,
        preloaded,
        indices,
        std::slice::from_ref(search),
        backend,
    )?
    .into_iter()
    .next()
    .unwrap_or_default())
}

fn predict_subset_naive(
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    search: &SearchConfig,
    backend: Backend,
) -> Result<Vec<CocoPrediction>> {
    let config = search.detector_config();
    let mut predictions = Vec::<CocoPrediction>::new();
    let mut circle_mask = None::<CircleMask>;

    for &idx in indices {
        let info = &dataset.images[idx];
        let owned;
        let image = if let Some(cache) = preloaded {
            &cache[idx]
        } else {
            owned = load_gray_image(&image_dir.join(&info.file_name))?;
            &owned
        };
        let mask = circle_mask
            .get_or_insert_with(|| CircleMask::new(image.width, image.height, config.circle));
        let output = detect(image, mask, &config, backend)?;
        predictions.extend(format_predictions(info.id, &output.detections));
    }

    Ok(predictions)
}

fn predict_configs_grouped(
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    configs: &[SearchConfig],
    backend: Backend,
) -> Result<Vec<Vec<CocoPrediction>>> {
    if configs.is_empty() {
        return Ok(Vec::new());
    }

    let structural_key = configs[0].structural_key();
    let preprocess_config = structural_key.detector_config();
    let full_configs = configs
        .iter()
        .map(SearchConfig::detector_config)
        .collect::<Vec<_>>();
    let mut grouped_by_dilation = HashMap::<usize, Vec<usize>>::new();
    for (cfg_idx, config) in configs.iter().enumerate() {
        if config.structural_key() != structural_key {
            return Err(anyhow!(
                "predict_configs_grouped received mixed structural keys"
            ));
        }
        grouped_by_dilation
            .entry(config.dilation)
            .or_default()
            .push(cfg_idx);
    }

    let mut predictions = vec![Vec::<CocoPrediction>::new(); configs.len()];
    let mut circle_mask = None::<CircleMask>;
    let batch_size = if backend == Backend::Cuda && preloaded.is_none() {
        CUDA_PREPROCESS_BATCH
    } else {
        1
    };
    for batch_indices in indices.chunks(batch_size) {
        let owned_images;
        let image_refs = if let Some(cache) = preloaded {
            batch_indices
                .iter()
                .map(|&idx| &cache[idx])
                .collect::<Vec<_>>()
        } else {
            owned_images = batch_indices
                .iter()
                .map(|&idx| load_gray_image(&image_dir.join(&dataset.images[idx].file_name)))
                .collect::<Result<Vec<_>>>()?;
            owned_images.iter().collect::<Vec<_>>()
        };
        let first = *image_refs
            .first()
            .ok_or_else(|| anyhow!("empty batch in grouped prediction"))?;
        let mask = circle_mask.get_or_insert_with(|| {
            CircleMask::new(first.width, first.height, preprocess_config.circle)
        });
        let preprocessed_batch = if backend == Backend::Cuda {
            preprocess_batch(&image_refs, mask, &preprocess_config, backend)?
        } else {
            image_refs
                .iter()
                .map(|image| preprocess(image, mask, &preprocess_config, backend))
                .collect::<Result<Vec<_>>>()?
        };

        for ((&idx, preprocessed), image) in batch_indices
            .iter()
            .zip(preprocessed_batch.into_iter())
            .zip(image_refs.iter())
        {
            let mut components_cache = HashMap::<usize, _>::new();
            for &dilation in grouped_by_dilation.keys() {
                components_cache.insert(dilation, components_for_dilation(&preprocessed, dilation));
            }

            for (&dilation, cfg_indices) in &grouped_by_dilation {
                let components = components_cache
                    .get(&dilation)
                    .ok_or_else(|| anyhow!("missing component cache for dilation {dilation}"))?;
                for &cfg_idx in cfg_indices {
                    let detections = detections_from_components(
                        components,
                        preprocessed.width,
                        preprocessed.height,
                        &full_configs[cfg_idx],
                    );
                    predictions[cfg_idx]
                        .extend(format_predictions(dataset.images[idx].id, &detections));
                }
            }
            let _ = image;
        }
    }

    Ok(predictions)
}

fn group_configs_by_structural_key(
    configs: &[SearchConfig],
) -> Vec<(StructuralKey, Vec<SearchConfig>)> {
    let mut groups = Vec::<(StructuralKey, Vec<SearchConfig>)>::new();
    let mut positions = HashMap::<StructuralKey, usize>::new();
    for config in configs {
        let key = config.structural_key();
        if let Some(&group_idx) = positions.get(&key) {
            groups[group_idx].1.push(config.clone());
        } else {
            positions.insert(key, groups.len());
            groups.push((key, vec![config.clone()]));
        }
    }
    groups
}

fn group_scored_configs_by_structural_key(
    configs: &[(f32, SearchConfig)],
) -> Vec<(StructuralKey, Vec<(f32, SearchConfig)>)> {
    let mut groups = Vec::<(StructuralKey, Vec<(f32, SearchConfig)>)>::new();
    let mut positions = HashMap::<StructuralKey, usize>::new();
    for (score, config) in configs {
        let key = config.structural_key();
        if let Some(&group_idx) = positions.get(&key) {
            groups[group_idx].1.push((*score, config.clone()));
        } else {
            positions.insert(key, groups.len());
            groups.push((key, vec![(*score, config.clone())]));
        }
    }
    groups
}

fn benchmark_sweep_variant(
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    search_grid: &[SearchConfig],
    backend: Backend,
    warmup: usize,
    repeat: usize,
    variant: SweepVariant,
) -> Result<f64> {
    let mut timings = Vec::<f64>::with_capacity(repeat);
    for round in 0..(warmup + repeat) {
        let start = Instant::now();
        match variant {
            SweepVariant::Naive => {
                for config in search_grid {
                    let _ = predict_subset_naive(
                        dataset, image_dir, preloaded, indices, config, backend,
                    )?;
                }
            }
            SweepVariant::Grouped => {
                for (_, configs) in group_configs_by_structural_key(search_grid) {
                    let _ = predict_configs_grouped(
                        dataset, image_dir, preloaded, indices, &configs, backend,
                    )?;
                }
            }
        }
        if round >= warmup {
            timings.push(start.elapsed().as_secs_f64());
        }
    }

    timings
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("sweep benchmark did not run"))
}

fn preload_images(image_dir: &Path, infos: &[CocoImageInfo]) -> Result<Vec<GrayImage>> {
    infos
        .iter()
        .map(|info| load_gray_image(&image_dir.join(&info.file_name)))
        .collect()
}

fn format_predictions(
    image_id: u64,
    detections: &[wavelet::detector::Detection],
) -> Vec<CocoPrediction> {
    detections
        .iter()
        .map(|detection| CocoPrediction {
            image_id,
            category_id: 1,
            bbox: [
                detection.bbox_xyxy[0] as f32,
                detection.bbox_xyxy[1] as f32,
                (detection.bbox_xyxy[2] - detection.bbox_xyxy[0]) as f32,
                (detection.bbox_xyxy[3] - detection.bbox_xyxy[1]) as f32,
            ],
            score: detection.score,
        })
        .collect()
}

fn print_metrics(metrics: &Metrics) {
    println!(
        "images={} preds={} AP={:.4} AP50={:.4} AP75={:.4} recall@P90={:.4} FA/img={:.2}",
        metrics.n_images,
        metrics.n_pred,
        metrics.ap,
        metrics.ap50,
        metrics.ap75,
        metrics.recall_at_p90,
        metrics.fa_per_image
    );
}

fn slice_infos(infos: &[CocoImageInfo], limit: Option<usize>) -> &[CocoImageInfo] {
    match limit {
        Some(limit) => &infos[..infos.len().min(limit)],
        None => infos,
    }
}

fn round4(value: f32) -> f32 {
    (value * 10_000.0).round() / 10_000.0
}

fn round2(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}

fn round1(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}
