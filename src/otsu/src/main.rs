use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use otsu::config::{Backend, DetectorConfig, Method};
use otsu::cuda;
use otsu::detector::{
    candidates_from_preprocessed, detect, detections_from_candidates,
    detections_from_preprocessed, preprocess, DetectionCandidate, HeavyConfig,
    PreprocessOutput,
};
use otsu::eval::{build_gt_index, evaluate_predictions_with_index, fast_ap50, Metrics};
use otsu::image_io::{
    load_coco_dataset, load_gray_image, load_predictions, save_predictions, CircleMask,
    CocoDataset, CocoImageInfo, CocoPrediction, GrayImage,
};
use otsu::matrix::{
    build_all_experiments, build_search_grid, filter_experiment, subset_image_ids, Experiment,
    GridSize, SearchConfig,
};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "otsu")]
#[command(about = "Native Rust/CUDA Otsu and SSOTSU detector")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Eval(EvalArgs),
    Bench(BenchArgs),
    Matrix(MatrixArgs),
}

#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
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
    pred: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Method::Otsu)]
    method: Method,
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
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Args)]
struct MatrixArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value = "results/otsu")]
    out_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
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
    #[arg(long, value_enum, default_value_t = Method::Otsu)]
    method: Method,
    #[arg(long, default_value_t = 64)]
    block_size: usize,
    #[arg(long, default_value_t = 1.5)]
    seed_threshold_factor: f32,
    #[arg(long, default_value_t = 0.8)]
    region_grow_threshold: f32,
    #[arg(long, default_value_t = 10)]
    min_area: u32,
    #[arg(long, default_value_t = 50_000)]
    max_area: u32,
    #[arg(long, default_value_t = 0.0)]
    min_eccentricity: f32,
    #[arg(long, default_value_t = 1.0)]
    max_eccentricity: f32,
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

#[derive(Clone, Debug, Serialize)]
struct MatrixRow {
    exp_id: String,
    condition: String,
    method: String,
    tune_set: String,
    val_set: String,
    test_set: String,
    seed: u64,
    win_block_size: Option<usize>,
    win_seed_threshold_factor: Option<f32>,
    win_region_grow_threshold: Option<f32>,
    win_min_area: u32,
    win_max_area: u32,
    win_min_eccentricity: f32,
    win_max_eccentricity: f32,
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
            method: args.method,
            block_size: args.block_size,
            seed_threshold_factor: args.seed_threshold_factor,
            region_grow_threshold: args.region_grow_threshold,
            min_area: args.min_area,
            max_area: args.max_area,
            min_eccentricity: args.min_eccentricity,
            max_eccentricity: args.max_eccentricity,
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
        Commands::Matrix(args) => run_matrix(args),
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate layout should be <repo>/src/otsu")
        .to_path_buf()
}

fn default_predictions_path(method: Method) -> PathBuf {
    repo_root().join(format!("results/otsu/{}_predictions.json", method.slug()))
}

fn run(args: RunArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let infos = slice_infos(&dataset.images, args.limit);
    let config = DetectorConfig::from(args.config);
    let out_path = args
        .out
        .unwrap_or_else(|| default_predictions_path(config.method));
    let mut predictions = Vec::<CocoPrediction>::new();
    let mut circle_mask = None::<CircleMask>;

    for info in infos {
        let image = load_gray_image(&args.images.join(&info.file_name))?;
        let mask = circle_mask
            .get_or_insert_with(|| CircleMask::new(image.width, image.height, config.circle));
        let output = detect(&image, mask, &config, args.backend)?;
        predictions.extend(format_predictions(info.id, &output.detections));
    }

    save_predictions(&out_path, &predictions)?;
    let gt_index = build_gt_index(&dataset);
    let image_ids = infos.iter().map(|info| info.id).collect::<Vec<_>>();
    let metrics = evaluate_predictions_with_index(&predictions, &gt_index, &image_ids);
    println!(
        "saved {} predictions to {}",
        predictions.len(),
        out_path.display()
    );
    print_metrics(&metrics);
    Ok(())
}

fn eval_predictions(args: EvalArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let predictions = load_predictions(
        &args
            .pred
            .unwrap_or_else(|| default_predictions_path(args.method)),
    )?;
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
        "native {:?} {:?}{}: {:.3} ms/image over {} images (best of {})",
        config.method,
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

fn run_matrix(args: MatrixArgs) -> Result<()> {
    let dataset = load_coco_dataset(&args.ann)?;
    let gt_index = build_gt_index(&dataset);
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

    let mut completed = HashSet::<String>::new();
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
            "\n=== {}  [{}]  {}  {} -> {} -> {} ===",
            exp.exp_id,
            exp.condition,
            exp.method.label(),
            exp.tune_label,
            exp.val_label,
            exp.test_label
        );
        let search_grid = build_search_grid(exp.method, args.grid);
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
            "  winner: block={:?} seed={:?} grow={:?} min_area={} max_area={} dilation={} box_padding={}",
            row.win_block_size,
            row.win_seed_threshold_factor,
            row.win_region_grow_threshold,
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
    gt_index: &otsu::eval::GtIndex,
    search_grid: &[SearchConfig],
) -> Result<MatrixRow> {
    let (tune_idx, val_idx, test_idx) = exp.split(dataset)?;
    let tune_ids = subset_image_ids(dataset, &tune_idx);
    let val_ids = subset_image_ids(dataset, &val_idx);
    let test_ids = subset_image_ids(dataset, &test_idx);
    let t0 = Instant::now();

    let tune_ranked = score_configs(
        exp.method,
        dataset,
        image_dir,
        preloaded,
        &tune_idx,
        search_grid,
        backend,
        gt_index,
        &tune_ids,
    )?;
    let top_k = tune_ranked.into_iter().take(5).collect::<Vec<_>>();
    if top_k.is_empty() {
        return Err(anyhow!("search grid produced no candidates"));
    }

    let top_configs = top_k.iter().map(|(_, cfg)| cfg.clone()).collect::<Vec<_>>();
    let tune_lookup = top_k
        .iter()
        .map(|(score, cfg)| (config_key(exp.method, cfg), *score))
        .collect::<HashMap<_, _>>();
    let mut best_cfg = None::<SearchConfig>;
    let mut best_tune_ap50 = 0.0f32;
    let mut best_val_ap50 = -1.0f32;
    let val_ranked = score_configs(
        exp.method,
        dataset,
        image_dir,
        preloaded,
        &val_idx,
        &top_configs,
        backend,
        gt_index,
        &val_ids,
    )?;
    for (val_ap50, cfg) in val_ranked {
        if val_ap50 > best_val_ap50 {
            best_val_ap50 = val_ap50;
            best_tune_ap50 = *tune_lookup
                .get(&config_key(exp.method, &cfg))
                .ok_or_else(|| anyhow!("validation winner missing tune score"))?;
            best_cfg = Some(cfg.clone());
        }
    }
    let best_cfg = best_cfg.ok_or_else(|| anyhow!("no validation winner"))?;

    let predictions = predict_subset_cached(
        exp.method, dataset, image_dir, preloaded, &test_idx, &best_cfg, backend,
    )?;
    let metrics = evaluate_predictions_with_index(&predictions, gt_index, &test_ids);

    Ok(MatrixRow {
        exp_id: exp.exp_id.clone(),
        condition: exp.condition.clone(),
        method: exp.method.label().to_string(),
        tune_set: exp.tune_label.clone(),
        val_set: exp.val_label.clone(),
        test_set: exp.test_label.clone(),
        seed: exp.seed,
        win_block_size: best_cfg.block_size,
        win_seed_threshold_factor: best_cfg.seed_threshold_factor.map(round4),
        win_region_grow_threshold: best_cfg.region_grow_threshold.map(round4),
        win_min_area: best_cfg.min_area,
        win_max_area: best_cfg.max_area,
        win_min_eccentricity: round4(best_cfg.min_eccentricity),
        win_max_eccentricity: round4(best_cfg.max_eccentricity),
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

fn score_configs(
    method: Method,
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    configs: &[SearchConfig],
    backend: Backend,
    gt_index: &otsu::eval::GtIndex,
    image_ids: &[u64],
) -> Result<Vec<(f32, SearchConfig)>> {
    let mut grouped = HashMap::<HeavyConfig, Vec<SearchConfig>>::new();
    for config in configs {
        let heavy = HeavyConfig::from_detector(&config.detector_config(method));
        grouped.entry(heavy).or_default().push(config.clone());
    }

    let mut scored = Vec::<(f32, SearchConfig)>::with_capacity(configs.len());
    for group in grouped.into_values() {
        let preprocessed = preprocess_subset(
            dataset,
            image_dir,
            preloaded,
            indices,
            &group[0].detector_config(method),
            backend,
        )?;
        let mut candidate_cache = HashMap::<(usize, bool), Vec<Vec<DetectionCandidate>>>::new();
        for config in group {
            let detector_config = config.detector_config(method);
            let compute_shape = detector_config.min_eccentricity > 0.0
                || detector_config.max_eccentricity < 1.0;
            let cache_key = (detector_config.dilation, compute_shape);
            if !candidate_cache.contains_key(&cache_key) {
                let candidates = preprocessed
                    .iter()
                    .map(|output| {
                        candidates_from_preprocessed(
                            output,
                            &detector_config,
                            backend == Backend::Cuda,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                candidate_cache.insert(cache_key, candidates);
            }
            let predictions = predictions_from_candidate_subset(
                dataset,
                indices,
                &preprocessed,
                candidate_cache
                    .get(&cache_key)
                    .ok_or_else(|| anyhow!("candidate cache missing for {:?}", cache_key))?,
                &detector_config,
            );
            let ap50 = fast_ap50(&predictions, gt_index, image_ids);
            scored.push((ap50, config));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
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

fn predict_subset_cached(
    method: Method,
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    search: &SearchConfig,
    backend: Backend,
) -> Result<Vec<CocoPrediction>> {
    let config = search.detector_config(method);
    let preprocessed = preprocess_subset(dataset, image_dir, preloaded, indices, &config, backend)?;
    predictions_from_preprocessed_subset(dataset, indices, &preprocessed, &config, backend)
}

fn preprocess_subset(
    dataset: &CocoDataset,
    image_dir: &Path,
    preloaded: Option<&[GrayImage]>,
    indices: &[usize],
    config: &DetectorConfig,
    backend: Backend,
) -> Result<Vec<PreprocessOutput>> {
    let mut outputs = Vec::<PreprocessOutput>::with_capacity(indices.len());
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
        outputs.push(preprocess(image, mask, config, backend)?);
    }
    Ok(outputs)
}

fn predictions_from_preprocessed_subset(
    dataset: &CocoDataset,
    indices: &[usize],
    preprocessed: &[PreprocessOutput],
    config: &DetectorConfig,
    backend: Backend,
) -> Result<Vec<CocoPrediction>> {
    let mut predictions = Vec::<CocoPrediction>::new();
    for (slot, &idx) in indices.iter().enumerate() {
        let info = &dataset.images[idx];
        let detections =
            detections_from_preprocessed(&preprocessed[slot], config, backend == Backend::Cuda)?;
        predictions.extend(format_predictions(info.id, &detections));
    }
    Ok(predictions)
}

fn predictions_from_candidate_subset(
    dataset: &CocoDataset,
    indices: &[usize],
    preprocessed: &[PreprocessOutput],
    candidate_sets: &[Vec<DetectionCandidate>],
    config: &DetectorConfig,
) -> Vec<CocoPrediction> {
    let mut predictions = Vec::<CocoPrediction>::new();
    for (slot, &idx) in indices.iter().enumerate() {
        let info = &dataset.images[idx];
        let detections =
            detections_from_candidates(&candidate_sets[slot], &preprocessed[slot], config);
        predictions.extend(format_predictions(info.id, &detections));
    }
    predictions
}

fn preload_images(image_dir: &Path, infos: &[CocoImageInfo]) -> Result<Vec<GrayImage>> {
    infos
        .iter()
        .map(|info| load_gray_image(&image_dir.join(&info.file_name)))
        .collect()
}

fn format_predictions(
    image_id: u64,
    detections: &[otsu::detector::Detection],
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

fn config_key(method: Method, config: &SearchConfig) -> String {
    format!(
        "{}:{}:{:.3}:{:.3}:{}:{}:{:.3}:{:.3}:{}:{}",
        method.slug(),
        config.block_size.unwrap_or(0),
        config.seed_threshold_factor.unwrap_or(0.0),
        config.region_grow_threshold.unwrap_or(0.0),
        config.min_area,
        config.max_area,
        config.min_eccentricity,
        config.max_eccentricity,
        config.dilation,
        config.box_padding,
    )
}
