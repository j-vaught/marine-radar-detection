use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use morphological::config::{Backend, DetectorConfig};
use morphological::image_io::{load_gray_image, CircleMask, CocoImageInfo, CocoImageList, CocoPrediction};
use morphological::{cuda, detect};

#[derive(Parser)]
#[command(name = "morphological")]
#[command(about = "Native Rust/CUDA morphology detector")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Eval(EvalArgs),
    Bench(BenchArgs),
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
    #[arg(long)]
    compare_python: bool,
    #[arg(long)]
    python: Option<PathBuf>,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Args)]
struct EvalArgs {
    #[arg(long)]
    ann: PathBuf,
    #[arg(long)]
    pred: Option<PathBuf>,
    #[arg(long)]
    python: Option<PathBuf>,
}

#[derive(Args, Clone)]
struct ConfigArgs {
    #[arg(long, default_value_t = 3)]
    opening_kernel_size: usize,
    #[arg(long, default_value_t = 1)]
    opening_iterations: usize,
    #[arg(long, default_value_t = 5)]
    closing_kernel_size: usize,
    #[arg(long, default_value_t = 1)]
    closing_iterations: usize,
    #[arg(long, default_value_t = 20)]
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

impl From<ConfigArgs> for DetectorConfig {
    fn from(args: ConfigArgs) -> Self {
        DetectorConfig {
            opening_kernel_size: args.opening_kernel_size,
            opening_iterations: args.opening_iterations,
            closing_kernel_size: args.closing_kernel_size,
            closing_iterations: args.closing_iterations,
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
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate layout should be <repo>/src/morphological")
        .to_path_buf()
}

fn default_predictions_path() -> PathBuf {
    repo_root().join("results/morphological/predictions.json")
}

fn default_python_path() -> PathBuf {
    repo_root().join(".venv/bin/python")
}

fn run(args: RunArgs) -> Result<()> {
    let config = DetectorConfig::from(args.config);
    let manifest = load_manifest(&args.ann)?;
    let images = slice_infos(&manifest.images, args.limit);
    let mut predictions = Vec::<CocoPrediction>::new();
    let mut circle_mask = None::<CircleMask>;
    let out_path = args.out.unwrap_or_else(default_predictions_path);

    for info in images {
        let image_path = args.images.join(&info.file_name);
        let image = load_gray_image(&image_path)?;
        let mask = circle_mask.get_or_insert_with(|| {
            CircleMask::new(image.width, image.height, config.circle)
        });
        let output = detect(&image, mask, &config, args.backend)?;
        predictions.extend(output.detections.into_iter().map(|detection| {
            CocoPrediction {
                image_id: info.id,
                category_id: 1,
                bbox: [
                    detection.bbox_xyxy[0] as f32,
                    detection.bbox_xyxy[1] as f32,
                    (detection.bbox_xyxy[2] - detection.bbox_xyxy[0]) as f32,
                    (detection.bbox_xyxy[3] - detection.bbox_xyxy[1]) as f32,
                ],
                score: detection.score,
            }
        }));
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
    }
    std::fs::write(&out_path, serde_json::to_vec_pretty(&predictions)?)?;
    println!("saved {} predictions to {}", predictions.len(), out_path.display());
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    let config = DetectorConfig::from(args.config.clone());
    let manifest = load_manifest(&args.ann)?;
    let infos = slice_infos(&manifest.images, Some(args.count));
    let rust = benchmark_native(
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
        rust.ms_per_image,
        rust.images,
        args.repeat
    );

    if args.compare_python {
        let python = args.python.unwrap_or_else(default_python_path);
        let python = benchmark_python(
            &python,
            &args.ann,
            &args.images,
            args.count,
            args.warmup,
            args.repeat,
            args.preload,
        )?;
        println!(
            "legacy python{}: {:.3} ms/image over {} images (best of {})",
            if args.preload { " preload" } else { "" },
            python.ms_per_image,
            python.images,
            args.repeat
        );
        println!("speedup: {:.2}x", python.ms_per_image / rust.ms_per_image);
    }

    if args.backend == Backend::Cuda {
        println!("cuda available: {}", cuda::is_available());
    }
    Ok(())
}

fn eval_predictions(args: EvalArgs) -> Result<()> {
    let pred_path = args.pred.unwrap_or_else(default_predictions_path);
    let python = args.python.unwrap_or_else(default_python_path);
    let script = r#"
from pathlib import Path
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann = Path(sys.argv[1])
pred = Path(sys.argv[2])
gt = COCO(str(ann))
dt = gt.loadRes(str(pred))
ev = COCOeval(gt, dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
"#;
    let status = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(&args.ann)
        .arg(&pred_path)
        .status()
        .with_context(|| format!("run COCO eval via {}", python.display()))?;
    if !status.success() {
        return Err(anyhow!("COCO eval command failed"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BenchSummary {
    images: usize,
    ms_per_image: f64,
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
    let preloaded = if preload {
        Some(
            infos.iter()
                .map(|info| load_gray_image(&image_dir.join(&info.file_name)))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let mut timings = Vec::with_capacity(repeat);
    for round in 0..(warmup + repeat) {
        let mut circle_mask = None::<CircleMask>;
        let start = Instant::now();
        for (idx, info) in infos.iter().enumerate() {
            let owned;
            let image = if let Some(images) = preloaded.as_ref() {
                &images[idx]
            } else {
                owned = load_gray_image(&image_dir.join(&info.file_name))?;
                &owned
            };
            let mask = circle_mask
                .get_or_insert_with(|| CircleMask::new(image.width, image.height, config.circle));
            let _ = detect(&image, mask, config, backend)?;
        }
        let elapsed = start.elapsed();
        if round >= warmup {
            timings.push(elapsed);
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

fn benchmark_python(
    python: &Path,
    ann: &Path,
    images: &Path,
    count: usize,
    warmup: usize,
    repeat: usize,
    preload: bool,
) -> Result<BenchSummary> {
    let script = format!(
        r#"
import json, sys, time
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage import measure

ann_path = Path(sys.argv[1])
img_dir = Path(sys.argv[2])
count = int(sys.argv[3])
warmup = int(sys.argv[4])
repeat = int(sys.argv[5])
preload = int(sys.argv[6])
CENTER=(531,515)
RADIUS=466

def create_mask(h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, CENTER, RADIUS, 255, -1)
    return m

def dilate_mask(mask, dilation=0):
    if dilation <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

def pad_bboxes(bboxes, padding=0, img_shape=None):
    if padding <= 0 or len(bboxes) == 0:
        return bboxes
    bboxes = bboxes.copy()
    bboxes[:,0] -= padding
    bboxes[:,1] -= padding
    bboxes[:,2] += padding
    bboxes[:,3] += padding
    if img_shape is not None:
        h, w = img_shape[:2]
        bboxes[:, [0,2]] = np.clip(bboxes[:, [0,2]], 0, w)
        bboxes[:, [1,3]] = np.clip(bboxes[:, [1,3]], 0, h)
    return bboxes

def binary_mask_to_bboxes_with_scores(mask, intensity_image, threshold_map=None, min_area=20, max_area=50000, dilation=0, box_padding=0):
    if dilation > 0:
        mask = dilate_mask(mask, dilation)
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels, intensity_image=intensity_image.astype(np.float32))
    bboxes = []
    raw_scores = []
    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue
        y1, x1, y2, x2 = region.bbox
        bboxes.append([x1, y1, x2, y2])
        peak = float(region.intensity_image.max())
        cy, cx = int(round(region.centroid[0])), int(round(region.centroid[1]))
        cy = np.clip(cy, 0, threshold_map.shape[0] - 1)
        cx = np.clip(cx, 0, threshold_map.shape[1] - 1)
        raw_scores.append(peak / (float(threshold_map[cy, cx]) + 1e-9))
    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    bboxes_arr = np.array(bboxes, dtype=np.float32)
    scores_arr = np.array(raw_scores, dtype=np.float32)
    s_max = scores_arr.max()
    if s_max > 0:
        scores_arr = np.clip(scores_arr / s_max, 0.0, 1.0)
    if box_padding > 0:
        bboxes_arr = pad_bboxes(bboxes_arr, box_padding, mask.shape)
    return bboxes_arr, scores_arr

def binary_mask_to_bboxes(mask, min_area=20, max_area=50000, dilation=0, box_padding=0):
    if dilation > 0:
        mask = dilate_mask(mask, dilation)
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels)
    bboxes = []
    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue
        y1, x1, y2, x2 = region.bbox
        bboxes.append([x1, y1, x2, y2])
    result = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    if box_padding > 0 and len(result) > 0:
        result = pad_bboxes(result, box_padding, mask.shape)
    return result

def detect(image):
    img_u8 = (image * 255).astype(np.uint8)
    nonzero = img_u8[img_u8 > 0]
    if len(nonzero) < 100:
        return np.zeros((0,4), dtype=np.float32)
    thresh = threshold_otsu(nonzero)
    binary = ((img_u8 > thresh) & (img_u8 > 0)).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    threshold_map = np.full(closed.shape, float(thresh), dtype=np.float32)
    _ = binary_mask_to_bboxes_with_scores(closed, img_u8.astype(np.float32), threshold_map=threshold_map)
    return binary_mask_to_bboxes(closed)

ann = json.loads(ann_path.read_text())
infos = ann["images"][:count]
mask = None
preloaded = []
if preload:
    for info in infos:
        img = cv2.imread(str(img_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = create_mask(img.shape[0], img.shape[1])
        preloaded.append(cv2.bitwise_and(img, mask).astype(np.float32) / 255.0)
times = []
for round_idx in range(warmup + repeat):
    t0 = time.time()
    if preload:
        for imgf in preloaded:
            _ = detect(imgf)
    else:
        for info in infos:
            img = cv2.imread(str(img_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = create_mask(img.shape[0], img.shape[1])
            img = cv2.bitwise_and(img, mask)
            _ = detect(img.astype(np.float32) / 255.0)
    dt = time.time() - t0
    if round_idx >= warmup:
        times.append(dt)
best = min(times)
print(best * 1000.0 / len(infos))
"#
    );
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(ann)
        .arg(images)
        .arg(count.to_string())
        .arg(warmup.to_string())
        .arg(repeat.to_string())
        .arg(u8::from(preload).to_string())
        .output()
        .with_context(|| format!("run python benchmark via {}", python.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "python benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)?;
    let ms_per_image = stdout
        .trim()
        .parse::<f64>()
        .with_context(|| format!("parse python benchmark output: {stdout}"))?;
    Ok(BenchSummary {
        images: count,
        ms_per_image,
    })
}

fn load_manifest(path: &Path) -> Result<CocoImageList> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?)
}

fn slice_infos(infos: &[CocoImageInfo], limit: Option<usize>) -> &[CocoImageInfo] {
    match limit {
        Some(n) => &infos[..infos.len().min(n)],
        None => infos,
    }
}
