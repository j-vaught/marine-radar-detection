use std::path::Path;

use anyhow::{Context, Result};
use image::ImageReader;
use serde::{Deserialize, Serialize};

use crate::config::CircleSpec;

#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CircleMask {
    values: Vec<u8>,
}

impl CircleMask {
    pub fn new(width: usize, height: usize, spec: CircleSpec) -> Self {
        let mut values = vec![0u8; width * height];
        let cx = i64::from(spec.center_x);
        let cy = i64::from(spec.center_y);
        let radius_sq = i64::from(spec.radius) * i64::from(spec.radius);
        for y in 0..height {
            for x in 0..width {
                let dx = x as i64 - cx;
                let dy = y as i64 - cy;
                if dx * dx + dy * dy <= radius_sq {
                    values[y * width + x] = 1;
                }
            }
        }
        Self { values }
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn apply(&self, image: &mut GrayImage) {
        for (pixel, keep) in image.pixels.iter_mut().zip(self.values.iter()) {
            if *keep == 0 {
                *pixel = 0;
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CocoImageInfo {
    pub id: u64,
    pub file_name: String,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub dataset: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CocoImageList {
    pub images: Vec<CocoImageInfo>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CocoAnnotation {
    pub id: u64,
    pub image_id: u64,
    pub category_id: u32,
    pub bbox: [f32; 4],
    #[serde(default)]
    pub area: Option<f32>,
    #[serde(default)]
    pub iscrowd: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CocoCategory {
    pub id: u32,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CocoDataset {
    pub images: Vec<CocoImageInfo>,
    pub annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    pub categories: Vec<CocoCategory>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CocoPrediction {
    pub image_id: u64,
    pub category_id: u32,
    pub bbox: [f32; 4],
    pub score: f32,
}

pub fn load_gray_image(path: &Path) -> Result<GrayImage> {
    let image = ImageReader::open(path)
        .with_context(|| format!("open image {}", path.display()))?
        .decode()
        .with_context(|| format!("decode image {}", path.display()))?
        .to_luma8();
    let (width, height) = image.dimensions();
    Ok(GrayImage::new(
        width as usize,
        height as usize,
        image.into_raw(),
    ))
}

pub fn load_coco_dataset(path: &Path) -> Result<CocoDataset> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn load_predictions(path: &Path) -> Result<Vec<CocoPrediction>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn save_predictions(path: &Path, predictions: &[CocoPrediction]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let payload = serde_json::to_vec_pretty(predictions)?;
    std::fs::write(path, payload).with_context(|| format!("write {}", path.display()))
}
