use anyhow::{anyhow, Result};
use rayon::prelude::*;

use crate::ccl::{extract_components, ComponentStat};
use crate::config::{Backend, CircleSpec, DetectorConfig, Method};
use crate::cuda;
use crate::image_io::{CircleMask, GrayImage};

const REGION_UNSET: u8 = u8::MAX;

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox_xyxy: [u32; 4],
    pub score: f32,
    pub area: u32,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub orientation: f32,
    pub eccentricity: f32,
}

#[derive(Clone, Debug)]
pub struct DetectionCandidate {
    pub bbox_xyxy: [u32; 4],
    pub area: u32,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub orientation: f32,
    pub eccentricity: f32,
    pub raw_score: f32,
}

#[derive(Clone, Debug)]
pub enum ThresholdField {
    Constant(u16),
    Map(Vec<u16>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Roi {
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug)]
pub struct PreprocessOutput {
    pub image_width: usize,
    pub image_height: usize,
    pub roi: Roi,
    pub roi_image: Vec<u8>,
    pub binary_mask: Vec<u8>,
    pub threshold_field: ThresholdField,
    pub cuda_cache: Option<cuda::CudaPreprocessCache>,
}

#[derive(Clone, Debug)]
pub struct DetectionOutput {
    pub width: usize,
    pub height: usize,
    pub binary_mask: Vec<u8>,
    pub detections: Vec<Detection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HeavyConfig {
    Otsu,
    Ssotsu {
        block_size: usize,
        seed_threshold_milli: u16,
        region_grow_milli: u16,
    },
}

impl HeavyConfig {
    pub fn from_detector(config: &DetectorConfig) -> Self {
        match config.method {
            Method::Otsu => Self::Otsu,
            Method::Ssotsu => Self::Ssotsu {
                block_size: config.block_size,
                seed_threshold_milli: (config.seed_threshold_factor * 1000.0).round() as u16,
                region_grow_milli: (config.region_grow_threshold * 1000.0).round() as u16,
            },
        }
    }
}

pub fn detect(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
    backend: Backend,
) -> Result<DetectionOutput> {
    let preprocessed = preprocess(image, circle_mask, config, backend)?;
    let detections = detections_from_preprocessed(&preprocessed, config, backend == Backend::Cuda)?;
    Ok(DetectionOutput {
        width: preprocessed.roi.width,
        height: preprocessed.roi.height,
        binary_mask: preprocessed.binary_mask,
        detections,
    })
}

pub fn preprocess(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
    backend: Backend,
) -> Result<PreprocessOutput> {
    if image.width == 0 || image.height == 0 {
        return Err(anyhow!("image must be non-empty"));
    }
    if image.pixels.len() != image.width * image.height {
        return Err(anyhow!("image pixel buffer has invalid size"));
    }
    if circle_mask.values().len() != image.pixels.len() {
        return Err(anyhow!("circle mask size does not match image"));
    }

    let roi = circle_roi(image.width, image.height, config.circle);
    let roi_image = crop_roi(&image.pixels, image.width, roi);
    let (binary_mask, threshold_field) = match (config.method, backend) {
        (Method::Otsu, Backend::Cpu) => preprocess_otsu_cpu(image, circle_mask, roi),
        (Method::Otsu, Backend::Cuda) => cuda::preprocess_otsu(image, circle_mask, roi)?,
        (Method::Ssotsu, Backend::Cpu) => preprocess_ssotsu_cpu(image, circle_mask, roi, config),
        (Method::Ssotsu, Backend::Cuda) => {
            cuda::preprocess_ssotsu(image, circle_mask, roi, config)?
        }
    };

    let cuda_cache = if backend == Backend::Cuda {
        Some(cuda::create_preprocess_cache(
            &roi_image,
            &binary_mask,
            &threshold_field,
            roi.width,
            roi.height,
        )?)
    } else {
        None
    };

    Ok(PreprocessOutput {
        image_width: image.width,
        image_height: image.height,
        roi,
        roi_image,
        binary_mask,
        threshold_field,
        cuda_cache,
    })
}

pub fn detections_from_preprocessed(
    preprocessed: &PreprocessOutput,
    config: &DetectorConfig,
    prefer_cuda: bool,
) -> Result<Vec<Detection>> {
    let candidates = candidates_from_preprocessed(preprocessed, config, prefer_cuda)?;
    Ok(detections_from_candidates(&candidates, preprocessed, config))
}

pub fn candidates_from_preprocessed(
    preprocessed: &PreprocessOutput,
    config: &DetectorConfig,
    prefer_cuda: bool,
) -> Result<Vec<DetectionCandidate>> {
    let compute_shape = config.min_eccentricity > 0.0 || config.max_eccentricity < 1.0;
    Ok(if prefer_cuda && !compute_shape {
        let gpu_result = if let Some(cache) = preprocessed.cuda_cache.as_ref() {
            cuda::postprocess_candidates_cached(preprocessed, cache, config.dilation)
        } else {
            cuda::postprocess_candidates(preprocessed, config.dilation)
        };
        match gpu_result {
            Ok(candidates) => candidates,
            Err(_) => build_detection_candidates_cpu(preprocessed, config.dilation, compute_shape),
        }
    } else {
        build_detection_candidates_cpu(preprocessed, config.dilation, compute_shape)
    })
}

pub fn detections_from_candidates(
    candidates: &[DetectionCandidate],
    preprocessed: &PreprocessOutput,
    config: &DetectorConfig,
) -> Vec<Detection> {
    let filtered = candidates
        .iter()
        .filter(|candidate| keep_candidate(candidate, config))
        .cloned()
        .collect::<Vec<_>>();
    finalize_detections(
        filtered,
        preprocessed.roi,
        config.box_padding,
        preprocessed.image_width as u32,
        preprocessed.image_height as u32,
    )
}

fn build_detection_candidates_cpu(
    preprocessed: &PreprocessOutput,
    dilation: usize,
    compute_shape: bool,
) -> Vec<DetectionCandidate> {
    let components = components_for_dilation(preprocessed, dilation, compute_shape);
    components
        .into_iter()
        .map(|component| {
            let threshold = threshold_at_centroid(
                &preprocessed.threshold_field,
                preprocessed.roi.width,
                preprocessed.roi.height,
                component.centroid_x,
                component.centroid_y,
            )
            .max(1.0);
            DetectionCandidate {
                bbox_xyxy: component.bbox_xyxy,
                area: component.area,
                centroid_x: component.centroid_x,
                centroid_y: component.centroid_y,
                orientation: component.orientation,
                eccentricity: component.eccentricity,
                raw_score: component.peak as f32 / threshold,
            }
        })
        .collect()
}

fn finalize_detections(
    candidates: Vec<DetectionCandidate>,
    roi: Roi,
    box_padding: i32,
    image_width: u32,
    image_height: u32,
) -> Vec<Detection> {
    let max_raw_score = candidates
        .iter()
        .map(|candidate| candidate.raw_score)
        .fold(0.0f32, f32::max);
    let mut detections = candidates
        .into_iter()
        .map(|candidate| Detection {
            bbox_xyxy: pad_bbox_and_offset(
                candidate.bbox_xyxy,
                roi,
                box_padding,
                image_width,
                image_height,
            ),
            score: normalize_score(candidate.raw_score, max_raw_score),
            area: candidate.area,
            centroid_x: candidate.centroid_x + roi.x0 as f32,
            centroid_y: candidate.centroid_y + roi.y0 as f32,
            orientation: candidate.orientation,
            eccentricity: candidate.eccentricity,
        })
        .collect::<Vec<_>>();
    detections.sort_by(|a, b| {
        a.bbox_xyxy
            .cmp(&b.bbox_xyxy)
            .then_with(|| a.area.cmp(&b.area))
            .then_with(|| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    detections
}

pub fn components_for_dilation(
    preprocessed: &PreprocessOutput,
    dilation: usize,
    compute_shape: bool,
) -> Vec<ComponentStat> {
    let binary;
    let active_binary = if dilation > 0 {
        binary = dilate_binary(
            &preprocessed.binary_mask,
            preprocessed.roi.width,
            preprocessed.roi.height,
            2 * dilation + 1,
            1,
        );
        binary.as_slice()
    } else {
        preprocessed.binary_mask.as_slice()
    };

    if compute_shape {
        extract_components::<true>(
            active_binary,
            &preprocessed.roi_image,
            preprocessed.roi.width,
            preprocessed.roi.height,
            preprocessed.roi.width,
            0,
            0,
        )
    } else {
        extract_components::<false>(
            active_binary,
            &preprocessed.roi_image,
            preprocessed.roi.width,
            preprocessed.roi.height,
            preprocessed.roi.width,
            0,
            0,
        )
    }
}

fn preprocess_otsu_cpu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
) -> (Vec<u8>, ThresholdField) {
    let threshold = otsu_nonzero_roi(&image.pixels, circle_mask.values(), image.width, roi);
    let threshold_u16 = threshold.map_or(1u16, u16::from);
    let mut binary_mask = vec![0u8; roi.width * roi.height];

    if let Some(threshold) = threshold {
        for y in 0..roi.height {
            let src_row = (roi.y0 + y) * image.width + roi.x0;
            let dst_row = y * roi.width;
            for x in 0..roi.width {
                let src_idx = src_row + x;
                let pixel = image.pixels[src_idx];
                if circle_mask.values()[src_idx] != 0 && pixel > 0 && pixel > threshold {
                    binary_mask[dst_row + x] = 1;
                }
            }
        }
    }

    (binary_mask, ThresholdField::Constant(threshold_u16))
}

fn preprocess_ssotsu_cpu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
    config: &DetectorConfig,
) -> (Vec<u8>, ThresholdField) {
    let mut threshold_map = vec![1u16; roi.width * roi.height];
    let mut best_requirements = vec![REGION_UNSET; roi.width * roi.height];

    let block_size = config.block_size.max(1);
    for by in (0..roi.height).step_by(block_size) {
        let bh = (roi.height - by).min(block_size);
        for bx in (0..roi.width).step_by(block_size) {
            let bw = (roi.width - bx).min(block_size);
            let local_thresh = local_otsu_block(image, circle_mask, roi, bx, by, bw, bh);
            let seed_threshold = local_thresh
                .map(|value| (value as f32 * config.seed_threshold_factor).ceil() as u16)
                .unwrap_or(1)
                .max(1);

            fill_block_u16(
                &mut threshold_map,
                roi.width,
                bx,
                by,
                bw,
                bh,
                seed_threshold,
            );

            if local_thresh.is_none() {
                continue;
            }

            for y in 0..bh {
                let global_y = roi.y0 + by + y;
                let dst_row = (by + y) * roi.width;
                let src_row = global_y * image.width + roi.x0 + bx;
                for x in 0..bw {
                    let src_idx = src_row + x;
                    let pixel = image.pixels[src_idx];
                    if circle_mask.values()[src_idx] == 0
                        || pixel == 0
                        || pixel as u16 <= seed_threshold
                    {
                        continue;
                    }
                    let grow_req = grow_requirement(pixel, config.region_grow_threshold);
                    let dst_idx = dst_row + bx + x;
                    if grow_req < best_requirements[dst_idx] {
                        best_requirements[dst_idx] = grow_req;
                    }
                }
            }
        }
    }

    let binary_mask = region_grow_cpu(image, circle_mask, roi, &mut best_requirements);
    (binary_mask, ThresholdField::Map(threshold_map))
}

fn region_grow_cpu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
    best_requirements: &mut [u8],
) -> Vec<u8> {
    let mut binary_mask = vec![0u8; roi.width * roi.height];
    let mut stack = Vec::<usize>::new();
    for (idx, &requirement) in best_requirements.iter().enumerate() {
        if requirement != REGION_UNSET {
            binary_mask[idx] = 1;
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        let requirement = best_requirements[idx];
        let x = idx % roi.width;
        let y = idx / roi.width;
        let neighbors = [
            (x.wrapping_sub(1), y, x > 0),
            (x + 1, y, x + 1 < roi.width),
            (x, y.wrapping_sub(1), y > 0),
            (x, y + 1, y + 1 < roi.height),
        ];

        for (nx, ny, valid) in neighbors {
            if !valid {
                continue;
            }
            let nidx = ny * roi.width + nx;
            let src_idx = (roi.y0 + ny) * image.width + roi.x0 + nx;
            if circle_mask.values()[src_idx] == 0 {
                continue;
            }
            let pixel = image.pixels[src_idx];
            if pixel == 0 || pixel < requirement {
                continue;
            }
            if requirement < best_requirements[nidx] {
                best_requirements[nidx] = requirement;
                binary_mask[nidx] = 1;
                stack.push(nidx);
            }
        }
    }

    binary_mask
}

fn keep_candidate(candidate: &DetectionCandidate, config: &DetectorConfig) -> bool {
    candidate.area >= config.min_area
        && candidate.area <= config.max_area
        && candidate.eccentricity >= config.min_eccentricity
        && candidate.eccentricity <= config.max_eccentricity
}

fn threshold_at_centroid(
    field: &ThresholdField,
    width: usize,
    height: usize,
    centroid_x: f32,
    centroid_y: f32,
) -> f32 {
    match field {
        ThresholdField::Constant(value) => *value as f32,
        ThresholdField::Map(values) => {
            let x = centroid_x
                .round()
                .clamp(0.0, (width.saturating_sub(1)) as f32) as usize;
            let y = centroid_y
                .round()
                .clamp(0.0, (height.saturating_sub(1)) as f32) as usize;
            values[y * width + x] as f32
        }
    }
}

fn normalize_score(score: f32, max_score: f32) -> f32 {
    if max_score <= 0.0 {
        0.0
    } else {
        (score / max_score).clamp(0.0, 1.0)
    }
}

fn pad_bbox_and_offset(
    mut bbox: [u32; 4],
    roi: Roi,
    padding: i32,
    width: u32,
    height: u32,
) -> [u32; 4] {
    bbox[0] += roi.x0 as u32;
    bbox[1] += roi.y0 as u32;
    bbox[2] += roi.x0 as u32;
    bbox[3] += roi.y0 as u32;
    if padding <= 0 {
        return bbox;
    }
    let pad = padding as i64;
    bbox[0] = (i64::from(bbox[0]) - pad).clamp(0, i64::from(width)) as u32;
    bbox[1] = (i64::from(bbox[1]) - pad).clamp(0, i64::from(height)) as u32;
    bbox[2] = (i64::from(bbox[2]) + pad).clamp(0, i64::from(width)) as u32;
    bbox[3] = (i64::from(bbox[3]) + pad).clamp(0, i64::from(height)) as u32;
    bbox
}

fn local_otsu_block(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
) -> Option<u8> {
    let mut histogram = [0u32; 256];
    let mut count = 0u32;
    let mut sum = 0f64;
    let mut sq_sum = 0f64;

    for y in 0..bh {
        let global_y = roi.y0 + by + y;
        let src_row = global_y * image.width + roi.x0 + bx;
        for x in 0..bw {
            let src_idx = src_row + x;
            if circle_mask.values()[src_idx] == 0 {
                continue;
            }
            let pixel = image.pixels[src_idx];
            if pixel == 0 {
                continue;
            }
            histogram[pixel as usize] += 1;
            count += 1;
            let value = f64::from(pixel);
            sum += value;
            sq_sum += value * value;
        }
    }

    if count < 10 {
        return None;
    }
    let mean = sum / count as f64;
    let variance = (sq_sum / count as f64 - mean * mean).max(0.0);
    if variance.sqrt() < 1.0 {
        return None;
    }
    otsu_threshold_from_hist(&histogram, count)
}

fn fill_block_u16(
    values: &mut [u16],
    width: usize,
    x0: usize,
    y0: usize,
    block_w: usize,
    block_h: usize,
    value: u16,
) {
    for y in y0..(y0 + block_h) {
        let row = y * width + x0;
        for entry in &mut values[row..row + block_w] {
            *entry = value;
        }
    }
}

#[inline(always)]
fn grow_requirement(pixel: u8, region_grow_threshold: f32) -> u8 {
    (pixel as f32 * region_grow_threshold)
        .ceil()
        .clamp(1.0, 255.0) as u8
}

fn otsu_nonzero_roi(pixels: &[u8], circle_mask: &[u8], width: usize, roi: Roi) -> Option<u8> {
    let mut histogram = [0u32; 256];
    let mut count = 0u32;
    for y in roi.y0..(roi.y0 + roi.height) {
        let row_start = y * width + roi.x0;
        let row_end = row_start + roi.width;
        for (&pixel, &keep) in pixels[row_start..row_end]
            .iter()
            .zip(circle_mask[row_start..row_end].iter())
        {
            if keep != 0 && pixel > 0 {
                histogram[pixel as usize] += 1;
                count += 1;
            }
        }
    }
    if count < 100 {
        return None;
    }
    otsu_threshold_from_hist(&histogram, count)
}

fn otsu_threshold_from_hist(histogram: &[u32; 256], count: u32) -> Option<u8> {
    if count == 0 {
        return None;
    }
    let mut total_sum = 0f64;
    for (value, &freq) in histogram.iter().enumerate() {
        total_sum += value as f64 * f64::from(freq);
    }

    let mut background_weight = 0f64;
    let mut background_sum = 0f64;
    let mut best_threshold = 0usize;
    let mut best_variance = -1f64;
    let total_weight = f64::from(count);

    for (threshold, &freq) in histogram.iter().enumerate() {
        background_weight += f64::from(freq);
        if background_weight == 0.0 {
            continue;
        }
        let foreground_weight = total_weight - background_weight;
        if foreground_weight == 0.0 {
            break;
        }
        background_sum += threshold as f64 * f64::from(freq);
        let background_mean = background_sum / background_weight;
        let foreground_mean = (total_sum - background_sum) / foreground_weight;
        let mean_delta = background_mean - foreground_mean;
        let variance = background_weight * foreground_weight * mean_delta * mean_delta;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    Some(best_threshold as u8)
}

fn dilate_binary(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
) -> Vec<u8> {
    let offsets = ellipse_offsets(kernel_size);
    let mut current = binary.to_vec();
    let mut next = vec![0u8; binary.len()];
    for _ in 0..iterations {
        next.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut value = 0u8;
                for &(dx, dy) in &offsets {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx >= 0
                        && ny >= 0
                        && (nx as usize) < width
                        && (ny as usize) < height
                        && current[ny as usize * width + nx as usize] != 0
                    {
                        value = 1;
                        break;
                    }
                }
                *out = value;
            }
        });
        std::mem::swap(&mut current, &mut next);
    }
    current
}

fn crop_roi(pixels: &[u8], width: usize, roi: Roi) -> Vec<u8> {
    let mut cropped = Vec::with_capacity(roi.width * roi.height);
    for y in roi.y0..(roi.y0 + roi.height) {
        let row_start = y * width + roi.x0;
        let row_end = row_start + roi.width;
        cropped.extend_from_slice(&pixels[row_start..row_end]);
    }
    cropped
}

fn ellipse_offsets(size: usize) -> Vec<(isize, isize)> {
    if size <= 1 {
        return vec![(0, 0)];
    }
    let radius = (size / 2) as isize;
    let mut offsets = Vec::new();
    for dy in -radius..=radius {
        let dx = if radius == 0 {
            0
        } else {
            let radius_f = radius as f32;
            let dy_f = dy as f32;
            (radius_f * (1.0 - (dy_f * dy_f) / (radius_f * radius_f)).sqrt()).round() as isize
        };
        for dx_i in -dx..=dx {
            offsets.push((dx_i, dy));
        }
    }
    offsets
}

fn circle_roi(width: usize, height: usize, circle: CircleSpec) -> Roi {
    let x0 = (circle.center_x - circle.radius).max(0) as usize;
    let y0 = (circle.center_y - circle.radius).max(0) as usize;
    let x1 = ((circle.center_x + circle.radius + 1) as usize).min(width);
    let y1 = ((circle.center_y + circle.radius + 1) as usize).min(height);
    let bounded_x0 = x0.min(width.saturating_sub(1));
    let bounded_y0 = y0.min(height.saturating_sub(1));
    let bounded_x1 = x1.max(bounded_x0 + 1);
    let bounded_y1 = y1.max(bounded_y0 + 1);
    Roi {
        x0: bounded_x0,
        y0: bounded_y0,
        width: bounded_x1 - bounded_x0,
        height: bounded_y1 - bounded_y0,
    }
}

#[cfg(test)]
mod tests {
    use super::{ellipse_offsets, grow_requirement, otsu_threshold_from_hist};

    #[test]
    fn ellipse_kernel_matches_opencv_shapes() {
        let k3 = ellipse_offsets(3);
        assert_eq!(k3.len(), 5);
        let k5 = ellipse_offsets(5);
        assert_eq!(k5.len(), 17);
        let k7 = ellipse_offsets(7);
        assert_eq!(k7.len(), 33);
    }

    #[test]
    fn otsu_hist_returns_threshold() {
        let mut histogram = [0u32; 256];
        histogram[10] = 100;
        histogram[200] = 100;
        let threshold = otsu_threshold_from_hist(&histogram, 200).unwrap();
        assert!(threshold >= 10);
        assert!(threshold < 200);
    }

    #[test]
    fn grow_threshold_is_ceiled() {
        assert_eq!(grow_requirement(100, 0.8), 80);
        assert_eq!(grow_requirement(101, 0.8), 81);
    }
}
