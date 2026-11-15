use anyhow::{anyhow, Result};
use rayon::prelude::*;

use crate::ccl::{extract_components, ComponentStat};
use crate::config::{Backend, DetectorConfig};
use crate::cuda;
use crate::image_io::{CircleMask, GrayImage};

const DB4_LOW: [f32; 8] = [
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171888114,
    -0.027983769416859854,
    0.6308807679298589,
    0.7148465705529157,
    0.2303778133088965,
];

const DB4_HIGH: [f32; 8] = [
    -0.2303778133088965,
    0.7148465705529157,
    -0.6308807679298589,
    -0.027983769416859854,
    0.18703481171888114,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
];

const SCORE_CLAMP: f32 = 8.0;

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox_xyxy: [u32; 4],
    pub score: f32,
    pub area: u32,
    pub centroid_x: f32,
    pub centroid_y: f32,
}

#[derive(Clone, Debug)]
pub struct PreprocessOutput {
    pub width: usize,
    pub height: usize,
    pub binary_mask: Vec<u8>,
    pub score_map: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DetectionOutput {
    pub width: usize,
    pub height: usize,
    pub binary_mask: Vec<u8>,
    pub score_map: Vec<u8>,
    pub detections: Vec<Detection>,
}

pub fn detect(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
    backend: Backend,
) -> Result<DetectionOutput> {
    let preprocessed = preprocess(image, circle_mask, config, backend)?;
    let detections = detections_from_preprocessed(&preprocessed, config);
    Ok(DetectionOutput {
        width: preprocessed.width,
        height: preprocessed.height,
        binary_mask: preprocessed.binary_mask,
        score_map: preprocessed.score_map,
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
        return Err(anyhow!("circle mask size does not match image size"));
    }

    let (binary_mask, score_map) = match backend {
        Backend::Cpu => preprocess_wavelet_cpu(image, circle_mask, config)?,
        Backend::Cuda => cuda::detect_mask_and_scores(image, circle_mask, config)?,
    };

    Ok(PreprocessOutput {
        width: image.width,
        height: image.height,
        binary_mask,
        score_map,
    })
}

pub fn preprocess_batch(
    images: &[&GrayImage],
    circle_mask: &CircleMask,
    config: &DetectorConfig,
    backend: Backend,
) -> Result<Vec<PreprocessOutput>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }

    match backend {
        Backend::Cpu => images
            .iter()
            .map(|image| preprocess(image, circle_mask, config, backend))
            .collect(),
        Backend::Cuda => {
            cuda::detect_batch_mask_and_scores(images, circle_mask, config).map(|items| {
                items
                    .into_iter()
                    .map(|(binary_mask, score_map)| PreprocessOutput {
                        width: images[0].width,
                        height: images[0].height,
                        binary_mask,
                        score_map,
                    })
                    .collect()
            })
        }
    }
}

pub fn detections_from_preprocessed(
    preprocessed: &PreprocessOutput,
    config: &DetectorConfig,
) -> Vec<Detection> {
    let components = components_for_dilation(preprocessed, config.dilation);
    detections_from_components(&components, preprocessed.width, preprocessed.height, config)
}

pub fn components_for_dilation(
    preprocessed: &PreprocessOutput,
    dilation: usize,
) -> Vec<ComponentStat> {
    let dilated;
    let binary_mask = if dilation > 0 {
        dilated = dilate_binary(
            &preprocessed.binary_mask,
            preprocessed.width,
            preprocessed.height,
            2 * dilation + 1,
            1,
        );
        dilated.as_slice()
    } else {
        preprocessed.binary_mask.as_slice()
    };

    extract_components::<false>(
        binary_mask,
        &preprocessed.score_map,
        preprocessed.width,
        preprocessed.height,
        preprocessed.width,
        0,
        0,
    )
}

pub fn detections_from_components(
    components: &[ComponentStat],
    width: usize,
    height: usize,
    config: &DetectorConfig,
) -> Vec<Detection> {
    let kept = components
        .iter()
        .filter(|component| keep_component(component, config))
        .collect::<Vec<_>>();
    let max_peak = kept
        .iter()
        .map(|component| component.peak)
        .max()
        .unwrap_or(0);
    kept.into_iter()
        .map(|component| Detection {
            bbox_xyxy: pad_bbox(
                component.bbox_xyxy,
                config.box_padding,
                width as u32,
                height as u32,
            ),
            score: normalize_score(component.peak, max_peak),
            area: component.area,
            centroid_x: component.centroid_x,
            centroid_y: component.centroid_y,
        })
        .collect()
}

fn preprocess_wavelet_cpu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let width = image.width;
    let height = image.height;
    let level = config.level.max(1);
    let scale = 1usize << level;
    let padded_width = round_up(width, scale);
    let padded_height = round_up(height, scale);

    let mut approx = vec![0.0f32; padded_width * padded_height];
    for y in 0..padded_height {
        let src_y = reflect_index(y as isize, height);
        for x in 0..padded_width {
            let src_x = reflect_index(x as isize, width);
            let src_idx = src_y * width + src_x;
            approx[y * padded_width + x] = if circle_mask.values()[src_idx] != 0 {
                image.pixels[src_idx] as f32 / 255.0
            } else {
                0.0
            };
        }
    }

    let mut score_accum = vec![0.0f32; padded_width * padded_height];
    let mut low_rows = vec![0.0f32; padded_width * padded_height];
    let mut high_rows = vec![0.0f32; padded_width * padded_height];
    let mut ll = vec![0.0f32; padded_width * padded_height];
    let mut lh = vec![0.0f32; padded_width * padded_height];
    let mut hl = vec![0.0f32; padded_width * padded_height];
    let mut hh = vec![0.0f32; padded_width * padded_height];

    for level_idx in 0..level {
        let stride = 1usize << level_idx;
        convolve_rows(
            &approx,
            &mut low_rows,
            padded_width,
            padded_height,
            &DB4_LOW,
            stride,
        );
        convolve_rows(
            &approx,
            &mut high_rows,
            padded_width,
            padded_height,
            &DB4_HIGH,
            stride,
        );

        convolve_cols(
            &low_rows,
            &mut ll,
            padded_width,
            padded_height,
            &DB4_LOW,
            stride,
        );
        convolve_cols(
            &low_rows,
            &mut lh,
            padded_width,
            padded_height,
            &DB4_HIGH,
            stride,
        );
        convolve_cols(
            &high_rows,
            &mut hl,
            padded_width,
            padded_height,
            &DB4_LOW,
            stride,
        );
        convolve_cols(
            &high_rows,
            &mut hh,
            padded_width,
            padded_height,
            &DB4_HIGH,
            stride,
        );

        let (mean, stddev) = response_stats(
            &lh,
            &hl,
            &hh,
            circle_mask.values(),
            &image.pixels,
            width,
            height,
            padded_width,
        );
        let threshold = mean + config.threshold_sigma * stddev.max(1.0e-6);
        if threshold > 0.0 {
            for y in 0..height {
                let row = y * padded_width;
                let src_row = y * width;
                for x in 0..width {
                    let idx = row + x;
                    let src_idx = src_row + x;
                    if circle_mask.values()[src_idx] == 0 || image.pixels[src_idx] == 0 {
                        continue;
                    }
                    let response = band_response(lh[idx], hl[idx], hh[idx]);
                    let ratio = response / threshold;
                    if ratio > score_accum[idx] {
                        score_accum[idx] = ratio;
                    }
                }
            }
        }

        std::mem::swap(&mut approx, &mut ll);
    }
    let mut binary = vec![0u8; width * height];
    let mut score = vec![0u8; width * height];
    for y in 0..height {
        let row = y * padded_width;
        let dst_row = y * width;
        for x in 0..width {
            let idx = row + x;
            let dst_idx = dst_row + x;
            if circle_mask.values()[dst_idx] == 0 || image.pixels[dst_idx] == 0 {
                continue;
            }
            let value = score_accum[idx];
            if value > 1.0 {
                binary[dst_idx] = 1;
            }
            score[dst_idx] = ((value.min(SCORE_CLAMP) / SCORE_CLAMP) * 255.0).round() as u8;
        }
    }

    Ok((binary, score))
}

fn response_stats(
    lh: &[f32],
    hl: &[f32],
    hh: &[f32],
    circle_mask: &[u8],
    image: &[u8],
    width: usize,
    height: usize,
    padded_width: usize,
) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..height {
        let row = y * padded_width;
        let src_row = y * width;
        for x in 0..width {
            let src_idx = src_row + x;
            if circle_mask[src_idx] == 0 || image[src_idx] == 0 {
                continue;
            }
            let idx = row + x;
            let response = band_response(lh[idx], hl[idx], hh[idx]) as f64;
            sum += response;
            sq_sum += response * response;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / count as f64;
    let variance = (sq_sum / count as f64 - mean * mean).max(0.0);
    (mean as f32, variance.sqrt() as f32)
}

#[inline(always)]
fn band_response(lh: f32, hl: f32, hh: f32) -> f32 {
    lh.abs().max(hl.abs()).max(0.5 * hh.abs())
}

fn convolve_rows(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    _height: usize,
    filter: &[f32],
    stride: usize,
) {
    let anchor = (filter.len() / 2) as isize;
    dst.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * width..(y + 1) * width];
        for (x, out) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &coeff) in filter.iter().enumerate() {
                let offset = (k as isize - anchor) * stride as isize;
                let sx = reflect_index(x as isize + offset, width);
                sum += coeff * src_row[sx];
            }
            *out = sum;
        }
    });
}

fn convolve_cols(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    filter: &[f32],
    stride: usize,
) {
    let anchor = (filter.len() / 2) as isize;
    dst.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, out) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &coeff) in filter.iter().enumerate() {
                let offset = (k as isize - anchor) * stride as isize;
                let sy = reflect_index(y as isize + offset, height);
                sum += coeff * src[sy * width + x];
            }
            *out = sum;
        }
    });
}

#[inline(always)]
fn reflect_index(mut idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let limit = len as isize;
    while idx < 0 || idx >= limit {
        if idx < 0 {
            idx = -idx - 1;
        }
        if idx >= limit {
            idx = 2 * limit - idx - 1;
        }
    }
    idx as usize
}

#[inline(always)]
fn round_up(value: usize, multiple: usize) -> usize {
    if multiple <= 1 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}

fn keep_component(component: &ComponentStat, config: &DetectorConfig) -> bool {
    component.area >= config.min_area && component.area <= config.max_area
}

fn normalize_score(score: u8, max_score: u8) -> f32 {
    if max_score == 0 {
        0.0
    } else {
        (score as f32 / max_score as f32).clamp(0.0, 1.0)
    }
}

fn pad_bbox(mut bbox: [u32; 4], padding: i32, width: u32, height: u32) -> [u32; 4] {
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
                for (dx, dy) in &offsets {
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

#[cfg(test)]
mod tests {
    use super::response_stats;

    #[test]
    fn stats_include_only_nonzero_pixels_in_circle() {
        let lh = [2.0, 100.0, 6.0, 200.0];
        let zeros = [0.0; 4];
        let (mean, stddev) =
            response_stats(&lh, &zeros, &zeros, &[1, 1, 1, 0], &[1, 0, 1, 1], 4, 1, 4);
        assert_eq!((mean, stddev), (4.0, 2.0));
        assert_eq!(
            response_stats(&lh, &zeros, &zeros, &[1; 4], &[0; 4], 4, 1, 4),
            (0.0, 0.0)
        );
    }
}
