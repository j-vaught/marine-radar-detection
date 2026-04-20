use anyhow::{anyhow, Result};
use rayon::prelude::*;

use crate::ccl::{extract_components, ComponentStat};
use crate::config::{Backend, DetectorConfig};
use crate::cuda;
use crate::image_io::{CircleMask, GrayImage};

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
pub struct DetectionOutput {
    pub threshold: u8,
    pub width: usize,
    pub height: usize,
    pub binary_mask: Vec<u8>,
    pub detections: Vec<Detection>,
}

#[derive(Clone, Copy, Debug)]
struct Roi {
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
}

pub fn detect(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
    backend: Backend,
) -> Result<DetectionOutput> {
    if image.width == 0 || image.height == 0 {
        return Err(anyhow!("image must be non-empty"));
    }
    if image.pixels.len() != image.width * image.height {
        return Err(anyhow!("image pixel buffer has invalid size"));
    }

    let roi = circle_roi(image.width, image.height, config.circle);

    let (threshold, mut binary) = match backend {
        Backend::Cpu => preprocess_mask_cpu(image, circle_mask, roi, config)?,
        Backend::Cuda => {
            let (threshold, full_binary) = cuda::preprocess_mask(image, circle_mask, config)?;
            (threshold, crop_binary_roi(&full_binary, image.width, roi))
        }
    };

    if config.dilation > 0 {
        binary = dilate_binary(
            &binary,
            roi.width,
            roi.height,
            2 * config.dilation + 1,
            1,
        );
    }

    let compute_shape = config.min_eccentricity > 0.0 || config.max_eccentricity < 1.0;
    let mut components = if compute_shape {
        extract_components::<true>(
            &binary,
            &image.pixels,
            roi.width,
            roi.height,
            image.width,
            roi.x0,
            roi.y0,
        )
    } else {
        extract_components::<false>(
            &binary,
            &image.pixels,
            roi.width,
            roi.height,
            image.width,
            roi.x0,
            roi.y0,
        )
    };
    components.retain(|component| keep_component(component, config));

    let mut max_raw_score = 0.0f32;
    let mut raw_scores = Vec::with_capacity(components.len());
    for component in &components {
        let raw = if threshold == 0 {
            0.0
        } else {
            f32::from(component.peak) / f32::from(threshold)
        };
        max_raw_score = max_raw_score.max(raw);
        raw_scores.push(raw);
    }

    let detections = components
        .into_iter()
        .zip(raw_scores.into_iter())
        .map(|(component, raw_score)| Detection {
            bbox_xyxy: pad_bbox_and_offset(
                component.bbox_xyxy,
                roi,
                config.box_padding,
                image.width as u32,
                image.height as u32,
            ),
            score: normalize_score(raw_score, max_raw_score),
            area: component.area,
            centroid_x: component.centroid_x + roi.x0 as f32,
            centroid_y: component.centroid_y + roi.y0 as f32,
            orientation: component.orientation,
            eccentricity: component.eccentricity,
        })
        .collect();

    Ok(DetectionOutput {
        threshold,
        width: roi.width,
        height: roi.height,
        binary_mask: binary,
        detections,
    })
}

fn preprocess_mask_cpu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
    config: &DetectorConfig,
) -> Result<(u8, Vec<u8>)> {
    let threshold = otsu_nonzero_roi(&image.pixels, circle_mask.values(), image.width, roi)
        .ok_or_else(|| anyhow!("not enough non-zero pixels"))?;
    let mut binary = Vec::with_capacity(roi.width * roi.height);
    for y in roi.y0..(roi.y0 + roi.height) {
        let row_start = y * image.width + roi.x0;
        let row_end = row_start + roi.width;
        let pixels = &image.pixels[row_start..row_end];
        let mask = &circle_mask.values()[row_start..row_end];
        binary.extend(
            pixels
                .iter()
                .zip(mask.iter())
                .map(|(&pixel, &keep)| u8::from(keep != 0 && pixel > 0 && pixel > threshold)),
        );
    }

    binary = open_binary(
        &binary,
        roi.width,
        roi.height,
        config.opening_kernel_size,
        config.opening_iterations,
    );

    binary = close_binary(
        &binary,
        roi.width,
        roi.height,
        config.closing_kernel_size,
        config.closing_iterations,
    );

    Ok((threshold, binary))
}

fn keep_component(component: &ComponentStat, config: &DetectorConfig) -> bool {
    component.area >= config.min_area
        && component.area <= config.max_area
        && component.eccentricity >= config.min_eccentricity
        && component.eccentricity <= config.max_eccentricity
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

pub fn ellipse_offsets(size: usize) -> Vec<(isize, isize)> {
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

fn open_binary(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
) -> Vec<u8> {
    let eroded = erode_binary(binary, width, height, kernel_size, iterations);
    dilate_binary(&eroded, width, height, kernel_size, iterations)
}

fn close_binary(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
) -> Vec<u8> {
    let dilated = dilate_binary(binary, width, height, kernel_size, iterations);
    erode_binary(&dilated, width, height, kernel_size, iterations)
}

fn erode_binary(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
) -> Vec<u8> {
    apply_binary_morph(binary, width, height, kernel_size, iterations, false)
}

fn dilate_binary(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
) -> Vec<u8> {
    apply_binary_morph(binary, width, height, kernel_size, iterations, true)
}

fn apply_binary_morph(
    binary: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    iterations: usize,
    dilate: bool,
) -> Vec<u8> {
    let mut current = binary.to_vec();
    if iterations == 0 {
        return current;
    }
    let offsets = ellipse_offsets(kernel_size);
    let len = current.len();
    let mut next = vec![0u8; len];
    for _ in 0..iterations {
        next.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, out_ref) in row.iter_mut().enumerate() {
                let mut out = u8::from(!dilate);
                for &(dx, dy) in &offsets {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let inside = nx >= 0 && ny >= 0 && nx < width as isize && ny < height as isize;
                    if dilate {
                        if inside && current[ny as usize * width + nx as usize] != 0 {
                            out = 1;
                            break;
                        }
                    } else if !inside || current[ny as usize * width + nx as usize] == 0 {
                        out = 0;
                        break;
                    }
                }
                *out_ref = out;
            }
        });
        std::mem::swap(&mut current, &mut next);
    }
    current
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

fn circle_roi(width: usize, height: usize, circle: crate::config::CircleSpec) -> Roi {
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

fn crop_binary_roi(binary: &[u8], full_width: usize, roi: Roi) -> Vec<u8> {
    let mut cropped = Vec::with_capacity(roi.width * roi.height);
    for y in roi.y0..(roi.y0 + roi.height) {
        let row_start = y * full_width + roi.x0;
        let row_end = row_start + roi.width;
        cropped.extend_from_slice(&binary[row_start..row_end]);
    }
    cropped
}

#[cfg(test)]
mod tests {
    use super::ellipse_offsets;

    #[test]
    fn ellipse_kernel_matches_opencv_shapes() {
        let k3 = ellipse_offsets(3);
        assert_eq!(k3.len(), 5);
        let k5 = ellipse_offsets(5);
        assert_eq!(k5.len(), 17);
        let k7 = ellipse_offsets(7);
        assert_eq!(k7.len(), 33);
    }
}
