use anyhow::{anyhow, Result};

use crate::config::DetectorConfig;
use crate::image_io::{CircleMask, GrayImage};

#[cfg(feature = "cuda")]
mod ffi {
    unsafe extern "C" {
        pub fn wavelet_cuda_available() -> i32;
        pub fn wavelet_cuda_process_u8(
            image_ptr: *const u8,
            circle_mask_ptr: *const u8,
            width: i32,
            height: i32,
            level: i32,
            threshold_sigma: f32,
            binary_out: *mut u8,
            score_out: *mut u8,
        ) -> i32;
        pub fn wavelet_cuda_process_batch_u8(
            images_ptr: *const u8,
            circle_mask_ptr: *const u8,
            batch: i32,
            width: i32,
            height: i32,
            level: i32,
            threshold_sigma: f32,
            binary_out: *mut u8,
            score_out: *mut u8,
        ) -> i32;
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    unsafe {
        ffi::wavelet_cuda_available() > 0
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn detect_mask_and_scores(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
) -> Result<(Vec<u8>, Vec<u8>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = image;
        let _ = circle_mask;
        let _ = config;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }

        let total = image.width * image.height;
        let mut binary = vec![0u8; total];
        let mut score = vec![0u8; total];
        let status = unsafe {
            ffi::wavelet_cuda_process_u8(
                image.pixels.as_ptr(),
                circle_mask.values().as_ptr(),
                image.width as i32,
                image.height as i32,
                config.level as i32,
                config.threshold_sigma,
                binary.as_mut_ptr(),
                score.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(anyhow!("CUDA wavelet detector failed with status {status}"));
        }
        Ok((binary, score))
    }
}

pub fn detect_batch_mask_and_scores(
    images: &[&GrayImage],
    circle_mask: &CircleMask,
    config: &DetectorConfig,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = images;
        let _ = circle_mask;
        let _ = config;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }

        let width = images[0].width;
        let height = images[0].height;
        let total = width * height;
        if circle_mask.values().len() != total {
            return Err(anyhow!("circle mask size does not match image size"));
        }
        for image in images {
            if image.width != width || image.height != height {
                return Err(anyhow!("batch images must have uniform shape"));
            }
            if image.pixels.len() != total {
                return Err(anyhow!("image pixel buffer has invalid size"));
            }
        }

        let batch = images.len();
        let mut packed_images = vec![0u8; batch * total];
        for (batch_idx, image) in images.iter().enumerate() {
            let start = batch_idx * total;
            packed_images[start..start + total].copy_from_slice(&image.pixels);
        }

        let mut packed_binary = vec![0u8; batch * total];
        let mut packed_score = vec![0u8; batch * total];
        let status = unsafe {
            ffi::wavelet_cuda_process_batch_u8(
                packed_images.as_ptr(),
                circle_mask.values().as_ptr(),
                batch as i32,
                width as i32,
                height as i32,
                config.level as i32,
                config.threshold_sigma,
                packed_binary.as_mut_ptr(),
                packed_score.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(anyhow!(
                "CUDA wavelet batch detector failed with status {status}"
            ));
        }

        let mut outputs = Vec::with_capacity(batch);
        for batch_idx in 0..batch {
            let start = batch_idx * total;
            outputs.push((
                packed_binary[start..start + total].to_vec(),
                packed_score[start..start + total].to_vec(),
            ));
        }
        Ok(outputs)
    }
}
