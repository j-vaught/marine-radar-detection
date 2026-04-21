use anyhow::{anyhow, Result};
use std::ffi::c_void;

use crate::config::DetectorConfig;
use crate::detector::{DetectionCandidate, PreprocessOutput, Roi, ThresholdField};
use crate::image_io::{CircleMask, GrayImage};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawDetectionCandidate {
    bbox_xyxy: [u32; 4],
    area: u32,
    centroid_x: f32,
    centroid_y: f32,
    raw_score: f32,
}

#[derive(Debug)]
pub struct CudaPreprocessCache {
    handle: *mut c_void,
}

impl Drop for CudaPreprocessCache {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.handle.is_null() {
                ffi::otsu_cuda_destroy_preprocess_cache(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod ffi {
    use super::RawDetectionCandidate;
    use std::ffi::c_void;

    unsafe extern "C" {
        pub fn otsu_cuda_available() -> i32;
        pub fn otsu_cuda_preprocess_global(
            image_ptr: *const u8,
            circle_mask_ptr: *const u8,
            image_width: i32,
            image_height: i32,
            roi_x0: i32,
            roi_y0: i32,
            roi_width: i32,
            roi_height: i32,
            threshold_out: *mut u16,
            binary_out: *mut u8,
        ) -> i32;
        pub fn otsu_cuda_preprocess_ssotsu(
            image_ptr: *const u8,
            circle_mask_ptr: *const u8,
            image_width: i32,
            image_height: i32,
            roi_x0: i32,
            roi_y0: i32,
            roi_width: i32,
            roi_height: i32,
            block_size: i32,
            seed_threshold_factor: f32,
            region_grow_threshold: f32,
            threshold_map_out: *mut u16,
            binary_out: *mut u8,
        ) -> i32;
        pub fn otsu_cuda_postprocess_u8(
            roi_image_ptr: *const u8,
            binary_ptr: *const u8,
            threshold_map_ptr: *const u16,
            threshold_constant: u16,
            roi_width: i32,
            roi_height: i32,
            dilation: i32,
            min_area: u32,
            max_area: u32,
            detections_out: *mut RawDetectionCandidate,
            max_detections: i32,
            detections_count_out: *mut i32,
        ) -> i32;
        pub fn otsu_cuda_postprocess_device_u8(
            cache_handle: *mut c_void,
            threshold_constant: u16,
            dilation: i32,
            min_area: u32,
            max_area: u32,
            detections_out: *mut RawDetectionCandidate,
            max_detections: i32,
            detections_count_out: *mut i32,
        ) -> i32;
        pub fn otsu_cuda_create_preprocess_cache(
            roi_image_ptr: *const u8,
            binary_ptr: *const u8,
            threshold_map_ptr: *const u16,
            roi_width: i32,
            roi_height: i32,
            cache_out: *mut *mut c_void,
        ) -> i32;
        pub fn otsu_cuda_destroy_preprocess_cache(cache: *mut c_void);
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    unsafe {
        ffi::otsu_cuda_available() > 0
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn preprocess_otsu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
) -> Result<(Vec<u8>, ThresholdField)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = image;
        let _ = circle_mask;
        let _ = roi;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }
        let mut threshold = 1u16;
        let mut binary = vec![0u8; roi.width * roi.height];
        let status = unsafe {
            ffi::otsu_cuda_preprocess_global(
                image.pixels.as_ptr(),
                circle_mask.values().as_ptr(),
                image.width as i32,
                image.height as i32,
                roi.x0 as i32,
                roi.y0 as i32,
                roi.width as i32,
                roi.height as i32,
                &mut threshold,
                binary.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(anyhow!("CUDA global Otsu failed with status {status}"));
        }
        Ok((binary, ThresholdField::Constant(threshold.max(1))))
    }
}

pub fn preprocess_ssotsu(
    image: &GrayImage,
    circle_mask: &CircleMask,
    roi: Roi,
    config: &DetectorConfig,
) -> Result<(Vec<u8>, ThresholdField)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = image;
        let _ = circle_mask;
        let _ = roi;
        let _ = config;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }
        let mut threshold_map = vec![1u16; roi.width * roi.height];
        let mut binary = vec![0u8; roi.width * roi.height];
        let status = unsafe {
            ffi::otsu_cuda_preprocess_ssotsu(
                image.pixels.as_ptr(),
                circle_mask.values().as_ptr(),
                image.width as i32,
                image.height as i32,
                roi.x0 as i32,
                roi.y0 as i32,
                roi.width as i32,
                roi.height as i32,
                config.block_size as i32,
                config.seed_threshold_factor,
                config.region_grow_threshold,
                threshold_map.as_mut_ptr(),
                binary.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(anyhow!("CUDA SSOTSU failed with status {status}"));
        }
        Ok((binary, ThresholdField::Map(threshold_map)))
    }
}

pub fn postprocess_candidates(
    preprocessed: &PreprocessOutput,
    dilation: usize,
) -> Result<Vec<DetectionCandidate>> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = preprocessed;
        let _ = dilation;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }

        let max_detections = preprocessed.roi.width * preprocessed.roi.height;
        if max_detections == 0 {
            return Ok(Vec::new());
        }

        let (threshold_map_ptr, threshold_constant) = match &preprocessed.threshold_field {
            ThresholdField::Constant(value) => (std::ptr::null(), *value),
            ThresholdField::Map(values) => (values.as_ptr(), 1u16),
        };

        let mut raw = vec![RawDetectionCandidate::default(); max_detections];
        let mut count = 0i32;
        let status = unsafe {
            ffi::otsu_cuda_postprocess_u8(
                preprocessed.roi_image.as_ptr(),
                preprocessed.binary_mask.as_ptr(),
                threshold_map_ptr,
                threshold_constant,
                preprocessed.roi.width as i32,
                preprocessed.roi.height as i32,
                dilation as i32,
                0,
                (preprocessed.roi.width * preprocessed.roi.height) as u32,
                raw.as_mut_ptr(),
                max_detections as i32,
                &mut count,
            )
        };
        if status != 0 {
            return Err(anyhow!("CUDA postprocess failed with status {status}"));
        }
        raw.truncate(count.max(0) as usize);
        Ok(raw
            .into_iter()
            .map(|candidate| DetectionCandidate {
                bbox_xyxy: candidate.bbox_xyxy,
                area: candidate.area,
                centroid_x: candidate.centroid_x,
                centroid_y: candidate.centroid_y,
                orientation: 0.0,
                eccentricity: 0.0,
                raw_score: candidate.raw_score,
            })
            .collect())
    }
}

pub fn create_preprocess_cache(
    roi_image: &[u8],
    binary_mask: &[u8],
    threshold_field: &ThresholdField,
    roi_width: usize,
    roi_height: usize,
) -> Result<CudaPreprocessCache> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = roi_image;
        let _ = binary_mask;
        let _ = threshold_field;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if roi_image.len() != binary_mask.len() {
            return Err(anyhow!("ROI image and binary mask sizes differ"));
        }
        if roi_image.len() != roi_width * roi_height {
            return Err(anyhow!("ROI cache dimensions do not match buffer size"));
        }
        let threshold_map_ptr = match threshold_field {
            ThresholdField::Constant(_) => std::ptr::null(),
            ThresholdField::Map(values) => values.as_ptr(),
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            ffi::otsu_cuda_create_preprocess_cache(
                roi_image.as_ptr(),
                binary_mask.as_ptr(),
                threshold_map_ptr,
                roi_width as i32,
                roi_height as i32,
                &mut handle,
            )
        };
        if status != 0 {
            return Err(anyhow!(
                "CUDA preprocess cache creation failed with status {status}"
            ));
        }
        Ok(CudaPreprocessCache { handle })
    }
}

pub fn postprocess_candidates_cached(
    preprocessed: &PreprocessOutput,
    cache: &CudaPreprocessCache,
    dilation: usize,
) -> Result<Vec<DetectionCandidate>> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = preprocessed;
        let _ = cache;
        let _ = dilation;
        return Err(anyhow!("crate was built without the `cuda` feature"));
    }

    #[cfg(feature = "cuda")]
    {
        if !is_available() {
            return Err(anyhow!("no CUDA device available"));
        }
        let max_detections = preprocessed.roi.width * preprocessed.roi.height;
        if max_detections == 0 {
            return Ok(Vec::new());
        }
        let threshold_constant = match &preprocessed.threshold_field {
            ThresholdField::Constant(value) => *value,
            ThresholdField::Map(_) => 1u16,
        };

        let mut raw = vec![RawDetectionCandidate::default(); max_detections];
        let mut count = 0i32;
        let status = unsafe {
            ffi::otsu_cuda_postprocess_device_u8(
                cache.handle,
                threshold_constant,
                dilation as i32,
                0,
                (preprocessed.roi.width * preprocessed.roi.height) as u32,
                raw.as_mut_ptr(),
                max_detections as i32,
                &mut count,
            )
        };
        if status != 0 {
            return Err(anyhow!(
                "CUDA cached postprocess failed with status {status}"
            ));
        }
        raw.truncate(count.max(0) as usize);
        Ok(raw
            .into_iter()
            .map(|candidate| DetectionCandidate {
                bbox_xyxy: candidate.bbox_xyxy,
                area: candidate.area,
                centroid_x: candidate.centroid_x,
                centroid_y: candidate.centroid_y,
                orientation: 0.0,
                eccentricity: 0.0,
                raw_score: candidate.raw_score,
            })
            .collect())
    }
}
