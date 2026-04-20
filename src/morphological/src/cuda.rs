use anyhow::{anyhow, Result};

use crate::config::DetectorConfig;
use crate::image_io::{CircleMask, GrayImage};

#[cfg(feature = "cuda")]
mod ffi {
    unsafe extern "C" {
        pub fn morph_cuda_available() -> i32;
        pub fn morph_cuda_process_u8(
            image_ptr: *const u8,
            circle_mask_ptr: *const u8,
            width: i32,
            height: i32,
            opening_kernel_size: i32,
            opening_iterations: i32,
            closing_kernel_size: i32,
            closing_iterations: i32,
            threshold_out: *mut u8,
            binary_out: *mut u8,
        ) -> i32;
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    unsafe {
        ffi::morph_cuda_available() > 0
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn preprocess_mask(
    image: &GrayImage,
    circle_mask: &CircleMask,
    config: &DetectorConfig,
) -> Result<(u8, Vec<u8>)> {
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
        let mut threshold = 0u8;
        let mut binary = vec![0u8; image.pixels.len()];
        let status = unsafe {
            ffi::morph_cuda_process_u8(
                image.pixels.as_ptr(),
                circle_mask.values().as_ptr(),
                image.width as i32,
                image.height as i32,
                config.opening_kernel_size as i32,
                config.opening_iterations as i32,
                config.closing_kernel_size as i32,
                config.closing_iterations as i32,
                &mut threshold,
                binary.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(anyhow!("CUDA preprocessing failed with status {status}"));
        }
        Ok((threshold, binary))
    }
}
