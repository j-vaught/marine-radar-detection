use otsu::config::{Backend, CircleSpec, DetectorConfig, Method};
use otsu::detect;
use otsu::image_io::{CircleMask, GrayImage};

fn full_mask(width: usize, height: usize) -> CircleMask {
    CircleMask::new(width, height, full_circle_spec(width, height))
}

fn full_circle_spec(width: usize, height: usize) -> CircleSpec {
    CircleSpec {
        center_x: (width / 2) as i32,
        center_y: (height / 2) as i32,
        radius: width.max(height) as i32,
    }
}

#[test]
fn cpu_otsu_detects_bright_blob() {
    let width = 16usize;
    let height = 16usize;
    let mut pixels = vec![10u8; width * height];
    for y in 4..8 {
        for x in 5..10 {
            pixels[y * width + x] = 200;
        }
    }
    let image = GrayImage::new(width, height, pixels);
    let mask = full_mask(width, height);
    let config = DetectorConfig {
        method: Method::Otsu,
        min_area: 4,
        max_area: 100,
        circle: full_circle_spec(width, height),
        ..DetectorConfig::default()
    };

    let output = detect(&image, &mask, &config, Backend::Cpu).unwrap();
    assert_eq!(output.detections.len(), 1);
    assert_eq!(output.detections[0].bbox_xyxy, [5, 4, 10, 8]);
    assert!(output.detections[0].score > 0.99);
}

#[test]
fn cpu_ssotsu_grows_connected_region() {
    let width = 32usize;
    let height = 32usize;
    let mut pixels = vec![0u8; width * height];
    for y in 8..24 {
        for x in 8..24 {
            pixels[y * width + x] = 20;
        }
    }
    for y in 13..19 {
        for x in 13..19 {
            pixels[y * width + x] = 120;
        }
    }
    for y in 15..17 {
        for x in 15..17 {
            pixels[y * width + x] = 220;
        }
    }
    let image = GrayImage::new(width, height, pixels);
    let mask = full_mask(width, height);
    let config = DetectorConfig {
        method: Method::Ssotsu,
        block_size: 16,
        seed_threshold_factor: 1.5,
        region_grow_threshold: 0.5,
        min_area: 8,
        max_area: 256,
        circle: full_circle_spec(width, height),
        ..DetectorConfig::default()
    };

    let output = detect(&image, &mask, &config, Backend::Cpu).unwrap();
    assert_eq!(output.detections.len(), 1);
    assert_eq!(output.detections[0].bbox_xyxy, [13, 13, 19, 19]);
    assert!(output.detections[0].score > 0.99);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu_on_synthetic_inputs() {
    if !cuda::is_available() {
        return;
    }

    let width = 32usize;
    let height = 32usize;
    let mut pixels = vec![8u8; width * height];
    for y in 7..15 {
        for x in 9..18 {
            pixels[y * width + x] = 170;
        }
    }
    for y in 18..25 {
        for x in 19..27 {
            pixels[y * width + x] = 230;
        }
    }
    let image = GrayImage::new(width, height, pixels);
    let mask = full_mask(width, height);

    for method in [Method::Otsu, Method::Ssotsu] {
        let config = DetectorConfig {
            method,
            block_size: 16,
            seed_threshold_factor: 1.4,
            region_grow_threshold: 0.6,
            min_area: 8,
            max_area: 512,
            circle: full_circle_spec(width, height),
            ..DetectorConfig::default()
        };
        let cpu = detect(&image, &mask, &config, Backend::Cpu).unwrap();
        let gpu = detect(&image, &mask, &config, Backend::Cuda).unwrap();
        assert_eq!(cpu.binary_mask, gpu.binary_mask);
        assert_eq!(cpu.detections.len(), gpu.detections.len());
        for (lhs, rhs) in cpu.detections.iter().zip(gpu.detections.iter()) {
            assert_eq!(lhs.bbox_xyxy, rhs.bbox_xyxy);
            assert!((lhs.score - rhs.score).abs() < 1.0e-4);
        }
    }
}
#[cfg(feature = "cuda")]
use otsu::cuda;
