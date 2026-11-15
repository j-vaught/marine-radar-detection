#![cfg(feature = "cuda")]

use wavelet::{
    cuda, preprocess, preprocess_batch, Backend, CircleMask, CircleSpec, DetectorConfig, GrayImage,
};

#[test]
fn cpu_cuda_parity_for_sparse_constant_and_boundary_images() {
    assert!(cuda::is_available(), "CUDA parity test requires a GPU");
    for (width, height) in [(1023, 1024), (1024, 1024), (1050, 1024)] {
        let mask = CircleMask::new(
            width,
            height,
            CircleSpec {
                center_x: (width / 2) as i32,
                center_y: (height / 2) as i32,
                radius: 2000,
            },
        );
        let config = DetectorConfig {
            level: 1,
            threshold_sigma: 2.0,
            ..Default::default()
        };
        let mut sparse = vec![0; width * height];
        for &(x, y) in &[
            (0, 0),
            (width - 1, height - 1),
            (width / 2, height / 2),
            (width - 2, height - 2),
        ] {
            sparse[y * width + x] = 255;
        }
        let constant = vec![127; width * height];
        let mut patterned = vec![0; width * height];
        for (i, pixel) in patterned.iter_mut().enumerate() {
            if i % 7 != 0 {
                *pixel = ((i * 37 % 253) + 1) as u8;
            }
        }
        let images = [
            GrayImage::new(width, height, sparse),
            GrayImage::new(width, height, constant),
            GrayImage::new(width, height, patterned),
        ];
        let refs: Vec<_> = images.iter().collect();
        let batch = preprocess_batch(&refs, &mask, &config, Backend::Cuda).unwrap();
        for (image, gpu) in images.iter().zip(&batch) {
            let cpu = preprocess(image, &mask, &config, Backend::Cpu).unwrap();
            let single = preprocess(image, &mask, &config, Backend::Cuda).unwrap();
            assert_eq!(
                gpu.binary_mask, single.binary_mask,
                "batch mask {width}x{height}"
            );
            assert_eq!(
                gpu.score_map, single.score_map,
                "batch score {width}x{height}"
            );
            let score_max = cpu
                .score_map
                .iter()
                .zip(&gpu.score_map)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            let masks_differ = cpu
                .binary_mask
                .iter()
                .zip(&gpu.binary_mask)
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                score_max <= 1,
                "score difference {score_max} at {width}x{height}"
            );
            assert_eq!(masks_differ, 0, "mask differences at {width}x{height}");
        }
    }
}

#[test]
fn cuda_batch_clears_excluded_pixels_and_empty_image() {
    assert!(cuda::is_available(), "CUDA parity test requires a GPU");
    let (width, height) = (1050, 1024);
    let mask = CircleMask::new(
        width,
        height,
        CircleSpec {
            center_x: 531,
            center_y: 515,
            radius: 466,
        },
    );
    let config = DetectorConfig {
        level: 1,
        threshold_sigma: 2.0,
        ..Default::default()
    };
    let bright = GrayImage::new(width, height, vec![127; width * height]);
    let empty = GrayImage::new(width, height, vec![0; width * height]);
    let outputs = preprocess_batch(&[&bright, &empty], &mask, &config, Backend::Cuda).unwrap();
    for (index, &inside) in mask.values().iter().enumerate() {
        if inside == 0 {
            assert_eq!(outputs[0].binary_mask[index], 0);
            assert_eq!(outputs[0].score_map[index], 0);
        }
    }
    assert!(outputs[1].binary_mask.iter().all(|&v| v == 0));
    assert!(outputs[1].score_map.iter().all(|&v| v == 0));
    let cpu = preprocess(&bright, &mask, &config, Backend::Cpu).unwrap();
    assert_eq!(outputs[0].binary_mask, cpu.binary_mask);
    assert!(outputs[0]
        .score_map
        .iter()
        .zip(&cpu.score_map)
        .all(|(&a, &b)| a.abs_diff(b) <= 1));
}
