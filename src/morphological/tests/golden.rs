use anyhow::Result;
use morphological::config::{Backend, DetectorConfig};
use morphological::image_io::{load_gray_image, CircleMask};
use morphological::detect;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/processed/images")
}

fn load_case(file_name: &str) -> Result<morphological::GrayImage> {
    load_gray_image(&data_dir().join(file_name))
}

fn assert_boxes(actual: &[[u32; 4]], expected: &[[u32; 4]]) {
    assert_eq!(actual, expected);
}

fn assert_scores(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (got, want) in actual.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "score mismatch: got {got}, want {want}");
    }
}

#[test]
fn cpu_matches_legacy_daans() -> Result<()> {
    let config = DetectorConfig::default();

    let cases = [
        (
            "DAAN_000.png",
            &[
                [534, 141, 551, 152],
                [417, 245, 435, 253],
                [508, 417, 515, 422],
                [525, 513, 533, 520],
                [780, 838, 817, 862],
            ][..],
            &[1.0, 1.0, 1.0, 1.0, 1.0][..],
        ),
        (
            "DAAN_001.png",
            &[
                [534, 142, 551, 152],
                [418, 245, 434, 253],
                [508, 417, 515, 422],
                [525, 513, 534, 520],
                [780, 837, 820, 862],
            ][..],
            &[1.0, 1.0, 0.9803922, 1.0, 1.0][..],
        ),
        (
            "DAAN_003.png",
            &[
                [535, 143, 550, 152],
                [419, 245, 436, 253],
                [525, 513, 533, 520],
                [899, 698, 905, 705],
                [783, 837, 821, 862],
            ][..],
            &[1.0, 1.0, 1.0, 0.76862746, 1.0][..],
        ),
    ];

    for (file_name, expected_boxes, expected_scores) in cases {
        let image = load_case(file_name)?;
        let circle_mask = CircleMask::new(image.width, image.height, config.circle);
        let output = detect(&image, &circle_mask, &config, Backend::Cpu)?;
        let actual_boxes = output
            .detections
            .iter()
            .map(|d| d.bbox_xyxy)
            .collect::<Vec<_>>();
        let actual_scores = output.detections.iter().map(|d| d.score).collect::<Vec<_>>();
        assert_boxes(&actual_boxes, expected_boxes);
        assert_scores(&actual_scores, expected_scores);
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu_on_smoke_image() -> Result<()> {
    if !morphological::cuda::is_available() {
        return Ok(());
    }
    let config = DetectorConfig::default();
    let image = load_case("DAAN_000.png")?;
    let circle_mask = CircleMask::new(image.width, image.height, config.circle);
    let cpu = detect(&image, &circle_mask, &config, Backend::Cpu)?;
    let gpu = detect(&image, &circle_mask, &config, Backend::Cuda)?;

    assert_eq!(cpu.threshold, gpu.threshold);
    assert_eq!(cpu.binary_mask, gpu.binary_mask);

    let cpu_boxes = cpu.detections.iter().map(|d| d.bbox_xyxy).collect::<Vec<_>>();
    let gpu_boxes = gpu.detections.iter().map(|d| d.bbox_xyxy).collect::<Vec<_>>();
    assert_eq!(cpu_boxes, gpu_boxes);

    let cpu_scores = cpu.detections.iter().map(|d| d.score).collect::<Vec<_>>();
    let gpu_scores = gpu.detections.iter().map(|d| d.score).collect::<Vec<_>>();
    assert_scores(&cpu_scores, &gpu_scores);
    Ok(())
}
