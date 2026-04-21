pub mod ccl;
pub mod config;
pub mod cuda;
pub mod detector;
pub mod eval;
pub mod image_io;
pub mod matrix;

pub use config::{Backend, CircleSpec, DetectorConfig, Method};
pub use detector::{
    components_for_dilation, detect, detections_from_preprocessed, preprocess, Detection,
    DetectionOutput, HeavyConfig, PreprocessOutput, Roi, ThresholdField,
};
pub use eval::{evaluate_predictions, Metrics};
pub use image_io::{
    CircleMask, CocoAnnotation, CocoCategory, CocoDataset, CocoImageInfo, CocoImageList,
    CocoPrediction, GrayImage,
};
