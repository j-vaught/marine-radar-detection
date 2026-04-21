pub mod ccl;
pub mod config;
pub mod cuda;
pub mod detector;
pub mod eval;
pub mod image_io;
pub mod matrix;

pub use config::{Backend, CircleSpec, DetectorConfig};
pub use detector::{
    components_for_dilation, detect, detections_from_components, detections_from_preprocessed,
    preprocess, preprocess_batch, Detection, DetectionOutput, PreprocessOutput,
};
pub use eval::{evaluate_predictions, Metrics};
pub use image_io::{
    CircleMask, CocoAnnotation, CocoCategory, CocoDataset, CocoImageInfo, CocoImageList,
    CocoPrediction, GrayImage,
};
