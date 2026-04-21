use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, ValueEnum)]
pub enum Method {
    Otsu,
    Ssotsu,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Self::Otsu => "OTSU",
            Self::Ssotsu => "SSOTSU",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Otsu => "otsu",
            Self::Ssotsu => "ssotsu",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CircleSpec {
    pub center_x: i32,
    pub center_y: i32,
    pub radius: i32,
}

impl Default for CircleSpec {
    fn default() -> Self {
        Self {
            center_x: 531,
            center_y: 515,
            radius: 466,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DetectorConfig {
    pub method: Method,
    pub block_size: usize,
    pub seed_threshold_factor: f32,
    pub region_grow_threshold: f32,
    pub min_area: u32,
    pub max_area: u32,
    pub min_eccentricity: f32,
    pub max_eccentricity: f32,
    pub dilation: usize,
    pub box_padding: i32,
    pub circle: CircleSpec,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            method: Method::Otsu,
            block_size: 64,
            seed_threshold_factor: 1.5,
            region_grow_threshold: 0.8,
            min_area: 10,
            max_area: 50_000,
            min_eccentricity: 0.0,
            max_eccentricity: 1.0,
            dilation: 0,
            box_padding: 0,
            circle: CircleSpec::default(),
        }
    }
}
