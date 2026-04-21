use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    Cpu,
    Cuda,
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
    pub level: usize,
    pub threshold_sigma: f32,
    pub min_area: u32,
    pub max_area: u32,
    pub dilation: usize,
    pub box_padding: i32,
    pub circle: CircleSpec,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            level: 3,
            threshold_sigma: 6.0,
            min_area: 30,
            max_area: 3_000,
            dilation: 0,
            box_padding: 0,
            circle: CircleSpec::default(),
        }
    }
}
