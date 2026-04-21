use anyhow::{anyhow, Result};
use clap::ValueEnum;
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use serde::Serialize;

use crate::config::DetectorConfig;
use crate::image_io::{CocoDataset, CocoImageInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GridSize {
    Small,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Site {
    Manv,
    Darc,
    Daan,
}

impl Site {
    pub fn name(self) -> &'static str {
        match self {
            Self::Manv => "MANV",
            Self::Darc => "DARC",
            Self::Daan => "DAAN",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::Manv => "Mv",
            Self::Darc => "Dc",
            Self::Daan => "Da",
        }
    }
}

#[derive(Clone, Debug)]
enum SplitSpec {
    Pooled { seed: u64 },
    SitePermutation { tune: Site, val: Site, test: Site },
    Transfer { tune: Site, test: Site },
}

#[derive(Clone, Debug)]
pub struct Experiment {
    pub exp_id: String,
    pub condition: String,
    pub method: String,
    pub tune_label: String,
    pub val_label: String,
    pub test_label: String,
    pub seed: u64,
    split: SplitSpec,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchConfig {
    pub level: usize,
    pub threshold_sigma: f32,
    pub min_area: u32,
    pub max_area: u32,
    pub dilation: usize,
    pub box_padding: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StructuralKey {
    level: usize,
    threshold_sigma_bits: u32,
}

impl StructuralKey {
    pub fn threshold_sigma(self) -> f32 {
        f32::from_bits(self.threshold_sigma_bits)
    }

    pub fn detector_config(self) -> DetectorConfig {
        DetectorConfig {
            level: self.level,
            threshold_sigma: self.threshold_sigma(),
            ..DetectorConfig::default()
        }
    }
}

impl SearchConfig {
    pub fn detector_config(&self) -> DetectorConfig {
        DetectorConfig {
            level: self.level,
            threshold_sigma: self.threshold_sigma,
            min_area: self.min_area,
            max_area: self.max_area,
            dilation: self.dilation,
            box_padding: self.box_padding,
            ..DetectorConfig::default()
        }
    }

    pub fn structural_key(&self) -> StructuralKey {
        StructuralKey {
            level: self.level,
            threshold_sigma_bits: self.threshold_sigma.to_bits(),
        }
    }
}

impl Experiment {
    pub fn split(&self, dataset: &CocoDataset) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        match self.split {
            SplitSpec::Pooled { seed } => Ok(pooled_split(dataset.images.len(), seed)),
            SplitSpec::SitePermutation { tune, val, test } => Ok((
                indices_for_site(&dataset.images, tune)?,
                indices_for_site(&dataset.images, val)?,
                indices_for_site(&dataset.images, test)?,
            )),
            SplitSpec::Transfer { tune, test } => {
                let (tune_idx, val_idx, held_idx) = site_internal_split(&dataset.images, tune, 0)?;
                if tune == test {
                    Ok((tune_idx, val_idx, held_idx))
                } else {
                    Ok((tune_idx, val_idx, indices_for_site(&dataset.images, test)?))
                }
            }
        }
    }
}

pub fn build_all_experiments() -> Vec<Experiment> {
    let mut exps = Vec::<Experiment>::new();

    for seed in 0..5u64 {
        exps.push(Experiment {
            exp_id: format!("P-WAV-{seed}"),
            condition: "P".to_string(),
            method: "WAV".to_string(),
            tune_label: format!("pool70@{seed}"),
            val_label: format!("pool15@{seed}"),
            test_label: format!("pool15@{seed}"),
            seed,
            split: SplitSpec::Pooled { seed },
        });
    }

    let perms = [
        (Site::Manv, Site::Darc, Site::Daan),
        (Site::Manv, Site::Daan, Site::Darc),
        (Site::Darc, Site::Manv, Site::Daan),
        (Site::Darc, Site::Daan, Site::Manv),
        (Site::Daan, Site::Manv, Site::Darc),
        (Site::Daan, Site::Darc, Site::Manv),
    ];
    for (idx, (tune, val, test)) in perms.into_iter().enumerate() {
        exps.push(Experiment {
            exp_id: format!("S-WAV-{}", idx + 1),
            condition: "S".to_string(),
            method: "WAV".to_string(),
            tune_label: tune.name().to_string(),
            val_label: val.name().to_string(),
            test_label: test.name().to_string(),
            seed: 0,
            split: SplitSpec::SitePermutation { tune, val, test },
        });
    }

    for tune in [Site::Manv, Site::Darc, Site::Daan] {
        for test in [Site::Manv, Site::Darc, Site::Daan] {
            exps.push(Experiment {
                exp_id: format!("T-WAV-{}{}", tune.code(), test.code()),
                condition: "T".to_string(),
                method: "WAV".to_string(),
                tune_label: format!("{}[70]", tune.name()),
                val_label: format!("{}[15]", tune.name()),
                test_label: if tune == test {
                    format!("{}[15]", test.name())
                } else {
                    format!("{}[all]", test.name())
                },
                seed: 0,
                split: SplitSpec::Transfer { tune, test },
            });
        }
    }

    exps
}

pub fn build_search_grid(size: GridSize) -> Vec<SearchConfig> {
    let (levels, sigmas, min_areas, max_areas, dilations, paddings): (
        &[usize],
        &[f32],
        &[u32],
        &[u32],
        &[usize],
        &[i32],
    ) = match size {
        GridSize::Small => (
            &[2, 3],
            &[4.5, 6.0, 7.5],
            &[20, 30, 50],
            &[3_000],
            &[0, 1],
            &[0],
        ),
        GridSize::Full => (
            &[1, 2, 3],
            &[4.0, 5.0, 6.0, 7.0],
            &[20, 30, 50],
            &[2_000, 3_000, 6_000],
            &[0, 1],
            &[0, 2],
        ),
    };

    let mut out = Vec::<SearchConfig>::new();
    for &level in levels {
        for &threshold_sigma in sigmas {
            for &min_area in min_areas {
                for &max_area in max_areas {
                    if min_area > max_area {
                        continue;
                    }
                    for &dilation in dilations {
                        for &box_padding in paddings {
                            out.push(SearchConfig {
                                level,
                                threshold_sigma,
                                min_area,
                                max_area,
                                dilation,
                                box_padding,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn filter_experiment(exp: &Experiment, only: Option<&str>) -> bool {
    match only {
        None => true,
        Some("P") => exp.condition == "P",
        Some("S") => exp.condition == "S",
        Some("T") => exp.condition == "T",
        Some("WAV") => exp.method == "WAV",
        Some(other) => exp.exp_id == other,
    }
}

pub fn subset_image_ids(dataset: &CocoDataset, indices: &[usize]) -> Vec<u64> {
    indices.iter().map(|&idx| dataset.images[idx].id).collect()
}

fn pooled_split(n: usize, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut idx = (0..n).collect::<Vec<_>>();
    let mut rng = SmallRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_tune = ((n as f32) * 0.70).round() as usize;
    let n_val = ((n as f32) * 0.15).round() as usize;
    (
        idx[..n_tune].to_vec(),
        idx[n_tune..(n_tune + n_val).min(n)].to_vec(),
        idx[(n_tune + n_val).min(n)..].to_vec(),
    )
}

fn site_internal_split(
    images: &[CocoImageInfo],
    site: Site,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let mut idx = indices_for_site(images, site)?;
    let mut rng = SmallRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n = idx.len();
    let n_tune = ((n as f32) * 0.70).round() as usize;
    let n_val = ((n as f32) * 0.15).round() as usize;
    Ok((
        idx[..n_tune].to_vec(),
        idx[n_tune..(n_tune + n_val).min(n)].to_vec(),
        idx[(n_tune + n_val).min(n)..].to_vec(),
    ))
}

fn indices_for_site(images: &[CocoImageInfo], site: Site) -> Result<Vec<usize>> {
    let mut idx = Vec::<usize>::new();
    for (image_idx, info) in images.iter().enumerate() {
        if site_from_image(info)? == site {
            idx.push(image_idx);
        }
    }
    Ok(idx)
}

fn site_from_image(info: &CocoImageInfo) -> Result<Site> {
    if let Some(dataset) = info.dataset.as_deref() {
        return parse_site(dataset);
    }
    let prefix = info
        .file_name
        .split('_')
        .next()
        .ok_or_else(|| anyhow!("cannot infer site from {}", info.file_name))?;
    parse_site(prefix)
}

fn parse_site(raw: &str) -> Result<Site> {
    match raw {
        "MANV" => Ok(Site::Manv),
        "DARC" => Ok(Site::Darc),
        "DAAN" => Ok(Site::Daan),
        other => Err(anyhow!("unknown site {other}")),
    }
}
