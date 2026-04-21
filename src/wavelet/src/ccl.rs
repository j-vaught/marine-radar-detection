#[derive(Clone, Debug)]
pub struct ComponentStat {
    pub bbox_xyxy: [u32; 4],
    pub area: u32,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub orientation: f32,
    pub eccentricity: f32,
    pub peak: u8,
}

#[derive(Debug)]
struct UnionFind {
    parents: Vec<u32>,
    ranks: Vec<u8>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parents: vec![0],
            ranks: vec![0],
        }
    }

    fn make_set(&mut self) -> u32 {
        let label = self.parents.len() as u32;
        self.parents.push(label);
        self.ranks.push(0);
        label
    }

    #[inline(always)]
    fn find(&mut self, label: u32) -> u32 {
        let mut root = label;
        while self.parents[root as usize] != root {
            root = self.parents[root as usize];
        }
        let mut node = label;
        while self.parents[node as usize] != root {
            let parent = self.parents[node as usize];
            self.parents[node as usize] = root;
            node = parent;
        }
        root
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return root_a;
        }
        let rank_a = self.ranks[root_a as usize];
        let rank_b = self.ranks[root_b as usize];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parents[root_b as usize] = root_a;
        if rank_a == rank_b {
            self.ranks[root_a as usize] += 1;
        }
        root_a
    }
}

#[derive(Clone, Debug)]
struct ComponentAccum {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    area: u32,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
    peak: u8,
}

impl ComponentAccum {
    fn new() -> Self {
        Self {
            min_x: u32::MAX,
            min_y: u32::MAX,
            max_x: 0,
            max_y: 0,
            area: 0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_x2: 0.0,
            sum_y2: 0.0,
            sum_xy: 0.0,
            peak: 0,
        }
    }

    #[inline(always)]
    fn add_pixel<const COMPUTE_SHAPE: bool>(&mut self, x: usize, y: usize, intensity: u8) {
        let x_u32 = x as u32;
        let y_u32 = y as u32;
        self.min_x = self.min_x.min(x_u32);
        self.min_y = self.min_y.min(y_u32);
        self.max_x = self.max_x.max(x_u32 + 1);
        self.max_y = self.max_y.max(y_u32 + 1);
        self.area += 1;
        let xf = x as f64;
        let yf = y as f64;
        self.sum_x += xf;
        self.sum_y += yf;
        if COMPUTE_SHAPE {
            self.sum_x2 += xf * xf;
            self.sum_y2 += yf * yf;
            self.sum_xy += xf * yf;
        }
        self.peak = self.peak.max(intensity);
    }

    fn finalize<const COMPUTE_SHAPE: bool>(self) -> ComponentStat {
        let area_f = self.area as f64;
        let mean_x = self.sum_x / area_f;
        let mean_y = self.sum_y / area_f;
        let (orientation, eccentricity) = if COMPUTE_SHAPE {
            let mu20 = self.sum_x2 / area_f - mean_x * mean_x;
            let mu02 = self.sum_y2 / area_f - mean_y * mean_y;
            let mu11 = self.sum_xy / area_f - mean_x * mean_y;
            let trace = mu20 + mu02;
            let det_term = ((mu20 - mu02) * (mu20 - mu02) + 4.0 * mu11 * mu11).sqrt();
            let major = ((trace + det_term) * 0.5).max(0.0);
            let minor = ((trace - det_term) * 0.5).max(0.0);
            let eccentricity = if major > 0.0 {
                (1.0 - (minor / major)).max(0.0).sqrt() as f32
            } else {
                0.0
            };
            let orientation = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);
            (orientation as f32, eccentricity)
        } else {
            (0.0, 0.0)
        };
        ComponentStat {
            bbox_xyxy: [self.min_x, self.min_y, self.max_x, self.max_y],
            area: self.area,
            centroid_x: mean_x as f32,
            centroid_y: mean_y as f32,
            orientation,
            eccentricity,
            peak: self.peak,
        }
    }
}

pub fn extract_components<const COMPUTE_SHAPE: bool>(
    mask: &[u8],
    intensity: &[u8],
    width: usize,
    height: usize,
    intensity_stride: usize,
    intensity_offset_x: usize,
    intensity_offset_y: usize,
) -> Vec<ComponentStat> {
    let mut labels = vec![0u32; width * height];
    let mut uf = UnionFind::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask[idx] == 0 {
                continue;
            }
            let mut neighbors = [0u32; 4];
            let mut count = 0usize;
            if x > 0 {
                let label = labels[idx - 1];
                if label != 0 {
                    neighbors[count] = label;
                    count += 1;
                }
            }
            if y > 0 {
                let up_idx = idx - width;
                let label = labels[up_idx];
                if label != 0 {
                    neighbors[count] = label;
                    count += 1;
                }
                if x > 0 {
                    let label = labels[up_idx - 1];
                    if label != 0 {
                        neighbors[count] = label;
                        count += 1;
                    }
                }
                if x + 1 < width {
                    let label = labels[up_idx + 1];
                    if label != 0 {
                        neighbors[count] = label;
                        count += 1;
                    }
                }
            }
            if count == 0 {
                labels[idx] = uf.make_set();
                continue;
            }
            let first = neighbors[0];
            labels[idx] = first;
            for &other in neighbors[..count].iter().skip(1) {
                uf.union(first, other);
            }
        }
    }

    let mut root_to_component = vec![usize::MAX; uf.parents.len()];
    let mut accumulators = Vec::<ComponentAccum>::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let label = labels[idx];
            if label == 0 {
                continue;
            }
            let root = uf.find(label);
            labels[idx] = root;
            let component_idx = if root_to_component[root as usize] == usize::MAX {
                root_to_component[root as usize] = accumulators.len();
                accumulators.push(ComponentAccum::new());
                accumulators.len() - 1
            } else {
                root_to_component[root as usize]
            };
            let source_idx = (y + intensity_offset_y) * intensity_stride + (x + intensity_offset_x);
            let component = &mut accumulators[component_idx];
            component.add_pixel::<COMPUTE_SHAPE>(x, y, intensity[source_idx]);
        }
    }

    accumulators
        .into_iter()
        .filter(|component| component.area > 0)
        .map(ComponentAccum::finalize::<COMPUTE_SHAPE>)
        .collect()
}
