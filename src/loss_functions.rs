use core::f64;

pub enum LossFunc {
    HuberLoss,
}

pub trait Loss: Send + Sync {
    fn evaluate(&self, s: f64) -> [f64; 3];
}

#[derive(Debug, Clone)]
pub struct HuberLoss {
    scale: f64,
    scale2: f64,
}
impl HuberLoss {
    pub fn new(scale: f64) -> Self {
        if scale <= 0.0 {
            panic!("scale needs to be larger than zero");
        }
        HuberLoss {
            scale,
            scale2: scale * scale,
        }
    }
}

impl Loss for HuberLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s > self.scale2 {
            // Outlier region.
            // 'r' is always positive.
            let r = s.sqrt();
            let rho1 = (self.scale / r).max(f64::MIN);
            [2.0 * self.scale * r - self.scale2, rho1, -rho1 / (2.0 * s)]
        } else {
            // Inlier region.
            [s, 1.0, 0.0]
        }
    }
}
