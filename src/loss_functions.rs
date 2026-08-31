use core::f64;

pub trait Loss: Send + Sync {
    fn evaluate(&self, s: f64) -> [f64; 3];
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrivialLoss;

impl Loss for TrivialLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        [s, 1.0, 0.0]
    }
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

#[derive(Debug, Clone)]
pub struct SoftLOneLoss {
    scale2: f64,
    inverse_scale2: f64,
}

impl SoftLOneLoss {
    pub fn new(scale: f64) -> Self {
        assert!(scale > 0.0, "scale needs to be larger than zero");
        let scale2 = scale * scale;
        Self {
            scale2,
            inverse_scale2: 1.0 / scale2,
        }
    }
}

impl Loss for SoftLOneLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let sum = 1.0 + s * self.inverse_scale2;
        let square_root = sum.sqrt();
        let first_derivative = (1.0 / square_root).max(f64::MIN);
        [
            2.0 * self.scale2 * (square_root - 1.0),
            first_derivative,
            -(self.inverse_scale2 * first_derivative) / (2.0 * sum),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct CauchyLoss {
    scale2: f64,
    c: f64,
}
impl CauchyLoss {
    pub fn new(scale: f64) -> Self {
        assert!(scale > 0.0, "scale needs to be larger than zero");
        let scale2 = scale * scale;
        CauchyLoss {
            scale2,
            c: 1.0 / scale2,
        }
    }
}
impl Loss for CauchyLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let sum = 1.0 + s * self.c;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive, assuming that 's' is.
        [
            self.scale2 * sum.ln(),
            inv.max(f64::MIN),
            -self.c * (inv * inv),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct ArctanLoss {
    tolerance: f64,
    inv_of_squared_tolerance: f64,
}

impl ArctanLoss {
    pub fn new(tolerance: f64) -> Self {
        if tolerance <= 0.0 {
            panic!("scale needs to be larger than zero");
        }
        ArctanLoss {
            tolerance,
            inv_of_squared_tolerance: 1.0 / (tolerance * tolerance),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TolerantLoss {
    tolerance: f64,
    transition: f64,
    offset: f64,
}

impl TolerantLoss {
    pub fn new(tolerance: f64, transition: f64) -> Self {
        assert!(tolerance >= 0.0, "tolerance cannot be negative");
        assert!(transition > 0.0, "transition needs to be larger than zero");
        Self {
            tolerance,
            transition,
            offset: transition * (1.0 + (-tolerance / transition).exp()).ln(),
        }
    }
}

impl Loss for TolerantLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let normalized = (s - self.tolerance) / self.transition;
        if normalized > 36.7 {
            [s - self.tolerance - self.offset, 1.0, 0.0]
        } else {
            let exponential = normalized.exp();
            [
                self.transition * exponential.ln_1p() - self.offset,
                (exponential / (1.0 + exponential)).max(f64::MIN),
                0.5 / (self.transition * (1.0 + normalized.cosh())),
            ]
        }
    }
}

#[derive(Debug, Clone)]
pub struct TukeyLoss {
    scale2: f64,
}

impl TukeyLoss {
    pub fn new(scale: f64) -> Self {
        assert!(scale > 0.0, "scale needs to be larger than zero");
        Self {
            scale2: scale * scale,
        }
    }
}

impl Loss for TukeyLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s <= self.scale2 {
            let value = 1.0 - s / self.scale2;
            let value2 = value * value;
            [
                self.scale2 / 3.0 * (1.0 - value2 * value),
                value2,
                -2.0 / self.scale2 * value,
            ]
        } else {
            [self.scale2 / 3.0, 0.0, 0.0]
        }
    }
}

pub struct ComposedLoss {
    outer: Box<dyn Loss>,
    inner: Box<dyn Loss>,
}

impl ComposedLoss {
    pub fn new(outer: Box<dyn Loss>, inner: Box<dyn Loss>) -> Self {
        Self { outer, inner }
    }
}

impl Loss for ComposedLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let inner = self.inner.evaluate(s);
        let outer = self.outer.evaluate(inner[0]);
        [
            outer[0],
            outer[1] * inner[1],
            outer[2] * inner[1] * inner[1] + outer[1] * inner[2],
        ]
    }
}

pub struct ScaledLoss {
    loss: Option<Box<dyn Loss>>,
    scale: f64,
}

impl ScaledLoss {
    pub fn new(loss: Option<Box<dyn Loss>>, scale: f64) -> Self {
        Self { loss, scale }
    }
}

impl Loss for ScaledLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let rho = self
            .loss
            .as_ref()
            .map_or([s, 1.0, 0.0], |loss| loss.evaluate(s));
        rho.map(|value| value * self.scale)
    }
}

impl Loss for ArctanLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let sum = 1.0 + s * s * self.inv_of_squared_tolerance;
        let inv = 1.0 / sum;

        [
            self.tolerance * s.atan2(self.tolerance),
            inv.max(f64::MIN),
            -2.0 * s * self.inv_of_squared_tolerance * (inv * inv),
        ]
    }
}
