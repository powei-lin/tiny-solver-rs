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

pub struct CauchyLoss {
    scale2: f64,
    c: f64,
}
impl CauchyLoss {
    pub fn new(scale: f64) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_huber_loss() {
        let scale = 1.5;
        let loss = HuberLoss::new(scale);

        // Inlier region (s <= scale^2)
        // scale^2 = 2.25
        let s_inlier = 1.0;
        let res_inlier = loss.evaluate(s_inlier);
        // rho(s) = s
        assert!((res_inlier[0] - s_inlier).abs() < EPSILON);
        // rho'(s) = 1
        assert!((res_inlier[1] - 1.0).abs() < EPSILON);
        // rho''(s) = 0
        assert!((res_inlier[2] - 0.0).abs() < EPSILON);

        // Outlier region (s > scale^2)
        let s_outlier: f64 = 4.0;
        let _r = s_outlier.sqrt(); // 2.0
        let res_outlier = loss.evaluate(s_outlier);
        // rho(s) = 2 * scale * r - scale^2 = 2 * 1.5 * 2.0 - 2.25 = 6.0 - 2.25 = 3.75
        assert!((res_outlier[0] - 3.75).abs() < EPSILON);
        // rho'(s) = scale / r = 1.5 / 2.0 = 0.75
        assert!((res_outlier[1] - 0.75).abs() < EPSILON);
        // rho''(s) = -rho'(s) / (2s) = -0.75 / 8.0 = -0.09375
        assert!((res_outlier[2] - (-0.09375)).abs() < EPSILON);
    }

    #[test]
    fn test_cauchy_loss() {
        let scale = 1.0;
        let loss = CauchyLoss::new(scale); // c = 1.0, scale2 = 1.0

        let s = 2.0;
        let res = loss.evaluate(s);
        // rho(s) = scale^2 * ln(1 + s / scale^2) = 1.0 * ln(1 + 2) = ln(3)
        assert!((res[0] - 3.0f64.ln()).abs() < EPSILON);

        // rho'(s) = 1 / (1 + s / scale^2) = 1 / (1 + 2) = 1/3
        assert!((res[1] - 1.0 / 3.0).abs() < EPSILON);

        // rho''(s) = -1 / (scale^2 * (1 + s / scale^2)^2) = -1 / (1 * 3^2) = -1/9
        assert!((res[2] - (-1.0 / 9.0)).abs() < EPSILON);
    }

    #[test]
    fn test_arctan_loss() {
        let tolerance = 1.0;
        let loss = ArctanLoss::new(tolerance); // inv_tol2 = 1.0

        let s = 1.0;
        let res = loss.evaluate(s);
        // rho(s) = tolerance * atan2(s, tolerance) = 1 * atan2(1, 1) = pi/4
        // Note: The implementation uses s.atan2(tolerance) which is atan(tolerance / s) ?? Wait.
        // Rust atan2 is (y, x) -> atan(y/x). So s.atan2(tolerance) is atan(s/tolerance).
        // If s=1, tolerance=1, atan(1) = pi/4.
        assert!((res[0] - std::f64::consts::FRAC_PI_4).abs() < EPSILON);
    }
}
