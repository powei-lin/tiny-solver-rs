/// https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc

pub(crate) struct Corrector {
    sqrt_rho1: f64,
    residual_scaling: f64,
    alpha_sq_norm: f64,
}

impl Corrector {
    pub fn new(sq_norm: f64, rho: &[f64; 3]) -> Corrector {
        let sqrt_rho1 = rho[1].sqrt();
        // is always well formed.
        if (sq_norm == 0.0) || (rho[2] <= 0.0) {
            let residual_scaling = sqrt_rho1;
            let alpha_sq_norm = 0.0;
            Corrector {
                sqrt_rho1,
                residual_scaling,
                alpha_sq_norm,
            }
        } else {
            let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

            let alpha = 1.0 - d.sqrt();

            let residual_scaling = sqrt_rho1 / (1.0 - alpha);
            let alpha_sq_norm = alpha / sq_norm;
            Corrector {
                sqrt_rho1,
                residual_scaling,
                alpha_sq_norm,
            }
        }
    }
}
