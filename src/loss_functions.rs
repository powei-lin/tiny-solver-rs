use std::ops::Mul;

use nalgebra as na;

pub trait Loss: Send + Sync {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>);
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DMatrix<f64>,
    );
}

#[derive(Debug, Clone)]
pub struct HuberLoss {
    scale: f64,
}
impl HuberLoss {
    pub fn new(scale: f64) -> Self {
        if scale <= 0.0 {
            panic!("scale needs to be larger than zero");
        }
        HuberLoss { scale }
    }
    fn weight(&self, abs_err: f64) -> f64 {
        if abs_err < self.scale {
            1.0
        } else {
            self.scale / abs_err * (2.0 * abs_err / self.scale - 1.0).sqrt()
        }
    }
}

impl Loss for HuberLoss {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>) {
        let weight = self.weight(residual.norm());
        *residual = residual.clone().mul(weight);
    }
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DMatrix<f64>,
    ) {
        let weight = self.weight(residual.norm());
        *residual = residual.clone().mul(weight);
        *jac = jac.clone().mul(weight);
    }
}
