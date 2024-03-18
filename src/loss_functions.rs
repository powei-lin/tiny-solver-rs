use std::{borrow::Borrow, ops::Mul};

use nalgebra as na;
use pyo3::prelude::*;

pub trait Loss: Send + Sync {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>);
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DVector<f64>,
    );
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct HuberLoss {}
impl HuberLoss {
    fn weight(err: f64) -> f64 {
        // return (err < k_) ? 1.0 : (k_ / std::fabs(err));
        1.0
    }
}

impl Loss for HuberLoss {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>) {
        let sqrt_weight = HuberLoss::weight(residual.norm()).sqrt();
        // println!("wwwwwww");
        *residual = residual.clone().mul(sqrt_weight);
    }
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DVector<f64>,
    ) {
    }
}
