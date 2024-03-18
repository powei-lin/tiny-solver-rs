use std::{borrow::Borrow, ops::{DerefMut, Mul}};

use na::ComplexField;
use nalgebra as na;
use pyo3::prelude::*;

pub trait Loss: Send + Sync {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>);
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DMatrix<f64>,
    );
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct HuberLoss {
    pub scale: f64,
}
impl HuberLoss {
    fn weight(&self, err: f64) -> f64 {
        if err < self.scale {
            1.0
        } else {
            self.scale / err.abs()
        }
    }
}

impl Loss for HuberLoss {
    fn weight_residual_in_place(&self, residual: &mut na::DVector<f64>) {
        let sqrt_weight = self.weight(residual.norm()).sqrt();
        *residual = residual.clone().mul(sqrt_weight);
    }
    fn weight_residual_jacobian_in_place(
        &self,
        residual: &mut na::DVector<f64>,
        jac: &mut na::DMatrix<f64>,
    ) {
        let sqrt_weight = self.weight(residual.norm()).sqrt();
        *residual = residual.clone().mul(sqrt_weight);
        *jac = jac.clone().mul(sqrt_weight);
    }
}
