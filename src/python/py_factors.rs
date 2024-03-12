use pyo3::prelude::*;

use crate::factors;

#[pyclass(name = "FactorSE2")]
pub struct PyFactorSE2(factors::CostFactorSE2);

#[pymethods]
impl PyFactorSE2 {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self(factors::CostFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        })
    }

    #[getter]
    pub fn dx(&self) -> f64 {
        self.0.dx
    }
    #[getter]
    pub fn dy(&self) -> f64 {
        self.0.dy
    }
    #[getter]
    pub fn dtheta(&self) -> f64 {
        self.0.dtheta
    }
}
