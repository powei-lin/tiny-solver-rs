use pyo3::prelude::*;

use crate::factors;

#[pyclass(name = "Factor")]
pub struct PyFactor(factors::CostFactorSE2);

#[pymethods]
impl PyFactor {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self(factors::CostFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        })
    }

    #[getter]
    pub fn get_first_derivative(&self) -> f64 {
        1.0
    }
}
