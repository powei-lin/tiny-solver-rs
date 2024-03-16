use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::factors::*;

#[pymethods]
impl CostFactorSE2 {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        CostFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        }
    }
}

#[pymethods]
impl PriorFactor {
    #[new]
    pub fn new(x: PyReadonlyArray1<f64>) -> Self {
        PriorFactor {
            v: x.as_matrix().column(0).into(),
        }
    }
}
