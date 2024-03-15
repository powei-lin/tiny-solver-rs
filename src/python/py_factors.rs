use nalgebra as na;
use numpy::{pyarray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
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
        println!("pypy {:?}", x);
        let d: na::DVector<f64> = x.as_matrix().column(0).into();
        PriorFactor { v: d }
    }
}
