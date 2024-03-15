use pyo3::prelude::*;

use crate::optimizer::Optimizer;
use crate::GaussNewtonOptimizer;

#[pyclass(name = "GaussNewtonOptimizer")]
pub struct PyGaussNewtonOptimizer(GaussNewtonOptimizer);

#[pymethods]
impl PyGaussNewtonOptimizer {
    #[new]
    pub fn new() -> Self {
        println!("init GaussNewtonOptimizer");
        Self(GaussNewtonOptimizer {})
    }
}
