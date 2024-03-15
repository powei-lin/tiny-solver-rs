use numpy::{pyarray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::optimizer::Optimizer;
use crate::problem::Problem;
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

    pub fn optimize(&self, problem: &Problem, initial_values: &PyDict) -> PyResult<()> {
        let init_values: HashMap<String, PyReadonlyArray1<f64>> = initial_values.extract().unwrap();
        let init_values: HashMap<String, nalgebra::DVector<f64>> = init_values
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_matrix().column(0).into()))
            .collect();
        println!("{}", initial_values);
        self.0.optimize(problem, &init_values);
        Ok(())
    }
}
