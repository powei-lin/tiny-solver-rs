use numpy::{pyarray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::optimizer::Optimizer;
use crate::problem::Problem;
use crate::GaussNewtonOptimizer;

#[pymethods]
impl GaussNewtonOptimizer {
    #[new]
    pub fn new() -> Self {
        println!("init GaussNewtonOptimizer");
        GaussNewtonOptimizer {}
    }

    #[pyo3(name = "optimize")]
    pub fn optimize_py(&self, problem: &Problem, initial_values: &PyDict) -> PyResult<()> {
        let init_values: HashMap<String, PyReadonlyArray1<f64>> = initial_values.extract().unwrap();
        let init_values: HashMap<String, nalgebra::DVector<f64>> = init_values
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_matrix().column(0).into()))
            .collect();
        println!("{}", initial_values);
        self.optimize(problem, &init_values);
        Ok(())
    }
}
