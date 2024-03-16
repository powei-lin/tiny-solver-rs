use std::collections::HashMap;

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    pub fn optimize_py(
        &self,
        py: Python<'_>,
        problem: &Problem,
        initial_values: &PyDict,
    ) -> PyResult<HashMap<String, Py<PyArray2<f64>>>> {
        let init_values: HashMap<String, PyReadonlyArray1<f64>> = initial_values.extract().unwrap();
        let init_values: HashMap<String, nalgebra::DVector<f64>> = init_values
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_matrix().column(0).into()))
            .collect();
        // println!("{}", initial_values);
        let result = self.optimize(problem, &init_values, None);

        let output_d: HashMap<String, Py<PyArray2<f64>>> = result
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_pyarray(py).to_owned().into()))
            .collect();
        Ok(output_d)
    }
}
