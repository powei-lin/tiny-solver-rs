use log::info;
use std::collections::HashMap;

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::optimizer::Optimizer;
use crate::problem::Problem;
use crate::{GaussNewtonOptimizer, LinearSolver, OptimizerOptions};

#[pymethods]
impl GaussNewtonOptimizer {
    #[new]
    pub fn new() -> Self {
        info!("init GaussNewtonOptimizer");
        GaussNewtonOptimizer {}
    }

    #[pyo3(name = "optimize")]
    #[pyo3(signature=(problem, initial_values, optimizer_options=None))]
    pub fn optimize_py(
        &self,
        py: Python<'_>,
        problem: &Problem,
        initial_values: &Bound<'_, PyDict>,
        optimizer_options: Option<OptimizerOptions>,
    ) -> PyResult<HashMap<String, Py<PyArray2<f64>>>> {
        let init_values: HashMap<String, PyReadonlyArray1<f64>> = initial_values.extract().unwrap();
        let init_values: HashMap<String, nalgebra::DVector<f64>> = init_values
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_matrix().column(0).into()))
            .collect();
        let result = self.optimize(problem, &init_values, optimizer_options);

        let output_d: HashMap<String, Py<PyArray2<f64>>> = result
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_pyarray_bound(py).to_owned().into()))
            .collect();
        Ok(output_d)
    }
}

#[pymethods]
impl OptimizerOptions {
    #[new]
    #[pyo3(signature = (
        max_iteration=100,
        linear_solver_type=LinearSolver::SparseCholesky,
        verbosity_level=0,
        min_abs_error_decrease_threshold=1e-5,
        min_rel_error_decrease_threshold=1e-5,
        min_error_threshold=1e-8,
    ))]
    pub fn new(
        max_iteration: usize,
        linear_solver_type: LinearSolver,
        verbosity_level: usize,
        min_abs_error_decrease_threshold: f64,
        min_rel_error_decrease_threshold: f64,
        min_error_threshold: f64,
    ) -> Self {
        OptimizerOptions {
            max_iteration,
            linear_solver_type,
            verbosity_level,
            min_abs_error_decrease_threshold,
            min_rel_error_decrease_threshold,
            min_error_threshold,
        }
    }
}
