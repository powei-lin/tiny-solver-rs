use log::info;
use std::collections::HashMap;

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::py_linear::PyLinearSolver;
use super::py_problem::PyProblem;
use crate::optimizer::Optimizer;
use crate::{GaussNewtonOptimizer, LinearSolver, OptimizerOptions};

#[pyclass(name = "GaussNewtonOptimizer")]
pub struct PyGaussNewtonOptimizer(GaussNewtonOptimizer);

#[pymethods]
impl PyGaussNewtonOptimizer {
    #[new]
    pub fn new() -> Self {
        info!("init GaussNewtonOptimizer");
        PyGaussNewtonOptimizer(GaussNewtonOptimizer {})
    }

    #[pyo3(name = "optimize")]
    #[pyo3(signature=(problem, initial_values, optimizer_options=None))]
    pub fn optimize_py(
        &self,
        py: Python<'_>,
        problem: &PyProblem,
        initial_values: &Bound<'_, PyDict>,
        optimizer_options: Option<PyOptimizerOptions>,
    ) -> PyResult<HashMap<String, Py<PyArray2<f64>>>> {
        let init_values: HashMap<String, PyReadonlyArray1<f64>> = initial_values.extract().unwrap();
        let init_values: HashMap<String, nalgebra::DVector<f64>> = init_values
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_matrix().column(0).into()))
            .collect();
        let result = self
            .0
            .optimize(&problem.0, &init_values, Some(optimizer_options.unwrap().0));

        let output_d: HashMap<String, Py<PyArray2<f64>>> = result
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_pyarray_bound(py).to_owned().into()))
            .collect();
        Ok(output_d)
    }
}

#[pyclass(name = "OptimizerOptions")]
#[derive(Clone)]
pub struct PyOptimizerOptions(pub OptimizerOptions);

#[pymethods]
impl PyOptimizerOptions {
    #[new]
    #[pyo3(signature = (
        max_iteration=100,
        linear_solver_type=PyLinearSolver(LinearSolver::SparseCholesky),
        verbosity_level=0,
        min_abs_error_decrease_threshold=1e-5,
        min_rel_error_decrease_threshold=1e-5,
        min_error_threshold=1e-8,
    ))]
    pub fn new(
        max_iteration: usize,
        linear_solver_type: PyLinearSolver,
        verbosity_level: usize,
        min_abs_error_decrease_threshold: f64,
        min_rel_error_decrease_threshold: f64,
        min_error_threshold: f64,
    ) -> Self {
        PyOptimizerOptions(OptimizerOptions {
            max_iteration,
            linear_solver_type: linear_solver_type.0,
            verbosity_level,
            min_abs_error_decrease_threshold,
            min_rel_error_decrease_threshold,
            min_error_threshold,
        })
    }
}
