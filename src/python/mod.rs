use pyo3::prelude::*;
// use numpy::pyo3::Python;
use numpy::nalgebra::Matrix3;
use numpy::{pyarray, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyResult, Python};

use crate::*;
use nalgebra::{DVector, SVector};
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

mod py_factors;
mod py_optimizer;
mod py_problem;
pub use py_factors::*;
pub use py_optimizer::*;
pub use py_problem::*;

use self::factors::{CostFactorSE2, PriorFactor};
#[pyclass(name = "Dual64")]
pub struct PyDual64(tiny_solver_old::SolverParameters);

#[pymethods]
impl PyDual64 {
    #[new]
    pub fn new() -> Self {
        Self(tiny_solver_old::SolverParameters::defualt())
    }

    #[getter]
    pub fn get_first_derivative(&self) -> f64 {
        1.0
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn register_child_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "factors")?;
    child_module.add_class::<CostFactorSE2>()?;
    child_module.add_class::<PriorFactor>()?;
    parent_module.add_submodule(child_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tiny_solver.factors", child_module)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn tiny_solver<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyDual64>()?;
    m.add_class::<Problem>()?;
    m.add_class::<GaussNewtonOptimizer>()?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    register_child_module(_py, m)?;

    #[pyfn(m)]
    #[pyo3(name = "mult")]
    pub fn npp<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<i32>> {
        println!("{:?}", x);
        let py_array = pyarray![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];

        let py_array = py_array.readwrite();
        let mut na_matrix = py_array.as_matrix_mut();

        na_matrix.add_scalar_mut(1);

        let py_array_square = na_matrix.pow(2).to_pyarray(py);
        Ok(py_array_square)
    }
    Ok(())
}
