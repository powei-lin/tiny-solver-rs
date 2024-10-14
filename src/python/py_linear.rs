use crate::linear::LinearSolver;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass(name = "LinearSolver")]
pub struct PyLinearSolver(pub LinearSolver);
