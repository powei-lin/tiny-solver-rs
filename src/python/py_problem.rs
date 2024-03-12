use pyo3::prelude::*;

use crate::problem::Problem;

#[pyclass(name = "Problem")]
pub struct PyProblem(Problem);

#[pymethods]
impl PyProblem {
    #[new]
    pub fn new() -> Self {
        Self(Problem::new())
    }

    // #[getter]
    // pub fn dx(&self) -> f64 {
    //     self.0
    // }
}
