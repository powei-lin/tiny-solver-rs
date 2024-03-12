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

    pub fn add_residual_block(&self) -> PyResult<()> {
        println!("add residual block");
        Ok(())
    }

    #[getter]
    pub fn get_num(&self) -> PyResult<usize> {
        Ok(self.0.thread_num)
    }
    #[setter]
    pub fn set_num(&mut self, value: usize) -> PyResult<()> {
        self.0.thread_num = value;
        Ok(())
    }
}
