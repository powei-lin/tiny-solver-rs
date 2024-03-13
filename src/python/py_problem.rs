use pyo3::prelude::*;

use crate::factors::BetweenFactor;
use crate::factors::Factor;
use crate::problem::Problem;

use super::py_factors::DynFactor;

#[pyclass(name = "Problem")]
pub struct PyProblem(Problem);

pub fn from_py_generic<T: for<'a> FromPyObject<'a>>(obj: Py<PyAny>) -> PyResult<T> {
    Python::with_gil(|py| obj.extract(py))
}

#[pymethods]
impl PyProblem {
    #[new]
    pub fn new() -> Self {
        Self(Problem::new())
    }

    pub fn add_residual_block(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: Vec<(String, usize)>,
    ) -> PyResult<()> {
        self.0.add_residual_block(
            dim_residual,
            variable_key_size_list,
            Box::new(BetweenFactor {}),
        );

        // let factor: DynFactor = py_factor.extract()?;
        // self.0.add_residual_block(dim_residual, variable_key_size_list, Box::new(factor));
        println!("add residual block {}", self.0.total_residual_dimension);
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
