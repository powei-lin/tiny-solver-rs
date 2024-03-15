use pyo3::prelude::*;

use crate::factors::BetweenFactor;
use crate::factors::Factor;
use crate::problem::Problem;

#[pyclass(name = "Problem")]
pub struct PyProblem(Problem);

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
        pyfactor: &PyAny,
    ) -> PyResult<()> {
        self.0.add_residual_block(
            dim_residual,
            variable_key_size_list,
            Box::new(BetweenFactor {}),
        );

        println!(
            "total residual {}, total keys {}",
            self.0.total_residual_dimension,
            self.0.variable_name_to_col_idx_dict.len()
        );
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
