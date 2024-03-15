use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::factors::*;
use crate::problem::Problem;

fn convert_pyany_to_factor(py_any: &PyAny) -> PyResult<Box<dyn Factor + Send>> {
    let factor_name: String = py_any.getattr("factor_name")?.extract()?;

    match factor_name.as_str() {
        "CostFactorSE2" => {
            println!("add se2");
            let factor: CostFactorSE2 = py_any.extract().unwrap();
            println!("ddd {} {} {}", factor.dx, factor.dy, factor.dtheta);
            Ok(Box::new(factor))
        }
        "BetweenFactor" => {
            let factor: BetweenFactor = py_any.extract().unwrap();
            println!("add between factor");
            Ok(Box::new(factor))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown factor type",
        )),
    }
}

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
            convert_pyany_to_factor(pyfactor).unwrap(),
        );
        // Box::new(BetweenFactor {}),

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
