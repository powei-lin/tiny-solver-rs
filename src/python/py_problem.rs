use pyo3::prelude::*;

use crate::factors::*;
use crate::loss_functions::*;
use crate::problem::Problem;

use super::py_factors::*;
use super::py_loss_functions::*;

fn convert_pyany_to_factor(py_any: &Bound<'_, PyAny>) -> PyResult<(bool, Box<dyn Factor + Send>)> {
    let factor_name: String = py_any.get_type().getattr("__name__")?.extract()?;
    match factor_name.as_str() {
        "BetweenFactorSE2" => {
            let factor: PyBetweenFactorSE2 = py_any.extract().unwrap();
            Ok((false, Box::new(factor.0)))
        }
        "PriorFactor" => {
            let factor: PyPriorFactor = py_any.extract().unwrap();
            Ok((false, Box::new(factor.0)))
        }
        "PyFactor" => {
            let factor: PyFactor = py_any.extract().unwrap();
            Ok((true, Box::new(factor)))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown factor type",
        )),
    }
}
fn convert_pyany_to_loss_function(
    py_any: &Bound<'_, PyAny>,
) -> PyResult<Option<Box<dyn Loss + Send>>> {
    let factor_name: String = py_any.get_type().getattr("__name__")?.extract()?;
    match factor_name.as_str() {
        "HuberLoss" => {
            let loss_func: PyHuberLoss = py_any.extract().unwrap();
            Ok(Some(Box::new(loss_func.0)))
        }
        "NoneType" => Ok(None),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown factor type",
        )),
    }
}

#[pyclass(name = "Problem")]
pub struct PyProblem(pub Problem);

#[pymethods]
impl PyProblem {
    #[new]
    pub fn new_py() -> Self {
        PyProblem(Problem::new())
    }

    #[pyo3(name = "add_residual_block")]
    pub fn add_residual_block_py(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: Vec<(String, usize)>,
        pyfactor: &Bound<'_, PyAny>,
        pyloss_func: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (is_pyfactor, factor) = convert_pyany_to_factor(pyfactor).unwrap();
        self.0.add_residual_block(
            dim_residual,
            variable_key_size_list,
            factor,
            convert_pyany_to_loss_function(pyloss_func).unwrap(),
        );
        if is_pyfactor {
            self.0.has_py_factor()
        }
        Ok(())
    }
}
