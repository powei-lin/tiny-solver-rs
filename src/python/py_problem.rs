use pyo3::prelude::*;

use crate::factors::*;
use crate::problem::Problem;

fn convert_pyany_to_factor(py_any: &PyAny) -> PyResult<Box<dyn Factor + Send>> {
    let factor_name: String = py_any.get_type().getattr("__name__")?.extract()?;
    println!("{}", factor_name);
    match factor_name.as_str() {
        "CostFactorSE2" => {
            println!("add se2");
            let factor: CostFactorSE2 = py_any.extract().unwrap();
            println!("ddd {} {} {}", factor.dx, factor.dy, factor.dtheta);
            Ok(Box::new(factor))
        }
        "PriorFactor" => {
            let factor: PriorFactor = py_any.extract().unwrap();
            println!("add prior factor {}", factor.v);
            Ok(Box::new(factor))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown factor type",
        )),
    }
}

#[pymethods]
impl Problem {
    #[new]
    pub fn new_py() -> Self {
        Problem::new()
    }

    #[pyo3(name = "add_residual_block")]
    pub fn add_residual_block_py(
        &mut self,
        dim_residual: usize,
        variable_key_size_list: Vec<(String, usize)>,
        pyfactor: &PyAny,
    ) -> PyResult<()> {
        self.add_residual_block(
            dim_residual,
            variable_key_size_list,
            convert_pyany_to_factor(pyfactor).unwrap(),
        );

        println!(
            "total residual {}, total keys {}",
            self.total_residual_dimension,
            self.variable_name_to_col_idx_dict.len()
        );
        Ok(())
    }

    #[getter]
    pub fn get_num(&self) -> PyResult<usize> {
        Ok(self.thread_num)
    }
    #[setter]
    pub fn set_num(&mut self, value: usize) -> PyResult<()> {
        self.thread_num = value;
        Ok(())
    }
}
