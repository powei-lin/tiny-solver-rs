use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::factors::*;
use crate::problem::Problem;

fn convert_pyany_to_factor(py_any: &PyAny) -> PyResult<Box<dyn Factor + Send>> {
    let obj_type: String = py_any.getattr("factor_name")?.extract()?;

    // Box::new(PyFactor { factor: py_any })
    match obj_type.as_str() {
        "CostFactorSE2" => {
            // 假设 PyFactor 类型的对象有一个 `value` 属性
            println!("add se2");
            let f: CostFactorSE2 = py_any.extract().unwrap();
            println!("ddd {} {} {}", f.dx, f.dy, f.dtheta);
            // let value: i32 = py_any.getattr("value")?.extract()?;
            Ok(Box::new(f))
            // Ok(Box::new(CostFactorSE2 { dx: 1.0 , dy:0.1, dtheta:0.0 }))
        }
        "BetweenFactor" => {
            // 假设 TFactor 类型的对象也有一个 `value` 属性
            println!("fff");
            // let value: i32 = py_any.getattr("value")?.extract()?;
            Ok(Box::new(BetweenFactor {}))
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
