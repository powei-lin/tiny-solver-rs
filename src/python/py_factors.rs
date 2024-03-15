use nalgebra as na;
use num_dual;
use numpy::{pyarray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::factors::*;

#[pyclass(name = "Factor")]
#[derive(Debug)]
pub struct PyFactor {
    pub factor: Py<PyAny>,
}

#[pymethods]
impl PyFactor {
    #[new]
    pub fn new(factor: Py<PyAny>) -> Self {
        PyFactor { factor }
    }
}

#[pymethods]
impl CostFactorSE2 {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        CostFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        }
    }
    #[getter]
    pub fn get_factor_name(&self) -> String {
        "CostFactorSE2".to_string()
    }
}

#[pymethods]
impl BetweenFactor {
    #[new]
    pub fn new(x: PyReadonlyArray1<f64>) -> Self {
        println!("pypy {:?}", x);
        BetweenFactor {
            v: na::dvector![0.0, 0.0, 0.0],
        }
    }
    #[getter]
    pub fn get_factor_name(&self) -> String {
        "BetweenFactor".to_string()
    }
}
// pub fn npp<'py>(
//     py: Python<'py>,
// ) -> PyResult<&'py PyArray2<i32>> {
//     let py_array = pyarray![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];

//     let py_array = py_array.readwrite();

// #[pyclass(name = "FactorSE2")]
// pub struct PyFactorSE2(factors::CostFactorSE2);

// #[pymethods]
// impl PyFactorSE2 {
//     #[new]
//     pub fn new(x: f64, y: f64, theta: f64) -> Self {
//         Self(factors::CostFactorSE2 {
//             dx: x,
//             dy: y,
//             dtheta: theta,
//         })
//     }
//     #[getter]
//     pub fn get_type(&self) -> String {
//         "se2".to_string()
//     }

//     #[getter]
//     pub fn dx(&self) -> f64 {
//         self.0.dx
//     }
//     #[getter]
//     pub fn dy(&self) -> f64 {
//         self.0.dy
//     }
//     #[getter]
//     pub fn dtheta(&self) -> f64 {
//         self.0.dtheta
//     }
// }

// pub fn ttt(&self) -> PyResult<()> {
//     let x0 = na::dvector![1.0, 2.0];
//     let x0 = x0.map(num_dual::DualDVec64::from_re);
//     let a = vec![x0];
//     let r = self.0.residual_func(&a);
//     println!("{:?}", r);
//     Ok(())
// }
