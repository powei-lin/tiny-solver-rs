use nalgebra as na;
use num_dual;
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
    pub fn get_type(&self) -> String {
        "se2".to_string()
    }
}

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

#[pyclass(name = "BetweenFactor")]
#[derive(Clone)]
pub struct PyBetweenFactor(BetweenFactor);

#[pymethods]
impl PyBetweenFactor {
    #[new]
    pub fn new() -> Self {
        Self(BetweenFactor {})
    }
    #[getter]
    pub fn get_name(&self) -> String {
        "between".to_string()
    }

    // pub fn ttt(&self) -> PyResult<()> {
    //     let x0 = na::dvector![1.0, 2.0];
    //     let x0 = x0.map(num_dual::DualDVec64::from_re);
    //     let a = vec![x0];
    //     let r = self.0.residual_func(&a);
    //     println!("{:?}", r);
    //     Ok(())
    // }
}
