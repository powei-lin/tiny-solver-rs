use nalgebra as na;
use num_dual::python::PyDual64Dyn;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::factors::*;

#[pymethods]
impl BetweenFactorSE2 {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        BetweenFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        }
    }
}

#[pymethods]
impl PriorFactor {
    #[new]
    pub fn new(x: PyReadonlyArray1<f64>) -> Self {
        PriorFactor {
            v: x.as_matrix().column(0).into(),
        }
    }
}

// #[pyclass(name = "DualDVec64")]
// #[derive(Clone)]
// pub struct PyDualDVec64(DualDVec64);
// #[pymethods]
// impl PyDualDVec64 {
//     #[new]
//     pub fn new(re: f64) -> Self {
//         Self(DualDVec64::from_re(re))
//     }
// }

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyFactor {
    pub func: Py<PyAny>,
}

#[pymethods]
impl PyFactor {
    #[new]
    pub fn new(f: Py<PyAny>) -> Self {
        PyFactor { func: f }
    }
}

impl Factor for PyFactor {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        let residual_py = Python::with_gil(|py| -> PyResult<Vec<PyDual64Dyn>> {
            let py_params: Vec<Py<PyAny>> = params
                .iter()
                .map(|param| {
                    param
                        .data
                        .as_vec()
                        .iter()
                        .map(|x| PyDual64Dyn::from(x.clone()))
                        .collect::<Vec<PyDual64Dyn>>()
                })
                .map(|x| x.into_py(py))
                .collect();
            let args = PyTuple::new(py, py_params);
            let result = self.func.call1(py, args);
            let residual_py = result.unwrap().extract::<Vec<PyDual64Dyn>>(py);
            residual_py
        });
        let residual_py: Vec<num_dual::DualDVec64> = residual_py
            .unwrap()
            .iter()
            .map(|x| <PyDual64Dyn as Clone>::clone(x).into())
            .collect();
        na::DVector::from_vec(residual_py)
    }
}
