use nalgebra as na;
use num_dual::python::PyDual64Dyn;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::factors::*;

#[derive(Debug, Clone)]
#[pyclass(name = "BetweenFactorSE2")]
pub struct PyBetweenFactorSE2(pub BetweenFactorSE2);

#[pymethods]
impl PyBetweenFactorSE2 {
    #[new]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        PyBetweenFactorSE2(BetweenFactorSE2 {
            dx: x,
            dy: y,
            dtheta: theta,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(name = "PriorFactor")]
pub struct PyPriorFactor(pub PriorFactor);

#[pymethods]
impl PyPriorFactor {
    #[new]
    pub fn new(x: PyReadonlyArray1<f64>) -> Self {
        PyPriorFactor(PriorFactor {
            v: x.as_matrix().column(0).into(),
        })
    }
}

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
        params: &[na::DVector<num_dual::DualDVec64>],
    ) -> na::DVector<num_dual::DualDVec64> {
        // this can not be called with par_iter
        // TODO find a way to deal with multi threading
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
            let args = PyTuple::new_bound(py, py_params);
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
