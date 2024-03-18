use num_dual::{try_first_derivative, Dual64};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyTypeError, prelude::*};

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

#[pyclass(name = "Dual64")]
#[derive(Clone, Debug)]
pub struct PyDual64(Dual64);
#[pymethods]
impl PyDual64 {
    #[new]
    pub fn new(re: f64, eps: f64) -> Self {
        Self(Dual64::new(re, eps))
    }

    #[getter]
    pub fn get_first_derivative(&self) -> f64 {
        self.0.eps
    }
}
#[pyfunction]
pub fn first_derivative(f: &PyAny, x: f64) -> PyResult<(f64, f64)> {
    let g = |x| {
        let res = f.call1((PyDual64::from(x),))?;
        if let Ok(res) = res.extract::<PyDual64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                    .to_string(),
            ))
        }
    };
    Ok((1.0, 2.0))
    // try_first_derivative(g, x)
}
