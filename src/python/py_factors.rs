use num_dual::{python::PyDual64Dyn, Dual64, DualDVec64};
use numpy::PyReadonlyArray1;
use numpy::{PyArray, PyReadonlyArrayDyn};

use nalgebra as na;
use pyo3::types::PyTuple;
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

    pub fn call_func(&self, py: Python) -> PyResult<()> {
        // self.func.call1(([],));
        // let a: na::DVector<f64> = x.as_matrix().column(0).into();
        let p0 = PyDual64Dyn::from(num_dual::DualDVec64::new(
            1.0,
            num_dual::Derivative::some(na::dvector![1.0, 0.0]),
        ));
        let p1 = PyDual64Dyn::from(num_dual::DualDVec64::new(
            1.0,
            num_dual::Derivative::some(na::dvector![1.0, 0.0]),
        ));
        let result = self.func.call1(py, ([p0, p1],));
        // self.func.g(py)
        println!("{}", result.unwrap());

        println!("fff");
        Ok(())
    }
}

impl Factor for PyFactor {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        na::dvector![]
    }
}

#[pyfunction]
pub fn first_derivative_test(f: &PyAny, x: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let a: na::DVector<f64> = x.as_matrix().column(0).into();
    let p0 = PyDual64Dyn::from(num_dual::DualDVec64::new(
        a[0],
        num_dual::Derivative::some(na::dvector![1.0, 0.0]),
    ));
    let p1 = PyDual64Dyn::from(num_dual::DualDVec64::new(
        a[1],
        num_dual::Derivative::some(na::dvector![0.0, 1.0]),
    ));
    // let a = p.get_first_derivative().unwrap();
    // println!("{:?}", a);
    let rr = f.call1(([p0, p1],)).unwrap();
    println!("{}", rr);
    // let g = |x| {
    //     println!("Aa");
    //     let res = f.call1((Dua::from(x),))?;
    //     if let Ok(res) = res.extract::<PyDual64>() {
    //         println!("abc {:?}", res.0);
    //         Ok(res.0)
    //     } else {
    //         println!("eeeeeee");
    //         Err(PyErr::new::<PyTypeError, _>(
    //             "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
    //                 .to_string(),
    //         ))
    //     }
    // };
    // let _ = g(PyDual64::new(1.0, 0.0));
    Ok((1.0, 2.0))
    // try_first_derivative(g, x)
}
