use num_dual::{python::PyDual64Dyn, Dual64, DualDVec64};
use numpy::{PyArray, PyReadonlyArrayDyn};
use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use std::collections::HashMap;

use nalgebra as na;
use pyo3::types::{PyDict, PyList, PyTuple};

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
    // pub kwarg_names: Vec<String>
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
        let p0 = num_dual::DualDVec64::new(1.0, num_dual::Derivative::some(na::dvector![1.0, 0.0]));
        let p1 = num_dual::DualDVec64::new(1.0, num_dual::Derivative::some(na::dvector![1.0, 0.0]));
        let params = vec![na::dvector![p0.clone(), p1.clone()], na::dvector![p0, p1]];
        // let pp: HashMap::<String, Vec<PyDual64Dyn>> = params.iter().map(|param| param
        //     .data
        //     .as_vec()
        //     .iter()
        //     .map(|x| PyDual64Dyn::from(x.clone()))
        //     .collect::<Vec<PyDual64Dyn>>()).zip(self.kwarg_names.iter()).map(|(v, k)|(k.to_string(), v) ).collect();
        // let pp1: Vec<(String, Py<PyAny>)> = params.iter().map(|param| param
        //     .data
        //     .as_vec()
        //     .iter()
        //     .map(|x| PyDual64Dyn::from(x.clone()))
        //     .collect::<Vec<PyDual64Dyn>>()).zip(self.kwarg_names.iter()).map(|(v, k)|(k.to_string(), v.into_py(py)) ).collect();
        let pp1: Vec<Py<PyAny>> = params
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
        // let pp1: Vec<PyDual64Dyn> = params[0]
        //     .data
        //     .as_vec()
        //     .iter()
        //     .map(|x| PyDual64Dyn::from(x.clone()))
        //     .collect::<Vec<PyDual64Dyn>>();
        // let ss = PyList::new(py, pp1);
        let args = PyTuple::new(py, pp1);

        // let mut pp = PyDict::from_sequence(py, pp1).unwrap();
        // pp.set_item("x", pp1.into_py(py));
        // let a = PyList::append(py, [pp1[0]]);
        // let args = PyTuple::new(py, &pp);
        // let pp = params[0].map(PyDual64Dyn::from);
        // let py_params = params.iter().map(|x| x.map(PyDual64Dyn::from).to_owned().to_pyarray(py)).collect();
        // let tt = (pp1, );
        // println!("{:?}", tt);
        let result = self.func.call1(py, args);
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
        Python::with_gil(|py| {
            self.func.call1(py, ());
        });
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
