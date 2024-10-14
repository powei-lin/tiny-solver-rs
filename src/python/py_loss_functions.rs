use pyo3::prelude::*;

use crate::loss_functions::*;

#[pyclass(name = "HuberLoss")]
#[derive(Clone)]
pub struct PyHuberLoss(pub HuberLoss);

#[pymethods]
impl PyHuberLoss {
    #[new]
    #[pyo3(signature=(scale=1.0))]
    pub fn new_py(scale: f64) -> Self {
        PyHuberLoss(HuberLoss::new(scale))
    }
}
