use pyo3::prelude::*;

use crate::loss_functions::*;

#[pymethods]
impl HuberLoss {
    #[new]
    #[pyo3(signature=(scale=1.0))]
    pub fn new_py(scale: f64) -> Self {
        HuberLoss::new(scale)
    }
}
