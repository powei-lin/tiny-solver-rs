use pyo3::prelude::*;

use crate::loss_functions::*;

#[pymethods]
impl HuberLoss {
    #[new]
    pub fn new() -> Self {
        HuberLoss {}
    }
}
