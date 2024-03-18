use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::loss_functions::*;

#[pymethods]
impl HuberLoss {
    #[new]
    pub fn new() -> Self {
        HuberLoss {}
    }
}
